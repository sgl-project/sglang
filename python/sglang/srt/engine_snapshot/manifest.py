# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Artifact contract for initialized engine snapshots.

The manifest is the only thing a restore trusts before touching CRIU: it
records what was captured, the identity that must still hold, and the hash of
every file the artifact carries.

Portions of the path-validation and rollback helpers are derived from vLLM's
initialized-snapshot work (Apache-2.0).
"""

import ast
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import msgspec

from sglang.srt.engine_snapshot.errors import (
    SnapshotCompatibilityError,
    SnapshotRuntimeFailure,
    SnapshotSecurityError,
    SnapshotUsageError,
)

# The artifact format this tree writes and reads. This is the first version of
# the format, so there is nothing to negotiate or migrate yet.
MANIFEST_FORMAT = 1
MANIFEST_NAME = "manifest.json"
FAILURE_NAME = "failure.json"
LOCK_NAME = ".lock"

# Environment variables that participate in the identity contract. Values are
# stored as digests: the manifest travels between containers, and a plaintext
# path or token adds nothing a hash does not already prove.
IDENTITY_ENV_PREFIXES = (
    "SGLANG_",
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "TRITON_",
    "FLASHINFER_",
    "GLOO_",
)
# Names carrying credentials are neither hashed nor compared: they differ
# between operators by design and must not gate a restore.
_REDACTED_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
# Set per artifact for the create child only; never part of the contract.
_TRANSPORT_ENV_NAMES = ("SGLANG_SNAPSHOT_DIR",)


class SnapshotFile(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """A file carried inside the artifact, with its content hash."""

    path: str
    sha256: str


class SnapshotModelFile(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    path: str
    size: int
    mtime_ns: int


class SnapshotEnvironmentEntry(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    name: str
    digest: str


class SnapshotCanary(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """One greedy token sampled at the initialization boundary."""

    prompt: str
    token_id: int


class SnapshotIdentity(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Everything a restore must still match before CRIU is allowed to run."""

    sglang_code: str
    model: list[SnapshotModelFile]
    host_id: str
    gpu_name: str
    gpu_uuid: str
    driver_version: str
    python: str
    kernel: str
    libc: list[str]
    torch: str
    cuda_runtime: str
    sglang_kernel: str
    transformers: str
    triton: str
    criu: str
    cuda_checkpoint: str
    environment: list[SnapshotEnvironmentEntry]


class SnapshotManifest(
    msgspec.Struct, forbid_unknown_fields=True, frozen=True, kw_only=True
):
    format: int
    artifact_path: str
    created_at: str
    artifact_bytes: int
    model_path: str
    host: str
    port: int
    identity: SnapshotIdentity
    root_pid: int
    pids: list[int]
    cuda_pids: list[int]
    # The engine's captured standard output and error, as CRIU resource ids;
    # restore maps them onto the restoring caller's streams.
    stdio: list[str]
    files: list[SnapshotFile]
    dev_shm: list[SnapshotFile]
    canary: SnapshotCanary


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _cuda_runtime():
    """CUDA version torch was built against, parsed without importing torch.

    The restore CLI must stay import-light, so the version file is read as
    source rather than through ``torch.version.cuda``.
    """
    spec = importlib.util.find_spec("torch")
    locations = spec and spec.submodule_search_locations
    if not locations:
        return "missing"
    version_file = Path(next(iter(locations))) / "version.py"
    try:
        tree = ast.parse(version_file.read_text())
    except (OSError, SyntaxError) as error:
        raise SnapshotRuntimeFailure(
            f"cannot read the CUDA version from {version_file}"
        ) from error
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "cuda" and value is not None:
            try:
                cuda = ast.literal_eval(value)
            except ValueError as error:
                raise SnapshotRuntimeFailure(
                    f"cannot evaluate torch.version.cuda in {version_file}"
                ) from error
            if not isinstance(cuda, str) or not cuda:
                raise SnapshotRuntimeFailure(
                    f"torch reports no built CUDA version in {version_file}"
                )
            return cuda
    raise SnapshotRuntimeFailure(f"no CUDA version found in {version_file}")


def _host_id():
    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            return Path(candidate).read_text().strip()
        except OSError:
            continue
    return ""


def _code_digest(package):
    """Digest every shipped module.

    The artifact holds process memory compiled from exactly this source, so a
    changed Python file invalidates it. Compiled extensions are covered by the
    package versions in the same identity.
    """
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_files(model_path):
    model = Path(model_path)
    files = [
        SnapshotModelFile(
            str(p.relative_to(model)), p.stat().st_size, p.stat().st_mtime_ns
        )
        for p in sorted(model.rglob("*"))
        if p.is_file()
    ]
    if not files:
        raise SnapshotCompatibilityError(f"model files are missing: {model}")
    return files


def _environment_entries(environment):
    return [
        SnapshotEnvironmentEntry(
            name, hashlib.sha256(name.encode() + b"\0" + value.encode()).hexdigest()
        )
        for name, value in sorted(environment.items())
        if name.startswith(IDENTITY_ENV_PREFIXES)
        and name not in _TRANSPORT_ENV_NAMES
        and not any(marker in name for marker in _REDACTED_ENV_MARKERS)
    ]


def build_identity(
    model_path,
    gpu_uuid,
    gpu_name,
    driver_version,
    criu_version,
    cuda_checkpoint_sha256,
    environment=None,
):
    """Describe the runtime a captured engine may be restored into."""
    package = Path(__file__).resolve().parents[2]
    return SnapshotIdentity(
        sglang_code=_code_digest(package),
        model=_model_files(model_path),
        host_id=_host_id(),
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        driver_version=driver_version,
        python=platform.python_version(),
        kernel=platform.release(),
        libc=list(platform.libc_ver()),
        torch=_package_version("torch"),
        cuda_runtime=_cuda_runtime(),
        sglang_kernel=_package_version("sglang-kernel"),
        transformers=_package_version("transformers"),
        triton=_package_version("triton"),
        criu=criu_version,
        cuda_checkpoint=cuda_checkpoint_sha256,
        environment=_environment_entries(
            os.environ if environment is None else environment
        ),
    )


def validate_identity(expected, actual):
    """Require an exact match and name every field that differs."""
    expected_values = msgspec.structs.asdict(expected)
    actual_values = msgspec.structs.asdict(actual)
    differing = [
        name for name in expected_values if expected_values[name] != actual_values[name]
    ]
    if differing:
        raise SnapshotCompatibilityError(
            f"snapshot identity mismatch: {', '.join(differing)}"
        )


def resolve_artifact_path(artifact_path, relative):
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(p in ("..", ".") for p in path.parts)
    ):
        raise SnapshotSecurityError(f"Invalid snapshot file path: {relative}")
    target = artifact_path / path
    if any(
        parent.is_symlink()
        for parent in (target, *target.parents)
        if parent != artifact_path.parent
    ):
        raise SnapshotSecurityError(f"Symlink in snapshot file path: {target}")
    return target


def validate_artifact_path(artifact_path):
    """Reject an artifact path another user could replace or read."""
    for path in (artifact_path, *artifact_path.parents):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise SnapshotSecurityError(
                f"Snapshot path must contain real directories: {path}"
            )
        if metadata.st_uid not in (0, os.geteuid()):
            raise SnapshotSecurityError(f"Snapshot path has an untrusted owner: {path}")
        sticky_root = metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX
        if metadata.st_mode & 0o022 and not sticky_root:
            raise SnapshotSecurityError(
                f"Snapshot path has a writable ancestor: {path}"
            )
    if artifact_path.exists():
        metadata = artifact_path.stat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise SnapshotSecurityError("Snapshot directory must be owner-only (0700)")


@contextmanager
def locked(artifact_path):
    """Serialize create/restore on one artifact for this host."""
    validate_artifact_path(artifact_path)
    fd = os.open(
        artifact_path / LOCK_NAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SnapshotUsageError(
                f"another snapshot operation is already running on {artifact_path}"
            ) from error
        yield


def write_json_atomic(path, data, overwrite=False):
    """Publish a private JSON file, optionally refusing to replace one."""
    if not overwrite and (path.exists() or path.is_symlink()):
        raise SnapshotUsageError(f"refusing to replace an existing file: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(data, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_manifest(artifact_path, manifest):
    """Write ``manifest.json`` last: its presence means the artifact is complete."""
    write_json_atomic(
        artifact_path / MANIFEST_NAME, msgspec.to_builtins(manifest), overwrite=False
    )


def record_failure(artifact_path, error, secondary=()):
    """Best-effort note explaining why an artifact has no manifest.

    The artifact directory is kept after a failed create so the logs next to it
    stay available; this file is what makes that directory self-explaining.
    """
    payload = {"error": str(error) or type(error).__name__}
    if secondary:
        payload["cleanup"] = list(secondary)
    try:
        write_json_atomic(artifact_path / FAILURE_NAME, payload, overwrite=True)
    except OSError:
        pass


def load_manifest(artifact_path, identity=None):
    """Read and validate the artifact contract.

    ``identity`` is the runtime identity to compare against; ``None`` skips the
    comparison so callers such as ``inspect`` can report the difference instead
    of failing on it.
    """
    try:
        manifest = msgspec.json.decode(
            (artifact_path / MANIFEST_NAME).read_bytes(), type=SnapshotManifest
        )
    except FileNotFoundError as error:
        raise SnapshotCompatibilityError(
            f"snapshot has no {MANIFEST_NAME}: {artifact_path}"
        ) from error
    except msgspec.ValidationError as error:
        raise SnapshotCompatibilityError(f"invalid {MANIFEST_NAME}: {error}") from error
    if manifest.format != MANIFEST_FORMAT or manifest.artifact_path != str(
        artifact_path
    ):
        raise SnapshotCompatibilityError(
            "Unsupported snapshot format or changed artifact mount path"
        )
    if (
        manifest.root_pid <= 1
        or manifest.root_pid not in manifest.pids
        or not manifest.cuda_pids
    ):
        raise SnapshotCompatibilityError("Invalid snapshot process tree")
    if len(set(manifest.pids)) != len(manifest.pids) or any(
        pid <= 1 for pid in manifest.pids
    ):
        raise SnapshotCompatibilityError("Invalid snapshot process list")
    if any(pid not in manifest.pids for pid in manifest.cuda_pids):
        raise SnapshotCompatibilityError("Invalid snapshot CUDA process list")
    if manifest.canary.token_id < 0:
        raise SnapshotCompatibilityError("Invalid snapshot canary")
    if identity is not None:
        validate_identity(manifest.identity, identity)
    return manifest


def created_at_now():
    return datetime.now(timezone.utc).isoformat()


def artifact_bytes(artifact_path):
    return sum(
        path.stat().st_size for path in artifact_path.rglob("*") if path.is_file()
    )
