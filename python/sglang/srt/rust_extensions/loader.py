"""Load bundled Rust extensions or build them from an SGLang source tree."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Literal, Mapping

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

RustBuildMode = Literal["auto", "never", "force"]

_FINGERPRINT_VERSION = 1
_IGNORED_SOURCE_DIRECTORIES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", "__pycache__", "target"}
)
_BUILD_ENVIRONMENT_VARIABLES = (
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTFLAGS",
    "RUSTFLAGS",
)

_RUST_WORKSPACE = Path(__file__).resolve().parents[4] / "rust"


@dataclass(frozen=True)
class _CrateSpec:
    """One extension crate, discovered from its Cargo manifest."""

    package: str
    library: str
    python_module: str
    manifest: Path
    workspace: Path
    features: tuple[str, ...]


@dataclass(frozen=True)
class _BuildContext:
    source_digest: str
    fingerprint: str
    target_fingerprint: str


def load_rust_extension(
    python_module: str,
    *,
    mode: RustBuildMode | None = None,
    cache_dir: Path | None = None,
    workspace: Path | None = None,
    additional_features: tuple[str, ...] = (),
    extension_module: str | None = None,
    build_environment: Mapping[str, str] | None = None,
    build_fingerprint: Mapping[str, object] | None = None,
) -> ModuleType:
    """Import a PyO3 extension, compiling it locally when permitted and needed.

    The crate is discovered from the workspace under ``rust/``: the one whose
    Cargo manifest declares ``[package.metadata.sglang] python-module`` equal
    to ``python_module`` (the same metadata setup.py uses for wheel builds), so
    new crates need no registration here.

    ``auto`` prefers a module bundled in an installed wheel. In a source tree,
    it ignores unverified in-package artifacts and uses the fingerprinted cache
    before invoking Cargo. ``never`` explicitly trusts a bundled module, then
    permits the cache but never invokes Cargo. ``force`` rebuilds from source.
    A same-name feature variant is always sourced from the fingerprinted cache.
    A distinctly named variant may be supplied by test infrastructure and is
    otherwise built into that cache after its bundled import misses.
    ``mode`` defaults to ``SGLANG_RUST_BUILD_MODE``.
    """
    if mode is None:
        mode = envs.SGLANG_RUST_BUILD_MODE.get()
    if mode not in ("auto", "never", "force"):
        raise ValueError(
            f"invalid Rust extension build mode {mode!r}; expected auto, never, or force"
        )

    load_module = extension_module or python_module
    same_name_feature_variant = (
        bool(additional_features) and load_module == python_module
    )
    if loaded := sys.modules.get(load_module):
        if mode != "force":
            return loaded
        raise RuntimeError(
            f"cannot force-build {load_module} after it has been imported; "
            "start a new Python process"
        )
    if workspace is None:
        workspace = _RUST_WORKSPACE
    source_checkout = (Path(workspace) / "Cargo.toml").is_file()
    trust_bundled = mode == "never" or not source_checkout
    if mode != "force" and trust_bundled and not same_name_feature_variant:
        module = _import_bundled_extension(load_module)
        if module is not None:
            return module

    crate = _discover_crate(workspace, python_module)
    features = tuple(dict.fromkeys((*crate.features, *additional_features)))
    context = _build_context(
        crate,
        features=features,
        build_fingerprint=build_fingerprint,
        extension_module=load_module,
    )
    cache_root = _cache_root(cache_dir)
    extension_path = _cached_extension_path(
        cache_root, crate, context.fingerprint, load_module
    )
    lock_path = (
        cache_root / "locks" / f"{crate.package}-{context.target_fingerprint}.lock"
    )

    with _filesystem_lock(lock_path):
        if mode != "force" and extension_path.is_file():
            return _load_extension_from_path(load_module, extension_path)

        if mode == "never":
            raise ModuleNotFoundError(
                f"{crate.python_module} is not bundled or cached, and Rust extension "
                "build mode is 'never'",
                name=crate.python_module,
            )

        target_dir = cache_root / "targets" / context.target_fingerprint
        artifact = _cargo_build(
            crate,
            target_dir,
            features=features,
            build_environment=build_environment,
        )
        if _source_digest(crate.workspace) != context.source_digest:
            raise RuntimeError(
                f"Rust sources under {crate.workspace} changed during the build; "
                "the result was not cached"
            )
        _stage_atomically(artifact, extension_path)
        return _load_extension_from_path(load_module, extension_path)


def _import_bundled_extension(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def _discover_crate(workspace: Path, python_module: str) -> _CrateSpec:
    workspace = Path(workspace).resolve()
    workspace_manifest = workspace / "Cargo.toml"
    if not workspace_manifest.is_file():
        raise FileNotFoundError(
            f"Rust workspace for {python_module} was not found at {workspace}"
        )
    matches: list[_CrateSpec] = []
    declared_modules: list[str] = []
    for manifest in _source_files(workspace):
        if manifest.name != "Cargo.toml":
            continue
        with manifest.open("rb") as file:
            document = tomllib.load(file)
        package = document.get("package")
        if not isinstance(package, dict):
            continue
        sglang_metadata = package.get("metadata", {}).get("sglang", {})
        declared_module = sglang_metadata.get("python-module")
        if declared_module is None:
            continue
        declared_modules.append(declared_module)
        if declared_module != python_module:
            continue

        package_name = package.get("name")
        library = document.get("lib", {}).get("name")
        if not package_name or not library:
            raise ValueError(
                f"{manifest} declares python-module {python_module!r} but must "
                "also set `package.name` and `lib.name`"
            )
        crate_workspace = manifest.parent if "workspace" in document else workspace
        lockfile = crate_workspace / "Cargo.lock"
        if not lockfile.is_file():
            raise FileNotFoundError(
                f"{lockfile} is required for reproducible `cargo build --locked` builds"
            )
        matches.append(
            _CrateSpec(
                package=package_name,
                library=library,
                python_module=python_module,
                manifest=manifest,
                workspace=crate_workspace,
                features=tuple(sglang_metadata.get("features", ())),
            )
        )

    if not matches:
        raise ModuleNotFoundError(
            f"no Cargo package under {workspace} declares "
            f'`[package.metadata.sglang] python-module = "{python_module}"`; '
            f"declared modules: {sorted(declared_modules)}",
            name=python_module,
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple Cargo packages under {workspace} declare python module "
            f"{python_module!r}: {sorted(crate.package for crate in matches)}"
        )
    return matches[0]


def _build_context(
    crate: _CrateSpec,
    *,
    features: tuple[str, ...] | None = None,
    build_fingerprint: Mapping[str, object] | None = None,
    extension_module: str | None = None,
) -> _BuildContext:
    if features is None:
        features = crate.features
    if extension_module is None:
        extension_module = crate.python_module
    source_digest = _source_digest(crate.workspace)
    toolchain = {
        "cargo": _command_version(
            "cargo", "--version", "--verbose", cwd=crate.workspace
        ),
        "rustc": _command_version("rustc", "-vV", cwd=crate.workspace),
    }
    python_abi = {
        "cache_tag": sys.implementation.cache_tag,
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "platform": sysconfig.get_platform(),
        "pointer_bits": struct.calcsize("P") * 8,
        "soabi": sysconfig.get_config_var("SOABI"),
        "version": list(sys.version_info[:3]),
    }
    build_environment = {
        name: os.environ.get(name) for name in _BUILD_ENVIRONMENT_VARIABLES
    }
    target_inputs = {
        "build_environment": build_environment,
        "extension_build": dict(build_fingerprint or {}),
        "python_abi": python_abi,
        "toolchain": toolchain,
    }
    target_fingerprint = _json_digest(target_inputs)[:24]
    fingerprint = _json_digest(
        {
            "fingerprint_version": _FINGERPRINT_VERSION,
            "package": crate.package,
            "library": crate.library,
            "python_module": crate.python_module,
            "extension_module": extension_module,
            "features": features,
            "source_digest": source_digest,
            **target_inputs,
        }
    )
    return _BuildContext(
        source_digest=source_digest,
        fingerprint=fingerprint,
        target_fingerprint=target_fingerprint,
    )


def _source_digest(workspace: Path) -> str:
    digest = hashlib.sha256()
    for path in _source_files(workspace):
        relative_path = path.relative_to(workspace).as_posix().encode()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        if path.is_symlink():
            contents = os.readlink(path).encode()
        else:
            contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _source_files(workspace: Path) -> Iterator[Path]:
    for root, directories, filenames in os.walk(workspace):
        directories[:] = sorted(
            name for name in directories if name not in _IGNORED_SOURCE_DIRECTORIES
        )
        root_path = Path(root)
        for filename in sorted(filenames):
            yield root_path / filename


def _command_version(command: str, *arguments: str, cwd: Path) -> str:
    try:
        result = subprocess.run(
            [command, *arguments],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"failed to query the Rust toolchain with `{command} {' '.join(arguments)}`"
        ) from exc
    return result.stdout.strip()


def _json_digest(value: object) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _cache_root(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    sglang_cache = envs.SGLANG_CACHE_DIR.get()
    return Path(sglang_cache).expanduser().resolve() / "rust_extensions"


def _cached_extension_path(
    cache_root: Path,
    crate: _CrateSpec,
    fingerprint: str,
    extension_module: str | None = None,
) -> Path:
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an EXT_SUFFIX for native extensions")
    module_leaf = (extension_module or crate.python_module).rsplit(".", 1)[-1]
    return (
        cache_root
        / "artifacts"
        / crate.package
        / fingerprint
        / (module_leaf + extension_suffix)
    )


@contextmanager
def _filesystem_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _cargo_build(
    crate: _CrateSpec,
    target_dir: Path,
    *,
    features: tuple[str, ...] | None = None,
    build_environment: Mapping[str, str] | None = None,
) -> Path:
    if features is None:
        features = crate.features
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--package",
        crate.package,
    ]
    if features:
        command.extend(("--features", ",".join(features)))

    environment = dict(os.environ if build_environment is None else build_environment)
    environment["CARGO_TARGET_DIR"] = os.fspath(target_dir)
    environment["PYO3_PYTHON"] = sys.executable
    logger.info("Building %s with `%s`", crate.python_module, " ".join(command))
    try:
        subprocess.run(command, cwd=crate.workspace, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"failed to build {crate.python_module} with Cargo") from exc

    release_dir = target_dir / "release"
    if target := environment.get("CARGO_BUILD_TARGET"):
        release_dir = target_dir / target / "release"
    artifact = release_dir / _cargo_library_filename(crate.library)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Cargo completed but did not produce the expected artifact {artifact}"
        )
    return artifact


def _cargo_library_filename(library: str) -> str:
    if sys.platform == "win32":
        return f"{library}.dll"
    if sys.platform == "darwin":
        return f"lib{library}.dylib"
    return f"lib{library}.so"


def _stage_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with (
            os.fdopen(descriptor, "wb") as destination_file,
            source.open("rb") as source_file,
        ):
            shutil.copyfileobj(source_file, destination_file)
            destination_file.flush()
            os.fsync(destination_file.fileno())
        temporary_path.chmod(0o755)
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_extension_from_path(module_name: str, path: Path) -> ModuleType:
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(
            f"could not create an import spec for {module_name} at {path}"
        )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
