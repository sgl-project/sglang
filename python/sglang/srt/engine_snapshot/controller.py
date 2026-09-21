# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Transactional orchestration of snapshot create/restore/inspect.

The controller is the only place that knows the order of the steps; the
process, CRIU, cuda-checkpoint and artifact-file operations belong to
:class:`SnapshotRuntime`.
"""

import os
from pathlib import Path

import msgspec

from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import (
    SnapshotCompatibilityError,
    SnapshotError,
    SnapshotRuntimeFailure,
    SnapshotUsageError,
    error_detail,
)
from sglang.srt.engine_snapshot.manifest import (
    MANIFEST_FORMAT,
    SnapshotManifest,
    artifact_bytes,
    created_at_now,
    load_manifest,
    locked,
    publish_manifest,
    record_failure,
    validate_artifact_path,
    validate_identity,
)
from sglang.srt.engine_snapshot.runtime import SnapshotRuntime
from sglang.srt.engine_snapshot.startup import (
    validate_listen_host,
    validate_server_args,
)
from sglang.srt.environ import envs, third_party_cache_defaults


class RestoreOutcome(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """Where the restored engine ended up listening."""

    root_pid: int
    host: str
    port: int


_ARTIFACT_SUBDIRS = (
    "control",
    "images",
    "work",
    "files",
    # /dev/shm objects replayed on restore: files the engine held open there
    # and the CRIU link_remap ghosts produced by the dump.
    "dev_shm",
    "runtime/cache",
    "runtime/tmp",
)

# Cache directories the captured engine must keep inside the artifact, beyond
# the ones the repository's own redirect table derives from SGLANG_CACHE_DIR.
# Each name is the environment variable its reader honours.
_EXTRA_CACHE_DIRS = {
    "DG_JIT_CACHE_DIR": "cache/deep_gemm",  # deep_gemm JIT (native side)
    "SGLANG_DG_CACHE_DIR": "cache/deep_gemm",  # sglang's own default for it
    "TILELANG_CACHE_DIR": "cache/tilelang",  # tilelang
    "HUMMING_CACHE_DIR": "cache/humming",  # humming-kernels launcher/kernel cache
    "HUMMING_TMP_DIR": "tmp/humming",  # humming-kernels build scratch
    "CUTE_DSL_CACHE_DIR": "cache/cutlass-dsl",  # vendored CuTeDSL generated IR
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "cache/flash-attn-cute",  # FA cute DSL
}

# Every cache variable the controller owns for its child. Importing sglang in
# the controller's own process already redirects some of these (and writes
# DG_JIT_CACHE_DIR), so they are dropped from the inherited environment instead
# of being passed through: the captured engine may only write inside the
# artifact, whatever the operator's shell had set.
_MANAGED_CACHE_VARS = (
    "TMPDIR",
    "SGLANG_CACHE_DIR",
    "SGLANG_JIT_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    *_EXTRA_CACHE_DIRS,
)


def _child_environment(launch_environment, artifact_path):
    """Environment for the captured engine, with its state inside the artifact.

    Triton, Inductor, CUDA and FlashInfer roots come from the repository's own
    ``third_party_cache_defaults`` so this cannot drift from what importing
    ``sglang`` redirects; the JIT caches that table does not cover are listed in
    ``_EXTRA_CACHE_DIRS``.

    The values recorded in the manifest identity come from the controller's own
    environment, not from this one.
    """
    cache_root = artifact_path / "runtime/cache"
    environment = {
        name: value
        for name, value in launch_environment.items()
        if name not in _MANAGED_CACHE_VARS
    }
    environment.update(
        SGLANG_SNAPSHOT_DIR=str(artifact_path),
        USE_LIBUV="0",
        GLOO_SOCKET_IFNAME="lo",
        TMPDIR=str(artifact_path / "runtime/tmp"),
        SGLANG_CACHE_DIR=str(cache_root),
        SGLANG_JIT_CACHE_DIR=str(cache_root / "jit"),
    )
    # Resolved against the artifact path rather than the controller's
    # environment: the helper reads SGLANG_CACHE_DIR when it is called.
    with envs.SGLANG_CACHE_DIR.override(str(cache_root)):
        environment.update(third_party_cache_defaults())
    for name, suffix in _EXTRA_CACHE_DIRS.items():
        environment[name] = str(artifact_path / "runtime" / suffix)
    return environment


def _assemble_manifest(
    artifact_path, engine_info, inventory, launch_environment, runtime, stdio
):
    scheduler_info = control.read_json(
        artifact_path / control.CONTROL_DIRNAME,
        control.SCHEDULER,
        control.SchedulerInfo,
    )
    return SnapshotManifest(
        format=MANIFEST_FORMAT,
        artifact_path=str(artifact_path),
        created_at=created_at_now(),
        artifact_bytes=artifact_bytes(artifact_path),
        model_path=engine_info.model_path,
        host=engine_info.host,
        port=engine_info.port,
        identity=runtime.current_identity(
            engine_info.model_path, engine_info.gpu_uuid, launch_environment
        ),
        root_pid=inventory.root_pid,
        pids=inventory.pids,
        cuda_pids=inventory.cuda_pids,
        stdio=stdio,
        files=inventory.files,
        dev_shm=inventory.dev_shm,
        canary=scheduler_info.canary,
    )


def create_snapshot(artifact, server_argv, timeout=600, runtime=None):
    """Initialize an engine, checkpoint it and publish the artifact."""
    runtime = runtime or SnapshotRuntime()
    launch_environment = dict(os.environ)
    validate_server_args(server_argv)
    artifact_path = Path(os.path.abspath(artifact))
    runtime.preflight("create", artifact_path)
    stdio = runtime.stdio_resources()
    try:
        artifact_path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise SnapshotUsageError(
            f"snapshot artifact directory already exists: {artifact_path}"
        ) from error
    except FileNotFoundError as error:
        raise SnapshotUsageError(
            f"snapshot artifact parent directory does not exist: {artifact_path.parent}"
        ) from error
    with locked(artifact_path):
        for directory in _ARTIFACT_SUBDIRS:
            (artifact_path / directory).mkdir(parents=True, exist_ok=True)
        control_dir = artifact_path / control.CONTROL_DIRNAME
        remaps_before = set(runtime.SHM_DIR.glob("link_remap.*"))
        root_pid = None
        try:
            root_pid = runtime.launch_child(
                server_argv,
                _child_environment(launch_environment, artifact_path),
            )
            engine_info = runtime.wait_ready(artifact_path, root_pid, timeout)
            inventory = runtime.inventory(
                artifact_path, root_pid, engine_info.model_path, engine_info.gpu_uuid
            )
            inventory = runtime.dump(artifact_path, inventory, timeout)
            runtime.verify_dead(inventory)
            manifest = _assemble_manifest(
                artifact_path,
                engine_info,
                inventory,
                launch_environment,
                runtime,
                stdio,
            )
            runtime.flush_artifact(artifact_path)
            publish_manifest(artifact_path, manifest)
            return manifest
        except BaseException as error:
            control.write_abort(control_dir)
            failures = []
            runtime.abort_create(root_pid, artifact_path, failures)
            runtime.discard_new_link_remaps(remaps_before)
            record_failure(artifact_path, error, failures)
            raise


def restore_snapshot(artifact, timeout=300, runtime=None, host=None, port=None):
    """Restore an artifact on this host; returns the engine PID and address."""
    runtime = runtime or SnapshotRuntime()
    artifact_path = Path(os.path.abspath(artifact))
    with locked(artifact_path):
        runtime.preflight("restore", artifact_path)
        manifest = load_manifest(artifact_path)
        validate_identity(
            manifest.identity,
            runtime.current_identity(manifest.model_path, manifest.identity.gpu_uuid),
        )
        control_dir = artifact_path / control.CONTROL_DIRNAME
        if not control_dir.is_dir():
            raise SnapshotCompatibilityError(
                f"snapshot has no {control.CONTROL_DIRNAME} directory: {artifact_path}"
            )
        effective_host = manifest.host if host is None else host
        effective_port = manifest.port if port is None else port
        validate_listen_host(effective_host)
        control.clear_handshake(control_dir)
        runtime.verify_restorable(manifest, effective_host, effective_port)
        root_pid = None
        try:
            root_pid = runtime.restore(artifact_path, manifest, timeout)
            # CRIU restore takes tens of seconds; re-check the address it is
            # about to bind so a listener that appeared meanwhile is reported
            # here instead of as a readiness timeout.
            runtime.check_port_free(effective_host, effective_port)
            control.write_release(control_dir, host=host, port=port)
            runtime.wait_listener(
                artifact_path, manifest, effective_host, effective_port, timeout
            )
            _verify_resumed(manifest, control_dir)
            runtime.complete_restore(root_pid)
            return RestoreOutcome(root_pid, effective_host, effective_port)
        except BaseException as error:
            control.write_abort(control_dir)
            failures = []
            if root_pid is not None:
                runtime.stop_restored_tree(root_pid, artifact_path, failures)
            if failures:
                raise SnapshotRuntimeFailure(
                    f"Snapshot restore failed: {error_detail(error)}; "
                    + "; ".join(failures)
                ) from error
            raise


def _verify_resumed(manifest, control_dir):
    """Require the restored engine's own canary report to match the artifact.

    The engine compares the canary before it writes this marker; checking the
    value here as well means the controller judges the evidence instead of
    trusting that the comparison happened.
    """
    resumed = control.read_json(control_dir, control.RESUMED, control.ResumedInfo)
    if not manifest.canary.matches(resumed.token_id, resumed.logprob):
        raise SnapshotRuntimeFailure(
            "Snapshot canary mismatch: the restored engine sampled token "
            f"{resumed.token_id} at logprob {resumed.logprob:.6g}, but the "
            f"artifact recorded token {manifest.canary.token_id} at "
            f"{manifest.canary.logprob:.6g}"
        )


def inspect_snapshot(artifact, runtime=None):
    """Report what an artifact claims and whether this host could restore it.

    Read-only: nothing is written and no lock is taken, so an operator can look
    at an artifact while a create or restore is running.
    """
    runtime = runtime or SnapshotRuntime()
    artifact_path = Path(os.path.abspath(artifact))
    validate_artifact_path(artifact_path)
    manifest = load_manifest(artifact_path)
    checks = {
        "identity": _identity_check(runtime, manifest),
        "occupied_pids": runtime.occupied_pids(manifest.pids),
        "listen_address": _listen_check(runtime, manifest.host, manifest.port),
    }
    return manifest, checks


def _identity_check(runtime, manifest):
    try:
        validate_identity(
            manifest.identity,
            runtime.current_identity(manifest.model_path, manifest.identity.gpu_uuid),
        )
    except SnapshotError as error:
        return str(error)
    return "match"


def _listen_check(runtime, host, port):
    try:
        runtime.check_port_free(host, port)
    except SnapshotUsageError as error:
        return str(error)
    return "free"
