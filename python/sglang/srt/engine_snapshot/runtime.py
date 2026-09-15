# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Low-level runtime for initialized engine snapshots.

Groups the process, CRIU, cuda-checkpoint, artifact-file and /dev/shm
operations the controller needs, so the controller stays a thin, transactional
orchestrator. Everything CRIU reopens is carried inside the artifact, and the
runtime only signals processes it owns: children it started, trees it pinned,
or processes carrying the artifact's marker.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import msgspec
import psutil
import requests

from sglang.srt.engine_snapshot import control
from sglang.srt.engine_snapshot.errors import (
    SnapshotRuntimeFailure,
    SnapshotSecurityError,
    SnapshotUsageError,
    error_detail,
)
from sglang.srt.engine_snapshot.manifest import (
    SnapshotFile,
    build_identity,
    resolve_artifact_path,
    sha256_file,
    validate_artifact_path,
)
from sglang.srt.environ import envs

# Entry module of the captured engine; used to prove a pinned process tree is
# ours before any signal is sent to it.
_ENGINE_MODULE = "sglang.srt.engine_snapshot.startup"


def _cache_root_prefixes():
    """Default locations of the caches the controller redirects into the artifact.

    A cache file still referenced outside the artifact means a redirect was
    missed, and CRIU would reopen a path that can rotate between create and
    restore. Failing at create is the only cheap moment to notice.
    """
    home = os.path.expanduser("~").rstrip(os.sep)
    return (
        f"{home}/.cache/",
        f"{home}/.triton/",
        f"{home}/.humming/",
        f"{home}/.tilelang/",
        f"{home}/.nv/",
        f"{home}/.deep_gemm/",
        os.path.join(tempfile.gettempdir(), "torchinductor_"),
    )


_CACHE_ROOT_PREFIXES = _cache_root_prefixes()
_ALIVE_POLL_SECONDS = 0.05
_CLEANUP_TIMEOUT_SECONDS = 10.0


class EngineInventory(msgspec.Struct, forbid_unknown_fields=True, frozen=True):
    """What the host side knows about a captured engine."""

    root_pid: int
    pids: list[int]
    cuda_pids: list[int]
    gpu_uuid: str
    shared_paths: set[Path]
    files: list[SnapshotFile] = []
    dev_shm: list[SnapshotFile] = []


def _find_tool(name, configured, variable=None):
    path = shutil.which(configured)
    if path is None:
        hint = f" or set {variable} to its executable" if variable else ""
        raise SnapshotRuntimeFailure(f"Install {name}{hint}")
    return path


def _inside(path, artifact_path):
    prefix = f"{os.path.abspath(artifact_path).rstrip(os.sep)}{os.sep}"
    return path.startswith(prefix)


def _stdio_resource(fd):
    """The ``--inherit-fd`` resource id CRIU gives the target of ``fd``.

    Terminals, pipes and sockets are identified by device or inode numbers;
    everything CRIU reopens by path must be a file the process can still name.
    """
    metadata = os.fstat(fd)
    target = os.readlink(f"/proc/self/fd/{fd}")
    if os.isatty(fd):
        return f"tty[{metadata.st_rdev:x}:{metadata.st_dev:x}]"
    if target.startswith("pipe:["):
        return f"pipe:[{metadata.st_ino}]"
    if target.startswith("socket:["):
        return f"socket:[{metadata.st_ino}]"
    if target.startswith("/"):
        return target.removeprefix("/").removesuffix(" (deleted)")
    raise SnapshotUsageError(
        f"snapshot create requires a terminal, pipe or file standard stream: {target}"
    )


class SnapshotRuntime:
    """CRIU / cuda-checkpoint / process / artifact-file / /dev/shm operations."""

    SHM_DIR = Path("/dev/shm")
    CRIU_ARGS = [
        "--shell-job",
        "--file-locks",
        "--link-remap",
        "--tcp-established",
        "--network-lock",
        "iptables",
    ]

    def __init__(self):
        # Resolve external tools lazily so the file, /dev/shm and process
        # helpers stay usable without criu/cuda-checkpoint on PATH (unit tests);
        # preflight resolves all of them before an operation starts.
        self._criu = None
        self._cuda_checkpoint = None
        self._nvidia_smi = None
        self._children = {}
        self._restored = {}

    @property
    def criu(self):
        if self._criu is None:
            self._criu = _find_tool("criu", envs.SGLANG_CRIU.get(), "SGLANG_CRIU")
        return self._criu

    @property
    def cuda_checkpoint(self):
        if self._cuda_checkpoint is None:
            self._cuda_checkpoint = _find_tool(
                "cuda-checkpoint",
                envs.SGLANG_CUDA_CHECKPOINT.get(),
                "SGLANG_CUDA_CHECKPOINT",
            )
        return self._cuda_checkpoint

    @property
    def nvidia_smi(self):
        if self._nvidia_smi is None:
            self._nvidia_smi = _find_tool("nvidia-smi", "nvidia-smi")
        return self._nvidia_smi

    # ------------------------------------------------------------------ #
    # subprocess helpers
    # ------------------------------------------------------------------ #
    def _run(self, command, log_path, timeout, pass_fds=(), env=None):
        with log_path.open("ab") as output:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=output,
                pass_fds=pass_fds,
                start_new_session=True,
                env=env,
            )
            try:
                code = process.wait(timeout=timeout)
                if code:
                    raise SnapshotRuntimeFailure(
                        f"{command[0]} exited with {code}; see {log_path}"
                    )
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise

    def _capture(self, command, timeout=30):
        """Run a query command, turning every failure mode into one error type."""
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as error:
            raise SnapshotRuntimeFailure(f"{command[0]} is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise SnapshotRuntimeFailure(
                f"{command[0]} did not answer within {timeout:g}s"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "").strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise SnapshotRuntimeFailure(
                f"{command[0]} failed with {error.returncode}{suffix}"
            ) from error
        return result.stdout

    # ------------------------------------------------------------------ #
    # preflight and identity
    # ------------------------------------------------------------------ #
    def preflight(self, action, artifact_path):
        """Refuse to start when this host cannot finish the operation."""
        unavailable = []
        for name, resolve in (
            ("criu", lambda: self.criu),
            ("cuda-checkpoint", lambda: self.cuda_checkpoint),
            ("nvidia-smi", lambda: self.nvidia_smi),
        ):
            try:
                resolve()
            except SnapshotRuntimeFailure as error:
                unavailable.append(f"{name}: {error}")
        if unavailable:
            raise SnapshotRuntimeFailure(
                "snapshot dependencies are unavailable: " + "; ".join(unavailable)
            )
        if action == "restore" and not (
            callable(getattr(os, "pidfd_open", None))
            and callable(getattr(signal, "pidfd_send_signal", None))
        ):
            raise SnapshotRuntimeFailure(
                "snapshot restore requires Linux pidfd support"
            )
        validate_artifact_path(artifact_path)

    def current_identity(self, model_path, gpu_uuid, environment=None):
        gpu_name, driver_version = self._gpu_details(gpu_uuid)
        return build_identity(
            model_path=model_path,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            driver_version=driver_version,
            criu_version=self._tool_version([self.criu, "--version"]),
            cuda_checkpoint_sha256=sha256_file(Path(self.cuda_checkpoint)),
            environment=environment,
        )

    def _gpu_details(self, gpu_uuid):
        output = self._capture(
            [
                self.nvidia_smi,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
                f"--id={gpu_uuid}",
            ]
        ).strip()
        try:
            name, uuid, driver = (item.strip() for item in output.split(",", 2))
        except ValueError as error:
            raise SnapshotRuntimeFailure(
                f"nvidia-smi did not describe GPU {gpu_uuid}"
            ) from error
        if uuid != gpu_uuid:
            raise SnapshotRuntimeFailure(
                f"nvidia-smi reported GPU {uuid} for {gpu_uuid}"
            )
        return name, driver

    def _tool_version(self, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        output = (result.stdout or result.stderr).strip().splitlines()
        return output[0] if output else "unknown"

    # ------------------------------------------------------------------ #
    # process tree
    # ------------------------------------------------------------------ #
    def tree_pids(self, pid):
        try:
            children = psutil.Process(pid).children(recursive=True)
        except psutil.Error as error:
            raise SnapshotRuntimeFailure(
                f"cannot read the process tree of {pid}"
            ) from error
        return [pid] + [child.pid for child in children]

    def cuda_holders(self, pids):
        """Return the captured pids that hold GPU memory, and that GPU's UUID."""
        rows = self._capture(
            [
                self.nvidia_smi,
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        wanted = set(pids)
        holders = set()
        uuids = set()
        for row in rows:
            pid_text, _, uuid = row.partition(",")
            try:
                pid = int(pid_text.strip())
            except ValueError as error:
                raise SnapshotRuntimeFailure(
                    f"unreadable nvidia-smi row: {row!r}"
                ) from error
            if pid in wanted:
                holders.add(pid)
                uuids.add(uuid.strip())
        if not holders:
            raise SnapshotRuntimeFailure(
                "no process in the engine tree holds GPU memory"
            )
        if len(uuids) != 1:
            raise SnapshotRuntimeFailure(
                f"engine tree uses {len(uuids)} GPUs; exactly one is supported"
            )
        return sorted(holders), uuids.pop()

    def alive(self, pid):
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def descriptor_targets(self, pid):
        """Every target the process has open as a descriptor."""
        try:
            descriptors = tuple(Path(f"/proc/{pid}/fd").iterdir())
        except FileNotFoundError as error:
            raise SnapshotRuntimeFailure(
                f"engine process {pid} exited during inventory"
            ) from error
        targets = []
        for descriptor in descriptors:
            try:
                targets.append(os.readlink(descriptor))
            except (FileNotFoundError, PermissionError):
                continue
        return targets

    def _mapped_paths(self, pid):
        try:
            lines = Path(f"/proc/{pid}/maps").read_text().splitlines()
        except (FileNotFoundError, PermissionError):
            return []
        paths = []
        for line in lines:
            fields = line.split(None, 5)
            if len(fields) == 6:
                paths.append(fields[5].removesuffix(" (deleted)"))
        return paths

    def referenced_paths(self, pids):
        """Union of the descriptor targets and mappings the tree holds open."""
        paths = set()
        for pid in pids:
            paths.update(self.descriptor_targets(pid))
            paths.update(self._mapped_paths(pid))
        return paths

    def io_uring_pids(self, pids):
        """Pids holding io_uring state, which CRIU cannot dump."""
        return [
            pid
            for pid in pids
            if "anon_inode:[io_uring]" in self.descriptor_targets(pid)
        ]

    def cache_escapes(self, targets, artifact_path, model_path):
        """Cache files still referenced outside the artifact."""
        escapes = set()
        for target in targets:
            if target.endswith(" (deleted)") or _inside(target, artifact_path):
                continue
            if _inside(target, model_path):
                continue
            if target.startswith(_CACHE_ROOT_PREFIXES):
                escapes.add(target)
        return sorted(escapes)

    # ------------------------------------------------------------------ #
    # cleanup
    # ------------------------------------------------------------------ #
    def _attempt(self, failures, label, action, *arguments):
        """Run one cleanup step without hiding an earlier failure."""
        try:
            action(*arguments)
        except BaseException as error:
            failures.append(f"{label}: {error_detail(error)}")

    def cleanup_tagged(self, artifact_path):
        """Kill every process carrying this artifact's marker.

        Retained as the fallback owner check: the marker is inherited only by
        the engine tree and the helpers this runtime starts for it.
        """
        marker = f"SGLANG_SNAPSHOT_DIR={artifact_path}".encode()
        killed = []
        candidates = set()
        candidates.update(
            int(path.name) for path in Path("/proc").iterdir() if path.name.isdecimal()
        )
        for pid in sorted(candidates, reverse=True):
            if pid <= 1 or pid == os.getpid():
                continue
            try:
                if marker not in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                    continue
                pidfd = os.pidfd_open(pid)
                try:
                    if marker not in Path(f"/proc/{pid}/environ").read_bytes().split(
                        b"\0"
                    ):
                        continue
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    killed.append(pid)
                finally:
                    os.close(pidfd)
            except OSError:
                continue
        deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
        while any(self.alive(pid) for pid in killed):
            if time.monotonic() >= deadline:
                raise SnapshotRuntimeFailure(
                    f"Snapshot process cleanup timed out: {killed}"
                )
            time.sleep(_ALIVE_POLL_SECONDS)

    # ------------------------------------------------------------------ #
    # cuda-checkpoint
    # ------------------------------------------------------------------ #
    def cuda_action(self, pids, action, work, timeout):
        for pid in pids:
            command = [self.cuda_checkpoint, "--action", action, "--pid", str(pid)]
            if action == "lock":
                # cuda-checkpoint takes milliseconds here, the wait takes seconds.
                command += ["--timeout", str(int(timeout * 1000))]
            self._run(command, work / f"cuda-{action}-{pid}.log", timeout)

    # ------------------------------------------------------------------ #
    # criu
    # ------------------------------------------------------------------ #
    def dump_process_tree(self, root_pid, artifact_path, timeout):
        self._run(
            [
                self.criu,
                "dump",
                "--tree",
                str(root_pid),
                "--images-dir",
                str(artifact_path / "images"),
                "--work-dir",
                str(artifact_path / "work"),
                *self.CRIU_ARGS,
                "-o",
                "dump.log",
                "-v4",
            ],
            artifact_path / "work/dump-command.log",
            timeout,
            env=dict(os.environ, SGLANG_SNAPSHOT_DIR=str(artifact_path)),
        )

    # ------------------------------------------------------------------ #
    # artifact runtime files
    # ------------------------------------------------------------------ #
    def save_files(self, artifact_path):
        records = []
        for path in sorted((artifact_path / "runtime").rglob("*")):
            if path.is_symlink():
                raise SnapshotSecurityError(
                    f"Symlinks are not supported in snapshot runtime files: {path}"
                )
            relative = str(path.relative_to(artifact_path / "runtime"))
            copy = resolve_artifact_path(artifact_path / "files", relative)
            if path.is_dir():
                copy.mkdir(parents=True, exist_ok=True)
                continue
            if not path.is_file():
                continue
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            records.append(SnapshotFile(relative, sha256_file(copy)))
        return records

    # ------------------------------------------------------------------ #
    # /dev/shm references
    # ------------------------------------------------------------------ #
    def save_dev_shm(self, artifact_path, paths):
        records = []
        for path in sorted(paths):
            if path.is_symlink() or not path.is_file():
                raise SnapshotSecurityError(f"Invalid /dev/shm file: {path}")
            target = artifact_path / "dev_shm" / path.name
            shutil.copy2(path, target)
            records.append(SnapshotFile(path.name, sha256_file(target)))
        return records

    # ------------------------------------------------------------------ #
    # readiness
    # ------------------------------------------------------------------ #
    def wait_until(self, artifact_path, predicate, pid, timeout, description):
        deadline = time.monotonic() + timeout
        while True:
            if message := control.read_error(artifact_path / control.CONTROL_DIRNAME):
                raise SnapshotRuntimeFailure(message)
            if not self.alive(pid):
                raise SnapshotRuntimeFailure(
                    f"{description}: engine process {pid} exited"
                )
            if predicate():
                return
            if time.monotonic() >= deadline:
                raise SnapshotRuntimeFailure(
                    f"{description}: timed out after {timeout:g}s"
                )
            time.sleep(_ALIVE_POLL_SECONDS)

    def health_ok(self, host, port):
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1" if host == "0.0.0.0" else "[::1]"
        try:
            return (
                requests.get(f"http://{host}:{port}/health", timeout=2).status_code
                == 200
            )
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------ #
    # create steps
    # ------------------------------------------------------------------ #
    def stdio_resources(self):
        """The CRIU resource ids of this process's standard output and error.

        The engine child inherits these descriptors, so restore can map the
        captured streams back onto the restoring caller's streams.
        """
        return [_stdio_resource(1), _stdio_resource(2)]

    def launch_child(self, server_argv, environment):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                _ENGINE_MODULE,
                *server_argv,
            ],
            env=environment,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._children[process.pid] = process
        return process.pid

    def wait_ready(self, artifact_path, root_pid, timeout):
        control_dir = artifact_path / control.CONTROL_DIRNAME
        self.wait_until(
            artifact_path,
            lambda: (control_dir / control.READY).is_file(),
            root_pid,
            timeout,
            "engine did not finish initializing",
        )
        return control.read_json(control_dir, control.READY, control.EngineInfo)

    def inventory(self, artifact_path, root_pid, model_path, reported_gpu_uuid):
        """Collect the process tree, GPU holders and every external reference."""
        pids = self.tree_pids(root_pid)
        cuda_pids, gpu_uuid = self.cuda_holders(pids)
        if gpu_uuid != reported_gpu_uuid:
            raise SnapshotRuntimeFailure(
                f"engine reports {reported_gpu_uuid} but holds {gpu_uuid}"
            )
        if io_uring := self.io_uring_pids(pids):
            raise SnapshotUsageError(
                "engine owns io_uring state that CRIU cannot dump (pids: "
                f"{', '.join(map(str, io_uring))}); start it with USE_LIBUV=0, "
                "which snapshot create sets for its own child"
            )
        targets = self.referenced_paths(pids)
        if escapes := self.cache_escapes(targets, artifact_path, model_path):
            raise SnapshotUsageError(
                "engine keeps caches outside the artifact: "
                f"{', '.join(escapes)}; redirect them with the matching cache "
                "environment variable (SGLANG_CACHE_DIR, TRITON_CACHE_DIR, "
                "TORCHINDUCTOR_CACHE_DIR, CUDA_CACHE_PATH, FLASHINFER_WORKSPACE_BASE)"
            )
        return EngineInventory(
            root_pid=root_pid,
            pids=pids,
            cuda_pids=cuda_pids,
            gpu_uuid=gpu_uuid,
            shared_paths={
                Path(target)
                for target in targets
                if Path(target).parent == self.SHM_DIR and Path(target).is_file()
            },
        )

    def dump(self, artifact_path, inventory, timeout):
        work = artifact_path / "work"
        remaps_before = set(self.SHM_DIR.glob("link_remap.*"))
        self.cuda_action(inventory.cuda_pids, "lock", work, timeout)
        self.cuda_action(inventory.cuda_pids, "checkpoint", work, timeout)
        self.dump_process_tree(inventory.root_pid, artifact_path, timeout)
        new_remaps = set(self.SHM_DIR.glob("link_remap.*")) - remaps_before
        return msgspec.structs.replace(
            inventory,
            files=self.save_files(artifact_path),
            dev_shm=self.save_dev_shm(
                artifact_path, inventory.shared_paths | new_remaps
            ),
        )

    def flush_artifact(self, artifact_path):
        """Flush the artifact to storage before the manifest marks it complete."""
        directories = [artifact_path.parent, artifact_path]
        for path in artifact_path.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_dir():
                directories.append(path)
                continue
            if not path.is_file():
                # Sockets, FIFOs and device nodes are not fsynced: CRIU
                # restores those descriptors itself.
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        directories.sort(key=lambda item: len(item.parts), reverse=True)
        for path in directories:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def verify_dead(self, inventory):
        """CRIU must have taken the whole tree down before the manifest is written."""
        process = self._children.pop(inventory.root_pid, None)
        if process is not None:
            try:
                process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                raise SnapshotRuntimeFailure(
                    f"engine process {inventory.root_pid} survived the dump"
                ) from error
        surviving = [pid for pid in inventory.pids if self.alive(pid)]
        if surviving:
            raise SnapshotRuntimeFailure(
                f"processes survived the dump: {', '.join(map(str, surviving))}"
            )

    def abort_create(self, root_pid, artifact_path, failures):
        """Stop the launched tree after a failed create."""
        process = self._children.pop(root_pid, None)
        if root_pid is not None:
            try:
                os.killpg(root_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None:
            try:
                process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                failures.append(f"engine process {root_pid} survived SIGKILL")
        self._attempt(failures, "process sweep", self.cleanup_tagged, artifact_path)

    def discard_new_link_remaps(self, before):
        for path in set(self.SHM_DIR.glob("link_remap.*")) - before:
            path.unlink(missing_ok=True)
