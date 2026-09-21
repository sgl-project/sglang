# SPDX-License-Identifier: Apache-2.0
"""Standalone diffusion weight owner; tensor transport remains owned by SRT.

Run with ``python -m sglang.multimodal_gen.runtime.weight_cache.daemon
--model-path ...``. Scope is one resident, explicitly admitted native DiT.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import msgspec

from sglang.multimodal_gen.runtime.distributed.bootstrap import (
    bootstrap_diffusion_runtime,
)
from sglang.multimodal_gen.runtime.pipelines_core import resolve_pipeline_class
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.server_args import prepare_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weight_cache.client import (
    PROTOCOL,
    WeightCacheClient,
    decode_generation,
    peer_identity,
)
from sglang.multimodal_gen.runtime.weight_cache.identity import (
    compatibility_plan,
    default_runtime_dir,
    locate,
)
from sglang.multimodal_gen.runtime.weight_cache.placement import local_device_index
from sglang.multimodal_gen.runtime.weight_cache.plan import plan_diff
from sglang.srt.utils.network import NetworkAddress, get_free_port
from sglang.srt.weight_cache.common.liveness import ProcessHandle, ProcessIdentity
from sglang.srt.weight_cache.common.transport import CudaIpcExporter
from sglang.srt.weight_cache.protocol import recv_msg, send_msg

logger = init_logger(__name__)
MAX_CONTROL_CONNECTIONS = 16
DRAIN_GRACE_SECONDS = 5.0


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = path.lstat()
    if (
        not stat.S_ISDIR(st.st_mode)
        or st.st_uid != os.getuid()
        or stat.S_IMODE(st.st_mode) & 0o077
    ):
        raise PermissionError(
            f"Weight-cache runtime directory must be owned and private (0700): {path}"
        )


@contextmanager
def owner_lock(path):
    """Never unlink lock files: every contender must flock the same inode."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        st = os.fstat(fd)
        if st.st_uid != os.getuid() or not stat.S_ISREG(st.st_mode):
            raise PermissionError(f"Not an owned regular owner lock: {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Weight-cache owner already holds resource: {path}"
            ) from error
        yield
    finally:
        os.close(fd)


class DiffusionWeightCacheDaemon:
    def __init__(self, args):
        self.args = args
        self.prepared = prepare_pipeline(
            resolve_pipeline_class(args), args, required=True
        )
        self.plan = compatibility_plan(self.prepared, args, verify_checkpoint=True)
        self.path = locate(self.plan, args)
        self.ready_path = self.path.with_suffix(".ready")
        self._initialize_control()

    def _initialize_control(self):
        self.stopping = False
        self.consumers = {}
        self._consumers_lock = threading.Lock()
        self.exporter = None
        self._connections = {}
        self._connections_lock = threading.Lock()

    def stop(self, *_):
        # Signal handlers must not acquire locks also used by request threads.
        # run() wakes accepted sockets and joins handlers before draining.
        if self.stopping:
            logger.warning(
                "Owner is draining; allocations remain pinned until consumers exit"
            )
        self.stopping = True

    def _live_consumers(self):
        with self._consumers_lock:
            for identity, handle in list(self.consumers.items()):
                if not handle.is_alive():
                    handle.close()
                    del self.consumers[identity]
            return list(self.consumers.values())

    def _request(self, request, peer):
        if self.stopping:
            raise RuntimeError("Weight-cache owner is draining")
        if any(request.get(key) != value for key, value in PROTOCOL.items()):
            raise ValueError("Weight-cache protocol family/version/ABI mismatch")
        if request.get("compatibility") != self.plan.to_dict():
            raise ValueError(
                f"Weight-cache compatibility mismatch: {plan_diff(self.plan.to_dict(), request.get('compatibility'))}"
            )
        generation = self.exporter.generation
        kind = request.get("type")
        if kind == "query_status":
            # Observation never exports handles, registers a consumer, or
            # admits a worker. It remains available after budget exhaustion.
            stats = self.exporter.stats()
            return {
                "compatibility": self.plan.to_dict(),
                "generation": msgspec.to_builtins(generation),
                "cache_status": {
                    **stats,
                    "components": self.plan.to_dict()["requested"],
                    "active_consumers": len(self._live_consumers()),
                    "accepting_fetches": not stats["admission_stopped"]
                    and not stats["budget_exhausted"],
                },
            }
        if kind == "query_manifest":
            if self.exporter.stats()["budget_exhausted"]:
                raise RuntimeError(
                    "Weight-cache generation budget exhausted; drain and restart owner"
                )
            return {
                "compatibility": self.plan.to_dict(),
                "generation": msgspec.to_builtins(generation),
                "manifest": self.exporter.manifest.to_dict(),
            }
        if (
            kind != "fetch_bundle"
            or request.get("components") != self.plan.to_dict()["requested"]
        ):
            raise ValueError(
                "Unsupported diffusion weight-cache request/component bundle"
            )
        if decode_generation(request["generation"]) != generation:
            raise ValueError("Weight-cache fetch generation mismatch")
        if (
            msgspec.convert(request["consumer"], type=ProcessIdentity, strict=True)
            != peer
            or not peer.is_alive()
        ):
            raise ProcessLookupError("Consumer exited before fetch")
        # Register actual socket credentials before creating the first handle,
        # including clients which abandon a response or partially import it.
        self._live_consumers()
        with self._consumers_lock:
            if peer not in self.consumers:
                self.consumers[peer] = ProcessHandle(peer)
        return msgspec.to_builtins(
            self.exporter.export(request["request_id"], generation=generation)
        )

    def _drain(self):
        if self.exporter is None:
            return
        self.exporter.stop_admission()

        # The process identity alone is not a lease on its allocations. Retain
        # the exporter and its model until EVERY actual consumer has exited.
        def signal_consumers(signum):
            for handle in self.consumers.values():
                try:
                    handle.send_signal(signum)
                except ProcessLookupError:
                    pass
                except OSError:
                    logger.exception(
                        "Cannot signal consumer %s; retaining allocations",
                        handle.identity,
                    )

        signal_consumers(signal.SIGTERM)
        deadline = time.monotonic() + DRAIN_GRACE_SECONDS
        escalated = False
        next_report = deadline
        while True:
            try:
                if not self._live_consumers():
                    return
            except Exception:
                # Unknown is NOT dead. A bad pidfd must never authorize freeing
                # allocations potentially still referenced by a consumer.
                logger.exception(
                    "Cannot confirm consumer exit; retaining owner allocations"
                )
            now = time.monotonic()
            if now >= deadline and not escalated:
                signal_consumers(signal.SIGKILL)
                escalated = True
            if escalated and now >= next_report:
                logger.critical(
                    "Owner drain stalled; retaining allocations for %s. "
                    "Forced owner death is NOT graceful recovery; investigate consumers/GPU health.",
                    list(self.consumers),
                )
                next_report = now + 10
            time.sleep(0.1)

    def _serve_connection(self, conn):
        try:
            with conn:
                conn.settimeout(self.args.weight_cache_timeout)
                try:
                    peer = peer_identity(conn)
                    while not self.stopping:
                        request = recv_msg(conn)
                        response = self._request(request, peer)
                        send_msg(conn, {**PROTOCOL, "status": "ok", **response})
                except TimeoutError:
                    logger.warning("Weight-cache control connection timed out")
                except (EOFError, ConnectionError):
                    pass
                except Exception as error:
                    logger.warning("Weight-cache request failed: %s", error)
                    try:
                        send_msg(
                            conn, {**PROTOCOL, "status": "error", "error": str(error)}
                        )
                    except OSError:
                        pass
        finally:
            with self._connections_lock:
                self._connections.pop(conn, None)

    def _dispatch_connection(self, conn):
        with self._connections_lock:
            if self.stopping or len(self._connections) >= MAX_CONTROL_CONNECTIONS:
                conn.close()
                return
            thread = threading.Thread(
                target=self._serve_connection, args=(conn,), daemon=True
            )
            self._connections[conn] = thread
            try:
                thread.start()
            except BaseException:
                del self._connections[conn]
                conn.close()
                raise

    def _close_connections(self):
        with self._connections_lock:
            connections = list(self._connections.items())
        for conn, _ in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # Do not free an exporter while a handler could still be exporting or
        # registering a consumer. A wedged CUDA export retains its producer.
        for _, thread in connections:
            thread.join()

    def _cleanup_stale_files(self):
        # Both device and path locks are held. Never signal a PID from .ready:
        # it may have been recycled since the previous owner's crash.
        for path, kind in ((self.path, stat.S_ISSOCK), (self.ready_path, stat.S_ISREG)):
            try:
                st = path.lstat()
            except FileNotFoundError:
                continue
            if st.st_uid != os.getuid() or not kind(st.st_mode):
                raise PermissionError(
                    f"Refusing to remove non-owned cache endpoint: {path}"
                )
            if path == self.path:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.2)
                    try:
                        probe.connect(str(path))
                    except (ConnectionRefusedError, FileNotFoundError):
                        pass
                    else:
                        raise RuntimeError(
                            f"Live weight-cache socket already exists: {path}"
                        )
            path.unlink(missing_ok=True)

    def _remove_endpoints(self):
        for path in (self.ready_path, self.path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # An endpoint failure must not bypass consumer draining and
                # release allocations that may still be imported elsewhere.
                logger.exception("Cannot remove weight-cache endpoint %s", path)

    def _owner_lock_paths(self):
        device_uuid = self.plan.to_dict()["rank"]["device_uuid"]
        lock_root = (
            default_runtime_dir()
            / hashlib.sha256(device_uuid.encode()).hexdigest()[:16]
        )
        private_directory(default_runtime_dir())
        private_directory(lock_root)
        private_directory(self.path.parent)
        return sorted(
            {
                lock_root / "owner.lock",
                self.path.with_name(self.path.name + ".lock"),
                self.ready_path.with_name(self.ready_path.name + ".lock"),
            }
        )

    def run(self):
        locks = ExitStack()
        listener = None
        published = False
        handlers = {}
        try:
            for path in self._owner_lock_paths():
                locks.enter_context(owner_lock(path))
            self._cleanup_stale_files()
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, self.stop)
            bootstrap_diffusion_runtime(
                self.args,
                local_rank=local_device_index(self.args),
                rank=0,
                rendezvous=NetworkAddress("127.0.0.1", get_free_port()),
                role="diffusion_weight_cache_daemon",
            )
            from sglang.multimodal_gen.runtime.weight_cache.bundle import build_bundle

            model = build_bundle(self.prepared.cached_components)
            self.exporter = CudaIpcExporter(
                model, max_deliveries=self.args.weight_cache_max_deliveries
            )
            if compatibility_plan(self.prepared, self.args) != self.plan:
                raise ValueError("Checkpoint/build changed while loading cache owner")
            if self.stopping:
                return
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.path))
            published = True
            os.chmod(self.path, 0o600)
            listener.listen(16)
            listener.settimeout(0.2)
            # Exporter construction synchronized all finalized writes. Publish
            # readiness only after the socket can accept manifest requests.
            fd, temp = tempfile.mkstemp(prefix=".ready-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as out:
                    out.write(f"pid={os.getpid()}\n")
                    out.write(
                        json.dumps(
                            {
                                "compatibility_digest": self.plan.digest,
                                "generation": msgspec.to_builtins(
                                    self.exporter.generation
                                ),
                            }
                        )
                    )
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp, self.ready_path)
            finally:
                Path(temp).unlink(missing_ok=True)
            logger.info(
                "[WeightCache] ready: %s (%d shared bytes)",
                self.path,
                self.exporter.manifest.unique_storage_bytes,
            )
            while not self.stopping:
                try:
                    conn, _ = listener.accept()
                except TimeoutError:
                    continue
                self._dispatch_connection(conn)
        finally:
            self.stopping = True
            if listener is not None:
                listener.close()
            self._close_connections()
            if published:
                self._remove_endpoints()
            self._drain()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            locks.close()
            # Keep self.exporter strongly owned even after run returns. The
            # standalone owner process lifetime defines the allocation lifetime.


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--status", action="store_true")
    command, remaining = parser.parse_known_args(sys.argv[1:])
    args = prepare_server_args([*remaining, "--weight-cache-mode", "client"])
    if command.status:
        prepared = prepare_pipeline(resolve_pipeline_class(args), args, required=True)
        plan = compatibility_plan(prepared, args)
        with WeightCacheClient(plan, args) as client:
            print(json.dumps(client.status(), sort_keys=True))
        return
    owner = DiffusionWeightCacheDaemon(args)
    try:
        owner.run()
    finally:
        from sglang.multimodal_gen.runtime.distributed import (
            cleanup_dist_env_and_memory,
        )

        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
