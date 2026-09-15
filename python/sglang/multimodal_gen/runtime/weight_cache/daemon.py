# SPDX-License-Identifier: Apache-2.0
"""Standalone diffusion weight owner; tensor transport remains owned by SRT.

Run with ``python -m sglang.multimodal_gen.runtime.weight_cache.daemon
--model-path ...``. Initial scope is one resident Wan2.1 1.3B transformer.
"""

import dataclasses
import fcntl
import hashlib
import json
import os
import signal
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

from sglang.multimodal_gen.runtime.distributed.bootstrap import (
    bootstrap_diffusion_runtime,
)
from sglang.multimodal_gen.runtime.pipelines_core import resolve_pipeline_class
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.server_args import prepare_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weight_cache.adapters import dit_wan
from sglang.multimodal_gen.runtime.weight_cache.client import (
    PROTOCOL,
    decode_generation,
    peer_identity,
)
from sglang.multimodal_gen.runtime.weight_cache.identity import (
    compatibility_plan,
    locate,
)
from sglang.multimodal_gen.runtime.weight_cache.placement import local_device_index
from sglang.multimodal_gen.runtime.weight_cache.plan import plan_diff
from sglang.srt.utils.network import NetworkAddress, get_free_port
from sglang.srt.weight_cache.protocol import (
    CLIENT_CONNECTION_TIMEOUT,
    cleanup_stale_daemon_files,
    recv_msg,
    send_msg,
)
from sglang.weight_cache_common.identity import default_runtime_dir
from sglang.weight_cache_common.transport import CudaIpcExporter

logger = init_logger(__name__)


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


class DiffusionWeightCacheDaemon:
    def __init__(self, args):
        self.args = args
        self.prepared = prepare_pipeline(
            resolve_pipeline_class(args), args, required=True
        )
        self.plan = compatibility_plan(self.prepared, args, verify_checkpoint=True)
        self.path = locate(self.plan, args)
        self.ready_path = self.path.with_suffix(".ready")
        self.stopping = False
        self.consumers = set()
        self.exporter = None
        self._connection = None

    def stop(self, *_):
        self.stopping = True
        # Wake an idle/partial control exchange without waiting for its timeout.
        # The generation and registered consumers still go through _drain().
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

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
        if kind == "query_manifest":
            if self.exporter.stats()["budget_exhausted"]:
                raise RuntimeError(
                    "Weight-cache generation budget exhausted; drain and restart owner"
                )
            return {
                "compatibility": self.plan.to_dict(),
                "generation": dataclasses.asdict(generation),
                "manifest": self.exporter.manifest.to_dict(),
            }
        if kind != "fetch_component" or request.get("component") != "transformer":
            raise ValueError("Unsupported diffusion weight-cache request/component")
        if decode_generation(request["generation"]) != generation:
            raise ValueError("Weight-cache fetch generation mismatch")
        if not peer.is_alive():
            raise ProcessLookupError("Consumer exited before fetch")
        # Register actual socket credentials before creating the first handle,
        # including clients which abandon a response or partially import it.
        self.consumers = {p for p in self.consumers if p.is_alive()}
        self.consumers.add(peer)
        return dataclasses.asdict(
            self.exporter.export(request["request_id"], generation=generation)
        )

    def _drain(self):
        if self.exporter is None:
            return
        self.exporter.stop_admission()
        # The process identity alone is not a lease on its allocations. Retain
        # the exporter and its model until EVERY actual consumer has exited.
        for peer in self.consumers:
            if peer.is_alive():
                try:
                    os.kill(peer.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while any(peer.is_alive() for peer in self.consumers):
            if time.monotonic() >= deadline:
                for peer in self.consumers:
                    if peer.is_alive():
                        try:
                            os.kill(peer.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            time.sleep(0.05)

    def run(self):
        device_uuid = self.plan.to_dict()["rank"]["device_uuid"]
        lock_root = (
            default_runtime_dir()
            / hashlib.sha256(device_uuid.encode()).hexdigest()[:16]
        )
        private_directory(default_runtime_dir())
        private_directory(lock_root)
        private_directory(self.path.parent)
        lock_fd = os.open(
            lock_root / "owner.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        lock_st = os.fstat(lock_fd)
        if lock_st.st_uid != os.getuid() or not stat.S_ISREG(lock_st.st_mode):
            os.close(lock_fd)
            raise PermissionError(
                "Weight-cache owner lock is not an owned regular file"
            )
        listener = None
        published = False
        handlers = {}
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cleanup_stale_daemon_files(
                device_uuid, socket_path=str(self.path), ready_path=str(self.ready_path)
            )
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, self.stop)
            bootstrap_diffusion_runtime(
                self.args,
                local_rank=local_device_index(self.args),
                rank=0,
                rendezvous=NetworkAddress("127.0.0.1", get_free_port()),
                role="diffusion_weight_cache_daemon",
            )
            model = dit_wan.load_ordinary(self.prepared.transformer)
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
                                "generation": dataclasses.asdict(
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
                with conn:
                    self._connection = conn
                    conn.settimeout(
                        min(CLIENT_CONNECTION_TIMEOUT, self.args.weight_cache_timeout)
                    )
                    try:
                        peer = peer_identity(conn)
                        while not self.stopping:
                            request = recv_msg(conn)
                            response = self._request(request, peer)
                            send_msg(conn, {**PROTOCOL, "status": "ok", **response})
                    except (EOFError, ConnectionError, TimeoutError):
                        pass
                    except Exception as error:
                        logger.warning("Weight-cache request failed: %s", error)
                        try:
                            send_msg(
                                conn,
                                {**PROTOCOL, "status": "error", "error": str(error)},
                            )
                        except (OSError, EOFError):
                            pass
                    finally:
                        self._connection = None
        finally:
            self.stopping = True
            if listener is not None:
                listener.close()
            if published:
                self.ready_path.unlink(missing_ok=True)
                self.path.unlink(missing_ok=True)
            self._drain()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            os.close(lock_fd)
            # Keep self.exporter strongly owned even after run returns. The
            # standalone owner process lifetime defines the allocation lifetime.


def main():
    args = prepare_server_args([*sys.argv[1:], "--weight-cache-mode", "client"])
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
