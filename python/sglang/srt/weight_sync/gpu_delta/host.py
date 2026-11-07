# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verified engine-host encoded cache and persistent pinned input arenas.

Each engine host shares a prepared tmpfs image built by its persistent CPU child.
CUDA decoders consume it directly; DE decoders stage only their local spans into
rank-private hardware-decompression-capable input storage.
"""

import fcntl
import hashlib
import json
import mmap
import os
import socket
import stat
import subprocess
import sys
import time
import uuid
from functools import cached_property
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import orjson

from sglang.srt.weight_sync.gpu_delta import profiling
from sglang.srt.weight_sync.gpu_delta.codec import decoder_backend
from sglang.srt.weight_sync.gpu_delta.host_worker import (
    _CAPACITY_ALIGNMENT,
    _PREPARE_METRICS,
    _identity,
    _metadata_views,
    configured_cpu_workers,
    copy_bytes,
    new_metrics,
    publication_token,
)
from sglang.srt.weight_sync.gpu_delta.memory import HostAllocation, HostMapping


def _cache_base():
    root = Path(
        os.environ.get(
            "GPU_DELTA_HOST_CACHE_DIR", f"/dev/shm/sglang-gpu-delta-{os.getuid()}"
        )
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("GPU-delta host cache must be private to the current user")
    mounts = []
    with open("/proc/self/mountinfo") as source:
        for line in source:
            left, right = line.rstrip().split(" - ", 1)
            mount = left.split()[4]
            for escaped, value in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
                mount = mount.replace(escaped, value)
            mount = Path(mount)
            if root.is_relative_to(mount):
                mounts.append((len(mount.parts), right.split()[0]))
    if not mounts or max(mounts)[1] != "tmpfs":
        raise ValueError("GPU_DELTA_HOST_CACHE_DIR must be on host-shared tmpfs")
    return root


def _cache_root(engine_id):
    root = _cache_base() / hashlib.sha256(engine_id.encode()).hexdigest()
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def host_cache_id(engine_id):
    """An engine-scoped tmpfs root, not hostname, defines the sharing domain."""
    root = _cache_root(engine_id)
    with (root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / ".host-id"
        if not path.exists():
            path.write_text(uuid.uuid4().hex)
            path.chmod(0o400)
        value = path.read_text()
        if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid GPU-delta host cache identity")
        return value


class CpuWorker:
    """One clean interpreter per owner, with descriptor-only request/reply IPC."""

    def __init__(self, directory, device):
        self.configuration = {
            "directory": str(directory),
            "workers": configured_cpu_workers(),
            "backends": {
                name: decoder_backend(device, name)
                for name in ("snappy", "lz4", "bitcomp")
            },
        }
        local, child = socket.socketpair()
        parent_fd = None
        started = time.perf_counter()
        try:
            parent_fd = os.pidfd_open(os.getpid())
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("host_worker.py")),
                    "--socket-fd",
                    str(child.fileno()),
                    "--parent-fd",
                    str(parent_fd),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                pass_fds=(child.fileno(), parent_fd),
                start_new_session=True,
                env=os.environ
                | {
                    "SGLANG_ENABLE_NVTX_OPERATIONS": "1" if profiling.ENABLED else "0",
                    # Array construction/copies need no BLAS pool in this CPU child.
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
        except BaseException:
            local.close()
            raise
        finally:
            child.close()
            if parent_fd is not None:
                os.close(parent_fd)
        self.connection = Connection(local.detach())
        self.spawn_s = time.perf_counter() - started

    def _request(self, request, cancelled=None):
        if self.connection is None:
            raise RuntimeError(
                "GPU-delta CPU worker is closed; preparation cannot restart"
            )
        try:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("GPU-delta host preparation aborted")
            self.connection.send_bytes(orjson.dumps(request))
            while not self.connection.poll(0.05):
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("GPU-delta host preparation aborted")
            response = orjson.loads(self.connection.recv_bytes(1 << 20))
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("GPU-delta host preparation aborted")
            if response["token"] != request["token"]:
                raise RuntimeError(
                    "GPU-delta CPU worker returned a different publication"
                )
            if response["status"] != "OK":
                error = (
                    ValueError
                    if response["error_type"] == "ValueError"
                    else RuntimeError
                )
                raise error(response["error"])
            return response
        except (EOFError, BrokenPipeError) as error:
            self.close(terminate=True)
            raise RuntimeError(
                "GPU-delta CPU worker exited during preparation"
            ) from error
        except BaseException:
            self.close(terminate=True)
            raise

    @profiling.annotate("gpu_delta.host.cpu_worker_wait")
    def prepare(self, expected, namespace, metrics, cancelled):
        started = time.perf_counter()
        response = self._request(
            {
                "command": "prepare",
                "token": publication_token(expected),
                "configuration": self.configuration,
                "expected": expected,
                "namespace": namespace,
            },
            cancelled,
        )
        metrics.update(response["metrics"])
        metrics["host_cpu_worker_spawn_s"] = self.spawn_s
        metrics["host_cpu_worker_wait_s"] = time.perf_counter() - started
        self.spawn_s = 0.0
        return response["index"]

    def release(self, ready_state):
        if self.connection is None:
            return
        self._request(
            {
                "command": "release",
                "token": ready_state["token"],
                "ready_state": ready_state,
            }
        )

    def close(self, terminate=False):
        try:
            if terminate:
                self.process.kill()
            elif self.connection is not None:
                self.connection.send_bytes(b"null")
        except OSError:
            self.process.kill()
        finally:
            # Both paths reap before releasing the publication lock or mappings.
            self.process.wait()
            if self.connection is not None:
                self.connection.close()
                self.connection = None


class HostArena:
    """Persistent input storage with an engine-host encoded cache.

    The original engine cohort shares encoded files under each publication's hash
    policy. Successful all-rank apply/resume permits the next publication to
    overwrite encoded and prepared bytes; failed preparation never grants release.
    """

    def __init__(self, engine_id, device, identity):
        self.engine_id, self.device = engine_id, device
        self.identity = identity
        self.allocation = self.mapping = self.tensor = None
        self.capacity = None
        self.hardware = None
        self.shared_allocation = self.shared_capacity = None
        self.directory = None
        self._namespace = None
        self.owner = False
        self._metadata_mapping = self._metadata_capacity = None
        self._worker = None

    @cached_property
    def root(self):
        return _cache_root(self.engine_id)

    def _reserve_rank_arena(self, size, metrics):
        import torch

        if (
            self.capacity is not None
            and size <= self.capacity["capacity"]
            and self.hardware
        ):
            metrics["host_rank_mapping_reused"] = 1
            return
        previous = self.capacity if self.hardware else None
        capacity = size if previous is None else 2 * size
        capacity = (
            (capacity + _CAPACITY_ALIGNMENT - 1)
            // _CAPACITY_ALIGNMENT
            * _CAPACITY_ALIGNMENT
        )
        started = time.perf_counter()
        allocation = None
        if capacity:
            allocation = HostAllocation(capacity, self.device)
        self._close_private()
        self.allocation = allocation
        self.hardware = True
        self.mapping = allocation.view if allocation is not None else None
        self.tensor = (
            torch.frombuffer(self.mapping, dtype=torch.uint8)
            if self.mapping is not None
            else torch.empty(0, dtype=torch.uint8)
        )
        generation = previous["generation"] + 1 if previous else 1
        self.capacity = {
            "generation": generation,
            "capacity": allocation.capacity if allocation is not None else 0,
            "identity": uuid.uuid4().hex,
        }
        metrics["host_rank_allocation_s"] = time.perf_counter() - started
        metrics["host_rank_allocation_calls"] = int(allocation is not None)
        metrics["host_rank_allocation_bytes"] = self.capacity["capacity"]

    def _attach_shared_arena(self, prepared, metrics, hardware):
        replace = self.shared_capacity != prepared or (
            hardware
            and self.shared_allocation is not None
            and self.shared_allocation.registered
        )
        started = time.perf_counter()
        if replace:
            self._close_shared()
            if prepared["capacity"]:
                with (self.directory / prepared["file"]).open("r+b") as source:
                    if _identity(os.fstat(source.fileno())) != prepared["identity"]:
                        raise ValueError("prepared cache inode or extent changed")
                    mapping = mmap.mmap(
                        source.fileno(),
                        prepared["capacity"],
                        flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                    )
                self.shared_allocation = HostMapping(mapping, pinned=False)
            self.shared_capacity = prepared
        if not hardware:
            if (
                self.shared_allocation is not None
                and not self.shared_allocation.registered
            ):
                import torch

                # A fresh preparation thread may have no current CUDA context.
                torch.cuda.set_device(self.device)
                self.shared_allocation.pin()
                metrics["host_rank_allocation_calls"] = 1
                metrics["host_rank_allocation_bytes"] = prepared["capacity"]
                metrics["host_rank_allocation_s"] += time.perf_counter() - started
            else:
                metrics["host_rank_mapping_reused"] = 1

    def _attach_metadata(self, allocation):
        if self._metadata_capacity != allocation:
            with (self.directory / allocation["file"]).open("r+b") as source:
                if _identity(os.fstat(source.fileno())) != allocation["identity"]:
                    raise ValueError("metadata cache inode or extent changed")
                self._metadata_mapping = mmap.mmap(
                    source.fileno(), allocation["capacity"]
                )
            self._metadata_capacity = allocation

    @profiling.annotate("gpu_delta.host.prepare_encoded")
    def prepare_publication(
        self, manifest_path, manifest_sha256, metadata, timings, cancelled=None
    ):
        """The original owner authenticates and prepares one shared publication."""
        started = time.perf_counter()
        participants = sorted(
            metadata["participants"], key=lambda value: value["rank_id"]
        )
        namespace = {"stream_id": metadata["stream_id"], "participants": participants}
        if self._namespace is None:
            self._namespace = namespace
            key = hashlib.sha256(
                json.dumps(namespace, sort_keys=True).encode()
            ).hexdigest()
            self.directory = self.root / key
            self.directory.mkdir(mode=0o700, exist_ok=True)
            owner = min(
                (
                    p
                    for p in participants
                    if p["engine_id"] == self.engine_id
                    and p["host_cache_id"] == self.identity["host_cache_id"]
                ),
                key=lambda p: (p["tp_rank"], p["dp_rank"], p["pid"], p["rank_id"]),
            )
            self.owner = self.identity == owner
        elif namespace != self._namespace:
            raise ValueError("host cache original cohort or stream changed")
        publication = Path(manifest_path).resolve(strict=True)
        expected = {
            "manifest_path": str(publication),
            "manifest_sha256": manifest_sha256,
            **{
                key: metadata[key]
                for key in (
                    "session_id",
                    "base_version",
                    "target_version",
                    "plan_digest",
                )
            },
            "skip_payload_hash": not metadata["validate_payload_hash"],
        }
        token = publication_token(expected)
        metrics = new_metrics()
        if self.owner:
            if self._worker is None:
                self._worker = CpuWorker(self.directory, self.device)
            index = self._worker.prepare(expected, namespace, metrics, cancelled)
        else:
            waiting = time.perf_counter()
            with (self.directory / ".lock").open("a+b") as lock:
                while True:
                    if cancelled is not None and cancelled.is_set():
                        raise RuntimeError("GPU-delta host preparation aborted")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        state_path = self.directory / "state.json"
                        state = (
                            orjson.loads(state_path.read_bytes())
                            if state_path.exists()
                            else None
                        )
                        if state is not None and state["token"] == token:
                            if state["state"] != "READY":
                                raise ValueError(
                                    "encoded publication is failed or already released"
                                )
                            index = orjson.loads(
                                (self.directory / "index.json").read_bytes()
                            )
                            if (
                                index["namespace"] != namespace
                                or index["token"] != token
                            ):
                                raise ValueError("encoded cache publication differs")
                            break
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    if cancelled is None:
                        time.sleep(0.01)
                    else:
                        cancelled.wait(0.01)
            metrics.update(
                host_encoded_cache_wait_s=time.perf_counter() - waiting,
                host_encoded_cache_reused=1,
                host_prepared_cache_reused=1,
            )
            metrics["host_prepared_cache_wait_s"] = metrics["host_encoded_cache_wait_s"]
        self._attach_metadata(index["metadata"])
        result = HostPublication(self, index)
        metrics["host_encoded_cache_access_s"] = time.perf_counter() - started
        metrics.update(
            host_encoded_cache_capacity_bytes=index["encoded"]["capacity"],
            host_encoded_cache_capacity_generation=index["encoded"]["generation"],
            host_encoded_cache_build_s=index["build_s"],
            host_encoded_cache_skip_payload_hash=int(expected["skip_payload_hash"]),
            host_prepared_cache_capacity_bytes=index["prepared"]["capacity"],
            host_prepared_cache_capacity_generation=index["prepared"]["generation"],
            host_prepared_cache_arena_bytes=index["prepared_size"],
            host_prepared_cache_build_s=index["prepared_build_s"],
            host_metadata_cache_capacity_bytes=index["metadata"]["capacity"],
            host_metadata_cache_arena_bytes=index["metadata_size"],
            host_cpu_worker_pid=index["cpu_worker"]["pid"],
            host_cpu_worker_start_ticks=index["cpu_worker"]["start_ticks"],
            host_rank_cpu_workers=index["cpu_worker"]["workers"] if self.owner else 0,
        )
        timings.update(metrics)
        return result

    @profiling.annotate("gpu_delta.host.prepare_local")
    def prepare_local(self, publication, tensor_ids, timings, hardware=True):
        """Bind rank inputs using canonical ordinals, with DE-local rebasing."""
        if publication.header["backend"] != ("hardware" if hardware else "cuda"):
            raise ValueError("engine-host decoder backends differ")
        index, tensors = publication.index, publication.tensors
        metrics = {
            name: timings[name]
            for name in (
                "host_rank_allocation_s",
                "host_rank_allocation_calls",
                "host_rank_allocation_bytes",
                "host_rank_mapping_reused",
                "host_rank_outer_zstd_decode_s",
                "host_rank_encoded_copy_s",
                *_PREPARE_METRICS,
            )
        }
        self._attach_shared_arena(index["prepared"], metrics, hardware)
        layout_started = time.perf_counter()
        selected = np.asarray(tensor_ids, dtype=np.int64)
        sizes = tensors["prepared_nbytes"][selected]
        size = int(sizes.sum())
        staging_bytes = 0
        offsets = tensors["prepared_offset"]
        if hardware:
            selected = selected[np.argsort(offsets[selected], kind="stable")]
            sizes = tensors["prepared_nbytes"][selected]
            aligned = (sizes + 15) // 16 * 16
            positions = np.cumsum(aligned) - aligned
            offsets = np.zeros(len(tensors), dtype=np.int64)
            offsets[selected] = positions
            size = int(np.max((positions + sizes)[sizes != 0], initial=0))
        metrics["host_rank_layout_s"] = time.perf_counter() - layout_started
        if hardware:
            self._reserve_rank_arena(size, metrics)
            prepare_started = time.perf_counter()
            for tensor_id in selected:
                count = int(tensors["prepared_nbytes"][tensor_id])
                if count:
                    source = int(tensors["prepared_offset"][tensor_id])
                    target = int(offsets[tensor_id])
                    copy_bytes(
                        self.mapping[target : target + count],
                        self.shared_allocation.view[source : source + count],
                    )
                    staging_bytes += count
            staging_s = time.perf_counter() - prepare_started
        else:
            import torch

            prepare_started = time.perf_counter()
            if self.hardware is not False or self.capacity != index["prepared"]:
                self._close_private()
                self.mapping = (
                    self.shared_allocation.view
                    if self.shared_allocation is not None
                    else None
                )
                self.tensor = (
                    torch.frombuffer(self.mapping, dtype=torch.uint8)
                    if self.mapping is not None
                    else torch.empty(0, dtype=torch.uint8)
                )
                self.hardware, self.capacity = False, index["prepared"]
            staging_s = 0.0
        metrics["host_rank_prepare_call_s"] = time.perf_counter() - prepare_started
        metrics.update(
            host_rank_arena_bytes=size,
            host_rank_capacity_bytes=self.capacity["capacity"],
            host_rank_capacity_generation=self.capacity["generation"],
            host_rank_cpu_workers=timings["host_rank_cpu_workers"],
            host_rank_private_capacity_bytes=self.capacity["capacity"]
            if hardware
            else 0,
            host_rank_staging_copy_s=staging_s,
            host_rank_staging_copy_bytes=staging_bytes,
        )
        timings.update(metrics)
        return HostPreparedSnapshot(self, publication, offsets)

    def _close_private(self):
        if self.hardware:
            self.tensor = self.mapping = self.capacity = None
        if self.allocation is not None:
            self.allocation.close()
            self.allocation = None

    def _close_shared(self):
        if self.hardware is False:
            self.tensor = self.mapping = self.capacity = None
        if self.shared_allocation is not None:
            self.shared_allocation.close()
            self.shared_allocation = None
        self.shared_capacity = None

    def close(self):
        if self._worker is not None:
            # Retain the closed client so a poisoned backend cannot respawn it.
            self._worker.close()
        self._close_private()
        self._close_shared()
        self._metadata_mapping = self._metadata_capacity = None


class HostPublication:
    """Immutable numeric views, protected by the payload's READY/release lease."""

    def __init__(self, arena, index):
        self.directory, self.index = arena.directory, index
        self.header = index["publication"]
        self.tensors, self.frames, self.gaps = _metadata_views(
            arena._metadata_mapping, index["metadata_counts"]
        )
        for array in (self.tensors, self.frames, self.gaps):
            array.flags.writeable = False

    def read_catalog(self):
        """Cold local admission only; never retain the parsed static graph."""
        descriptor = self.index["catalog"]
        raw = (self.directory / descriptor["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("shared canonical catalog changed")
        return orjson.loads(raw)


class HostPreparedSnapshot:
    """One publication's local views into backend-retained pinned host storage."""

    def __init__(self, arena, publication, offsets):
        self.arena, self.publication, self.offsets = arena, publication, offsets
        self.directory = arena.directory
        self.worker = arena._worker if arena.owner else None
        index = publication.index
        self.ready_state = {
            "token": index["token"],
            "state": "READY",
            "generation": index["encoded"]["generation"],
        }

    def get(self, ordinal):
        start = int(self.offsets[ordinal])
        count = int(self.publication.tensors["prepared_nbytes"][ordinal])
        return self.arena.mapping[start : start + count]

    def mark_reusable(self):
        # Resume follows all original engine ranks applying successfully. The
        # child's exact-state check prevents stale callbacks releasing newer data.
        if self.worker is not None:
            self.worker.release(self.ready_state)

    def close(self):
        # Backend retains its mapping after all DE or H2D readers drain.
        self.arena = None
        self.publication = self.offsets = None
