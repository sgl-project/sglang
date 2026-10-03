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

"""Host-local, verified immutable publication snapshots for GPU-delta preparation.

All engines on a host must share this tmpfs directory. The namespace, rather than
container hostname, defines sharing. No distributed or inference collectives run.
"""

import fcntl
import hashlib
import json
import mmap
import os
import shutil
import stat
import time
import uuid
from pathlib import Path


def _cache_root():
    root = Path(
        os.environ.get(
            "WEIGHT_DELTA_HOST_CACHE_DIR", f"/dev/shm/sglang-gpu-delta-{os.getuid()}"
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
        raise ValueError("WEIGHT_DELTA_HOST_CACHE_DIR must be on host-shared tmpfs")
    return root


def _write_snapshot(source, destination, expected_size, expected_sha):
    """Read directly into the retained snapshot; hash exactly those bytes once."""
    read_s = hash_s = 0.0
    with source.open("rb", buffering=0) as incoming, destination.open("xb+") as target:
        before = os.fstat(incoming.fileno())
        if before.st_size != expected_size:
            raise ValueError("delta payload size mismatch")
        started = time.perf_counter()
        if expected_size:
            # Reserve tmpfs pages before writing, so insufficient space raises
            # ENOSPC rather than SIGBUS while touching a sparse mmap.
            os.posix_fallocate(target.fileno(), 0, expected_size)
            with mmap.mmap(target.fileno(), expected_size) as output:
                view = memoryview(output)
                try:
                    position = 0
                    while position < expected_size:
                        count = incoming.readinto(view[position:])
                        if not count:
                            raise ValueError("truncated delta payload")
                        position += count
                    read_s = time.perf_counter() - started
                    started = time.perf_counter()
                    actual_sha = hashlib.sha256(view).hexdigest()
                    hash_s = time.perf_counter() - started
                finally:
                    view.release()
        else:
            actual_sha = hashlib.sha256(b"").hexdigest()
        after = os.fstat(incoming.fileno())
        if (
            incoming.read(1)
            or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or actual_sha != expected_sha
        ):
            raise ValueError("delta payload SHA256/size mismatch or source changed")
    destination.chmod(0o400)
    return read_s, hash_s


def host_cache_id():
    """A shared tmpfs root, not hostname, defines the physical sharing domain."""
    root = _cache_root()
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


def _identity(info):
    # RW mapping metadata timestamps can change on attachment. The feature owns
    # immutable contents after READY; bind the exact retained inode and extent.
    return [info.st_dev, info.st_ino, info.st_size]


def _decode_arena(path, files, entries, pool, metrics):
    layout, size = {}, 0
    for entry in entries:
        count = (
            (entry["nbytes"] if entry["changed_bytes"] else 0)
            if entry["encoding"] == "raw_bytes"
            else (entry["outer"]["decoded_bytes"] if entry["frames"] else 0)
        )
        if count:
            size = (size + 15) // 16 * 16
            layout[entry["name"]] = {"offset": size, "nbytes": count}
            size += count
    futures = []
    with path.open("xb+") as target:
        if size:
            os.posix_fallocate(target.fileno(), 0, size)
            arena = mmap.mmap(target.fileno(), size)
            destination = memoryview(arena)
            started = time.perf_counter()
            error = None
            try:
                for entry in entries:
                    record = layout.get(entry["name"])
                    if record is None:
                        continue
                    offset, count = record["offset"], record["nbytes"]
                    selected = destination[offset : offset + count]
                    if entry["encoding"] == "raw_bytes":
                        raw = entry["raw"]
                        start = raw["encoded_offset"]
                        selected[:] = memoryview(files[raw["file"]])[
                            start : start + count
                        ]
                        continue
                    outer = entry["outer"]
                    start, length = outer["encoded_offset"], outer["encoded_bytes"]
                    payload = memoryview(files[outer["file"]])[start : start + length]
                    futures.append(
                        (
                            outer,
                            pool.executor.submit(
                                pool.decode, payload, outer["frames"], selected
                            ),
                        )
                    )
            except BaseException as exc:
                error = exc
            # Always join every task before allowing source/destination owners to
            # disappear. Failed tasks can retain views through their tracebacks.
            for outer, future in futures:
                try:
                    validation_s, decode_s = future.result()
                    metrics["host_outer_zstd_validate_s"] += validation_s
                    metrics["host_outer_zstd_worker_decode_sum_s"] += decode_s
                    metrics["host_outer_zstd_encoded_bytes"] += outer["encoded_bytes"]
                    metrics["host_outer_zstd_decoded_bytes"] += outer["decoded_bytes"]
                    metrics["host_outer_zstd_tensors"] += 1
                    metrics["host_outer_zstd_frames"] += len(outer["frames"])
                except BaseException as exc:
                    if error is None:
                        error = exc
            metrics["host_outer_zstd_decode_s"] = time.perf_counter() - started
            if error is not None:
                raise error
    return {"tensors": layout, "arena_bytes": size}


class HostDecodedSnapshot:
    """One verified, expanded Snappy/raw arena shared across host-local engines.

    The READY arena is logically immutable. Every process maps the same physical
    tmpfs pages with MAP_SHARED (never COW), registers its own VA for CUDA, and
    retains the registration until all H2D work completes. No allocator or CUDA
    operations run on CPU decode threads.
    """

    def __init__(self, manifest_path, manifest_sha256, manifest, names, pool, timings):
        self.mapping = self.tensor = None
        self.registered = False
        self.root = _cache_root()
        publication = Path(manifest_path).resolve(strict=True)
        records = manifest["files"]
        definitions = {}
        for record in records:
            name, size = record["name"], record["nbytes"]
            if (
                name in definitions
                or Path(name).name != name
                or name in {".", ".."}
                or type(size) is not int
                or size < 0
            ):
                raise ValueError("invalid or duplicate delta payload path/size")
            definitions[name] = {"nbytes": size, "sha256": record["sha256"]}
        expected = {
            "manifest_path": str(publication),
            "manifest_sha256": manifest_sha256,
            "host_tensor_names": names,
            "files": definitions,
        }
        key = hashlib.sha256(
            json.dumps([str(publication), manifest_sha256, names]).encode()
        ).hexdigest()
        self.directory = self.root / key
        self.expected = expected
        metrics = {
            name: 0
            for name in (
                "host_payload_read_s",
                "host_payload_sha256_s",
                "host_payload_hash_bytes",
                "host_payload_hash_files",
                "host_payload_cache_created",
                "host_payload_cache_reused",
                "host_outer_zstd_validate_s",
                "host_outer_zstd_decode_s",
                "host_outer_zstd_worker_decode_sum_s",
                "host_outer_zstd_encoded_bytes",
                "host_outer_zstd_decoded_bytes",
                "host_outer_zstd_tensors",
                "host_outer_zstd_frames",
            )
        }
        waiting = time.perf_counter()
        # Only construction/attachment/removal takes this stable lock. Never hold
        # it while waiting for another scheduler, engine, or GPU stream.
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metrics["host_payload_cache_wait_s"] = time.perf_counter() - waiting
            if self.directory.exists():
                index = json.loads((self.directory / "index.json").read_bytes())
                if index["publication"] != expected:
                    raise ValueError("verified host payload identity differs")
                metrics["host_payload_cache_reused"] = 1
            else:
                temporary = self.root / f".{key}.{uuid.uuid4().hex}.pending"
                temporary.mkdir(mode=0o700)
                encoded_dir = temporary / "encoded"
                encoded_dir.mkdir()
                files = {}
                build_started = time.perf_counter()
                for name, entry in definitions.items():
                    source = (publication.parent / name).resolve(strict=True)
                    if source.parent != publication.parent:
                        raise ValueError("delta payload escapes immutable publication")
                    destination = encoded_dir / name
                    read_s, hash_s = _write_snapshot(
                        source, destination, entry["nbytes"], entry["sha256"]
                    )
                    metrics["host_payload_read_s"] += read_s
                    metrics["host_payload_sha256_s"] += hash_s
                    metrics["host_payload_hash_bytes"] += entry["nbytes"]
                    metrics["host_payload_hash_files"] += 1
                    with destination.open("rb") as payload:
                        files[name] = (
                            mmap.mmap(payload.fileno(), 0, access=mmap.ACCESS_READ)
                            if entry["nbytes"]
                            else b""
                        )
                entries = {entry["name"]: entry for entry in manifest["tensors"]}
                index = _decode_arena(
                    temporary / "arena.bin",
                    files,
                    [entries[name] for name in names],
                    pool,
                    metrics,
                )
                files.clear()
                shutil.rmtree(encoded_dir)
                # RW mapping is required for portable cudaHostRegister support.
                # Only construction writes it; consumers treat all bytes as immutable.
                (temporary / "arena.bin").chmod(0o600)
                index.update(
                    publication=expected,
                    arena_identity=_identity((temporary / "arena.bin").stat()),
                    cpu_workers=pool.workers,
                    build_s=time.perf_counter() - build_started,
                )
                (temporary / "index.json").write_text(json.dumps(index, sort_keys=True))
                (temporary / "index.json").chmod(0o400)
                temporary.rename(self.directory)
                metrics["host_payload_cache_created"] = 1
            arena = self.directory / "arena.bin"
            with arena.open("r+b") as source:
                if _identity(os.fstat(source.fileno())) != index["arena_identity"]:
                    raise ValueError("verified host decoded arena changed")
                if index["arena_bytes"]:
                    self.mapping = mmap.mmap(
                        source.fileno(), 0, access=mmap.ACCESS_WRITE
                    )
            self.index = index
        metrics.update(
            host_shared_arena_bytes=index["arena_bytes"],
            host_shared_build_s=index["build_s"],
            host_outer_zstd_cpu_workers=index["cpu_workers"],
        )
        timings.update(metrics)

    def register(self, device, timings):
        import torch

        self.device = device
        started = time.perf_counter()
        if self.mapping is not None:
            self.tensor = torch.frombuffer(self.mapping, dtype=torch.uint8)
            self.pointer = self.tensor.data_ptr()
            with torch.cuda.device(device):
                result = torch.cuda.cudart().cudaHostRegister(
                    self.pointer, self.tensor.numel(), 1
                )
                if int(result) != 0:
                    raise RuntimeError(
                        f"shared delta cudaHostRegister failed: {result}"
                    )
                self.registered = True
                if not self.tensor.is_pinned():
                    raise RuntimeError(
                        "registered delta mapping is not recognized as CUDA pinned memory"
                    )
        else:
            self.tensor = torch.empty(0, dtype=torch.uint8, device="cpu")
        timings.update(
            host_shared_register_s=time.perf_counter() - started,
            host_shared_registered_bytes=self.tensor.numel(),
            host_shared_register_calls=int(self.registered),
        )

    def get(self, name):
        record = self.index["tensors"][name]
        return self.tensor[record["offset"] : record["offset"] + record["nbytes"]]

    def close(self, *, discard=False):
        if self.registered:
            import torch

            with torch.cuda.device(self.device):
                result = torch.cuda.cudart().cudaHostUnregister(self.pointer)
                if int(result) != 0:
                    raise RuntimeError(
                        f"shared delta cudaHostUnregister failed: {result}"
                    )
            self.registered = False
        # Tensor/memoryview references retain mmap owners; never invalidate a
        # surviving view or mask the original decode error with BufferError.
        self.tensor = self.mapping = None
        if discard:
            with (self.root / ".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.directory.exists():
                    index = json.loads((self.directory / "index.json").read_bytes())
                    if index["publication"] != self.expected:
                        raise ValueError(
                            "host decoded snapshot identity changed at cleanup"
                        )
                    shutil.rmtree(self.directory)
