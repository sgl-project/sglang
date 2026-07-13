"""Host-local pull of published weights (the /pull_weights endpoint).

A trainer publishes each weight sync as a version directory ``weight_v{N:06d}/``
under a shared ``source_dir``. Each version is a canonical HF checkpoint
directory of one of two kinds, distinguished by its index metadata:

- **full**: an ordinary checkpoint. Pulling it copies it into the host-local
  ``local_checkpoint_dir``, replacing whatever is there — no history needed.
- **delta** (index metadata carries ``delta_encoding``): safetensors files
  holding zstd-compressed per-tensor diffs against version N-1, plus per-tensor
  checksums of the new state. Pulling it patches the local checkpoint in place.

Version 0 is the engine's own base checkpoint (``model_path``). Every host of a
(possibly multi-node) deployment runs the same pull; the engine then reloads the
local checkpoint through the ordinary ``update_weights_from_disk`` path.

``pull()`` is safe to call concurrently from every scheduler rank on a host: a
per-host file lock serializes the work and an applied-version marker makes the
extra calls no-ops.
"""

from __future__ import annotations

import fcntl
import glob
import importlib
import json
import logging
import mmap
import os
import shutil
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional

import numpy as np
import zstandard

logger = logging.getLogger(__name__)

# The delta-apply phases (decompress, XOR/scatter, checksum) are memory-bandwidth
# bound and release the GIL, so a thread pool over tensors recovers the
# bandwidth one thread leaves idle.
NUM_WORKERS = min(32, (os.cpu_count() or 8))

# Per-checkpoint dir holding the applied-version marker and the pull lock.
SYNC_DIR = ".weight_sync"


def pull(
    local_checkpoint_dir: str,
    base_dir: str,
    source_dir: str,
    target_version: int,
    pre_read_hook: Optional[str] = None,
) -> None:
    """Bring the host-local checkpoint up to ``target_version``.

    Seeds from the newest full checkpoint at or below the target — the engine's
    own base (``base_dir``) for a pure-delta stream, a published full version
    otherwise — then applies the remaining deltas in order. A local checkpoint
    already past the seed point just continues its delta chain. Raises on any
    per-tensor checksum mismatch (fail loud, never serve bad weights).
    """
    # Object-store-backed shared filesystems lack cross-host read-after-write
    # consistency: the publisher's files only appear here after an explicit
    # refresh, which the deployment supplies as this hook. POSIX shared
    # filesystems (NFS, Lustre, ...) need none.
    if target_version > 0 and pre_read_hook:
        _load_hook(pre_read_hook)(source_dir, target_version)
    with _pull_lock(local_checkpoint_dir):
        applied = _read_applied_version(local_checkpoint_dir)  # None on a fresh host
        # Scan back from the target for the newest full version. Stop at the
        # local state — below it a reset can never be needed (or, on a fresh
        # host, at 0 = the engine's base).
        floor = applied if applied is not None else 0
        start = target_version
        while start > floor and _is_delta(_version_dir(source_dir, start)):
            start -= 1
        if applied is None or start > applied:
            seed_dir = base_dir if start == 0 else _version_dir(source_dir, start)
            _reset_checkpoint(seed_dir, local_checkpoint_dir, start)
        else:
            start = applied
        for version in range(start + 1, target_version + 1):
            _apply_delta(local_checkpoint_dir, _version_dir(source_dir, version))


def _load_hook(path: str):
    module_path, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


def _version_dir(source_dir: str, version: int) -> str:
    return os.path.join(source_dir, f"weight_v{version:06d}")


def _is_delta(version_dir: str) -> bool:
    """A version is a delta iff its index metadata declares an encoding; an
    ordinary HF checkpoint (with or without an index) is a full version."""
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"published weight version missing: {version_dir}")
    try:
        with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
            return "delta_encoding" in json.load(f).get("metadata", {})
    except FileNotFoundError:
        return False


class _Adler32:
    """adler32 behind the incremental .update / .hexdigest interface the hash objects expose."""

    def __init__(self):
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


def _checksum(algorithm: str, buf) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()


@contextmanager
def _pull_lock(local_checkpoint_dir: str):
    sync = os.path.join(local_checkpoint_dir, SYNC_DIR)
    os.makedirs(sync, exist_ok=True)
    with open(os.path.join(sync, "lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_applied_version(local_checkpoint_dir: str) -> Optional[int]:
    try:
        with open(os.path.join(local_checkpoint_dir, SYNC_DIR, "state.json")) as f:
            return int(json.load(f)["version"])
    except FileNotFoundError:
        return None


def _write_applied_version(local_checkpoint_dir: str, version: int) -> None:
    path = os.path.join(local_checkpoint_dir, SYNC_DIR, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": f"{version:06d}"}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _drop_page_cache(path: str) -> None:
    """Evict a file from the page cache (POSIX_FADV_DONTNEED)."""
    if not hasattr(os, "posix_fadvise"):  # POSIX-only (absent on macOS/Windows)
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _reset_checkpoint(src_dir: str, local_checkpoint_dir: str, version: int) -> None:
    """Make local_checkpoint_dir an exact copy of the full checkpoint in src_dir
    (files the new checkpoint doesn't have — e.g. differently-sharded old ones —
    are pruned). Later deltas chain on top of this state."""
    logger.info(
        "Pulling full checkpoint v%d %s -> %s", version, src_dir, local_checkpoint_dir
    )
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    src_files = [entry for entry in os.scandir(src_dir) if entry.is_file()]
    for entry in src_files:
        shutil.copy2(entry.path, os.path.join(local_checkpoint_dir, entry.name))
        # don't let the source evict the local copy we keep resident
        _drop_page_cache(entry.path)
    names = {entry.name for entry in src_files}
    for entry in os.scandir(local_checkpoint_dir):
        if entry.is_file() and entry.name not in names:
            os.remove(entry.path)
    # a truncated copy (e.g. an object-store mount surfacing metadata before
    # bytes) must fail loud, not serve bad weights
    for entry in src_files:
        copied = os.path.getsize(os.path.join(local_checkpoint_dir, entry.name))
        if copied != entry.stat().st_size:
            raise RuntimeError(
                f"size mismatch copying {entry.name}: src {entry.stat().st_size} != local {copied}"
            )
    _write_applied_version(local_checkpoint_dir, version)


def _tensor_locations(ckpt_dir: str) -> dict:
    """Map each tensor name to (file, byte offset, nbytes) by reading every safetensors header."""
    locations = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations


def _apply_delta(local_checkpoint_dir: str, version_dir: str) -> None:
    """Apply one version's delta in place: decompress + apply + checksum each tensor across a thread
    pool (each writes a distinct mmap region, so the writes don't conflict). Any mismatch raises.
    """
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        meta = json.load(f)["metadata"]
    applied = _read_applied_version(local_checkpoint_dir)
    if applied == int(meta["version"]):
        return
    if applied != int(meta["base_version"]):
        raise RuntimeError(
            f"out-of-order delta: local at {applied}, delta builds on {meta['base_version']}"
        )
    if meta["compression_format"] != "zstd":
        raise NotImplementedError(
            f"compression {meta['compression_format']!r} not supported"
        )
    encoding = meta["delta_encoding"]
    algorithm = meta["checksum_format"]
    locations = _tensor_locations(local_checkpoint_dir)
    open_mmaps = {}
    mismatches = []
    lock = threading.Lock()
    file_bytes = []  # keep alive: items hold zero-copy views into these
    items = []  # (name, compressed_view, path, offset, nbytes, want_checksum)
    try:
        for delta_file in sorted(glob.glob(os.path.join(version_dir, "*.safetensors"))):
            with open(delta_file, "rb") as f:
                blob = f.read()
            file_bytes.append(blob)
            (header_len,) = struct.unpack("<Q", blob[:8])
            header = json.loads(blob[8 : 8 + header_len])
            want_checksums = header.get("__metadata__", {})
            view = memoryview(blob)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                path, offset, nbytes = locations[name]
                if path not in open_mmaps:
                    fh = open(path, "r+b")
                    open_mmaps[path] = (fh, mmap.mmap(fh.fileno(), 0))
                data_start = 8 + header_len
                items.append(
                    (
                        name,
                        view[data_start + begin : data_start + end],
                        path,
                        offset,
                        nbytes,
                        want_checksums.get(name),
                    )
                )

        # prefetch into page cache (evicted during the rollout) so the apply
        # doesn't fault from cold storage
        for _, mm in open_mmaps.values():
            try:
                mm.madvise(mmap.MADV_WILLNEED)
            except (OSError, AttributeError, ValueError):
                pass

        def apply_xor(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            region = np.ndarray(
                (nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset
            )
            hasher = _new_hasher(algorithm)
            reader = zstandard.ZstdDecompressor().stream_reader(compressed)
            pos = 0
            # 2 MB chunks stay L2-resident across decompress -> XOR -> checksum
            while pos < nbytes:
                block = reader.read(min(2 << 20, nbytes - pos))
                if not block:
                    break
                chunk = np.frombuffer(block, dtype=np.uint8)
                region[pos : pos + chunk.size] ^= chunk
                hasher.update(region[pos : pos + chunk.size])
                pos += chunk.size
            if hasher.hexdigest() != want:
                with lock:
                    mismatches.append(name)

        def apply_overwrite(item) -> None:
            name, compressed, path, offset, nbytes, want = item
            delta = np.frombuffer(
                zstandard.ZstdDecompressor().decompress(compressed), dtype=np.uint8
            )
            region = np.ndarray(
                (nbytes,), dtype=np.uint8, buffer=open_mmaps[path][1], offset=offset
            )
            count = int.from_bytes(delta[:4], "little")
            positions = np.frombuffer(delta[4 : 4 + 4 * count], dtype="<u4")
            region[positions] = delta[4 + 4 * count :]
            if _checksum(algorithm, region) != want:
                with lock:
                    mismatches.append(name)

        if encoding == "xor":
            apply_tensor = apply_xor
        elif encoding == "overwrite":
            apply_tensor = apply_overwrite
        else:
            raise NotImplementedError(f"delta encoding {encoding!r} not supported")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            list(pool.map(apply_tensor, items))
        # no msync: the engine reads these pages via the shared cache; durability
        # isn't needed (a host that loses the cache rebuilds from base)
    finally:
        for fh, mm in open_mmaps.values():
            mm.close()
            fh.close()
    if mismatches:
        raise RuntimeError(
            f"checksum mismatch for {len(mismatches)} tensors after applying {version_dir}: "
            f"{sorted(mismatches)[:20]}"
        )
    _write_applied_version(local_checkpoint_dir, int(meta["version"]))
