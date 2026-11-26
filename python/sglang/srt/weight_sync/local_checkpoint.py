"""Host-local pull of published weights (the /pull_weights endpoint).

A trainer publishes each weight sync as a version directory ``weight_v{N:06d}/``
under a shared ``source_dir``. Each version is a canonical HF checkpoint
directory of one of two kinds, distinguished by its index metadata:

- **full**: an ordinary checkpoint. Pulling it copies it into the host-local
  ``local_checkpoint_dir``, replacing whatever is there; no history needed.
- **delta** (index metadata carries ``delta_encoding``): safetensors files
  holding zstd-compressed per-tensor diffs against version N-1, plus per-tensor
  checksums of the new state. Pulling it patches the local checkpoint in place.

Version 0 is the engine's own base checkpoint (``model_path``). Every host of a
(possibly multi-node) deployment runs the same pull; the engine then reloads the
local checkpoint through the ordinary ``update_weights_from_disk`` path.

``pull()`` is safe to call concurrently from every scheduler rank on a host: a
per-host file lock serializes the work and an applied-version marker makes the
extra calls no-ops. The marker is dropped before the local files are touched, so
a pull that fails midway leaves a host that reseeds on its next pull.
"""

from __future__ import annotations

import fcntl
import functools
import glob
import json
import logging
import mmap
import os
import shutil
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from typing import List, Optional

import msgspec
import numpy as np
import zstandard
from safetensors import safe_open

from sglang.srt.utils import dynamic_import

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
    snapshot: bool = False,
) -> None:
    """Bring the host-local checkpoint up to ``target_version``: seed from the newest full
    version at or below it (v0 is ``base_dir``), then apply the deltas after it in order."""
    if snapshot:
        destination = snapshot_path(local_checkpoint_dir, target_version)
        with _pull_lock(local_checkpoint_dir):
            if os.path.isdir(destination):
                return
            if target_version > 0 and pre_read_hook:
                dynamic_import(pre_read_hook)(source_dir, target_version)
            _publish_snapshot(destination, base_dir, source_dir, target_version)
        return

    if target_version > 0 and pre_read_hook:
        dynamic_import(pre_read_hook)(source_dir, target_version)
    with _pull_lock(local_checkpoint_dir):
        _materialize(local_checkpoint_dir, base_dir, source_dir, target_version)


def _materialize(
    local_checkpoint_dir: str,
    base_dir: str,
    source_dir: str,
    target_version: int,
    *,
    validate_index: bool = False,
) -> None:
    applied = _read_applied_version(local_checkpoint_dir)  # None on a fresh host
    if applied is not None and target_version < applied:
        raise RuntimeError(
            f"{local_checkpoint_dir} is at v{applied}, past the requested v{target_version}; "
            "deltas only apply forward, so pull an older version into a fresh local_checkpoint_dir"
        )
    # below the local state a reseed is never needed; a fresh host bottoms out at v0
    floor = applied if applied is not None else 0
    start = target_version
    while start > floor and _is_delta(_version_dir(source_dir, start)):
        start -= 1
    if applied is None or start > applied:
        seed_dir = base_dir if start == 0 else _version_dir(source_dir, start)
        _reset_checkpoint(seed_dir, local_checkpoint_dir, start)
        if validate_index:
            _validate_snapshot_seed(local_checkpoint_dir)
    else:
        start = applied
    for version in range(start + 1, target_version + 1):
        _apply_delta(
            local_checkpoint_dir,
            _version_dir(source_dir, version),
            validate_index=validate_index,
        )


def snapshot_path(local_checkpoint_dir: str, version: int) -> str:
    if not os.path.isabs(local_checkpoint_dir):
        raise ValueError("snapshot requires an absolute local_checkpoint_dir")
    if version < 0:
        raise ValueError("snapshot requires a nonnegative target_version")
    return os.path.join(local_checkpoint_dir, SYNC_DIR, "snapshots", f"v{version:06d}")


def _validate_snapshot_seed(directory: str) -> None:
    # Check the files actually copied. Opening the header validates tensor offsets
    # and file length without reading a second copy of the tensor data.
    shards = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            expected = json.load(f)["weight_map"]
        if not isinstance(expected, dict):
            raise ValueError("Snapshot checkpoint index requires a weight_map object")
    elif shards == [os.path.join(directory, "model.safetensors")]:
        expected = None
    else:
        raise ValueError("Snapshot requires a checkpoint index or model.safetensors")
    found = {}
    for shard in shards:
        with safe_open(shard, framework="np") as reader:
            for name in reader.keys():
                if name in found:
                    raise ValueError(f"Duplicate tensor {name} in snapshot seed")
                found[name] = os.path.basename(shard)
    if not found or (expected is not None and found != expected):
        raise ValueError(
            "Incomplete snapshot seed: tensors do not match the checkpoint index"
        )


def _publish_snapshot(
    destination: str, base_dir: str, source_dir: str, version: int
) -> None:
    # Build from the trainer publication, independent of any unvalidated state
    # left by a legacy in-place pull. Only the final rename exposes it to readers.
    root = os.path.dirname(destination)
    os.makedirs(root, exist_ok=True)
    staging = os.path.join(root, ".staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, SYNC_DIR))
    try:
        _materialize(staging, base_dir, source_dir, version, validate_index=True)
        if _read_applied_version(staging) != version:
            raise RuntimeError(f"Checkpoint did not reach version {version}")
        shutil.rmtree(os.path.join(staging, SYNC_DIR))
        os.rename(staging, destination)
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging)


def _version_dir(source_dir: str, version: int) -> str:
    return os.path.join(source_dir, f"weight_v{version:06d}")


def _is_delta(version_dir: str) -> bool:
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"published weight version missing: {version_dir}")
    try:
        with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
            return "delta_encoding" in json.load(f).get("metadata", {})
    except FileNotFoundError:  # a single-file HF checkpoint has no index
        return False


# adler32 behind the incremental .update / .hexdigest interface of the hash objects
class _Adler32:
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
        try:
            import blake3
        except ImportError as e:
            raise ImportError("blake3 checksums need `pip install blake3`") from e

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


def _clear_applied_version(local_checkpoint_dir: str) -> None:
    # a host with no marker reseeds on its next pull, so drop it before mutating
    try:
        os.remove(os.path.join(local_checkpoint_dir, SYNC_DIR, "state.json"))
    except FileNotFoundError:
        pass


def _write_applied_version(local_checkpoint_dir: str, version: int) -> None:
    path = os.path.join(local_checkpoint_dir, SYNC_DIR, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": f"{version:06d}"}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _drop_page_cache(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _reset_checkpoint(src_dir: str, local_checkpoint_dir: str, version: int) -> None:
    logger.info(
        "Pulling full checkpoint v%d %s -> %s", version, src_dir, local_checkpoint_dir
    )
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    _clear_applied_version(local_checkpoint_dir)
    src_files = [entry for entry in os.scandir(src_dir) if entry.is_file()]
    for entry in src_files:
        shutil.copy2(entry.path, os.path.join(local_checkpoint_dir, entry.name))
        # don't let the source evict the local copy we keep resident
        _drop_page_cache(entry.path)
    # an exact copy: drop files the new checkpoint lacks, e.g. a different sharding
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
    locations = {}  # name -> (file, byte offset, nbytes)
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


class _TensorPatch(msgspec.Struct, frozen=True):
    name: str
    payload: memoryview  # zstd-compressed delta
    target: mmap.mmap  # the local safetensors file holding the tensor
    offset: int
    nbytes: int
    checksum: str  # of the patched tensor

    def region(self) -> np.ndarray:
        return np.ndarray(
            (self.nbytes,), dtype=np.uint8, buffer=self.target, offset=self.offset
        )


def _patch_xor(patch: _TensorPatch, algorithm: str) -> bool:
    region = patch.region()
    hasher = _new_hasher(algorithm)
    reader = zstandard.ZstdDecompressor().stream_reader(patch.payload)
    pos = 0
    # 2 MB chunks stay L2-resident across decompress -> XOR -> checksum
    while pos < patch.nbytes:
        block = reader.read(min(2 << 20, patch.nbytes - pos))
        if not block:
            break
        chunk = np.frombuffer(block, dtype=np.uint8)
        region[pos : pos + chunk.size] ^= chunk
        hasher.update(region[pos : pos + chunk.size])
        pos += chunk.size
    return hasher.hexdigest() == patch.checksum


def _patch_overwrite(patch: _TensorPatch, algorithm: str) -> bool:
    delta = np.frombuffer(
        zstandard.ZstdDecompressor().decompress(patch.payload), dtype=np.uint8
    )
    region = patch.region()
    count = int.from_bytes(delta[:4], "little")
    positions = np.frombuffer(delta[4 : 4 + 4 * count], dtype="<u4")
    region[positions] = delta[4 + 4 * count :]
    return _checksum(algorithm, region) == patch.checksum


_PATCHERS = {"xor": _patch_xor, "overwrite": _patch_overwrite}


def _read_patches(
    *,
    version_dir: str,
    locations: dict,
    stack: ExitStack,
    expected: Optional[dict] = None,
) -> List[_TensorPatch]:
    mmaps = {}
    patches = []
    seen = set()
    for delta_file in sorted(glob.glob(os.path.join(version_dir, "*.safetensors"))):
        with open(delta_file, "rb") as f:
            blob = memoryview(f.read())
        (header_len,) = struct.unpack("<Q", blob[:8])
        header = json.loads(bytes(blob[8 : 8 + header_len]))
        checksums = header.pop("__metadata__")
        data_start = 8 + header_len
        for name, info in header.items():
            if expected is not None:
                if name in seen or expected.get(name) != os.path.basename(delta_file):
                    raise ValueError(
                        f"Delta tensor {name} does not match the checkpoint index"
                    )
                seen.add(name)
            begin, end = info["data_offsets"]
            path, offset, nbytes = locations[name]
            if path not in mmaps:
                fh = stack.enter_context(open(path, "r+b"))
                mmaps[path] = stack.enter_context(mmap.mmap(fh.fileno(), 0))
            patches.append(
                _TensorPatch(
                    name=name,
                    payload=blob[data_start + begin : data_start + end],
                    target=mmaps[path],
                    offset=offset,
                    nbytes=nbytes,
                    checksum=checksums[name],
                )
            )
    if expected is not None and seen != expected.keys():
        raise ValueError(
            "Incomplete snapshot delta: tensors do not match the checkpoint index"
        )
    # prefetch into page cache (evicted during the rollout) so the apply
    # doesn't fault from cold storage
    for mm in mmaps.values():
        mm.madvise(mmap.MADV_WILLNEED)
    return patches


def _apply_delta(
    local_checkpoint_dir: str, version_dir: str, *, validate_index: bool = False
) -> None:
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    meta = index["metadata"]
    if validate_index and not isinstance(index.get("weight_map"), dict):
        raise ValueError("Snapshot delta index requires a weight_map object")
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
    if meta["delta_encoding"] not in _PATCHERS:
        raise NotImplementedError(
            f"delta encoding {meta['delta_encoding']!r} not supported"
        )
    patch_tensor = functools.partial(
        _PATCHERS[meta["delta_encoding"]], algorithm=meta["checksum_format"]
    )
    # an unusable checksum format must fail before the local checkpoint is touched
    _new_hasher(meta["checksum_format"])
    locations = _tensor_locations(local_checkpoint_dir)
    # xor is not idempotent: a retry over half-patched bytes must reseed, not xor again
    _clear_applied_version(local_checkpoint_dir)
    with ExitStack() as stack:
        patches = _read_patches(
            version_dir=version_dir,
            locations=locations,
            stack=stack,
            expected=index["weight_map"] if validate_index else None,
        )
        # each patch writes a distinct mmap region, so the workers never conflict
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            matched = list(pool.map(patch_tensor, patches))
        # no msync: the engine reads these pages via the shared cache; durability
        # isn't needed (a host that loses the cache rebuilds from base)
    mismatches = sorted(p.name for p, ok in zip(patches, matched) if not ok)
    if mismatches:
        raise RuntimeError(
            f"checksum mismatch for {len(mismatches)} tensors after applying {version_dir}: "
            f"{mismatches[:20]}"
        )
    _write_applied_version(local_checkpoint_dir, int(meta["version"]))
