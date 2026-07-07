"""Materialize published full and delta weights in a host-local checkpoint."""

from __future__ import annotations

import fcntl
import glob
import importlib
import json
import logging
import mmap
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import NamedTuple, Optional

import numpy as np
import zstandard

logger = logging.getLogger(__name__)

_DELTA_WORKERS = min(32, (os.cpu_count() or 8))
_SEED_COPY_WORKERS = int(os.environ.get("SGLANG_SEED_COPY_WORKERS", "8"))
_SEED_COPY_CHUNK_SIZE = 16 << 20
_XOR_CHUNK_SIZE = 2 << 20
_SYNC_DIR = ".weight_sync"


class _DeltaItem(NamedTuple):
    name: str
    compressed_data: memoryview
    path: str
    offset: int
    num_bytes: int
    expected_checksum: Optional[str]


def pull(
    local_checkpoint_dir: str,
    base_dir: str,
    source_dir: str,
    target_version: int,
    pre_read_hook: Optional[str] = None,
) -> None:
    """Apply published versions through ``target_version``.

    Missing or incomplete source data raises ``FileNotFoundError``. Other apply
    failures trigger one clean reseed and replay before being raised.
    """
    with _pull_lock(local_checkpoint_dir):
        applied_version = _read_applied_version(local_checkpoint_dir)
        if applied_version is not None and applied_version >= target_version:
            return
        if target_version > 0 and pre_read_hook:
            _load_hook(pre_read_hook)(source_dir, target_version)
        try:
            _pull_locked(
                local_checkpoint_dir,
                base_dir,
                source_dir,
                target_version,
                force_reseed=False,
            )
        except FileNotFoundError:
            _log_pull_not_found(source_dir, target_version)
            raise
        except Exception:
            logger.exception(
                "Pull to version %d failed; reseeding and replaying",
                target_version,
            )
            _pull_locked(
                local_checkpoint_dir,
                base_dir,
                source_dir,
                target_version,
                force_reseed=True,
            )


def _pull_locked(
    local_checkpoint_dir: str,
    base_dir: str,
    source_dir: str,
    target_version: int,
    force_reseed: bool,
) -> None:
    applied_version = (
        None if force_reseed else _read_applied_version(local_checkpoint_dir)
    )
    search_floor = applied_version if applied_version is not None else 0
    base_version = target_version
    while base_version > search_floor and _is_delta(
        _version_dir(source_dir, base_version)
    ):
        base_version -= 1
    if applied_version is None or base_version > applied_version:
        seed_dir = (
            base_dir if base_version == 0 else _version_dir(source_dir, base_version)
        )
        _reset_checkpoint(seed_dir, local_checkpoint_dir, base_version)
    else:
        base_version = applied_version
    for version in range(base_version + 1, target_version + 1):
        _apply_delta(local_checkpoint_dir, _version_dir(source_dir, version))


def _load_hook(path: str):
    module_path, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


def _log_pull_not_found(source_dir: str, target_version: int) -> None:
    """Log source state after a missing or incomplete version."""
    target_dir = _version_dir(source_dir, target_version)
    try:
        versions = sorted(n for n in os.listdir(source_dir) if n.startswith("weight_v"))
    except OSError as exc:
        versions = [f"<listdir {source_dir} failed: {exc}>"]
    target_contents = None
    if os.path.isdir(target_dir):
        try:
            target_contents = sorted(os.listdir(target_dir))
        except OSError as exc:
            target_contents = [f"<listdir failed: {exc}>"]
    logger.error(
        "Missing weight version %d: versions=%s, target=%s, target_exists=%s, "
        "target_contents=%s, latest=%s",
        target_version,
        versions,
        target_dir,
        os.path.isdir(target_dir),
        target_contents,
        _read_latest_pointer_for_log(source_dir),
    )


def _read_latest_pointer_for_log(source_dir: str) -> str:
    for path in (
        os.path.join(source_dir, "latest"),
        os.path.join(os.path.dirname(source_dir.rstrip("/")), "latest"),
    ):
        try:
            with open(path) as f:
                return f"{path}={f.read().strip()!r}"
        except OSError:
            continue
    return "<no latest pointer found>"


def _version_dir(source_dir: str, version: int) -> str:
    return os.path.join(source_dir, f"weight_v{version:06d}")


def _is_delta(version_dir: str) -> bool:
    """Return whether a version index declares a delta encoding."""
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"published weight version missing: {version_dir}")
    try:
        with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
            return "delta_encoding" in json.load(f).get("metadata", {})
    except FileNotFoundError:
        return False


class _Adler32:
    """Expose Adler-32 through the incremental hash interface."""

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


def _checksum(algorithm: str, buffer) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buffer)
    return hasher.hexdigest()


@contextmanager
def _pull_lock(local_checkpoint_dir: str):
    sync_dir = os.path.join(local_checkpoint_dir, _SYNC_DIR)
    os.makedirs(sync_dir, exist_ok=True)
    with open(os.path.join(sync_dir, "lock"), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read_applied_version(local_checkpoint_dir: str) -> Optional[int]:
    try:
        with open(os.path.join(local_checkpoint_dir, _SYNC_DIR, "state.json")) as f:
            return int(json.load(f)["version"])
    except FileNotFoundError:
        return None


def _write_applied_version(local_checkpoint_dir: str, version: int) -> None:
    path = os.path.join(local_checkpoint_dir, _SYNC_DIR, "state.json")
    temp_path = path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump({"version": f"{version:06d}"}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _drop_page_cache(path: str) -> None:
    """Evict a file from the page cache when supported."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _reset_checkpoint(source_dir: str, local_checkpoint_dir: str, version: int) -> None:
    """Replace the local checkpoint with a full checkpoint."""
    logger.info(
        "Pulling full checkpoint v%d %s -> %s",
        version,
        source_dir,
        local_checkpoint_dir,
    )
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    source_files = [entry for entry in os.scandir(source_dir) if entry.is_file()]
    total_gb = sum(entry.stat().st_size for entry in source_files) / 1e9
    copied_gb = 0.0
    progress_lock = threading.Lock()

    def copy_file(entry) -> None:
        nonlocal copied_gb
        destination = os.path.join(local_checkpoint_dir, entry.name)
        if not (
            os.path.exists(destination) and os.path.samefile(entry.path, destination)
        ):
            with open(entry.path, "rb") as source, open(destination, "wb") as output:
                while chunk := source.read(_SEED_COPY_CHUNK_SIZE):
                    output.write(chunk)
            _drop_page_cache(entry.path)
        with progress_lock:
            copied_gb += entry.stat().st_size / 1e9
            logger.info("Seeding base v%d: %.0f/%.0f GB", version, copied_gb, total_gb)

    worker_count = min(_SEED_COPY_WORKERS, len(source_files) or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(copy_file, source_files))
    source_names = {entry.name for entry in source_files}
    for entry in os.scandir(local_checkpoint_dir):
        if entry.is_file() and entry.name not in source_names:
            os.remove(entry.path)
    for entry in source_files:
        copied_size = os.path.getsize(os.path.join(local_checkpoint_dir, entry.name))
        source_size = entry.stat().st_size
        if copied_size != source_size:
            raise RuntimeError(
                f"size mismatch copying {entry.name}: "
                f"source={source_size}, local={copied_size}"
            )
    _write_applied_version(local_checkpoint_dir, version)


def _tensor_locations(checkpoint_dir: str) -> dict:
    """Map tensor names to file paths, byte offsets, and sizes."""
    locations = {}
    for path in glob.glob(os.path.join(checkpoint_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations


def _expected_safetensors_size(blob: bytes) -> Optional[int]:
    """Return the size declared by a safetensors payload."""
    if len(blob) < 8:
        return None
    header_len = struct.unpack("<Q", blob[:8])[0]
    if len(blob) < 8 + header_len:
        return None
    try:
        header = json.loads(blob[8 : 8 + header_len])
    except ValueError:
        return None
    end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        end = max(end, info["data_offsets"][1])
    return 8 + header_len + end


def _apply_delta(local_checkpoint_dir: str, version_dir: str) -> None:
    """Apply and verify one delta version in place."""
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    metadata = index["metadata"]
    applied_version = _read_applied_version(local_checkpoint_dir)
    if applied_version == int(metadata["version"]):
        return
    for blob_name in sorted(set(index.get("weight_map", {}).values())):
        if not os.path.exists(os.path.join(version_dir, blob_name)):
            raise FileNotFoundError(
                f"incomplete source version {version_dir}: missing blob {blob_name}"
            )
    if applied_version != int(metadata["base_version"]):
        raise RuntimeError(
            f"out-of-order delta: local at {applied_version}, "
            f"delta builds on {metadata['base_version']}"
        )
    if metadata["compression_format"] != "zstd":
        raise NotImplementedError(
            f"compression {metadata['compression_format']!r} not supported"
        )
    encoding = metadata["delta_encoding"]
    checksum_algorithm = metadata["checksum_format"]
    tensor_locations = _tensor_locations(local_checkpoint_dir)
    mapped_files = {}
    checksum_mismatches = []
    mismatch_lock = threading.Lock()
    delta_blobs = []  # Keep blobs alive while memoryviews reference them.
    delta_items = []
    try:
        for delta_file in sorted(glob.glob(os.path.join(version_dir, "*.safetensors"))):
            with open(delta_file, "rb") as f:
                blob = f.read()
            expected_size = _expected_safetensors_size(blob)
            if expected_size is None or len(blob) != expected_size:
                raise FileNotFoundError(
                    f"incomplete source blob {delta_file}: {len(blob)}B, header "
                    f"declares {expected_size}B"
                )
            delta_blobs.append(blob)
            (header_len,) = struct.unpack("<Q", blob[:8])
            header = json.loads(blob[8 : 8 + header_len])
            expected_checksums = header.get("__metadata__", {})
            blob_view = memoryview(blob)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                path, offset, num_bytes = tensor_locations[name]
                if path not in mapped_files:
                    file_handle = open(path, "r+b")
                    mapped_files[path] = (
                        file_handle,
                        mmap.mmap(file_handle.fileno(), 0),
                    )
                data_start = 8 + header_len
                delta_items.append(
                    _DeltaItem(
                        name,
                        blob_view[data_start + begin : data_start + end],
                        path,
                        offset,
                        num_bytes,
                        expected_checksums.get(name),
                    )
                )

        for _, mapped_file in mapped_files.values():
            try:
                mapped_file.madvise(mmap.MADV_WILLNEED)
            except (OSError, AttributeError, ValueError):
                pass

        def apply_xor(item: _DeltaItem) -> None:
            region = np.ndarray(
                (item.num_bytes,),
                dtype=np.uint8,
                buffer=mapped_files[item.path][1],
                offset=item.offset,
            )
            hasher = _new_hasher(checksum_algorithm)
            reader = zstandard.ZstdDecompressor().stream_reader(item.compressed_data)
            position = 0
            while position < item.num_bytes:
                block = reader.read(min(_XOR_CHUNK_SIZE, item.num_bytes - position))
                if not block:
                    break
                chunk = np.frombuffer(block, dtype=np.uint8)
                region[position : position + chunk.size] ^= chunk
                hasher.update(region[position : position + chunk.size])
                position += chunk.size
            if hasher.hexdigest() != item.expected_checksum:
                with mismatch_lock:
                    checksum_mismatches.append(item.name)

        def apply_overwrite(item: _DeltaItem) -> None:
            delta = np.frombuffer(
                zstandard.ZstdDecompressor().decompress(item.compressed_data),
                dtype=np.uint8,
            )
            region = np.ndarray(
                (item.num_bytes,),
                dtype=np.uint8,
                buffer=mapped_files[item.path][1],
                offset=item.offset,
            )
            count = int.from_bytes(delta[:4], "little")
            positions = np.frombuffer(delta[4 : 4 + 4 * count], dtype="<u4")
            region[positions] = delta[4 + 4 * count :]
            if _checksum(checksum_algorithm, region) != item.expected_checksum:
                with mismatch_lock:
                    checksum_mismatches.append(item.name)

        if encoding == "xor":
            apply_tensor = apply_xor
        elif encoding == "overwrite":
            apply_tensor = apply_overwrite
        else:
            raise NotImplementedError(f"delta encoding {encoding!r} not supported")
        with ThreadPoolExecutor(max_workers=_DELTA_WORKERS) as pool:
            list(pool.map(apply_tensor, delta_items))
        for _, mapped_file in mapped_files.values():
            mapped_file.flush()
    finally:
        for file_handle, mapped_file in mapped_files.values():
            mapped_file.close()
            file_handle.close()
    if checksum_mismatches:
        raise RuntimeError(
            f"checksum mismatch for {len(checksum_mismatches)} tensors after "
            f"applying {version_dir}: {sorted(checksum_mismatches)[:20]}"
        )
    _write_applied_version(local_checkpoint_dir, int(metadata["version"]))
