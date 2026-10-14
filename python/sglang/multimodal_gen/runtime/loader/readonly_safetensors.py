"""Read-only mappings of safetensors files.

safetensors maps a file for torch through ``UntypedStorage.from_file(shared=False)``:
a private, writable mapping. On a shared CPU/GPU pool that permission costs
memory: when the device copies from such a mapping the driver pins the pages
with write intent, the kernel breaks copy-on-write, and every page copied
becomes anonymous memory -- 1 GiB copied in 1 GiB of unreclaimable RAM, at a
fraction of the bandwidth (0.1-3.8 GiB/s against 27 GiB/s on a GB10). A
read-only mapping copies at full speed and leaves the pages as page cache.

Frozen weights never need the write permission; a write into one here faults
instead of silently copying the page, which is the invariant we want.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import warnings
from typing import Iterator

import torch

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

# Mappings stay for the life of the process: the tensors handed out are views
# into them, and the layerwise manager keeps such views as its host store.
_MAPPINGS: dict[str, mmap.mmap] = {}


def _mapping(path: str) -> mmap.mmap:
    real = os.path.realpath(path)
    mapped = _MAPPINGS.get(real)
    if mapped is None:
        fd = os.open(real, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            mapped = mmap.mmap(fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE)
        finally:
            os.close(fd)
        _MAPPINGS[real] = mapped
    return mapped


def _header(mapped: mmap.mmap) -> tuple[dict, int]:
    (n,) = struct.unpack("<Q", mapped[:8])
    return json.loads(mapped[8 : 8 + n]), 8 + n


def _tensor(mapped: mmap.mmap, base: int, meta: dict) -> torch.Tensor:
    dtype = _DTYPES[meta["dtype"]]
    start, end = meta["data_offsets"]
    shape = tuple(meta["shape"])
    if end == start:
        return torch.empty(shape, dtype=dtype)
    count = (end - start) // torch.empty((), dtype=dtype).element_size()
    with warnings.catch_warnings():
        # torch warns that the buffer is not writable; that is the point.
        warnings.simplefilter("ignore")
        flat = torch.frombuffer(mapped, dtype=dtype, count=count, offset=base + start)
    return flat.view(shape)


def safetensors_keys(path: str) -> list[str]:
    header, _ = _header(_mapping(path))
    return [name for name in header if name != "__metadata__"]


def iter_safetensors_readonly(path: str) -> Iterator[tuple[str, torch.Tensor]]:
    """(name, tensor) for every tensor in the file, as views of a read-only mapping."""
    mapped = _mapping(path)
    header, base = _header(mapped)
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        yield name, _tensor(mapped, base, meta)


def load_safetensors_readonly(path: str) -> dict[str, torch.Tensor]:
    return dict(iter_safetensors_readonly(path))
