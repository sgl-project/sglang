"""Shared-memory tensor transport: the shm rung of the TensorRef ladder.

A ref is a plain msgpack-safe dict: {"transport": "shm", "name": str,
"dtype": str, "shape": [int, ...]}. Readers copy out and close. The request
side unlinks its own request segments once the response arrives and unlinks
response segments after reading them. Segments are named through
make_shm_name so cleanup_stale_shm can reclaim them if the owning client
dies before unlinking.
"""

from contextlib import suppress
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Dict, List

import numpy as np
import torch

from sglang.srt.utils.stale_shm_cleanup import make_shm_name

SHM_TRANSPORT = "shm"


def _untrack(segment: shared_memory.SharedMemory) -> None:
    """https://stackoverflow.com/q/62748654: SharedMemory registers every
    segment with the process-local resource tracker, which unlinks it again
    at process exit. Lifecycle here is cross-process (the peer unlinks), so
    drop the registration on both create and attach. Leaked segments are
    reclaimed by cleanup_stale_shm via the pid embedded in the name."""
    with suppress(Exception):
        resource_tracker.unregister(segment._name, "shared_memory")


def is_shm_ref(value: Any) -> bool:
    return isinstance(value, dict) and value.get("transport") == SHM_TRANSPORT


def read_shm_tensor(ref: Dict[str, Any]) -> np.ndarray:
    """Copy a peer-created segment out into process-local memory."""
    segment = shared_memory.SharedMemory(name=ref["name"])
    _untrack(segment)
    try:
        view = np.ndarray(
            tuple(ref["shape"]), dtype=np.dtype(ref["dtype"]), buffer=segment.buf
        )
        return view.copy()
    finally:
        segment.close()


def write_shm_tensor(tensor: np.ndarray, *, kind: str) -> Dict[str, Any]:
    """Publish a tensor into a fresh segment and return its ref."""
    name = make_shm_name(kind)
    segment = shared_memory.SharedMemory(name=name, create=True, size=tensor.nbytes)
    _untrack(segment)
    try:
        np.ndarray(tensor.shape, dtype=tensor.dtype, buffer=segment.buf)[...] = tensor
    finally:
        segment.close()
    return {
        "transport": SHM_TRANSPORT,
        "name": name,
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def package_hidden_states(chunks: List[torch.Tensor], *, kind: str) -> Dict[str, Any]:
    """Ship accumulated per-chunk hidden states as one [n, hidden] segment."""
    rows = torch.cat([c if c.dim() == 2 else c.unsqueeze(0) for c in chunks])
    return write_shm_tensor(rows.float().numpy(), kind=kind)
