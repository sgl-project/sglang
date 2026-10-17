"""Cross-platform helpers for the multimodal feature transport.

CUDA and Ascend NPU pool implementations agree on a small set of primitives:
device introspection, stream/device guards, and the reduce/rebuild pair.
Ascend NPU does *not* use the pool: it follows the official torch_npu IPC docs
(``docs/memory_sharing_ipc.md``, method two) and transports a tensor through the
public ``torch.multiprocessing.reductions.reduce_tensor`` -> ``rebuild_npu_tensor``
pair, so no undocumented ACL API is needed.  The CUDA value-signal primitives
used by the bounded pool stay in ``transport/memory_pool.py``, and the
CUDA-only storage share/open helpers stay in ``transport/cuda_ipc.py``.

Every platform branch is kept in one place so callers stay host agnostic.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


def _npu_available() -> bool:
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None:
        return False
    return bool(getattr(npu_mod, "is_available", lambda: False)())


IS_NPU = _npu_available()

if IS_NPU:
    # Importing this module installs the ``reduce_tensor`` / ``reduce_storage``
    # patches onto ``torch.multiprocessing.reductions`` through torch_npu's
    # normal import path and triggers lazy device init.
    import torch_npu.multiprocessing.reductions  # noqa: F401

# Position of the storage device index inside the ``reduce_tensor`` argument
# tuple produced by torch_npu's multiprocessing reductions patch.
_NPU_IPC_DEVICE_ARG_INDEX = 6


def is_device_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` lives on the accelerator of this node."""
    return tensor.device.type == device_type()


def device_type() -> str:
    return "npu" if IS_NPU else "cuda"


def device_str(device_id: int) -> str:
    return f"{device_type()}:{device_id}"


def make_device(device_id: int) -> torch.device:
    return torch.device(device_str(device_id))


def current_device() -> int:
    if IS_NPU:
        return torch.npu.current_device()
    return torch.cuda.current_device()


@contextmanager
def device_guard(device_id: int) -> Iterator[None]:
    if IS_NPU:
        ctx = torch.npu.device(device_id)
    else:
        ctx = torch.cuda.device(device_id)
    with ctx:
        yield


def npu_ipc_pack_tensor(tensor: torch.Tensor) -> tuple:
    """Prepare a tensor for cross-process transport on this node.

    Returns ``(reconstruct_func, args)``.  On Ascend NPU this is the public
    ``reduce_tensor`` exported by ``torch.multiprocessing.reductions`` (patched
    by torch_npu), whose ``args[6]`` holds the storage device index.  On CUDA
    the tuple is returned as a fallback but the pool path is preferred there.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    return reduce_tensor(tensor)


def npu_ipc_rebuild_tensor(func, args: tuple, rebuild_device_idx: int) -> torch.Tensor:
    """Reconstruct a tensor previously prepared by :func:`npu_ipc_pack_tensor`.

    ``args[6]`` is the storage device index; it is redirected to
    ``rebuild_device_idx`` so the shared storage is opened on the consumer's
    device (single-node IPC).  The position is verified before being rewritten
    so that a layout change in ``reduce_tensor`` cannot silently corrupt an
    unrelated argument.

    When the storage is opened on a different device than the producer's, the
    returned tensor is backed by peer HBM.  Compute reads that mapping fine, but
    the DMA engine rejects host copies from it (``memcpy exception``, 507001),
    so anything that calls ``.cpu()`` on the feature later would abort the
    process.  Copy such tensors into the consumer's own memory to keep the
    rebuilt tensor indistinguishable from a locally allocated one.
    """
    redirected = list(args)
    producer_device_idx = None
    if len(redirected) > _NPU_IPC_DEVICE_ARG_INDEX and isinstance(
        redirected[_NPU_IPC_DEVICE_ARG_INDEX], int
    ):
        producer_device_idx = redirected[_NPU_IPC_DEVICE_ARG_INDEX]
        redirected[_NPU_IPC_DEVICE_ARG_INDEX] = rebuild_device_idx
    else:
        logger.warning(
            "NPU IPC payload args layout changed (len=%d); "
            "rebuilding on the producer device instead of %d.",
            len(redirected),
            rebuild_device_idx,
        )
    with device_guard(rebuild_device_idx):
        tensor = func(*redirected)
        if producer_device_idx is not None and producer_device_idx != rebuild_device_idx:
            tensor = tensor.clone()
    return tensor
