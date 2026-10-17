"""Ascend NPU device-side transport for multimodal feature tensors.

Ascend NPU does *not* use the CUDA feature pool.  It follows the official
torch_npu IPC docs (``docs/memory_sharing_ipc.md``, method two) and transports
a tensor through the public
``torch.multiprocessing.reductions.reduce_tensor`` -> ``rebuild_npu_tensor``
pair, so no undocumented ACL API is needed.

Keeping the proxy subclass, the producer-side transport, and the device helpers
in this module lets ``cuda_ipc`` stay free of platform branches.
"""

import logging
import pickle
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if _is_npu:
    # Importing this module installs the ``reduce_tensor`` / ``reduce_storage``
    # patches onto ``torch.multiprocessing.reductions`` through torch_npu's
    # normal import path and triggers lazy device init.
    import torch_npu.multiprocessing.reductions  # noqa: F401

# Position of the storage device index inside the ``reduce_tensor`` argument
# tuple produced by torch_npu's multiprocessing reductions patch.
_NPU_IPC_DEVICE_ARG_INDEX = 6


def is_device_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` lives on the accelerator of this node."""
    return tensor.device.type == ("npu" if _is_npu else "cuda")


def current_device() -> int:
    if _is_npu:
        return torch.npu.current_device()
    return torch.cuda.current_device()


@contextmanager
def _device_guard(device_id: int) -> Iterator[None]:
    ctx = torch.npu.device(device_id) if _is_npu else torch.cuda.device(device_id)
    with ctx:
        yield


def _pack_tensor(tensor: torch.Tensor) -> tuple:
    """Prepare a tensor for cross-process transport on this node.

    Returns ``(reconstruct_func, args)``.  On Ascend NPU this is the public
    ``reduce_tensor`` exported by ``torch.multiprocessing.reductions`` (patched
    by torch_npu), whose ``args[6]`` holds the storage device index.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    return reduce_tensor(tensor)


def _rebuild_tensor(func, args: tuple, rebuild_device_idx: int) -> torch.Tensor:
    """Reconstruct a tensor previously prepared by :func:`_pack_tensor`.

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
    with _device_guard(rebuild_device_idx):
        tensor = func(*redirected)
        if (
            producer_device_idx is not None
            and producer_device_idx != rebuild_device_idx
        ):
            tensor = tensor.clone()
    return tensor


class NpuIpcTensorTransportProxy(CudaIpcTensorTransportProxy):
    """Serializable view of one tensor shipped through torch_npu memory sharing.

    The producer fills :attr:`proxy_state` with the pickled ``reduce_tensor``
    pair and the msgpack decoder rebuilds this class from the serialized state.
    There is no pool lease and no control word, so the methods that touch pool
    state are overridden as no-ops; the release helpers are inherited and
    short-circuit because no storage was ever borrowed.
    """

    def __init__(self, payload: bytes):
        self.proxy_state = {
            "ipc_extra": {
                "transport_mode": "npu_ipc",
                "npu_ipc_payload": payload,
            },
            "tensor_data": None,
        }
        self._restore_npu_ipc_proxy_state()

    @classmethod
    def from_proxy_state(cls, proxy_state: dict) -> "NpuIpcTensorTransportProxy":
        """Rebuild the proxy on the consumer side after deserialization."""
        obj = cls.__new__(cls)
        obj.proxy_state = proxy_state
        obj._restore_npu_ipc_proxy_state()
        return obj

    def _restore_npu_ipc_proxy_state(self) -> None:
        ipc_extra = self.proxy_state["ipc_extra"]
        self.npu_ipc_payload = ipc_extra["npu_ipc_payload"]
        self.reconstruct_tensor = None
        self._consumer_acknowledged = True
        # npu_ipc owns no pool, so it has no control-word sync metadata; the
        # attributes still have to exist because the msgpack proxy encoder
        # serializes them.
        self.sync_data_meta = None
        self.sync_buffer = None
        self._pool_storage = None
        self._borrowed_storage = None
        self._borrowed_base_address = None
        self._borrowed_device_id = None

    def borrow_on_target_device(
        self, rebuild_device_idx: int
    ) -> Optional[torch.Tensor]:
        # npu_ipc transport has no pool lease; fall back to reconstruct.
        return None

    def acknowledge_consumption(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        # Nothing to release: the producer-side tensor stays owned by the
        # producer and the rebuilt copy is owned by the consumer.
        return

    def reconstruct_on_target_device(
        self,
        rebuild_device_idx,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
    ):
        rebuild_device = torch.device(f"npu:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        func, args = pickle.loads(self.npu_ipc_payload)
        self.reconstruct_tensor = _rebuild_tensor(func, args, rebuild_device_idx)
        logger.info(
            "NPU IPC rebuild success on device %s: shape=%s, dtype=%s, nbytes=%d.",
            rebuild_device,
            tuple(self.reconstruct_tensor.shape),
            self.reconstruct_tensor.dtype,
            self.reconstruct_tensor.numel() * self.reconstruct_tensor.element_size(),
        )
        return self.reconstruct_tensor


class NpuIpcFeatureTransport:
    """Producer-side wrapper that turns a device tensor into an IPC proxy.

    NPU ships every feature on its own through ``reduce_tensor``, so there is no
    bounded pool to reserve, no lease to track, and nothing to cancel.
    """

    def __init__(self):
        logger.info(
            "NPU IPC transport uses torch_npu reduce_tensor/rebuild memory "
            "sharing; no feature pool is reserved."
        )

    def wrap_tensor(
        self, tensor: torch.Tensor, *, use_pool_handle_cache: bool
    ) -> Optional[NpuIpcTensorTransportProxy]:
        try:
            func, args = _pack_tensor(tensor)
        except Exception as exc:
            logger.warning(
                "NPU IPC reduce_tensor failed (%s); falling back to CPU transport.",
                exc,
            )
            return None
        return NpuIpcTensorTransportProxy(pickle.dumps((func, args)))

    def cancel_proxy(self, proxy: NpuIpcTensorTransportProxy) -> None:
        """NPU IPC owns no lease, nothing to cancel."""
        return
