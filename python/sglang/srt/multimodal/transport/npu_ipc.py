"""Ascend NPU transport for multimodal feature tensors.

The producer pickles the public ``reduce_tensor`` payload ``(reconstruct_func,
args)`` and every consumer rebuilds the tensor on its own device with
``rebuild_npu_tensor`` instead of mapping a CUDA feature pool.  Proxies,
producer-side bookkeeping and device helpers live here so ``cuda_ipc`` stays
free of platform branches.
"""

import logging
import pickle
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if _is_npu:
    # Installs the reduce_tensor patches onto torch.multiprocessing.reductions.
    import torch_npu.multiprocessing.reductions  # noqa: F401

# Payload layout written by torch_npu's reductions patch: ``args[6]`` is the
# storage device index, ``args[13]`` the IPC event handle and ``args[14]`` the
# flag telling the consumer to wait on that event.
_NPU_IPC_DEVICE_ARG_INDEX = 6
_NPU_IPC_EVENT_ARG_INDEX = 13  # event handle; the next slot is the sync flag

# Producer-side keepalive.  torch_npu releases an exported block as soon as the
# producer drops its source tensor, which can happen while consumers are still
# rebuilding it, so sources are held until their request finishes.  Module level
# because proxies are pickled and shipped to other processes.
_keepalive_lock = threading.Lock()
# Wrapped features not yet bound to a request id (see ``bind_keepalive``).
_pending_source_tensors: Dict[int, torch.Tensor] = {}
# Sources held per request id until ``release_keepalive`` runs.
_keepalive_by_request: Dict[str, List[torch.Tensor]] = {}
# Bytes currently parked by per-request / not-yet-bound sources.
_keepalive_bytes = 0
_pending_bytes = 0
# Count of features that fell back to CPU transport instead of IPC.
_fallback_count = 0
_KEEPALIVE_BUDGET_DIVISOR = 16  # at most 1/16 of HBM parked by the producer
_MEMINFO_CACHE_TTL_S = 0.1  # mem_get_info costs ~0.4 ms on Ascend
_meminfo_cache: Dict[Optional[int], Tuple[float, int, int]] = {}
_REBUILD_LOG_INTERVAL = 1024  # sample the per-feature rebuild log
_rebuild_count = 0


def _device_mem_info(device_index: Optional[int]) -> Tuple[int, int]:
    """Return cached ``(free, total)`` bytes of *device_index*.

    ``(-1, -1)`` when the query fails; the short TTL hides the per-call cost of
    ``mem_get_info`` (about 0.4 ms on Ascend).
    """
    now = time.monotonic()
    cached = _meminfo_cache.get(device_index)
    if cached is not None and now - cached[0] < _MEMINFO_CACHE_TTL_S:
        return cached[1], cached[2]
    try:
        free, total = torch.npu.mem_get_info(device_index)
    except Exception:
        return -1, -1
    _meminfo_cache[device_index] = (now, free, total)
    return free, total


def _effective_budget_bytes(device_index: Optional[int]) -> Tuple[int, int]:
    """Return ``(budget, free)`` bytes the producer is allowed to park.

    The budget is ``total / 16`` so exported sources never take more than a
    sixteenth of HBM, then clamped to the currently free bytes.
    """
    free, total = _device_mem_info(device_index)
    if total <= 0:
        return 0, free
    cap = total // _KEEPALIVE_BUDGET_DIVISOR
    if free < 0:
        return cap, free
    return min(cap, free), free


def bind_keepalive(rid: str, mm_inputs) -> None:
    """Move wrapped sources of *mm_inputs* from *pending* to request *rid*.

    Runs once the scheduler knows the request id; the sources behind the
    proxies stay alive until ``release_keepalive(rid)``.
    """
    if not _is_npu or not rid or not mm_inputs:
        return
    global _keepalive_bytes, _pending_bytes

    items = getattr(mm_inputs, "mm_items", None)
    if items is None:
        items = mm_inputs if isinstance(mm_inputs, (list, tuple)) else [mm_inputs]

    tensors: List[torch.Tensor] = []
    for item in items:
        for field in ("feature", "precomputed_embeddings"):
            value = getattr(item, field, None)
            if isinstance(value, NpuIpcTensorTransportProxy):
                tensor = _pending_source_tensors.pop(id(value), None)
                if tensor is not None:
                    tensors.append(tensor)
    if not tensors:
        return

    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    # wrap_tensor already reserved these bytes; only move the accounting.
    with _keepalive_lock:
        _keepalive_by_request.setdefault(rid, []).extend(tensors)
        _keepalive_bytes += nbytes
        _pending_bytes -= nbytes
        logger.debug(
            "npu_ipc bind rid=%s bytes=%d total=%d pending=%d reqs=%d",
            rid,
            nbytes,
            _keepalive_bytes,
            len(_pending_source_tensors),
            len(_keepalive_by_request),
        )


def release_keepalive(rid: str) -> None:
    """Drop the source tensors held for *rid* and let torch_npu reclaim them."""
    if not _is_npu or not rid:
        return
    global _keepalive_bytes

    with _keepalive_lock:
        tensors = _keepalive_by_request.pop(rid, None)
        if not tensors:
            return
        _keepalive_bytes -= sum(t.numel() * t.element_size() for t in tensors)
        logger.debug(
            "npu_ipc release rid=%s total=%d pending=%d reqs=%d fallbacks=%d",
            rid,
            _keepalive_bytes,
            len(_pending_source_tensors),
            len(_keepalive_by_request),
            _fallback_count,
        )

    # Dropping the references only makes the blocks eligible; the caching
    # allocator reuses them after the next collect.
    tensors.clear()
    torch.npu.ipc_collect()


def is_device_tensor(tensor: torch.Tensor) -> bool:
    """Whether *tensor* lives on the accelerator (NPU or CUDA)."""
    return tensor.device.type == ("npu" if _is_npu else "cuda")


def current_device() -> int:
    """Index of the accelerator device bound to this process."""
    if _is_npu:
        return torch.npu.current_device()
    return torch.cuda.current_device()


@contextmanager
def _device_guard(device_id: int) -> Iterator[None]:
    """Run the enclosed block with *device_id* as the current device."""
    ctx = torch.npu.device(device_id) if _is_npu else torch.cuda.device(device_id)
    with ctx:
        yield


def _pack_tensor(tensor: torch.Tensor) -> tuple:
    """Return ``(reconstruct_func, args)`` from the public ``reduce_tensor``.

    Consumers unpickle the pair and feed it to ``rebuild_npu_tensor``.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    return reduce_tensor(tensor)


# Export granularity is the storage, not the view.  One request ships several
# views over a single allocation; exporting each view would keep the whole block
# pinned in torch_npu's limbo.  Export it once, then rewrite the payload's copy
# of shape / stride / nbytes / storage_offset so every view rebuilds from it.
_export_lock = threading.Lock()
_export_cache: Dict[int, Tuple[tuple, tuple, "weakref.ReferenceType"]] = {}


def _export_storage_once(tensor: torch.Tensor) -> Tuple[tuple, tuple]:
    """Export *tensor*'s storage once and return a per-view ``(func, args)``.

    The cached payload is copied and its view geometry patched: 1/2 shape and
    stride, 3 storage offset, 8 byte span, 9 offset in bytes.
    """
    storage = tensor.untyped_storage()
    base_ptr = storage.data_ptr()
    elem_size = tensor.element_size()

    with _export_lock:
        cached = _export_cache.get(base_ptr)
    reused = cached is not None and cached[2]() is not None
    if reused:
        func, base_args, _ = cached
    else:
        func, raw_args = _pack_tensor(tensor)
        base_args = _flush_before_export(raw_args)
        with _export_lock:
            if len(_export_cache) > 64:
                for key in [k for k, v in _export_cache.items() if v[2]() is None]:
                    _export_cache.pop(key, None)
            _export_cache[base_ptr] = (func, base_args, weakref.ref(tensor))

    # The payload mirrors the positional args of ``rebuild_npu_tensor`` (see
    # ``torch_npu.multiprocessing.reductions``); only the view geometry differs
    # between slices of one storage.  Copy first: ``base_args`` is shared.
    patched = list(base_args)
    patched[1] = tensor.size()  # shape
    patched[2] = tensor.stride()  # stride
    patched[3] = 0  # element offset; the byte offset at [9] carries it instead
    # Byte span this view covers: 1 + (size - 1) * stride per dim.  Using stride
    # rather than numel keeps padded / non-contiguous views correct.
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.size(), tensor.stride())
    )
    patched[8] = span * elem_size  # storage_size_bytes: this view's span only
    patched[9] = tensor.storage_offset() * elem_size  # where the view starts
    return func, tuple(patched)


def _flush_before_export(args: tuple) -> tuple:
    """Synchronize the producing stream and clear ``event_sync_required``.

    torch_npu attaches an IPC event that dies with the source storage and that
    the producer never waits on; once an early consumer drops its view, a later
    one opening the payload fails with ``AclIpcOpenEventHandle 107017``.  After
    draining the stream the data is complete, so clearing the flag (``args[14]``)
    keeps consumers off the dead handle.
    """
    if len(args) < _NPU_IPC_EVENT_ARG_INDEX + 2:
        return args
    if not args[_NPU_IPC_EVENT_ARG_INDEX + 1]:
        return args
    torch.npu.current_stream().synchronize()
    patched = list(args)
    patched[_NPU_IPC_EVENT_ARG_INDEX + 1] = False
    return tuple(patched)


def _rebuild_tensor(func, args: tuple, rebuild_device_idx: int) -> torch.Tensor:
    """Rebuild a producer payload on *rebuild_device_idx*.

    ``args[6]`` (storage device index) is redirected to the consumer device.
    A cross-device mapping points at peer HBM: kernels can read it, but host
    copies fail with ``memcpy exception`` (507001), so the tensor is cloned into
    local memory.  Same-device rebuilds stay zero-copy.
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
    # Older payloads may still carry the flag cleared by the producer.
    if len(redirected) >= _NPU_IPC_EVENT_ARG_INDEX + 2:
        redirected[_NPU_IPC_EVENT_ARG_INDEX + 1] = False

    with _device_guard(rebuild_device_idx):
        tensor = func(*redirected)
        if (
            producer_device_idx is not None
            and producer_device_idx != rebuild_device_idx
        ):
            tensor = tensor.clone()
    return tensor


class NpuIpcTensorTransportProxy(CudaIpcTensorTransportProxy):
    """Consumer-side proxy for one tensor shipped through torch_npu IPC."""

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
        obj = cls.__new__(cls)
        obj.proxy_state = proxy_state
        obj._restore_npu_ipc_proxy_state()
        return obj

    def _restore_npu_ipc_proxy_state(self) -> None:
        """Restore attributes from ``proxy_state`` after unpickling."""
        ipc_extra = self.proxy_state["ipc_extra"]
        self.npu_ipc_payload = ipc_extra["npu_ipc_payload"]
        self.reconstruct_tensor = None
        self._consumer_acknowledged = True
        # No pool, but the msgpack proxy encoder still serializes these.
        self.sync_data_meta = None
        self.sync_buffer = None
        self._pool_storage = None
        self._borrowed_storage = None
        self._borrowed_base_address = None
        self._borrowed_device_id = None

    def borrow_on_target_device(
        self, rebuild_device_idx: int
    ) -> Optional[torch.Tensor]:
        """There is no pool lease; consumers reconstruct instead."""
        return None

    def acknowledge_consumption(
        self, consumer_count: int = 1, consumer_rank: Optional[int] = None
    ) -> None:
        """Nothing is borrowed from a pool, so there is nothing to acknowledge."""
        return

    def reconstruct_on_target_device(
        self,
        rebuild_device_idx,
        consumer_count: int = 1,
        consumer_rank: Optional[int] = None,
    ):
        """Rebuild once per device and cache the result on the proxy."""
        rebuild_device = torch.device(f"npu:{rebuild_device_idx}")
        global _rebuild_count
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        func, args = pickle.loads(self.npu_ipc_payload)
        self.reconstruct_tensor = _rebuild_tensor(func, args, rebuild_device_idx)
        _rebuild_count += 1
        if _rebuild_count == 1 or _rebuild_count % _REBUILD_LOG_INTERVAL == 0:
            logger.debug(
                "NPU IPC rebuild success on device %s: shape=%s, dtype=%s, "
                "nbytes=%d, rebuilds=%d.",
                rebuild_device,
                tuple(self.reconstruct_tensor.shape),
                self.reconstruct_tensor.dtype,
                self.reconstruct_tensor.numel()
                * self.reconstruct_tensor.element_size(),
                _rebuild_count,
            )
        return self.reconstruct_tensor


class NpuIpcFeatureTransport:
    """Producer side: export tensors over IPC, keep them alive, cancel on abort."""

    def __init__(self):
        """Log the transport choice once at startup."""
        logger.info(
            "NPU IPC transport uses torch_npu reduce_tensor/rebuild memory "
            "sharing; no feature pool is reserved."
        )

    def wrap_tensor(
        self, tensor: torch.Tensor, *, use_pool_handle_cache: bool
    ) -> Optional[NpuIpcTensorTransportProxy]:
        """Export *tensor* over IPC, or return ``None`` to use CPU transport.

        Reserves keepalive budget first (over budget => ``None``), exports the
        storage once, then holds the source until it is bound to a request.
        """
        global _pending_bytes, _fallback_count
        nbytes = tensor.numel() * tensor.element_size()
        device_index = tensor.device.index
        with _keepalive_lock:
            budget, free = _effective_budget_bytes(device_index)
            if _keepalive_bytes + _pending_bytes + nbytes > budget:
                _fallback_count += 1
                if _fallback_count == 1:  # static quota: warn once
                    limit = (
                        "physical free memory"
                        if free >= 0 and budget == free
                        else "device quota"
                    )
                    logger.warning(
                        "NPU IPC keepalive %s reached (%d MiB effective, free "
                        "%.2f MiB) for a %.2f MiB feature (held %.2f MiB); "
                        "falling back to CPU transport; further features will "
                        "do the same silently.",
                        limit,
                        budget // (1024 * 1024),
                        free / (1024 * 1024) if free >= 0 else -1.0,
                        nbytes / (1024 * 1024),
                        (_keepalive_bytes + _pending_bytes) / (1024 * 1024),
                    )
                return None
            _pending_bytes += nbytes
        try:
            func, args = _export_storage_once(tensor)
        except Exception as exc:
            with _keepalive_lock:
                _pending_bytes -= nbytes
            logger.warning(
                "NPU IPC reduce_tensor failed (%s); falling back to CPU transport.",
                exc,
            )
            return None
        proxy = NpuIpcTensorTransportProxy(pickle.dumps((func, args)))
        with _keepalive_lock:
            # Bound to the request later; the source must live until then.
            _pending_source_tensors[id(proxy)] = tensor
        return proxy

    def cancel_proxy(self, proxy: NpuIpcTensorTransportProxy) -> None:
        """Release the reservation of a proxy whose request never reached dispatch."""
        global _pending_bytes
        with _keepalive_lock:
            tensor = _pending_source_tensors.pop(id(proxy), None)
            if tensor is not None:
                _pending_bytes -= tensor.numel() * tensor.element_size()
        tensor = None
        torch.npu.ipc_collect()
