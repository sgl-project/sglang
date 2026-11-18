"""NPU host DRAM mapped into device VA space for the AIV IO path.

torch pin_memory buffers are only reachable by the SDMA engine; AIV kernels
(e.g. kv_exchange) can only de-reference host VAs that were mapped into the
device VA space first. SGLANG_HICACHE_HOST_MEM_BACKEND selects the backend
(a NpuHostMemoryBackend subclass resolved by get_host_memory_backend()); ACL
is used when the variable is unset:

- ``memfabric`` (MemfabricHostMemoryBackend): a MemFabric ``Offload``
  instance owns the host pool; ``pool.initialize`` reserves and
  pre-registers the whole DRAM range and ``pool.empty`` hands out blocks.
  The HiCache host pool is allocated through the Offload instance and the
  L2<->L1 IO uses the AIV kv_exchange kernel.
- ``acl`` (AclHostMemoryBackend): standard CANN runtime registration
  (aclrtHostRegister with ACL_HOST_REGISTER_MAPPED, see the kernel-side
  ``host_memory`` module).  Plain pageable host tensors are allocated
  with torch.empty and registered so the AIV kv_exchange kernel can
  resolve their device pointers. The kernel-side auto routing sends every
  transfer to the AIV operator.

ascendc_io_enabled() gates the AIV IO path and is true for both backends.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.device_pool_info import (
    MHABufferInfo,
    MLABufferInfo,
    MLASplitBuffers,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo

logger = logging.getLogger(__name__)

_MEMFABRIC_GB = 1024**3


def host_memory_backend() -> str:
    """Use ACL by default; honor an explicitly selected host-memory backend."""
    return (os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND") or "acl").lower()


def memfabric_host_memory_enabled() -> bool:
    """MemFabric offload entity backs the host pool (offload.empty blocks)."""
    return host_memory_backend() == "memfabric"


def acl_host_memory_enabled() -> bool:
    """aclrtHostRegister backs the host pool (pageable torch.empty blocks)."""
    return host_memory_backend() == "acl"


def ascendc_io_enabled() -> bool:
    """Use the AIV kv_exchange kernel for HiCache L2<->L1 IO.

    Both registered-host-memory backends map host DRAM into the device VA
    space, which is what the AIV kernel de-references, so either one
    enables this IO path.
    """
    return host_memory_backend() in ("memfabric", "acl")


class NpuHostMemoryBackend:
    """Base class for host-DRAM backends mapped into the device VA space.

    A backend hands out host blocks that the AIV kv_exchange kernel can
    de-reference directly.  Host pools declare their combined block budget
    with ensure_capacity() before the first alloc(); MemFabric needs the
    declaration (the entity reservation is fixed by the first one) while
    the aclrt path registers each block at allocation time and keeps the
    inherited no-op.
    """

    def ensure_capacity(self, total_bytes: int, device_id: int) -> None:
        """Declare the combined host-pool budget before allocation."""

    def alloc(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str,
        pin_memory: bool,
        allocator: None,
    ) -> torch.Tensor:
        """Allocate one host block visible to the AIV kv_exchange kernel."""
        raise NotImplementedError


class MemfabricHostMemoryBackend(NpuHostMemoryBackend):
    """Host blocks from a MemFabric Offload instance (pool.empty).

    One ``memfabric_hybrid.offload.Offload`` instance owns the host pool:
    ``pool.initialize`` reserves and pre-registers the whole DRAM range, so
    every ``pool.malloc``/``pool.empty`` block is device-visible immediately
    (16-byte-bump allocation, no per-block GB rounding), and
    ``pool.host_register`` on a block is an idempotent in-pool check.  The
    instance is published via the kernel-side ``set_memfabric_pool``
    so its register/get-device-pointer calls resolve through the same pool.
    """

    def __init__(self):
        self._pool = None
        self._initialized = False
        self._reserved_bytes = 0
        self._allocated_bytes = 0
        self._device_id = None

    def _get_offload(self):
        if self._pool is None:
            try:
                from memfabric_hybrid import offload
            except ImportError as exc:
                raise ImportError(
                    "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires "
                    "the memfabric_hybrid package (provides the Offload "
                    "host-memory pool). Install it or unset the env var."
                ) from exc
            self._pool = offload.Offload()
            self._offload_module = offload
        return self._offload_module

    def _register_block(self, tensor: torch.Tensor) -> None:
        """Enter an MF-allocated block into the kernel DVA table.

        The pool itself is already device-mapped at initialize time; this
        registration makes the block (and its interior views) resolvable by
        the AIV kv_exchange kernel through the host_memory table. Registration
        happens once, at allocation time; interior views resolve by byte
        offset at transfer time.
        """
        try:
            from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
                register_host_memory,
            )
        except ImportError as exc:
            raise ImportError(
                "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires sgl-kernel-npu "
                "with host_memory.register_host_memory (AIV kv_exchange needs "
                "registered MF blocks)"
            ) from exc
        register_host_memory(tensor)

    def ensure_capacity(self, total_bytes: int, device_id: int) -> None:
        """Initialize the host pool, sized by total_bytes.

        total_bytes is the combined byte size of all buffers the calling
        host pool is about to allocate (ultimately derived from
        --hicache-size / --hicache-ratio). The entity is sized by the first
        declaration; later host pools in the same process must fit into
        what is left.
        """
        if not self._initialized:
            try:
                from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
                    set_memfabric_pool,
                )
            except ImportError as exc:
                raise ImportError(
                    "SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric requires sgl-kernel-npu "
                    "with host_memory.set_memfabric_pool"
                ) from exc

            offload = self._get_offload()
            config = offload.OffloadConfig()
            config.device_id = device_id
            config.reserve_size = total_bytes
            # LOCAL scene requires alloc_size == reserve_size.
            config.alloc_size = total_bytes
            config.scene = offload.Scene.LOCAL
            config.flags = 0
            ret = self._pool.initialize(config)
            if ret != 0:
                self._pool = None
                raise RuntimeError(
                    f"Offload.initialize failed: ret={ret}, "
                    f"device_id={device_id}, reserve_size={total_bytes}"
                )
            set_memfabric_pool(self._pool)
            self._initialized = True
            self._reserved_bytes = total_bytes
            self._device_id = device_id
            logger.info(
                "[HiCache] memfabric host pool initialized (Offload entry): "
                "reserve=%.2fGB device=%d (exact combined budget)",
                total_bytes / _MEMFABRIC_GB,
                device_id,
            )
        elif self._device_id != device_id:
            raise RuntimeError(
                f"memfabric host pool is already initialized for device "
                f"{self._device_id}, got device {device_id}"
            )
        remaining = self._reserved_bytes - self._allocated_bytes
        if total_bytes > remaining:
            raise RuntimeError(
                f"memfabric host memory exhausted: need "
                f"{total_bytes / _MEMFABRIC_GB:.2f}GB, "
                f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
                f"{self._reserved_bytes / _MEMFABRIC_GB:.2f}GB reserve "
                "(sized automatically from the first L2 host pool, i.e. from "
                "--hicache-size / --hicache-ratio). All host pools of the "
                "process share this reserve."
            )

    def alloc(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str,
        pin_memory: bool,
        allocator: None,
    ) -> torch.Tensor:
        """Allocate one host-pool block and enter it into the DVA table.

        The Python-side accounting tracks the exact block byte size.
        """
        if not self._initialized:
            raise RuntimeError("MemFabric ensure_capacity must run before alloc")
        nbytes = math.prod(dims) * dtype.itemsize
        remaining = self._reserved_bytes - self._allocated_bytes
        if nbytes > remaining:
            raise RuntimeError(
                f"memfabric host memory exhausted: block needs "
                f"{nbytes / _MEMFABRIC_GB:.2f}GB, "
                f"only {remaining / _MEMFABRIC_GB:.2f}GB left of the "
                f"{self._reserved_bytes / _MEMFABRIC_GB:.2f}GB reserve"
            )
        tensor = self._pool.empty(list(dims), dtype=dtype)
        self._register_block(tensor)
        self._allocated_bytes += nbytes
        return tensor


class AclHostMemoryBackend(NpuHostMemoryBackend):
    """Pageable torch.empty blocks registered with aclrtHostRegister.

    Standard CANN runtime registration: plain pageable host pages are
    mapped into the device VA space by aclrtHostRegister(ACL_HOST_REGISTER_MAPPED)
    inside the kernel-side ``host_memory`` module.  There is no
    pre-reserved entity and nothing to declare up front, so ensure_capacity
    stays the inherited no-op; the kernel-side host_memory table resolves
    interior views by byte offset at transfer time.
    """

    def _register_block(self, tensor: torch.Tensor) -> None:
        try:
            from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import (
                register_acl_host_memory,
            )
        except ImportError as exc:
            raise ImportError(
                "SGLANG_HICACHE_HOST_MEM_BACKEND=acl requires sgl-kernel-npu "
                "with host_memory.register_acl_host_memory (AIV kv_exchange "
                "needs aclrt-registered host blocks)"
            ) from exc
        register_acl_host_memory(tensor)

    def alloc(
        self,
        dims: tuple,
        dtype: torch.dtype,
        device: str,
        pin_memory: bool,
        allocator: None,
    ) -> torch.Tensor:
        """Allocate one pageable host block and map it for NPU access."""
        tensor = torch.empty(list(dims), dtype=dtype)
        self._register_block(tensor)
        return tensor


_backend: NpuHostMemoryBackend | None = None


def get_host_memory_backend() -> NpuHostMemoryBackend | None:
    """Return the backend selected by SGLANG_HICACHE_HOST_MEM_BACKEND.

    ACL is the default. The instance is cached, so the selection and the
    MemFabric entity state are process-wide singletons.
    """
    global _backend
    if _backend is None:
        if memfabric_host_memory_enabled():
            _backend = MemfabricHostMemoryBackend()
        elif acl_host_memory_enabled():
            _backend = AclHostMemoryBackend()
    return _backend


def reserve_hybrid_host_memory(*, kv_bytes: int, mamba_bytes: int) -> None:
    """Reserve all hybrid KV and Mamba blocks before the first MF allocation."""
    backend = get_host_memory_backend()
    if backend is not None:
        backend.ensure_capacity(kv_bytes + mamba_bytes, torch.npu.current_device())


def reserve_mla_split_host_memory(
    *,
    num_pages: int,
    page_size: int,
    num_layers: int,
    k_width: int,
    v_width: int,
    dtype: torch.dtype,
    buffers: MLASplitBuffers,
) -> None:
    """Reserve the split MLA host buffers before allocating them."""
    backend = get_host_memory_backend()
    if backend is None:
        return

    total_bytes = _mla_split_host_buffer_bytes(
        num_tokens=num_pages * page_size,
        num_layers=num_layers,
        k_width=k_width,
        v_width=v_width,
        dtype=dtype,
        buffers=buffers,
    )
    backend.ensure_capacity(total_bytes, torch.npu.current_device())


def _mla_split_host_buffer_bytes(
    *,
    num_tokens: int,
    num_layers: int,
    k_width: int,
    v_width: int,
    dtype: torch.dtype,
    buffers: MLASplitBuffers,
) -> int:
    num_kv_rows = num_tokens * num_layers
    total_bytes = num_kv_rows * k_width * dtype.itemsize
    if v_width > 0:
        total_bytes += num_kv_rows * v_width * dtype.itemsize

    num_indexer_rows = num_tokens * len(buffers.index_layer_indices)
    if buffers.index_k is not None:
        total_bytes += num_indexer_rows * buffers.index_width * dtype.itemsize
    if buffers.index_scales is not None:
        total_bytes += num_indexer_rows * 4  # Host scales are FP32.

    return total_bytes


def _mha_host_buffer_bytes(
    *,
    num_tokens: int,
    num_layers: int,
    head_num: int,
    k_width: int,
    v_width: int,
    dtype: torch.dtype,
) -> int:
    """Size the K and V host buffers for a token capacity."""
    return num_tokens * num_layers * head_num * (k_width + v_width) * dtype.itemsize


def reserve_mha_host_memory(
    *,
    num_pages: int,
    page_size: int,
    num_layers: int,
    head_num: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    """Declare the combined K/V host-buffer budget before MHA allocation."""
    backend = get_host_memory_backend()
    if backend is None:
        return

    total_bytes = _mha_host_buffer_bytes(
        num_tokens=num_pages * page_size,
        num_layers=num_layers,
        head_num=head_num,
        k_width=head_dim,
        v_width=head_dim,
        dtype=dtype,
    )
    backend.ensure_capacity(total_bytes, torch.npu.current_device())


def reserve_mamba_host_memory(*, num_slots: int, bytes_per_slot: int) -> None:
    """Declare Mamba's combined state bytes before allocating any buffers."""
    backend = get_host_memory_backend()
    total_bytes = num_slots * bytes_per_slot
    if backend is not None and total_bytes:
        backend.ensure_capacity(total_bytes, torch.npu.current_device())


def mamba_host_allocation_bytes(
    buffer_info,
    draft_infos,
    *,
    device_capacity: int,
    host_to_device_ratio: float,
    host_size: float,
) -> int:
    """Price Mamba's temporal, conv and side-state blocks before construction."""
    from sglang.srt.mem_cache.pool_host.base import sync_fixed_hicache_size

    num_layers = buffer_info.temporal.shape[0] + len(draft_infos)
    temporal_bytes = (
        math.prod(buffer_info.temporal.shape[2:])
        * buffer_info.temporal.dtype.itemsize
    )
    conv_bytes = sum(
        math.prod(conv.shape[2:]) * conv.dtype.itemsize for conv in buffer_info.conv
    )
    side_bytes = sum(
        math.prod(state.shape[1:]) * state.dtype.itemsize
        for info in (buffer_info, *draft_infos)
        for state in info.extra_checkpoint_buffers
    )
    bytes_per_slot = num_layers * (temporal_bytes + conv_bytes) + side_bytes
    if host_size > 0:
        slots = sync_fixed_hicache_size(
            int(host_size * 1e9 // bytes_per_slot), host_size
        )
    else:
        slots = int(device_capacity * host_to_device_ratio)
    # Mamba uses page_size=1 and retains one padding slot.
    return (slots + 1) * bytes_per_slot


def mha_host_allocation_bytes(
    buffer_info,
    *,
    draft_count: int,
    device_capacity: int,
    host_to_device_ratio: float,
    host_size: float,
    page_size: int,
) -> int:
    """Return the MHA host allocation including its padding page."""
    from sglang.srt.mem_cache.pool_host.base import sync_fixed_hicache_size

    if not isinstance(buffer_info, MHABufferInfo):
        raise NotImplementedError(
            "MemFabric hybrid reservation currently supports MHA + Mamba pools"
        )
    if buffer_info.scales is not None:
        raise NotImplementedError(
            "MemFabric hybrid reservation does not support MHA KV scales"
        )
    layers = len(buffer_info.buffers.k) + draft_count
    bytes_per_token = _mha_host_buffer_bytes(
        num_tokens=1,
        num_layers=layers,
        head_num=buffer_info.head_num,
        k_width=buffer_info.head_dim,
        v_width=buffer_info.v_head_dim,
        dtype=buffer_info.dtype,
    )
    if host_size > 0:
        slots = sync_fixed_hicache_size(
            int(host_size * 1e9 // bytes_per_token), host_size
        )
    else:
        slots = int(device_capacity * host_to_device_ratio)
    pages = slots // page_size + 1
    return pages * page_size * bytes_per_token


def mla_host_allocation_bytes(
    kv_pool,
    buffer_info: MLABufferInfo,
    *,
    draft_count: int,
    device_capacity: int,
    host_to_device_ratio: float,
    host_size: float,
    page_size: int,
) -> int:
    """Return the MLA split buffers that its host pool will allocate."""
    from sglang.srt.mem_cache.pool_host.base import sync_fixed_hicache_size

    buffers = buffer_info.buffers
    if not isinstance(buffers, MLASplitBuffers) or (
        get_memory().hicache_mem_layout
        not in ("page_first_kv_split", "layer_first_kv_split")
    ):
        raise NotImplementedError(
            "MemFabric MLA + Mamba reservation requires an MLA KV split layout"
        )
    parallel = get_parallel()
    if parallel.dcp_enabled:
        page_size //= parallel.attn_dcp_size
    target_layers = kv_pool.layer_num
    if kv_pool.layer_shard_enabled:
        target_layers = (
            target_layers + kv_pool.layer_shard_size - 1
        ) // kv_pool.layer_shard_size
    layers = target_layers + draft_count

    # HostKVCache sizes fixed hicache-size from the logical MLA row width.
    bytes_per_token = kv_pool.kv_cache_dim * buffer_info.dtype.itemsize * layers
    if buffers.index_k is not None:
        index_layers = len(buffers.index_layer_indices)
        bytes_per_token += (
            index_layers * buffers.index_width * buffer_info.dtype.itemsize
        )
        if buffers.index_scales is not None:
            bytes_per_token += index_layers * 4
    if host_size > 0:
        slots = sync_fixed_hicache_size(
            int(host_size * 1e9 // bytes_per_token), host_size
        )
    else:
        slots = int(device_capacity * host_to_device_ratio)
    pages = slots // page_size + 1
    return _mla_split_host_buffer_bytes(
        num_tokens=pages * page_size,
        num_layers=layers,
        k_width=buffers.k_width if buffer_info.packed_fp8 else buffer_info.kv_lora_rank,
        v_width=0 if buffer_info.packed_fp8 else buffers.v_width,
        dtype=buffer_info.dtype,
        buffers=buffers,
    )


def hybrid_host_allocation_bytes(
    *,
    kv_pool,
    kv_host_size: float | None,
    kv_draft_pools,
    page_size: int,
    mamba_pool,
    mamba_info,
    mamba_host_size: float,
    mamba_draft_pools,
) -> tuple[int, int]:
    """Estimate the MHA/MLA KV and Mamba blocks of one hybrid stack."""
    memory = get_memory()
    kv_capacity = kv_pool.host_capacity_tokens
    if kv_capacity is None:
        kv_capacity = kv_pool.size
    mamba_capacity = mamba_pool.host_capacity_tokens
    if mamba_capacity is None:
        mamba_capacity = mamba_pool.size
    kv_info = kv_pool.get_device_pool_infos()[0].buffer_info
    kv_options = dict(
        draft_count=len(kv_draft_pools),
        device_capacity=kv_capacity,
        host_to_device_ratio=memory.hicache_ratio,
        host_size=memory.hicache_size if kv_host_size is None else kv_host_size,
        page_size=page_size,
    )
    if isinstance(kv_info, MLABufferInfo):
        kv_bytes = mla_host_allocation_bytes(kv_pool, kv_info, **kv_options)
    else:
        kv_bytes = mha_host_allocation_bytes(kv_info, **kv_options)
    mamba_bytes = mamba_host_allocation_bytes(
        mamba_info.buffer_info,
        tuple(
            pool.get_device_pool_infos()[0].buffer_info for pool in mamba_draft_pools
        ),
        device_capacity=mamba_capacity,
        host_to_device_ratio=memory.hicache_ratio,
        host_size=mamba_host_size,
    )
    return kv_bytes, mamba_bytes


def reserve_dsv4_paged_host_memory(
    *,
    infos: Mapping[PoolName, DevicePoolInfo],
    num_host_pages: int,
    swa_num_host_pages: int,
    c128_num_host_pages: int,
    num_draft_swa_layers: int,
) -> None:
    """Reserve DSV4 paged buffers before constructing the host pools."""
    if not is_npu():
        return

    # Declare every DSV4 paged host block before constructing the SWA pool.
    # Per-block byte sizes; MemFabric fixes its reservation at the first
    # declaration, so the blocks are summed before declaring.
    paged_blocks = []
    layer_first = get_memory().hicache_mem_layout == "layer_first"

    def add_blocks(layer_bytes):
        if not layer_bytes:
            return
        # Match actual allocator calls: one block per layer for layer_first,
        # one block for the entire pool for page-first layouts.
        if layer_first:
            paged_blocks.extend(layer_bytes)
        else:
            paged_blocks.append(sum(layer_bytes))

    def add_paged(name: PoolName, pages: int, draft_layers: int = 0) -> None:
        info = infos.get(name)
        if info is not None:
            region = info.buffer_info
            add_blocks(
                [pages * region.page_bytes] * (len(region.buffers) + draft_layers)
            )

    add_paged(PoolName.SWA, swa_num_host_pages, num_draft_swa_layers)
    c4 = infos.get(PoolName.DEEPSEEK_V4_C4)
    if c4 is not None and c4.layer_ids:
        for name in (
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
            PoolName.DEEPSEEK_V4_C4_ROPE,
        ):
            add_paged(name, num_host_pages)
        for name in (
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        ):
            checkpoint = infos.get(name)
            if checkpoint is not None:
                add_blocks(
                    [
                        swa_num_host_pages * state.pages.shape[1]
                        for state in checkpoint.buffer_info.states
                    ]
                )
    c128 = infos.get(PoolName.DEEPSEEK_V4_C128)
    if c128 is not None and c128.layer_ids:
        add_paged(PoolName.DEEPSEEK_V4_C128, c128_num_host_pages)
        add_paged(PoolName.DEEPSEEK_V4_C128_ROPE, c128_num_host_pages)
    backend = get_host_memory_backend()
    if paged_blocks and backend is not None:
        backend.ensure_capacity(sum(paged_blocks), torch.npu.current_device())


# ---------------------------------------------------------------------------
# Sync-free H2D upload (NPU)
#
# A pageable .to(device) / torch.tensor(..., device=npu) completes with an
# aclrtStreamSynchronize that drains EVERYTHING queued on the current stream
# (profiler-confirmed on the HiCache load path).  When called on the default
# stream before entering the load/write stream, that sync stalls all queued
# compute; on the load stream it serializes the layer-group pipeline.  Stage
# through pinned memory instead: pinned + non_blocking=True is a genuinely
# async enqueue.  The pinned staging tensors are kept alive until their
# consumer copy retires, tracked via events (Event.query() is host-side and
# never synchronizes).
# ---------------------------------------------------------------------------
_pinned_inflight: list = []


def track_pinned_staging(pinned: torch.Tensor) -> None:
    """Keep a pinned staging tensor alive until its async consumer retires.

    Completed entries are dropped on each call so the list stays small.
    """
    done = torch.npu.Event()
    done.record()
    _pinned_inflight.append((pinned, done))
    _pinned_inflight[:] = [e for e in _pinned_inflight if not e[1].query()]


def to_device_no_sync(cpu_tensor: torch.Tensor, device) -> torch.Tensor:
    """Upload a CPU tensor to the NPU without synchronizing the stream.

    NPU-only helper (torch.npu.Event); callers are on the AscendC IO path.
    """
    pinned = cpu_tensor.pin_memory()
    out = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
    out.copy_(pinned, non_blocking=True)
    track_pinned_staging(pinned)
    return out
