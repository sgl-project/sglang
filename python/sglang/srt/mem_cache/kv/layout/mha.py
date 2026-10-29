"""The MHA layout: per-layer K and V rows in one token-major entry, and the
typed view that reads and stores them over a band of a `KVArena`."""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    UnquantizedKVCacheMethod,
)
from sglang.srt.mem_cache.device_pool_info import MHABufferInfo
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.layout.base import (
    HostPoolOptions,
    SubPoolSpec,
    _check_row_alignment,
    _store_dtype_for,
    check_physical_id_bound,
    dense_rows,
)
from sglang.srt.mem_cache.kv.memory.views import MHAKVView
from sglang.srt.mem_cache.kv.router import KVLayerRouter, PlainPoolRouter
from sglang.srt.mem_cache.layout.fused_draft import DenseDraftRegion, MLADraftRegion
from sglang.srt.mem_cache.layout.token_major import (
    DenseEntryLayout,
    DensePart,
    align_entry_bytes,
    align_part_offset,
    build_dense_views,
    replicated_token_layout,
)
from sglang.srt.mem_cache.memory_pool import (
    _set_kv_buffer_impl,
    _set_mha_kv_buffer_prefix_valid,
    masked_set_kv_buffer_kernel,
    unwrap_write_loc,
    write_loc_is_physical,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import is_cuda
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
    from sglang.srt.mem_cache.kv.memory.arena import KVArena


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` defaults to `head_dim`.

    With `draft_region` set, the draft model's K and V rows are two more
    parts of every slot's entry, after the host parts:

        [ K_0 | V_0 | ... | K_{Lh-1} | V_{Lh-1} | dK_0 | dV_0 | ... | pad ]

    The slot stride stays the one entry, so host and draft views are indexed
    by the same physical token id; only their offsets differ.
    """

    zero_units_on_handout: ClassVar[bool] = True

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    kv_cache_dtype: Optional[torch.dtype] = None
    v_head_dim: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.kv_cache_dtype is None:
            object.__setattr__(self, "kv_cache_dtype", self.store_dtype)
        expected_store_dtype = _store_dtype_for(self.kv_cache_dtype)
        assert self.store_dtype == expected_store_dtype, (
            "MHASubPoolSpec.store_dtype must match the storage dtype required by "
            f"kv_cache_dtype; got kv_cache_dtype={self.kv_cache_dtype}, "
            f"store_dtype={self.store_dtype}, expected={expected_store_dtype}"
        )
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert self.v_head_dim > 0, (
            f"v_head_dim must be positive; got {self.v_head_dim}"
        )
        if self.draft_region is not None:
            assert isinstance(self.draft_region, (DenseDraftRegion, MLADraftRegion)), (
                "dense entries carry dense draft K/V rows, never a state region"
            )
            self.draft_region.validate()

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot, before any draft parts."""
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

    def draft_offset_in_entry(self) -> int:
        """Byte offset of the fused draft parts inside one slot's entry."""
        assert self.draft_region is not None
        return align_part_offset(self.host_entry_bytes())

    def entry_bytes(self) -> int:
        if self.draft_region is not None and self.draft_region.token_replicas > 1:
            return replicated_token_layout(self._target_parts(), self.draft_region)[
                0
            ].entry_bytes
        if self.draft_region is None:
            return align_entry_bytes(self.host_entry_bytes())
        return align_entry_bytes(
            self.draft_offset_in_entry() + self.draft_region.entry_bytes()
        )

    def page_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def _target_parts(self):
        """Token-major entry ``[K_0 | V_0 | K_1 | V_1 | ...]`` per slot; a page
        is ``page_size`` such entries back to back."""
        if self.layer_num:
            _check_row_alignment(self.name, K=self.k_row_bytes(), V=self.v_row_bytes())
        layer_stride = self.k_row_bytes() + self.v_row_bytes()
        parts = (
            DensePart(
                name="k",
                offset_bytes=0,
                layer_stride_bytes=layer_stride,
                layer_num=self.layer_num,
                row_shape=(self.head_num, self.head_dim),
                dtype=self.store_dtype,
            ),
            DensePart(
                name="v",
                offset_bytes=self.k_row_bytes(),
                layer_stride_bytes=layer_stride,
                layer_num=self.layer_num,
                row_shape=(self.head_num, self.v_head_dim),
                dtype=self.store_dtype,
            ),
        )
        if not self.layer_num:
            parts = ()
        return parts

    def layout(self) -> DenseEntryLayout:
        parts = self._target_parts()
        if self.draft_region is not None:
            if self.draft_region.token_replicas > 1:
                return replicated_token_layout(parts, self.draft_region)[0]
            parts += self.draft_region.parts(self.draft_offset_in_entry())
        return DenseEntryLayout(entry_bytes=self.entry_bytes(), parts=parts)

    def views(self, arena) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K and V views over this sub-pool's band of ``arena``,
        indexed by the physical token id."""
        if not self.layer_num:
            return [], []
        rows = dense_rows(arena, self.name)
        check_physical_id_bound(sub_pool=self.name, n_rows=rows)
        return self._part_views(arena, rows, "k"), self._part_views(arena, rows, "v")

    def draft_views(self, arena) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K and V views of the draft parts fused into this
        sub-pool's entries: the same rows, the same ids."""
        assert self.draft_region is not None, (
            f"sub-pool {self.name!r} carries no fused draft region"
        )
        rows = dense_rows(arena, self.name)
        region = self.draft_region
        layout = (
            self.layout()
            if region.token_replicas == 1
            else replicated_token_layout(self._target_parts(), region)[1]
        )
        return tuple(
            build_dense_views(
                arena.raw,
                layout=layout,
                part_name=name,
                num_slots=rows * region.token_replicas,
            )
            for name in (
                ("draft_kv",)
                if isinstance(region, MLADraftRegion)
                else ("draft_k", "draft_v")
            )
        )

    def _part_views(self, arena, rows: int, part: str) -> List[torch.Tensor]:
        return build_dense_views(
            arena.raw, layout=self.layout(), part_name=part, num_slots=rows
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


class MHAView(MHAKVView):
    """One MHA sub-pool's per-layer K and V over its band of a `KVArena`:
    ``(rows, head_num, head_dim)`` tensors whose slot stride is the whole entry,
    indexed by the physical token id. The arena owns, moves and zeroes the
    bytes; writes take physical ids in this sub-pool's id space.

    Its per-layer methods take the view's layer index (the router's
    ``KVLayer.view_layer_id``) offset by ``start_layer``.
    """

    # Writes take physical ids only (`KVWriteLoc` marked physical).
    requires_physical_write_loc = True
    # The view's quantization method; unified memory stores BF16 or FP8 rows.
    is_quantized_kv_cache = False
    k_scale_buffer = None
    v_scale_buffer = None
    # Retraction moves state separately; this view copies token KV only.
    cpu_copy_carries_mamba = False
    layer_shard_enabled = False
    post_capture_active = False
    use_hnd = False
    kv_cache_layout = "page_major"
    # Host sizing reads these when a view's rows are not its token capacity.
    host_capacity_tokens: Optional[int] = None
    host_capacity_bytes: Optional[int] = None
    # Set by HiCache when it stages write-backs for this view.
    hicache_write_back_staging: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    enable_custom_mem_pool = False
    custom_mem_pool = None
    allocation_label = None
    mem_usage = 0.0  # the arena logs the real size
    cpu_offloading_chunk_size = 8192

    def __init__(
        self,
        *,
        arena: "KVArena",
        spec: MHASubPoolSpec,
        page_size: int,
        model_layer_ids: Tuple[int, ...],
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
    ):
        self._has_draft_parts = spec.draft_region is not None
        k_buffer, v_buffer = spec.views(arena)
        self._bind(
            arena=arena,
            sub_pool=spec.name,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            dtype=spec.kv_cache_dtype,
            page_size=page_size,
            entry_bytes=spec.entry_bytes(),
            model_layer_ids=model_layer_ids,
            start_layer=start_layer,
            end_layer=end_layer if end_layer is not None else spec.layer_num - 1,
            enable_alt_stream=enable_alt_stream,
        )

    def _bind(
        self,
        *,
        arena: "KVArena",
        sub_pool: str,
        k_buffer: List[torch.Tensor],
        v_buffer: List[torch.Tensor],
        head_num: int,
        head_dim: int,
        v_head_dim: int,
        dtype: torch.dtype,
        page_size: int,
        entry_bytes: int,
        model_layer_ids: Tuple[int, ...],
        start_layer: int,
        end_layer: int,
        enable_alt_stream: bool,
    ) -> None:
        self._arena = arena
        self._sub_pool_name = sub_pool
        self.k_buffer, self.v_buffer = k_buffer, v_buffer
        # The pages the view spans, and one page envelope's bytes (all layers).
        self.num_pages = (
            k_buffer[0].shape[0] if k_buffer else arena.extent_entries(sub_pool)
        ) // page_size
        self.page_bytes = page_size * entry_bytes
        # Rows 0..page_size-1 are the padding sink; physical ids run to the rows.
        self.size = self.num_pages * page_size - page_size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = _store_dtype_for(dtype)
        self.device = arena.device
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.layer_num = len(k_buffer)
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.model_layer_ids = model_layer_ids
        self.quant_method = UnquantizedKVCacheMethod()
        self.layer_transfer_counter: Optional[LayerDoneCounter] = None
        # The store_cache kernel's row widths.
        self.row_dim = self.head_num * self.head_dim
        self.v_row_dim = self.head_num * self.v_head_dim
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream()
            if enable_alt_stream and (is_cuda() or current_platform.is_cuda_alike())
            else None
        )
        self._init_data_ptrs_and_strides()

    def _init_data_ptrs_and_strides(self) -> None:
        """Per-layer pointers and row strides that HiCache's transfer kernels
        and slot moves read."""
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        buffers = [*self.k_buffer, *self.v_buffer]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in buffers], dtype=torch.uint64, device=self.device
        )
        self.data_strides = torch.tensor(
            [x.stride(0) * x.dtype.itemsize for x in buffers], device=self.device
        )

    # -- what the view is --

    def id_spaces(self) -> "dict[str, IdSpace]":
        """The view's own ids; under unified memory its allocator's tables
        name the slots instead (`ChainAllocator.id_spaces`)."""
        return {IdSpaceKind.FULL: IdSpace(key=(IdSpaceKind.FULL, None))}

    @cached_property
    def layer_router(self) -> KVLayerRouter:
        # A view is one sub-pool, its own router's only view.
        return PlainPoolRouter(self, IdSpaceKind.FULL)

    @property
    def grow_direction(self) -> str:
        """Which end of the arena this sub-pool's band grows from."""
        return self._arena.band(self._sub_pool_name).grow_direction

    def get_kv_cache_quant_method(self):
        return self.quant_method

    def get_v_head_dim(self) -> int:
        return self.v_head_dim

    def get_kv_size_bytes(self):
        return 0, 0  # the arena logs the total; per-sub-pool would double-count

    def get_kv_buffer_shape(self) -> Tuple[torch.Size, torch.Size]:
        return self.k_buffer[0].shape, self.v_buffer[0].shape

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def register_layer_transfer_counter(self, counter: "LayerDoneCounter") -> None:
        self.layer_transfer_counter = counter

    # -- reads --

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._typed(self.k_buffer[layer_id - self.start_layer])

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._typed(self.v_buffer[layer_id - self.start_layer])

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def _typed(self, buffer: torch.Tensor) -> torch.Tensor:
        # FP8 rows are stored as bytes; readers see the cache dtype.
        if self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        return buffer

    # -- writes --

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Stores one layer's K and V rows at physical ids ``loc_info``."""
        loc, _, _ = unwrap_write_loc(loc_info)
        if not write_loc_is_physical(loc_info):
            raise ValueError(
                "set_kv_buffer (MHA view): write loc is not marked physical. "
                "Hand the view KVWriteLoc(loc, physical=True) with ids in its "
                "sub-pool's space."
            )
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA view)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)
        k_buf = self.k_buffer[layer_id - self.start_layer]
        v_buf = self.v_buffer[layer_id - self.start_layer]
        if dcp_kv_mask is not None:
            n, h, d = cache_k.shape
            masked_set_kv_buffer_kernel[(n,)](
                cache_k,
                cache_v,
                k_buf,
                v_buf,
                loc,
                dcp_kv_mask,
                n,
                h,
                d,
                128,
                cache_k.stride(0),
                cache_k.stride(1),
                cache_v.stride(0),
                cache_v.stride(1),
                k_buf.stride(0),
                v_buf.stride(0),
            )
            return
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            k_buf,
            v_buf,
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            # The rows plus the padding sink; padded tokens write there.
            size_limit=self.size + self.page_size,
            alt_stream=self.alt_stream,
            v_row_dim=self.v_row_dim,
        )

    def set_kv_buffer_prefix_valid(
        self,
        layer,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Commit each request's live prefix at physical slot ids."""
        _set_mha_kv_buffer_prefix_valid(
            self,
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "a unified pool moves whole units in its arena; a per-slot move "
            "would corrupt the entries"
        )

    # -- host transfer and PD --

    def host_layout(self):
        from sglang.srt.mem_cache.kv.layout.page import page_envelope_host_layout

        if self._has_draft_parts:
            return page_envelope_host_layout(
                self._arena,
                self._sub_pool_name,
                self.size
                if self.host_capacity_tokens is None
                else self.host_capacity_tokens,
                self.page_size,
            )
        (info,) = self.get_device_pool_infos()
        return MHAHostLayout(
            buffer_info=info.buffer_info,
            capacity=self.size
            if self.host_capacity_tokens is None
            else self.host_capacity_tokens,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            write_back_staging=self.hicache_write_back_staging,
        )

    def get_device_pool_infos(self) -> "tuple[DevicePoolInfo, ...]":
        """The view's buffers, as host assembly reads them."""
        from sglang.srt.mem_cache.device_pool_info import (
            DevicePoolInfo,
            KVBufferLayout,
            KVBuffers,
            MHABufferInfo,
        )
        from sglang.srt.mem_cache.hicache_storage import PoolName

        buffers = KVBuffers(
            k=tuple(self.k_buffer),
            v=tuple(self.v_buffer),
            k_data_ptrs=self.k_data_ptrs,
            v_data_ptrs=self.v_data_ptrs,
            layout=KVBufferLayout.NHD,
        )
        return (
            DevicePoolInfo(
                pool_name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                layer_ids=self.model_layer_ids,
                buffer_info=MHABufferInfo(
                    buffers=buffers,
                    page_size=self.page_size,
                    head_num=self.head_num,
                    head_dim=self.head_dim,
                    v_head_dim=self.v_head_dim,
                    dtype=self.store_dtype,
                    device=(
                        buffers.k[0].device
                        if len(buffers.k)
                        else torch.device(self.device)
                    ),
                    scales=None,
                ),
            ),
        )

    def host_pool_decls(self):
        """The host pool HiCache keeps for this view's buffers."""
        from sglang.srt.mem_cache.pool_host.host_pool_decl import make_kv_pool_decl

        return (make_kv_pool_decl(self),)

    def get_contiguous_buf_infos(self):
        """PD registration: the raw arena as physical page envelopes. Every token
        sub-pool registers the same allocation with its own envelope size; the
        transfer backend deduplicates the memory registration."""
        return (
            [self._arena.raw.data_ptr()],
            [self._arena.accessible_bytes],
            [self.page_bytes],
        )

    def get_page_envelope_buffer(self) -> torch.Tensor:
        """This sub-pool's page-strided view of the arena."""
        page_count = self._arena.num_entries(self._sub_pool_name) // self.page_size
        return self._arena.raw[: page_count * self.page_bytes].view(
            page_count, self.page_bytes
        )

    # -- retraction copies, at physical ids --

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk].to("cpu", non_blocking=True)
                v_cpu = self.v_buffer[layer_id][chunk].to("cpu", non_blocking=True)
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i : i + chunk_size]
                k_cpu, v_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk)
                self.k_buffer[layer_id][chunk] = k_cpu.to(
                    self.k_buffer[0].device, non_blocking=True
                )
                self.v_buffer[layer_id][chunk] = v_cpu.to(
                    self.v_buffer[0].device, non_blocking=True
                )
        current_platform.synchronize()


class DraftMHAView(MHAView):
    """A fused draft runner's K and V: views of the draft parts inside every
    slot entry of its host sub-pool, one region lane per draft layer. Ids are
    the target's physical token ids in the host sub-pool's space; the host's
    unit moves carry the draft bytes, so the draft has no transfer or copy of
    its own. Its methods take the draft's model layer ids."""

    def __init__(
        self,
        *,
        arena: "KVArena",
        host_spec: SubPoolSpec,
        layer_lanes: "dict[int, int]",
        page_size: int,
    ):
        self._has_draft_parts = True
        region = host_spec.draft_region
        assert region is not None, (
            f"sub-pool {host_spec.name!r} carries no fused draft region"
        )
        self.token_replicas = region.token_replicas
        layer_ids = sorted(layer_lanes)
        assert layer_ids, "a fused draft view binds at least one layer"
        start_layer = layer_ids[0]
        assert layer_ids == list(range(start_layer, start_layer + len(layer_ids))), (
            f"fused draft layer ids must be contiguous; got {layer_ids}"
        )
        lanes = [layer_lanes[layer_id] for layer_id in layer_ids]
        assert len(set(lanes)) == len(lanes) and all(
            0 <= lane < region.lane_num for lane in lanes
        ), f"draft lanes {lanes} must be distinct and within range({region.lane_num})"
        k_views, v_views = host_spec.draft_views(arena)
        self.layer_lanes = dict(layer_lanes)
        self._bind(
            arena=arena,
            sub_pool=host_spec.name,
            k_buffer=[k_views[lane] for lane in lanes],
            v_buffer=[v_views[lane] for lane in lanes],
            head_num=region.head_num,
            head_dim=region.head_dim,
            v_head_dim=region.resolved_v_head_dim(),
            # The KV dtype, not its storage: stores cast to it and view the
            # result as the storage dtype (bytes for FP8).
            dtype=region.resolved_kv_dtype(),
            page_size=page_size,
            entry_bytes=host_spec.entry_bytes() // region.token_replicas,
            model_layer_ids=tuple(layer_ids),
            start_layer=start_layer,
            end_layer=start_layer + len(layer_ids) - 1,
            enable_alt_stream=True,
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no region of its own; KV transfer is unsupported"
        )

    def get_page_envelope_buffer(self) -> torch.Tensor:
        raise NotImplementedError("fused draft KV rides in its host's envelopes")

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV")

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV")


class MHAHostLayout(msgspec.Struct, frozen=True, kw_only=True):
    """MHA host rows and their layer order; transfers read the declared buffers."""

    buffer_info: MHABufferInfo
    capacity: int
    start_layer: int
    end_layer: int
    write_back_staging: object = None

    @property
    def device_layer_num(self) -> int:
        return len(self.buffer_info.buffers.k)

    def build(self, options: HostPoolOptions, *, draft_buffer_infos=()):
        from sglang.srt.mem_cache.hicache_key_scheme import resolve_head_group_num
        from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
        from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout

        if options.dcp_size != 1:
            raise ValueError("MHA host transfers require unsharded token ids")
        info = self.buffer_info
        page_layout = None
        if options.layout == "page_unified":
            page_layout = PageUnifiedLayout(
                page_size=options.page_size,
                layer_num=self.device_layer_num + len(draft_buffer_infos),
                head_num=info.head_num,
                head_group_num=resolve_head_group_num(info.head_num),
                head_dim=info.head_dim,
                itemsize=info.dtype.itemsize,
                is_mla=False,
            )
        return get_mha_host_pool_cls(info)(
            info,
            options.ratio,
            options.size,
            options.page_size,
            options.layout,
            pin_memory=options.pin_memory,
            allocator_type=options.allocator_type,
            pool_label=options.pool_label,
            device_capacity_tokens=self.capacity,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            draft_buffer_infos=draft_buffer_infos,
            write_back_staging=self.write_back_staging,
            page_unified_layout=page_layout,
            head_group_num=1 if page_layout is None else page_layout.head_group_num,
        )
