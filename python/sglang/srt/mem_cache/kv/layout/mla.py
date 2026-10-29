"""The MLA layout: one latent row per layer in a token-major entry, and the
typed view that reads and stores it over a band of a `KVArena`."""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.device_pool_info import MLABufferInfo
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.layout.base import (
    HostPoolOptions,
    SubPoolSpec,
    _check_row_alignment,
    _store_dtype_for,
    check_physical_id_bound,
    dense_rows,
)
from sglang.srt.mem_cache.kv.memory.views import MLAKVView
from sglang.srt.mem_cache.kv.router import KVLayerRouter, PlainPoolRouter
from sglang.srt.mem_cache.layout.fused_draft import DenseDraftRegion
from sglang.srt.mem_cache.layout.token_major import (
    DenseEntryLayout,
    DensePart,
    align_entry_bytes,
    align_part_offset,
    build_dense_views,
)
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc, write_loc_is_physical
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
    from sglang.srt.mem_cache.kv.memory.arena import KVArena


@dataclass(frozen=True, kw_only=True)
class MLASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MLA-shaped sub-pool.

    One latent row (``kv_lora_rank + qk_rope_head_dim``) per token per layer; V
    is a prefix slice of the same row, so there is no separate V region. Not a
    subclass of ``MHASubPoolSpec`` — the K+V byte math and the ``v_head_dim > 0``
    invariant there do not apply. With `draft_region` set, the draft's K and V
    parts follow the latent rows inside every slot's entry, as on an MHA host.
    """

    zero_units_on_handout: ClassVar[bool] = True

    kv_lora_rank: int
    qk_rope_head_dim: int
    store_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert self.kv_lora_rank > 0, (
            f"kv_lora_rank must be positive; got {self.kv_lora_rank}"
        )
        assert self.qk_rope_head_dim > 0, (
            f"qk_rope_head_dim must be positive; got {self.qk_rope_head_dim}"
        )
        if self.draft_region is not None:
            assert isinstance(self.draft_region, DenseDraftRegion), (
                "dense entries carry dense draft K/V rows, never a state region"
            )
            self.draft_region.validate()

    @property
    def kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def row_bytes(self) -> int:
        return self.kv_cache_dim * self.store_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot, before any draft parts."""
        return self.layer_num * self.row_bytes()

    def draft_offset_in_entry(self) -> int:
        """Byte offset of the fused draft parts inside one slot's entry."""
        assert self.draft_region is not None
        return align_part_offset(self.host_entry_bytes())

    def entry_bytes(self) -> int:
        if self.draft_region is None:
            return align_entry_bytes(self.host_entry_bytes())
        return align_entry_bytes(
            self.draft_offset_in_entry() + self.draft_region.entry_bytes()
        )

    def layout(self) -> DenseEntryLayout:
        _check_row_alignment(self.name, latent=self.row_bytes())
        parts = (
            DensePart(
                name="kv",
                offset_bytes=0,
                layer_stride_bytes=self.row_bytes(),
                layer_num=self.layer_num,
                row_shape=(1, self.kv_cache_dim),
                dtype=self.store_dtype,
            ),
        )
        if self.draft_region is not None:
            parts += self.draft_region.parts(self.draft_offset_in_entry())
        return DenseEntryLayout(entry_bytes=self.entry_bytes(), parts=parts)

    def views(self, arena) -> List[torch.Tensor]:
        """Per-layer latent views over this sub-pool's band of ``arena``,
        indexed by the physical token id."""
        rows = dense_rows(arena, self.name)
        check_physical_id_bound(sub_pool=self.name, n_rows=rows)
        return build_dense_views(
            arena.raw, layout=self.layout(), part_name="kv", num_slots=rows
        )

    def draft_views(self, arena) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K and V views of the draft parts fused into this
        sub-pool's entries: the same rows, the same ids."""
        assert self.draft_region is not None, (
            f"sub-pool {self.name!r} carries no fused draft region"
        )
        rows = dense_rows(arena, self.name)
        return tuple(
            build_dense_views(
                arena.raw, layout=self.layout(), part_name=part, num_slots=rows
            )
            for part in ("draft_k", "draft_v")
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype


class MLAView(MLAKVView):
    """One MLA sub-pool's per-layer latent rows over its band of a `KVArena`:
    ``(rows, 1, kv_lora_rank + qk_rope_head_dim)`` tensors whose slot stride is
    the whole entry, indexed by the physical token id. V is a prefix slice of the
    same row. The arena owns, moves and zeroes the bytes; writes take physical
    ids in this sub-pool's id space, already resolved under DCP by the write
    translation.

    Its per-layer methods take the view's layer index (the router's
    ``KVLayer.view_layer_id``).
    """

    requires_physical_write_loc = True
    # The write translation collapses a DCP-widened id and sends rows this rank
    # does not own to the padding sink, so a write id names a stored row.
    write_loc_is_dcp_resolved = True
    dcp_replicated = False
    use_dsa = False
    dsa_kv_cache_store_fp8 = False
    kv_scale_buffer = None
    cpu_copy_carries_mamba = False
    layer_shard_enabled = False
    hicache_write_back_staging = None
    post_capture_active = False
    host_capacity_tokens: Optional[int] = None
    host_capacity_bytes: Optional[int] = None
    enable_custom_mem_pool = False
    custom_mem_pool = None
    allocation_label = None
    mem_usage = 0.0  # the arena logs the real size
    cpu_offloading_chunk_size = 8192

    def __init__(
        self,
        *,
        arena: "KVArena",
        spec: MLASubPoolSpec,
        kv_cache_dtype: torch.dtype,
        page_size: int,
        model_layer_ids: Optional[Tuple[int, ...]] = None,
    ):
        assert spec.store_dtype == _store_dtype_for(kv_cache_dtype), (
            f"sub-pool {spec.name!r} store dtype {spec.store_dtype} does not "
            f"match kv cache dtype {kv_cache_dtype}"
        )
        self._arena = arena
        self._sub_pool_name = spec.name
        self.kv_buffer = spec.views(arena)
        self.num_pages = arena.extent_entries(spec.name) // page_size
        self.page_bytes = page_size * spec.entry_bytes()
        # Rows 0..page_size-1 are the padding sink; physical ids run to the rows.
        self.size = self.num_pages * page_size - page_size
        self.page_size = page_size
        self.dtype = kv_cache_dtype
        self.store_dtype = spec.store_dtype
        self.device = arena.device
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.kv_cache_dim = spec.kv_cache_dim
        self.layer_num = spec.layer_num
        # Addressed by the 0..N-1 index of its layers.
        self.start_layer = 0
        self.end_layer = spec.layer_num - 1
        self.model_layer_ids = (
            tuple(range(spec.layer_num)) if model_layer_ids is None else model_layer_ids
        )
        self.layer_transfer_counter: Optional[LayerDoneCounter] = None
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
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
        return None  # MLA rows carry no KV quantization method

    def get_kv_size_bytes(self) -> int:
        return 0  # the arena logs the total; per-sub-pool would double-count

    def get_kv_buffer_shape(self) -> Tuple[torch.Size, torch.Size]:
        return (
            self.kv_buffer[0].shape,
            self.kv_buffer[0][..., : self.kv_lora_rank].shape,
        )

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def register_layer_transfer_counter(self, counter: "LayerDoneCounter") -> None:
        self.layer_transfer_counter = counter

    # -- reads --

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._typed(self.kv_buffer[layer_id])

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._typed(self.kv_buffer[layer_id][..., : self.kv_lora_rank])

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def _typed(self, buffer: torch.Tensor) -> torch.Tensor:
        # FP8 rows are stored as bytes; readers see the cache dtype.
        if self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        return buffer

    def get_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
        layer_id_override: Optional[int] = None,
    ):
        """The NoPE and RoPE parts of the rows at ``loc``, cast to
        ``dst_dtype`` (default: the cache dtype)."""
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    # -- writes --

    def _check_write_loc(self, loc_info, loc: torch.Tensor, where: str) -> None:
        if not write_loc_is_physical(loc_info):
            raise ValueError(
                f"{where}: write loc is not marked physical. Hand the view "
                "KVWriteLoc(loc, physical=True) with ids in its sub-pool's space."
            )
        maybe_detect_oob(loc, 0, self.size + self.page_size, where)

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ) -> None:
        """Stores whole latent rows (NoPE and RoPE concatenated)."""
        loc, _, _ = unwrap_write_loc(loc_info)
        self._check_write_loc(loc_info, loc, "set_kv_buffer (MLA view)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
        self.kv_buffer[layer_id][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        layer_id_override: Optional[int] = None,
    ) -> None:
        """Stores the NoPE and RoPE parts of latent rows."""
        loc, _, _ = unwrap_write_loc(loc_info)
        self._check_write_loc(loc_info, loc, "set_mla_kv_buffer (MLA view)")
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            if cache_k_rope is not None and cache_k_rope.numel() > 0:
                cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            if cache_k_rope is not None and cache_k_rope.numel() > 0:
                cache_k_rope = cache_k_rope.view(self.store_dtype)
        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "a unified pool moves whole units in its arena; a per-slot move "
            "would corrupt the entries"
        )

    # -- host transfer and PD --

    def host_layout(self):
        (info,) = self.get_device_pool_infos()
        return MLAHostLayout(
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
        from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, MLABufferInfo
        from sglang.srt.mem_cache.hicache_storage import PoolName

        info = MLABufferInfo(
            buffers=tuple(self.kv_buffer),
            data_ptrs=self.data_ptrs,
            layer_indices=tuple(range(self.layer_num)),
            device_layer_num=self.layer_num,
            page_size=self.page_size,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            kv_cache_dim=self.kv_cache_dim,
            dtype=self.store_dtype,
            device=self.data_ptrs.device,
            packed_fp8=False,
            scales=None,
        )
        return (
            DevicePoolInfo(
                pool_name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                layer_ids=self.model_layer_ids,
                buffer_info=info,
            ),
        )

    def host_pool_decls(self):
        """The host pool HiCache keeps for this view's buffers."""
        from sglang.srt.mem_cache.pool_host.host_pool_decl import make_kv_pool_decl

        return (make_kv_pool_decl(self),)

    def get_contiguous_buf_infos(self):
        """PD registration: one entry, the raw arena, addressed as
        ``raw_ptr + physical_page_id * page_bytes``. The layers interleave in
        each slot's entry, so the transfer item is a whole page envelope and both
        sides must build the pool with identical specs."""
        raw = self._arena.raw
        return [raw.data_ptr()], [self._arena.accessible_bytes], [self.page_bytes]

    # -- retraction copies, at physical ids --

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i : i + chunk_size]
                kv_cache_cpu[-1].append(
                    self.kv_buffer[layer_id][chunk].to("cpu", non_blocking=True)
                )
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk)
                self.kv_buffer[layer_id][chunk] = kv_cpu.to(
                    self.kv_buffer[0].device, non_blocking=True
                )
        current_platform.synchronize()


class MLAHostLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Latent host rows, including the physical page width under DCP."""

    buffer_info: MLABufferInfo
    capacity: int
    start_layer: int
    end_layer: int
    write_back_staging: object = None

    @property
    def device_layer_num(self) -> int:
        return self.buffer_info.device_layer_num

    def build(self, options: HostPoolOptions, *, draft_buffer_infos=()):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
        from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout

        info = self.buffer_info
        page_layout = None
        if options.layout == "page_unified":
            page_layout = PageUnifiedLayout(
                page_size=options.page_size // options.dcp_size,
                layer_num=self.device_layer_num + len(draft_buffer_infos),
                head_num=1,
                head_group_num=1,
                head_dim=info.kv_cache_dim,
                itemsize=info.dtype.itemsize,
                is_mla=True,
            )
        return MLATokenToKVPoolHost(
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
            target_layer_num=self.device_layer_num,
            override_kv_cache_dim=info.kv_cache_dim,
            draft_buffer_infos=draft_buffer_infos,
            write_back_staging=self.write_back_staging,
            page_unified_layout=page_layout,
            dcp_size=options.dcp_size,
            dcp_rank=options.dcp_rank,
        )
