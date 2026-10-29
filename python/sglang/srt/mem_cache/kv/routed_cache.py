"""The token KV cache of a unified pool.

`RoutedKVCache` is what attention backends, HiCache and the scheduler hold as a
unified pool's `token_to_kv_pool`, for every set of sub-pools: each layer's
reads and writes go to the view of the sub-pool its route names, with the
layer id that view takes. It holds the views and the routes, never the
allocator: write ids arrive already in each sub-pool's id space, and the id
spaces are the allocator's (`ChainAllocator.id_spaces`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.kv.memory.arena import KVArena
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.mem_cache.kv.sizing import check_bs1_floor
from sglang.srt.mem_cache.memory_pool import KVCache


class RoutedKVCache(KVCache):
    """Routes every layer of a unified pool to its sub-pool's view.

    ``layer_maps`` lists the token KV sub-pools in plan order: each one's name,
    its view, and each routed model layer's index in that view, which is also
    the layer id the view's methods take. ``owner`` names the sub-pool whose
    write ids arrive in ``loc``. The owner's view answers for the pool where
    one answer serves every layer (the value head dim, the quantization method,
    the buffers a host stack anchors on); a fused draft that routes no layer to
    the owner has the first routed view answer instead.
    """

    # Every view's write door takes physical ids, in its sub-pool's id space.
    requires_physical_write_loc = True

    def __init__(
        self,
        *,
        arena: KVArena,
        owner: str,
        layer_maps: Sequence[Tuple[str, Any, Dict[int, int]]],
        page_size: int,
        start_layer: int,
        end_layer: Optional[int],
        size: int,
        head_num: int,
        head_dim: int,
        bs1_floor_terms: Sequence[Tuple[str, int]] = (),
    ):
        # KVCache.__init__ allocates buffers; these views alias the arena,
        # which owns the memory.
        self._arena = arena
        # Checked again on the final budget when the arena is backed after
        # capture; a draft over the target's arena leaves it to the target.
        self._bs1_floor_terms = tuple(bs1_floor_terms)
        self._owner = owner
        self._layer_maps = tuple(layer_maps)
        views = {name: view for name, view, _ in self._layer_maps}
        self._owner_view = views.get(owner, self._layer_maps[0][1])
        self.page_size = page_size
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.size = size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = sum(len(indices) for _, _, indices in self._layer_maps)
        self.device = arena.device
        self.dtype = self._owner_view.dtype
        self.store_dtype = self._owner_view.store_dtype
        for name, view, indices in self._layer_maps:
            if not indices:
                continue
            assert (view.dtype, view.store_dtype) == (self.dtype, self.store_dtype), (
                f"sub-pool {name!r} stores {view.store_dtype} for {view.dtype}; the "
                f"owner stores {self.store_dtype} for {self.dtype}"
            )
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.mem_usage = 0.0  # the arena logs the real size

    def _build_layer_router(self) -> KVLayerRouter:
        return KVLayerRouter.from_layer_maps(self, self._layer_maps, owner=self._owner)

    def id_spaces(self):
        raise NotImplementedError(
            "a unified pool's sub-pools name slots through its allocator's "
            "tables: read them from the allocator or the runner's `RunnerIds`"
        )

    # -- the arena --

    @property
    def backing_owner(self):
        return self._arena

    @property
    def post_capture_active(self) -> bool:
        return self._arena.backed_after_capture

    @property
    def post_capture_backed_bytes(self) -> int:
        return self._arena.backed_bytes

    def finalize_backing(self, config) -> None:
        """Back the final byte budget the configurator chose after capture."""
        total_bytes = config.unified_memory_pool_bytes
        if total_bytes is None:
            raise ValueError(
                "backing the KV arena after capture requires "
                "config.unified_memory_pool_bytes"
            )
        check_bs1_floor(
            total_bytes=int(total_bytes),
            floor_terms=self._bs1_floor_terms,
            where="RoutedKVCache.finalize_backing",
        )
        self._arena.finalize_backing(int(total_bytes))

    # -- answers of the owner's view --

    def get_kv_size_bytes(self):
        return self._owner_view.get_kv_size_bytes()

    def get_kv_buffer_shape(self):
        # The stage's first layer may hold no token KV (a recurrent layer), so
        # the owner's view, which holds the stage's token KV, answers the shape.
        return self._owner_view.get_kv_buffer_shape()

    def get_contiguous_buf_infos(self):
        return self._owner_view.get_contiguous_buf_infos()

    def host_pool_decls(self):
        return self._owner_view.host_pool_decls()

    def get_kv_cache_quant_method(self):
        """The first quantized view's method, else the views' unquantized one."""
        fallback = None
        for _, view, _ in self._layer_maps:
            quant_method = view.get_kv_cache_quant_method()
            if quant_method is None:
                continue
            if quant_method.name != "unquantized":
                return quant_method
            fallback = quant_method
        return fallback

    def get_v_head_dim(self):
        view = self._owner_view
        return view.get_value_buffer(view.start_layer).shape[-1]

    # -- per-layer reads and writes, through the layer's route --

    def get_key_buffer(self, layer_id: int, scale: Optional[float] = None):
        return self.layer_router.read(layer_id).key(scale)

    def get_value_buffer(self, layer_id: int, scale: Optional[float] = None):
        return self.layer_router.read(layer_id).value(scale)

    def get_kv_buffer(self, layer_id: int):
        return self.layer_router.read(layer_id).kv()

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        """Write ids arrive physical, in the routed sub-pool's id space: the
        owner's in ``full_loc`` (else ``loc``), another token sub-pool's in
        ``swa_loc``, both bound once per forward by the attention backend."""
        self.layer_router.layer(layer.layer_id).store(
            layer, loc_info, cache_k, cache_v, k_scale, v_scale, dcp_kv_mask=dcp_kv_mask
        )

    def set_mla_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        self.layer_router.layer(layer.layer_id).store_mla(
            layer, loc_info, cache_k_nope, cache_k_rope
        )

    def set_mla_kv_buffer_prefix_valid(
        self, layer, loc_info, commit_lens, cache_k_nope, cache_k_rope
    ):
        self.layer_router.layer(layer.layer_id).store_mla_prefix(
            layer, loc_info, commit_lens, cache_k_nope, cache_k_rope
        )

    def get_mla_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # Read ids arrive translated at their production site; the pool never
        # translates.
        return self.layer_router.read(layer.layer_id).read_mla(layer, loc, dst_dtype)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "a unified pool compacts per sub-pool: its allocator moves units "
            "in the arena"
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError(
            "a unified pool's retraction copies go through its allocator, "
            "which maps the ids per sub-pool"
        )

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError(
            "a unified pool's retraction copies go through its allocator, "
            "which maps the ids per sub-pool"
        )
