"""KV pools a DRAFT runner binds over the draft lanes fused into the target's
unified pool: same pages, same slot ids, same v2p table as the target -- one
allocation, one free, one relocation."""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from sglang.srt.mem_cache.layout.fused_draft import FusedDraftPlacement
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool


class UnifiedDraftKVPool(MHATokenToKVPool):
    """Dense draft KV over the draft parts of one host sub-pool's entries.

    Per-layer `k_buffer` / `v_buffer` are views of the draft parts inside
    every slot of the host sub-pool (`UnifiedKVPool.build_dense_draft_views`);
    ``layer_lanes`` maps each of this runner's layer ids to its region lane.
    Locs arriving through the KVCache API are the target's PHYSICAL token ids,
    produced by the allocator's translate (the id-space choke point binds it,
    see KVIndexTranslator); the pool exposes `host_allocator` for that
    binding. Relocation needs no method here: compaction moves whole page
    envelopes on the HOST pool, which carries the draft bytes; `move_kv_cache`
    raises so a stray per-slot move fails loudly instead of corrupting the
    fused layout.
    """

    requires_translated_write_loc = True

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        host_sub_pool_name: str,
        host_allocator,
        layer_lanes: Mapping[int, int],
        page_size: int = 1,
    ):
        region = unified_buffer.require_draft_host_spec(host_sub_pool_name).draft_region
        layer_ids = sorted(layer_lanes)
        assert layer_ids, "UnifiedDraftKVPool binds at least one layer"
        start_layer = layer_ids[0]
        assert layer_ids == list(range(start_layer, start_layer + len(layer_ids))), (
            f"fused draft layer ids must be contiguous; got {layer_ids}"
        )
        lanes = [layer_lanes[layer_id] for layer_id in layer_ids]
        assert len(set(lanes)) == len(lanes) and all(
            0 <= s < region.lane_num for s in lanes
        ), f"draft lanes {lanes} must be distinct and within range({region.lane_num})"
        k_views, v_views = unified_buffer.build_dense_draft_views(host_sub_pool_name)
        max_slots = unified_buffer.max_slots(host_sub_pool_name)

        self._unified_buffer = unified_buffer
        self._host_sub_pool_name = host_sub_pool_name
        self.host_allocator = host_allocator
        self.layer_lanes: Dict[int, int] = dict(layer_lanes)
        self._k_views: List[torch.Tensor] = [k_views[s] for s in lanes]
        self._v_views: List[torch.Tensor] = [v_views[s] for s in lanes]
        num_pages = max_slots // page_size

        super().__init__(
            size=num_pages * page_size - page_size,
            page_size=page_size,
            dtype=region.store_dtype,
            head_num=region.head_num,
            head_dim=region.head_dim,
            layer_num=len(layer_ids),
            device=unified_buffer.device,
            enable_memory_saver=False,  # buffer owned by UnifiedKVPool
            v_head_dim=region.resolved_v_head_dim(),
            start_layer=start_layer,
            end_layer=start_layer + len(layer_ids) - 1,
            enable_kv_cache_copy=False,
            kv_cache_layout="page_major",
        )

    def _create_buffers(self):
        self.k_buffer = self._k_views
        self.v_buffer = self._v_views

    def _clear_buffers(self):
        pass  # lifetime owned by UnifiedKVPool

    def get_kv_size_bytes(self):
        return 0, 0  # fused into the host entries; UnifiedKVPool logs the total

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError(
            "fused draft KV relocates with the HOST pool's whole-page move; a "
            "draft-side per-slot move would corrupt the fused layout"
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft KV has no per-layer contiguous regions; KV transfer / "
            "disaggregation is unsupported."
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("CPU offloading is unsupported for fused draft KV.")


def fused_draft_host_allocator(token_to_kv_pool: Any) -> Optional[Any]:
    """The host allocator a fused draft pool translates through, or None for
    any other pool."""
    if isinstance(token_to_kv_pool, UnifiedDraftKVPool):
        return token_to_kv_pool.host_allocator
    return None


def draft_kv_layer_ids(model) -> List[int]:
    """Layer ids owning attention KV in the BUILT draft model, in layer order.
    Window layers count -- a window is a parameter of the attention layer;
    linear and recurrent layers have no attention module and are absent."""
    from sglang.srt.layers.radix_attention import RadixAttention

    return sorted(
        {m.layer_id for m in model.modules() if isinstance(m, RadixAttention)}
    )


def bind_fused_draft(
    *,
    unified_buffer: UnifiedKVPool,
    host_allocator,
    placement: FusedDraftPlacement,
    runner: int,
    kv_layer_ids: Sequence[int],
    swa_layer_ids: Sequence[int],
    page_size: int,
) -> UnifiedDraftKVPool:
    """The KV pool draft runner ``runner`` binds over its fused slots.

    The placement sized the lanes from the draft config; the model's real
    layer ids fill them in layer order, so a count mismatch is a loud boot
    failure, never a silent alias.
    """
    swa = set(swa_layer_ids)
    full_ids = [layer_id for layer_id in kv_layer_ids if layer_id not in swa]
    assert len(full_ids) == len(kv_layer_ids), (
        f"draft layers {sorted(swa & set(kv_layer_ids))} are SWA-kind; fused "
        "SWA KV is not supported yet"
    )
    full_lanes = placement.lanes_for(runner)
    assert len(full_ids) == len(full_lanes), (
        f"draft runner {runner}: {len(full_ids)} full-attention layer(s) "
        f"{full_ids} vs {len(full_lanes)} placed lane(s) {list(full_lanes)}"
    )
    return UnifiedDraftKVPool(
        unified_buffer=unified_buffer,
        host_sub_pool_name="full",
        host_allocator=host_allocator,
        layer_lanes=dict(zip(full_ids, full_lanes)),
        page_size=page_size,
    )
