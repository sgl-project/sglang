"""PD adapters for NPU token KV and replicated indexer state."""

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import remap_draft_layer_ids


def register_npu_indexer_state(add, pool, draft_pool, total_kv_layers):
    """Register replicated index-K, scales and request tails beside token KV."""
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    if pool.index_head_dim is None:
        return
    ptrs, lens, item_lens = pool.get_state_buf_infos()
    layer_ids = pool.get_state_layer_ids()
    tail_ptrs, tail_lens, tail_items = pool.get_compress_tail_buf_infos()
    if draft_pool is not None:
        draft_pool = draft_pool.layer_router.wrapped_view() or draft_pool
        if (
            isinstance(draft_pool, NPUMLATokenToKVPool)
            and draft_pool.index_head_dim is not None
        ):
            draft_ptrs, draft_lens, draft_items = draft_pool.get_state_buf_infos()
            ptrs += draft_ptrs
            lens += draft_lens
            item_lens += draft_items
            layer_ids += remap_draft_layer_ids(
                draft_pool.get_state_layer_ids(), total_kv_layers
            )
            draft_ptrs, draft_lens, draft_items = (
                draft_pool.get_compress_tail_buf_infos()
            )
            tail_ptrs += draft_ptrs
            tail_lens += draft_lens
            tail_items += draft_items
    add(StateType.DSA, ptrs, lens, item_lens, layer_ids=layer_ids)
    if tail_ptrs:
        add(StateType.DSA_TAIL, tail_ptrs, tail_lens, tail_items)


class NPUPDLayoutMixin:
    _dsa_index_k_relayout = False

    def _init_dcp_pack_buffers_once(self, dcp_size, *, include_draft=False):
        from sglang.srt.hardware_backend.npu.dcp_pack import init_npu_dcp_pack_buffers

        super()._init_dcp_pack_buffers_once(
            dcp_size, include_draft=include_draft, allocator=init_npu_dcp_pack_buffers
        )


class NPUMooncakeKVManager(NPUPDLayoutMixin, MooncakeKVManager):
    pass
