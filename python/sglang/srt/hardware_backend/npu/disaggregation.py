"""PD adapters for NPU token KV and replicated indexer state."""

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import remap_draft_layer_ids


def register_npu_indexer_state(add, pool, draft_pool, total_kv_layers):
    """Keep index-K and its scales out of the token-KV entry list."""
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    if pool.index_head_dim is None:
        return
    ptrs, lens, item_lens = pool.get_state_buf_infos()
    layer_ids = pool.get_state_layer_ids()
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
    add(StateType.DSA, ptrs, lens, item_lens, layer_ids=layer_ids)


class NPUPDLayoutMixin:
    @property
    def _npu_token_widths(self):
        return getattr(self.kv_args, "npu_prepare_dcp_token_item_lens", None)

    def prepare_dcp_token_item_lens(
        self, dst_page_item_lens, dst_dcp_size, draft_rows_collapsed=False
    ):
        if self._npu_token_widths is not None:
            # Bootstrap only has a repeated first-entry width here. Validate
            # every actual peer width below, before publishing registration.
            dst_page_item_lens = [None] * len(self.kv_args.kv_item_lens)
        return super().prepare_dcp_token_item_lens(
            dst_page_item_lens, dst_dcp_size, draft_rows_collapsed
        )

    def _kv_layout_mismatch(self, info):
        mismatch = super()._kv_layout_mismatch(info)
        if mismatch is None and info.requires_dcp_relayout:
            prepare = self._npu_token_widths
            if prepare is not None:
                try:
                    info.dcp_token_item_lens = prepare(self.kv_args, info)
                except (ValueError, RuntimeError) as exc:
                    return str(exc)
        return mismatch

    def _init_dcp_pack_buffers_once(self, dcp_size, *, include_draft=False):
        if self._npu_token_widths is not None:
            self._dcp_pack_buffers = []
            return
        super()._init_dcp_pack_buffers_once(dcp_size, include_draft=include_draft)

    def dsa_index_k_dcp_entries(self, registration_info):
        if self._npu_token_widths is not None:
            # The GPU helper selects owner rows. NPU index-K uses global slots
            # and is copied independently by maybe_send_extra instead.
            return None
        return super().dsa_index_k_dcp_entries(registration_info)

    def maybe_send_extra(
        self, req, prefill_state_indices, executor, target_rank_registration_info=None
    ):
        rc = super().maybe_send_extra(
            req, prefill_state_indices, executor, target_rank_registration_info
        )
        info = target_rank_registration_info
        if (
            rc
            or self._npu_token_widths is None
            or info is None
            or not info.requires_dcp_relayout
            or StateType.DSA not in self.kv_args.state_types
        ):
            return rc
        # The common relayout skips DSA state because GPU index-K is moved
        # alongside target KV. NPU copies all indexer pages to every peer.
        i = self.kv_args.state_types.index(StateType.DSA)
        indices = prefill_state_indices[i] if i < len(prefill_state_indices) else None
        if indices is None or len(indices) == 0:
            return rc
        return self._send_kvcache_generic(
            mooncake_session_id=req.mooncake_session_id,
            src_data_ptrs=self.kv_args.state_data_ptrs[i],
            dst_data_ptrs=info.dst_state_data_ptrs[i],
            item_lens=self.kv_args.state_item_lens[i],
            prefill_data_indices=np.asarray(indices, dtype=np.int32),
            dst_data_indices=np.asarray(req.dst_state_indices[i], dtype=np.int32),
            executor=executor,
            state_type=StateType.DSA,
            force_flat=True,
            src_layer_ids=self.kv_args.state_layer_ids[i],
            dst_layer_ids=info.dst_state_layer_ids[i],
            dst_item_lens=info.dst_state_item_lens[i],
            bootstrap_room=req.room,
        )


class NPUMooncakeKVManager(NPUPDLayoutMixin, MooncakeKVManager):
    pass
