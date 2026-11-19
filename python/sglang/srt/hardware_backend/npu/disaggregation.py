"""PD adapters for NPU token KV and replicated indexer state."""

from copy import copy

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
    def _npu_kv_layout(self):
        return getattr(self.kv_args, "npu_kv_layout", None)

    def prepare_dcp_token_item_lens(
        self, dst_page_item_lens, dst_dcp_size, draft_rows_collapsed=False
    ):
        if self._npu_kv_layout is None:
            return super().prepare_dcp_token_item_lens(
                dst_page_item_lens, dst_dcp_size, draft_rows_collapsed
            )
        if self._npu_kv_layout != "ND" or draft_rows_collapsed:
            raise ValueError("NPU token DCP requires ND KV and replicated draft rows")
        # Bootstrap repeats the first peer width. Check the full list below.
        page_size = self.kv_args.page_size
        page_bytes = self.kv_args.kv_item_lens
        if any(n <= 0 or n % page_size for n in page_bytes):
            raise ValueError("Invalid NPU token DCP page geometry")
        return [n // page_size for n in page_bytes]

    def _kv_layout_mismatch(self, info):
        mismatch = super()._kv_layout_mismatch(info)
        if mismatch or self._npu_kv_layout is None or not info.requires_dcp_relayout:
            return mismatch
        if len(info.dst_kv_ptrs) != len(info.dst_kv_item_lens):
            return "NPU token DCP requires per-entry destination widths"
        num_target = len(self.kv_args.kv_item_lens) - self.kv_args.num_draft_entries
        expected_page_bytes = [
            n * (1 if i < num_target else info.dst_dcp_size)
            for i, n in enumerate(self.kv_args.kv_item_lens)
        ]
        try:
            return self._state_stride_mismatch(
                src_data_ptrs=self.kv_args.kv_data_ptrs,
                dst_data_ptrs=info.dst_kv_ptrs,
                item_lens=expected_page_bytes,
                dst_item_lens=info.dst_kv_item_lens,
                force_flat=True,
                src_layer_ids=self.kv_args.kv_layer_ids,
                dst_layer_ids=info.dst_kv_layer_ids,
            )
        except (ValueError, RuntimeError) as exc:
            return str(exc)

    def _init_dcp_pack_buffers_once(self, dcp_size, *, include_draft=False):
        if self._npu_kv_layout is not None:
            self._dcp_pack_buffers = []
            return
        super()._init_dcp_pack_buffers_once(dcp_size, include_draft=include_draft)

    def dsa_index_k_dcp_entries(self, registration_info):
        if self._npu_kv_layout is not None:
            # The GPU helper selects owner rows. NPU index-K uses global slots
            # and is copied independently by maybe_send_extra instead.
            return None
        return super().dsa_index_k_dcp_entries(registration_info)

    def maybe_send_extra(
        self, req, prefill_state_indices, executor, target_rank_registration_info=None
    ):
        info = target_rank_registration_info
        if (
            self._npu_kv_layout is not None
            and info is not None
            and info.requires_dcp_relayout
            and StateType.DSA in self.kv_args.state_types
        ):
            # Indexer state keeps global slots. Only target KV needs relayout;
            # give the common state sender its own view without changing that.
            info = copy(info)
            info.requires_dcp_relayout = False
        return super().maybe_send_extra(req, prefill_state_indices, executor, info)


class NPUMooncakeKVManager(NPUPDLayoutMixin, MooncakeKVManager):
    pass
