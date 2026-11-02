from __future__ import annotations

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class SailHybridAttnBackend(HybridAttnBackend):
    def _select_backend(self, forward_mode: ForwardMode) -> AttentionBackend:
        if forward_mode.is_draft_extend_v2() and self.spec_attn_is_decode:
            return self.decode_backend
        return super()._select_backend(forward_mode)
