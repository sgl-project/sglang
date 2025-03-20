from __future__ import annotations

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ComposedAttnBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        TODO

    def forward_extend(self, *args, **kwargs):
        TODO

    def forward_decode(self, *args, **kwargs):
        TODO
