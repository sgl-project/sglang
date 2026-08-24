"""LFM2-family DSpark draft model.

Its own arch so the LFM2 draft can later diverge from other DSpark drafts (e.g.
ShortConv layers) without touching shared classes. Current checkpoints are
attention-only Qwen3-style GQA with interleaved RoPE, so it is a thin
DSparkDraftModel subclass; config and weight names come from the checkpoint.
"""

from sglang.srt.models.dspark import DSparkDraftModel


class Lfm2DSparkDraftModel(DSparkDraftModel):
    pass


EntryClass = [Lfm2DSparkDraftModel]
