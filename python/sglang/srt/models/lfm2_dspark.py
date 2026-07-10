"""LFM2-family DSpark draft model.

Registered as its own architecture (rather than reusing Qwen3DSparkModel) so the
LFM2 draft can evolve independently of other DSpark drafts, e.g. adding
ShortConv / conv1d layers to the drafter, without touching the shared classes.

Today's LFM2 draft checkpoints use an attention-only Qwen3-style GQA backbone
with interleaved RoPE, so this is a thin subclass of DSparkDraftModel. All the
family-specific settings come from the checkpoint config (rope_is_neox_style,
enable_confidence_head) and its canonical weight names, so no adaptation is
needed here.
"""

from sglang.srt.models.dspark import DSparkDraftModel


class Lfm2DSparkDraftModel(DSparkDraftModel):
    pass


EntryClass = [Lfm2DSparkDraftModel]
