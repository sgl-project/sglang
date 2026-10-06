# SPDX-License-Identifier: Apache-2.0
"""GAP 3 regression: a request's ``max_sequence_length`` must never override
a CLIP text encoder's fixed 77-token context.

Flux v1's encoder 0 was already protected (``is_flux_v1() and i == 0`` in
``text_encoding.py``). Kandinsky6 TI2VA also pairs a variable-length tower
(Reason1/Qwen, encoder 0) with CLIP (encoder 1) -- but at the OPPOSITE
index from Flux -- so the pre-fix guard (hardcoded to index 0) never
protected it: a request's ``max_sequence_length`` silently reached CLIP's
tokenizer call, producing token ids far outside CLIP's 77-row position
table.

This exercises ``_text_encoder_max_length_is_fixed`` directly (the exact
predicate ``TextEncodingStage.encode_text`` guards on) against real pipeline
config objects, rather than the full stage (which needs real, multi-GB
tokenizers/encoders it may not download).
"""

from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.krea2 import Krea2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.stablediffusion3 import (
    StableDiffusion3PipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    _text_encoder_max_length_is_fixed,
)


def test_flux_v1_guard_is_byte_for_byte_unchanged():
    config = FluxPipelineConfig()
    assert config.is_flux_v1() is True
    # Flux v1: encoder 0 is CLIP (fixed 77-token context) -- must stay
    # protected exactly as before this change.
    assert _text_encoder_max_length_is_fixed(config, 0) is True
    # Flux v1: encoder 1 is T5 -- the request's max_sequence_length is meant
    # to control it, exactly as before this change.
    assert _text_encoder_max_length_is_fixed(config, 1) is False


def test_kandinsky6_clip_encoder_1_is_now_protected():
    config = Kandinsky6TI2VAPipelineConfig()
    assert config.is_flux_v1() is False
    # Kandinsky6's CLIP sits at encoder index 1 (Reason1/Qwen is index 0) --
    # the opposite of Flux -- so the fix must guard index 1, not index 0.
    assert _text_encoder_max_length_is_fixed(config, 1) is True
    # Encoder 0 (Reason1/Qwen) is meant to take the request's override.
    assert _text_encoder_max_length_is_fixed(config, 0) is False


def test_other_pipelines_are_unaffected():
    # A pipeline that is neither Flux v1 nor Kandinsky6 TI2VA keeps every
    # encoder overridable, exactly as before this change.
    config = Krea2PipelineConfig()
    assert config.is_flux_v1() is False
    assert _text_encoder_max_length_is_fixed(config, 0) is False
    assert _text_encoder_max_length_is_fixed(config, 1) is False


def test_other_pipelines_with_a_clip_encoder_are_also_unaffected():
    # A stronger version of the check above: Stable Diffusion 3 also pairs
    # CLIP encoders (indices 0 and 1) with a variable-length one (T5, index
    # 2) -- the same shape of pipeline as Flux v1 and Kandinsky6 -- but this
    # fix's scope is exactly Flux v1 and Kandinsky6 TI2VA, so SD3's CLIP
    # encoders must stay overridable exactly as before this change (this
    # fix does not generalize to "any CLIPTextConfig index", which would
    # have silently changed SD3's own behavior too).
    config = StableDiffusion3PipelineConfig()
    assert config.is_flux_v1() is False
    assert _text_encoder_max_length_is_fixed(config, 0) is False
    assert _text_encoder_max_length_is_fixed(config, 1) is False
    assert _text_encoder_max_length_is_fixed(config, 2) is False


def test_kandinsky6_large_max_sequence_length_no_longer_breaks_clip_tok_kwargs():
    """Reproduces the exact tok_kwargs construction
    ``TextEncodingStage.encode_text`` performs per encoder, using the real
    Kandinsky6 pipeline config's own ``text_encoder_extra_args`` defaults
    (Reason1: dynamic up to 641 tokens; CLIP: fixed 77): a large
    ``max_sequence_length`` from the request must reach Reason1's tok_kwargs
    but never CLIP's.
    """
    config = Kandinsky6TI2VAPipelineConfig()
    max_length = 1024  # a user request far above CLIP's fixed 77-token context

    qwen_tok_kwargs = dict(config.text_encoder_extra_args[0])
    if max_length is not None and not _text_encoder_max_length_is_fixed(config, 0):
        qwen_tok_kwargs["max_length"] = max_length
    assert qwen_tok_kwargs["max_length"] == max_length

    clip_tok_kwargs = dict(config.text_encoder_extra_args[1])
    assert clip_tok_kwargs["max_length"] == 77
    if max_length is not None and not _text_encoder_max_length_is_fixed(config, 1):
        clip_tok_kwargs["max_length"] = max_length
    assert clip_tok_kwargs["max_length"] == 77
