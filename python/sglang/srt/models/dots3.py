# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2024 SGLang Team

"""Registry entry point for the Dots3 model."""

from sglang.srt.models.dots3_common.modeling import (
    Dots3AttentionMLA,
    Dots3AttnForwardMethod,
    Dots3DecoderLayer,
    Dots3LanguageModelForCausalLM,
    Dots3MLP,
    Dots3Model,
    Dots3MoE,
    Dots3MoEGate,
    Dots3NoteForCausalLM,
    DotsNoteOmniForConditionalGeneration,
    DotsNoteOmniThinkerForConditionalGeneration,
    get_attention_sliding_window_size,
)

EntryClass = [Dots3NoteForCausalLM]

__all__ = [
    "Dots3AttentionMLA",
    "Dots3AttnForwardMethod",
    "Dots3DecoderLayer",
    "Dots3LanguageModelForCausalLM",
    "Dots3MLP",
    "Dots3Model",
    "Dots3MoE",
    "Dots3MoEGate",
    "Dots3NoteForCausalLM",
    "DotsNoteOmniForConditionalGeneration",
    "DotsNoteOmniThinkerForConditionalGeneration",
    "EntryClass",
    "get_attention_sliding_window_size",
]
