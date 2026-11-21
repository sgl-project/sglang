# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Attention phases for Nemotron diffusion drafting and causal verification."""

from typing import ClassVar

from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)


class NemotronLinearSpecGraphVariants:
    capture_labels: ClassVar[tuple[str, ...]] = ("draft", "verify")

    def select(self, forward_batch):
        return "verify" if forward_batch.dllm_is_causal else "draft"


def is_causal_verify(forward_batch):
    # Captures use a static batch, so the variant supplies its phase.
    variant = get_capture_attention_variant()
    if variant is not None:
        return variant == "verify"
    return forward_batch.dllm_is_causal
