# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2024 SGLang Team

"""Registry entry point for the Dots3 next-N model."""

from sglang.srt.models.dots3_common.nextn import (
    Dot3NoteModelNextN,
    Dots3MTPHead,
    Dots3NoteForCausalLMNextN,
)

EntryClass = [Dots3NoteForCausalLMNextN]

__all__ = [
    "Dot3NoteModelNextN",
    "Dots3MTPHead",
    "Dots3NoteForCausalLMNextN",
    "EntryClass",
]
