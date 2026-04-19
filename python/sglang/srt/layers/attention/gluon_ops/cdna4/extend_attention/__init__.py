# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon extend-attention kernel sources for MI350X (gfx950 / CDNA 4).

Drop-in replacement for the Triton ``extend_attention_fwd`` kernel in
``sglang.srt.layers.attention.triton_ops.extend_attention`` when running
on gfx950. Supports symmetric head dims (D64, D128, D256) with BF16 /
FP8 KV caches, causal + sliding-window, attention sinks, logit
capping, and GQA up to 32:8.

Public re-exports (consumed by
``sglang.srt.layers.attention.gluon_extend_attention``):

* ``gluon_extend_attention_fwd``  -- main entry point matching the
                                     Triton reference signature.
* ``prewarm_extend_attention``    -- warm the JIT cache for a single
                                     ``(head_dim, num_q_heads,
                                     num_kv_heads)`` tuple.
* ``prewarm_for_model``           -- HF-config-aware prewarm that emits
                                     one kernel per distinct attention
                                     pattern in the layer list.
* ``MODEL_PRESETS`` + ``spec_*``  -- known-good layer specs for
                                     GPT-OSS, Gemma 3, Qwen 3, Llama
                                     3 / 4, etc.

The canonical research copy of these sources lives in
``AMD-Triton/gluon-kernels`` branch
``tussingh/extend-attention-experiments``; this is the subset SGLang
imports at runtime.
"""

from .extend_attention_gfx950 import gluon_extend_attention_fwd  # noqa: F401
from .extend_attention_gfx950 import _get_num_CUs  # noqa: F401
from ._prewarm import (  # noqa: F401
    prewarm_extend_attention,
    prewarm_for_model,
    prewarm_preset,
    enumerate_basic_configs,
    enumerate_persistent_configs,
    enumerate_layer_patterns,
    spec_gpt_oss,
    spec_gemma3,
    spec_qwen3,
    spec_llama4,
    spec_llama3,
    MODEL_PRESETS,
)
