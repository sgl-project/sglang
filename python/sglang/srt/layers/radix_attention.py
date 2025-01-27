# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Radix attention."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        orig_context_len: Optional[int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.k_scale = None
        self.v_scale = None

        self.orig_context_len = orig_context_len

        # Store RoPE for context extension
        if rope is not None:
            if isinstance(rope, (list, tuple)):
                _, self.rope_cos, self.rope_sin = rope
            else:
                assert isinstance(rope, RotaryEmbedding)
                if hasattr(rope, "repeated_cos_sin_cache"):
                    self.rope_cos, self.rope_sin = rope.repeated_cos_sin_cache
                else:
                    cos_sin = rope.cos_sin_cache
                    cos, sin = cos_sin.chunk(2, dim=-1)
                    self.rope_cos = cos.repeat(1, 2)
                    self.rope_sin = sin.repeat(1, 2)
                    rope.repeated_cos_sin_cache = (self.rope_cos, self.rope_sin)
        else:
            self.rope_cos = self.rope_sin = None

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

        return forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache
        )
