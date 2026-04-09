# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/rotary_embedding.py

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Rotary Positional Embeddings — unified public API (drop-in replacement)."""

from .base import RotaryEmbedding
from .factory import get_rope, get_rotary_pos_embed
from .mrope import NDRotaryEmbedding
from .utils import (
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)

__all__ = [
    # _utils
    "_apply_rotary_emb",
    "apply_flashinfer_rope_qk_inplace",
    # _base
    "RotaryEmbedding",
    # _mrope
    "NDRotaryEmbedding",
    # _factory
    "get_rope",
    "get_rotary_pos_embed",
]
