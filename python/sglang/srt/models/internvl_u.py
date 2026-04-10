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
"""
InternVL-U model support for SGLang.

InternVL-U is a unified multimodal model by OpenGVLab that combines:
- InternViT-300M (vision encoder)
- Qwen3-1.7B (LLM backbone)
- MMDiT 1.7B (diffusion generation decoder)

This module adds VLM (text understanding) support by extending the existing
InternVL model adapter. The VLM portion of InternVL-U shares the same
InternViT + Qwen3 architecture as InternVL 3.5, so InternVLUChatModel
simply inherits from InternVLChatModel and filters out diffusion-only
weights during loading.

Model: https://huggingface.co/OpenGVLab/InternVL-U
Paper: https://arxiv.org/abs/2603.09877
"""

from typing import Iterable, Tuple

import torch

from sglang.srt.models.internvl import InternVLChatModel


class InternVLUChatModel(InternVLChatModel):
    """InternVL-U VLM model for text understanding.

    Inherits from InternVLChatModel since the VLM architecture is identical
    to InternVL 3.5 (InternViT + Qwen3 + MLP projector). The checkpoint
    config uses model_type='internvlu_chat' and architectures=['InternVLUChatModel'],
    with llm_config.architectures=['Qwen3ForCausalLM'].

    The only non-VLM weight in the vlm/ checkpoint is 'special_token_embedding.weight',
    which is used by the diffusion generation decoder and must be skipped.
    """

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Filter out the diffusion decoder's special_token_embedding weight,
        # which is not part of the VLM and would cause a KeyError in the parent.
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if "special_token_embedding" not in name
        )
        super().load_weights(filtered)


EntryClass = InternVLUChatModel
