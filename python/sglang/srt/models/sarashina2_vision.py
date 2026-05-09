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
"""Inference-only Sarashina2Vision model compatible with HuggingFace weights."""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultimodalDataItem,
    MultimodalInputs,
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader, WeightsMapper
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.qwen2_vl import Qwen2VisionTransformer
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Sarashina2VisionForCausalLM(nn.Module):
    """
    Sarashina2Vision model that combines:
    - Llama text backbone (sbintuitions/sarashina2-7b)
    - Qwen2VL vision encoder
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        # Extract text and vision configurations
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)

        # Create vision transformer first (like original model)
        if vision_config is not None:
            self.visual = Qwen2VisionTransformer(
                vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-5),
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
            )
        else:
            self.visual = None

        # Layer norm for vision outputs (matching original model)
        self.norm = nn.LayerNorm(text_config.hidden_size)

        # Create Llama text model (using 'llm' name to match original)
        if hasattr(text_config, "model_type") and text_config.model_type == "llama":
            llama_config = LlamaConfig(**text_config.__dict__)
            # Set vocab_size from main config if available
            if hasattr(config, "vocab_size"):
                llama_config.vocab_size = config.vocab_size
            self.llm = LlamaForCausalLM(
                llama_config,
                quant_config=quant_config,
                prefix=add_prefix("llm", prefix),
            )
        else:
            # Set vocab_size from main config if available
            if hasattr(config, "vocab_size"):
                config.vocab_size = config.vocab_size
            self.llm = LlamaForCausalLM(
                config,
                quant_config=quant_config,
                prefix=add_prefix("llm", prefix),
            )

        # Image token indices from config
        self.image_token_index = getattr(config, "image_token_index", 14)
        self.start_image_token_index = getattr(
            config, "start_image_token_index", 102397
        )
        self.end_image_token_index = getattr(config, "end_image_token_index", 102398)

        # Ensure vocabulary size matches
        if hasattr(config, "vocab_size"):
            self.llm.config.vocab_size = config.vocab_size

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input tokens with multimodal data hashes for RadixAttention."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self):
        """Get input embeddings from the language model."""
        return self.llm.get_input_embeddings()

    def get_image_embeds(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract image embeddings using the vision transformer."""
        if self.visual is None:
            raise ValueError("Visual encoder not initialized")

        # Use the existing Qwen2VisionTransformer forward method
        hidden_states = self.visual(pixel_values, image_grid_thw)

        # Apply normalization layer
        return self.norm(hidden_states)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract image features for SGLang compatibility."""
        if self.visual is None:
            raise ValueError("Visual encoder not initialized")

        # Concatenate pixel values and grid_thw from all items
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        # Use the get_image_embeds method
        return self.get_image_embeds(pixel_values, image_grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # Handles token-to-feature mapping for expanded tokens
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm.model,
            multimodal_model=self,
            positions=positions,
        )

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        else:
            return self.logits_processor(
                input_ids, hidden_states, self.llm.lm_head, forward_batch
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights via AutoWeightsLoader.

        Vision tower checkpoint exposes ``attn.qkv``; runtime ``VisionAttention``
        exposes ``attn.qkv_proj``. Llama text backbone fuses q/k/v and gate/up
        through its own ``LlamaForCausalLM.load_weights``.
        """
        mapper = WeightsMapper(
            orig_to_new_substr={".attn.qkv.": ".attn.qkv_proj."},
        )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)


# Register the model
EntryClass = Sarashina2VisionForCausalLM
