# Adapted from https://huggingface.co/nvidia/Eagle2.5-8B
# SPDX-License-Identifier: Apache-2.0

"""
Eagle2.5 Vision-Language Model Implementation.

This implements NVIDIA's Eagle2.5-8B vision-language model which combines:
- SigLIP vision encoder (27 layers, 1152 hidden size)
- Qwen3 language model (28 layers, 2048 hidden size)
- Multimodal fusion for image/video understanding and text generation
"""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from sglang.srt.configs.eagle2_5_vl import Eagle2_5_VLConfig
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# === Vision Components === #


# === Vision Projection === #


class Eagle2_5_VLVisionMLP(nn.Module):
    """MLP for projecting vision features to language space."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        hidden_act: str = "gelu_pytorch_tanh",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.act = ACT2FN[hidden_act]
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


class Eagle2_5_VLForConditionalGeneration(nn.Module):
    """Eagle2.5 Vision-Language Model for conditional generation."""

    def __init__(
        self,
        config: Eagle2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Vision components
        self.vision_model = SiglipVisionModel(
            config.vision_config,
            quant_config,
            prefix=add_prefix("vision_model", prefix),
        )

        # Language model
        self.language_model = Qwen3Model(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        # Vision projection
        self.visual_projection = Eagle2_5_VLVisionMLP(
            in_features=config.vision_config.hidden_size,
            hidden_features=config.vision_config.intermediate_size,
            out_features=config.text_config.hidden_size,
            hidden_act=config.vision_config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("visual_projection", prefix),
        )

        # Pooler for embedding extraction
        self.pooler = Pooler(
            pooling_type=PoolingType.LAST,
            normalize=True,
        )

        # LM Head
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        # Logits processor
        self.logits_processor = LogitsProcessor(self.config)

        # Special tokens
        self.image_token_index = config.image_token_index

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract image features using SigLIP vision model."""
        # Concatenate all image features
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        # SigLIP doesn't use grid_thw, it processes images directly
        image_embeds = self.vision_model(pixel_values)
        image_embeds = self.visual_projection(image_embeds)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract video features using SigLIP vision model."""
        # Concatenate all video frame features
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        # SigLIP doesn't use grid_thw, treat video frames like images
        video_embeds = self.vision_model(pixel_values)
        video_embeds = self.visual_projection(video_embeds)
        return video_embeds

    @property
    def device(self) -> torch.device:
        return self.language_model.device

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        """Forward pass for Eagle2.5 VLM."""
        # Use standard SGLang multimodal embedding routine
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
            get_embedding=get_embedding,
            logits_processor=self.logits_processor,
            lm_head=self.lm_head,
            pooler=self.pooler,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights with proper mapping."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle LM head tie
            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                if "visual" in name:
                    # Adapt vision model parameter names if needed
                    name = name.replace("vision_model.vision_model.", "vision_model.")

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                try:
                    param = params_dict[name]
                except KeyError:
                    logger.warning(f"Parameter {name} not found in model")
                    continue

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# Register the model for automatic discovery
EntryClass = Eagle2_5_VLForConditionalGeneration
