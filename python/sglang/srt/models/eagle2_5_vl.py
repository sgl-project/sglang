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
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


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
        self.language_model = Qwen2Model(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
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
        self.logits_processor = LogitsProcessor(self.config.text_config)

        # MLP connector (matches Eagle2.5 architecture from NVIDIA)
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        # Build MLP connector based on configuration
        # Note: use_pixel_shuffle determines if we need to handle pixel-shuffled dimensions
        input_dim = vit_hidden_size * int(1 / config.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # Special tokens
        self.image_token_index = config.image_token_index

    def pixel_shuffle(self, x, scale_factor=0.5):
        """Pixel shuffle operation for downsampling vision features."""
        # Reshape to spatial grid
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, -1)

        # Pixel shuffle
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )

        x = x.permute(0, 2, 1, 3).contiguous()

        # Reshape back to sequence
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x

    def feature_compression(self, vit_embeds):
        """Compress vision features using pixel shuffle and MLP."""
        if self.config.use_pixel_shuffle:
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.config.downsample_ratio
            )
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_feature(self, pixel_values):
        """Extract and compress vision features."""
        # Handle batched input: process each image individually since flatten_batch=True
        if pixel_values.dim() == 4 and pixel_values.shape[0] > 1:
            # Batched images: [batch_size, channels, height, width]
            batch_size = pixel_values.shape[0]
            vit_embeds_list = []
            for i in range(batch_size):
                single_image = pixel_values[i : i + 1]  # [1, channels, height, width]
                # Always grab the last hidden state
                single_vit_embeds = self.vision_model(pixel_values=single_image)
                single_vit_embeds = self.feature_compression(single_vit_embeds)
                vit_embeds_list.append(single_vit_embeds)
            vit_embeds = torch.cat(vit_embeds_list, dim=0)
        else:
            # Single image or already properly shaped
            vit_embeds = self.vision_model(pixel_values=pixel_values)
            # Always grab the last hidden state
            vit_embeds = self.feature_compression(vit_embeds)
        return vit_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract image features using SigLIP vision model with MLP connector."""
        # Concatenate all image features
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        # Extract and compress features
        image_embeds = self.extract_feature(pixel_values)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract video features using SigLIP vision model with MLP connector."""
        # Concatenate all video frame features
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        # Extract and compress features
        video_embeds = self.extract_feature(pixel_values)
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
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

        # Return embeddings or logits based on get_embedding flag
        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights with proper mapping."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "lm_head" in name:
                name = name.replace("language_model.lm_head.weight", "lm_head.weight")

            # Transform vision model parameter names
            if "vision_model" in name:
                # SigLIP weights use 'out_proj' but model uses 'proj', aside from for the head.
                if "out_proj" in name and "head" not in name:
                    name = name.replace("out_proj", "proj")

                # Transform head parameter names from checkpoint format to model format
                if "head.attention.in_proj.weight" in name:
                    name = name.replace(
                        "head.attention.in_proj.weight", "head.attention.in_proj_weight"
                    )
                if "head.attention.in_proj_bias" in name:
                    name = name.replace(
                        "head.attention.in_proj.bias", "head.attention.in_proj_bias"
                    )

            # Transform language model parameter names
            if "language_model" in name:
                name = name.replace(r"language_model.model.", r"language_model.")

            # Handle LM head tie using value from Eagle2.5 general config
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
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

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)


# Register the model for automatic discovery
EntryClass = Eagle2_5_VLForConditionalGeneration
