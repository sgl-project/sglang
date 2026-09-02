# Copyright 2026 Liquid AI. All rights reserved.
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
"""Inference-only LFM2-VL model compatible with HuggingFace weights.

LFM2-VL is a vision-language model that combines:
- SigLip2 vision encoder with NaFlex variable-resolution support
- LFM2 language model (hybrid attention + short convolution)
- Multimodal projector with pixel unshuffle downsampling
"""

import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.lfm2 import Lfm2ForCausalLM
from sglang.srt.models.siglip2 import Siglip2Model
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Lfm2VlMultiModalProjector(nn.Module):
    """Multimodal projector with pixel unshuffle downsampling and TP/DP support."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.factor = config.downsample_factor
        self.use_layer_norm = config.projector_use_layernorm
        self.layer_norm = (
            nn.LayerNorm(in_channels) if config.projector_use_layernorm else None
        )

        self.linear_1 = ColumnParallelLinear(
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
        )

    def forward(
        self,
        vision_features_packed: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Project packed vision features with pixel unshuffle.

        Args:
            vision_features_packed: (total_tokens, hidden_size) packed in tile order.
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile.

        Returns:
            projected_packed: (total_projected_tokens, text_hidden_size)
        """
        factor = self.factor
        hidden_size = vision_features_packed.shape[-1]

        # Compute tile lengths from spatial shapes
        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()

        # Split packed tensor into per-tile tensors
        tile_features = torch.split(vision_features_packed, lengths, dim=0)

        # Apply pixel unshuffle to each tile using reshape/permute (GPU operations)
        unshuffled_parts = []
        for tile, (h, w) in zip(tile_features, spatial_shapes.tolist()):
            if h == 0 or w == 0:
                continue
            # Reshape: (H*W, C) -> (H, W, C) -> (H/f, f, W/f, f, C)
            tile_2d = tile.view(h, w, hidden_size)
            tile_blocks = tile_2d.view(
                h // factor, factor, w // factor, factor, hidden_size
            )
            # Permute: (H/f, f, W/f, f, C) -> (H/f, W/f, f, f, C)
            tile_permuted = tile_blocks.permute(0, 2, 1, 3, 4)
            # Reshape: (H/f, W/f, f*f*C)
            tile_unshuffled = tile_permuted.reshape(
                (h // factor) * (w // factor), factor * factor * hidden_size
            )
            unshuffled_parts.append(tile_unshuffled)

        if unshuffled_parts:
            unshuffled = torch.cat(unshuffled_parts, dim=0)
        else:
            unshuffled = vision_features_packed.new_empty(
                (0, factor * factor * hidden_size)
            )

        if self.use_layer_norm:
            unshuffled = self.layer_norm(unshuffled)
        hidden_states, _ = self.linear_1(unshuffled)
        hidden_states = self.act(hidden_states)
        projected_packed, _ = self.linear_2(hidden_states)
        return projected_packed


class Lfm2VlForConditionalGeneration(nn.Module):
    """LFM2-VL Vision-Language Model."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Vision tower: Native Siglip2 implementation
        self.vision_tower = Siglip2Model(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        # Multimodal projector
        self.multi_modal_projector = Lfm2VlMultiModalProjector(
            config,
            quant_config=quant_config,
            prefix=add_prefix("multi_modal_projector", prefix),
        )

        # Language model: reuse SGLang's LFM2 implementation
        self.language_model = Lfm2ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        result = pattern.pad_input_tokens(input_ids, mm_inputs)
        return result

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Process images through vision tower and projector.

        Handles SigLip2's NaFlex variable-resolution output.
        Pixel values arrive padded from the base processor; we pack them
        using the attention mask before feeding into the vision tower.
        """
        # Collect data from all items
        all_pixel_values = []
        all_attention_masks = []
        all_spatial_shapes = []

        for item in items:
            pv = item.feature
            am = item.pixel_attention_mask
            ss = item.spatial_shapes

            if isinstance(pv, np.ndarray):
                pv = torch.from_numpy(pv)
            if isinstance(am, np.ndarray):
                am = torch.from_numpy(am)
            if isinstance(ss, np.ndarray):
                ss = torch.from_numpy(ss)

            all_pixel_values.append(pv)
            all_attention_masks.append(am)
            all_spatial_shapes.append(ss)

        pixel_values = torch.cat(all_pixel_values, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        spatial_shapes = torch.cat(all_spatial_shapes, dim=0)

        pixel_values = pixel_values.to(
            device=self.vision_tower.device,
            dtype=self.vision_tower.dtype,
        )
        spatial_shapes_cpu = spatial_shapes.cpu()

        # Pack padded pixel values using attention mask
        packed_list = []
        for i in range(pixel_values.shape[0]):
            mask = attention_mask[i].bool()
            packed_list.append(pixel_values[i][mask])

        if not packed_list:
            return torch.tensor(
                [], device=self.vision_tower.device, dtype=self.vision_tower.dtype
            )

        pixel_values_packed = torch.cat(packed_list, dim=0)

        # Compute cu_seqlens and max_seqlen for packed attention
        spatial_shapes_list = spatial_shapes_cpu.tolist()
        lengths_list = [int(h * w) for h, w in spatial_shapes_list]
        total_tokens = sum(lengths_list)

        if total_tokens == 0:
            return torch.tensor(
                [], device=self.vision_tower.device, dtype=self.vision_tower.dtype
            )

        lengths = torch.tensor(
            lengths_list, dtype=torch.int32, device=pixel_values_packed.device
        )
        cu_seqlens = torch.zeros(
            len(lengths_list) + 1,
            dtype=torch.int32,
            device=pixel_values_packed.device,
        )
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        max_seqlen = lengths.max()

        # Forward through vision tower
        vision_outputs = self.vision_tower(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes_cpu,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Get the packed features (remove batch dim if present)
        if vision_outputs.dim() == 3:
            vision_features_packed = vision_outputs[0]
        else:
            vision_features_packed = vision_outputs

        # Project through multimodal projector
        projected_packed = self.multi_modal_projector(
            vision_features_packed=vision_features_packed,
            spatial_shapes=spatial_shapes_cpu,
        )

        return projected_packed

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from HuggingFace format."""
        # Collect weights by destination
        vision_weights = []
        projector_weights = []
        lm_weights = []

        for name, loaded_weight in weights:
            if name.startswith("model.vision_tower."):
                # model.vision_tower.* → * (strip model.vision_tower. prefix)
                # siglip2.py expects names like "vision_model.embeddings.patch_embedding.weight"
                new_name = name.replace("model.vision_tower.", "", 1)
                vision_weights.append((new_name, loaded_weight))
            elif name.startswith("model.multi_modal_projector."):
                # model.multi_modal_projector.* → multi_modal_projector.*
                new_name = name.replace(
                    "model.multi_modal_projector.", "multi_modal_projector.", 1
                )
                projector_weights.append((new_name, loaded_weight))
            elif name.startswith("model.language_model."):
                # model.language_model.* → language_model.model.*
                new_name = name.replace(
                    "model.language_model.", "language_model.model.", 1
                )
                lm_weights.append((new_name, loaded_weight))
            elif name.startswith("lm_head."):
                # lm_head.* → language_model.lm_head.*
                new_name = name.replace("lm_head.", "language_model.lm_head.", 1)
                lm_weights.append((new_name, loaded_weight))
            else:
                # Try direct mapping
                lm_weights.append((name, loaded_weight))

        # Load vision tower weights using its own load_weights method
        self.vision_tower.load_weights(vision_weights)

        # Load projector weights
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in projector_weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # Load language model weights via Lfm2ForCausalLM.load_weights
        # Strip the "language_model." prefix since Lfm2ForCausalLM expects
        # names like "model.layers.0..." and "lm_head.weight"
        lm_weights_stripped = []
        for name, loaded_weight in lm_weights:
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            lm_weights_stripped.append((name, loaded_weight))
        self.language_model.load_weights(lm_weights_stripped)


EntryClass = Lfm2VlForConditionalGeneration
