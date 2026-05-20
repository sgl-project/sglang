# Copyright 2025 SGLang Team
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
Support for lightonai/LightOnOCR-2-1B.

LightOnOCR is a vision-language OCR model that combines:
- Pixtral vision encoder (24 layers, 1024 hidden dim)
- Spatial merge projection with RMSNorm + PatchMerger (2x2 = 4x token reduction)
- Qwen3 language decoder (28 layers, 1024 hidden dim)

Key differences from PixtralForConditionalGeneration:
- Uses Qwen3ForCausalLM instead of MistralLarge3ForCausalLM as the language model
- Has an RMSNorm applied to vision encoder output before patch merging
- Does not use image break/end tokens (single contiguous image token range)
- HuggingFace checkpoint uses a vision_projection namespace for norm, patch_merger,
  and adapter weights

References:
- https://huggingface.co/lightonai/LightOnOCR-2-1B
"""

from dataclasses import fields
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.pixtral import (
    PATCH_MERGE,
    PatchMerger,
    PixtralHFVisionModel,
    VisionEncoderArgs,
    VisionLanguageAdapter,
)
from sglang.srt.models.qwen3 import Qwen3ForCausalLM


class LightOnOCRForConditionalGeneration(nn.Module):
    """
    LightOnOCR model for SGLang inference.

    Architecture:
    - Pixtral-based vision encoder (PixtralHFVisionModel, 24 layers)
    - RMSNorm on vision encoder output
    - Spatial merge via PatchMerger (2x2 = 4x token reduction)
    - VisionLanguageAdapter projection to text hidden size
    - Qwen3-based decoder (28 layers) with QK norms
    """

    merge_by_field_config = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, config, prefix: str = "", **kwargs):
        super().__init__()
        self.config = config
        quant_config = kwargs.get("quant_config")

        # Build VisionEncoderArgs from config
        vision_config = config.vision_config
        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args = {
            key: value
            for key, value in vision_config.to_dict().items()
            if key in dataclass_fields
        }
        # LightOnOCR stores these at the top-level config
        if "image_token_id" not in vision_args:
            vision_args["image_token_id"] = getattr(config, "image_token_id", 151655)
        if "spatial_merge_size" not in vision_args:
            vision_args["spatial_merge_size"] = getattr(config, "spatial_merge_size", 2)
        if "adapter_bias" not in vision_args:
            vision_args["adapter_bias"] = getattr(
                config, "multimodal_projector_bias", True
            )
        # LightOnOCR uses patch merging for spatial merge
        vision_args["mm_projector_id"] = PATCH_MERGE
        self.vision_args = VisionEncoderArgs(**vision_args)

        # Vision encoder (Pixtral HF variant with SGLang parallel layers)
        self.vision_encoder = PixtralHFVisionModel(vision_config, quant_config=None)

        # RMSNorm applied to vision encoder output before patch merging
        self.vision_projection_norm = RMSNorm(self.vision_args.hidden_size, eps=1e-5)

        # Patch merger for spatial token reduction
        self.patch_merger = PatchMerger(
            vision_encoder_dim=self.vision_args.hidden_size,
            spatial_merge_size=self.vision_args.spatial_merge_size,
        )

        # Vision-to-language projection adapter
        self.vision_language_adapter = VisionLanguageAdapter(
            self.vision_args, dim=config.text_config.hidden_size
        )

        # Language model
        self.language_model = Qwen3ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
        )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Process images through vision encoder and projection pipeline."""
        images = [item.feature for item in items]

        # Extract image sizes from model-specific data or infer from tensor shape
        image_sizes_list = []
        for item in items:
            if item.model_specific_data and "image_sizes" in item.model_specific_data:
                sizes_tensor = item.model_specific_data["image_sizes"]
                for size in sizes_tensor:
                    image_sizes_list.append((int(size[0]), int(size[1])))
            else:
                img = item.feature
                for _ in range(img.shape[0]):
                    image_sizes_list.append((img.shape[-2], img.shape[-1]))

        # Stack pixel values
        if len(images) > 1:
            pixel_values = torch.cat(images, dim=0)
        else:
            pixel_values = images[0]

        # Vision encoder forward
        image_features = self.vision_encoder(pixel_values, image_sizes=image_sizes_list)
        image_features = image_features.view(-1, image_features.shape[-1])

        # Norm before patch merge (matches HF Mistral3MultiModalProjector order)
        image_features = self.vision_projection_norm(image_features)

        # Spatial merge via patch merger — use actual image sizes (not padded tensor
        # shape) because PixtralHFVisionModel crops embeddings to real dimensions.
        patch_size = self.vision_args.patch_size
        img_patch_dims = [
            (h // patch_size, w // patch_size) for (h, w) in image_sizes_list
        ]
        image_features = self.patch_merger(image_features, image_sizes=img_patch_dims)

        # Project to language model dimension
        image_embeds = self.vision_language_adapter(image_features)
        return image_embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from HuggingFace checkpoint.

        HF checkpoint weight layout (after stripping ``model.`` prefix):
        - ``vision_encoder.*`` -> self.vision_encoder
        - ``vision_projection.norm.*`` -> self.vision_projection_norm
        - ``vision_projection.patch_merger.*`` -> self.patch_merger
        - ``vision_projection.linear_1.*`` -> self.vision_language_adapter.w_in
        - ``vision_projection.linear_2.*`` -> self.vision_language_adapter.w_out
        - ``language_model.*`` -> self.language_model (Qwen3ForCausalLM)
        """
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
        patch_merger_dict = dict(self.patch_merger.named_parameters())
        norm_dict = dict(self.vision_projection_norm.named_parameters())
        adapter_dict = dict(self.vision_language_adapter.named_parameters())

        # PixtralHFVisionModel uses SGLang parallel layers with stacked params
        stacked_params_mapping = [
            (".attention.qkv_proj", ".attention.q_proj", "q"),
            (".attention.qkv_proj", ".attention.k_proj", "k"),
            (".attention.qkv_proj", ".attention.v_proj", "v"),
            (".feed_forward.gate_up_proj", ".feed_forward.gate_proj", 0),
            (".feed_forward.gate_up_proj", ".feed_forward.up_proj", 1),
        ]

        def llm_weights_generator():
            for name, w in weights:
                # HF checkpoint prefixes all weights with model.
                if name.startswith("model."):
                    name = name[len("model.") :]

                if name.startswith("vision_encoder."):
                    trimmed = name[len("vision_encoder.") :]

                    # Handle stacked params (QKV, gate/up)
                    loaded = False
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name in trimmed:
                            transformed = trimmed.replace(weight_name, param_name)
                            if transformed in vision_encoder_dict:
                                param = vision_encoder_dict[transformed]
                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                with torch.no_grad():
                                    weight_loader(param, w, shard_id)
                                loaded = True
                                break

                    if not loaded:
                        # Handle o_proj -> proj rename
                        if ".attention.o_proj" in trimmed:
                            trimmed = trimmed.replace(
                                ".attention.o_proj", ".attention.proj"
                            )
                        if trimmed in vision_encoder_dict:
                            param = vision_encoder_dict[trimmed]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            with torch.no_grad():
                                weight_loader(param, w)

                elif name.startswith("vision_projection."):
                    remaining = name[len("vision_projection.") :]

                    if remaining.startswith("patch_merger."):
                        trimmed = remaining[len("patch_merger.") :]
                        if trimmed in patch_merger_dict:
                            param = patch_merger_dict[trimmed]
                            with torch.no_grad():
                                default_weight_loader(param, w)

                    elif remaining.startswith("norm."):
                        trimmed = remaining[len("norm.") :]
                        if trimmed in norm_dict:
                            param = norm_dict[trimmed]
                            with torch.no_grad():
                                default_weight_loader(param, w)

                    else:
                        # linear_1 -> w_in, linear_2 -> w_out
                        trimmed = remaining.replace("linear_1.", "w_in.").replace(
                            "linear_2.", "w_out."
                        )
                        if trimmed in adapter_dict:
                            param = adapter_dict[trimmed]
                            with torch.no_grad():
                                default_weight_loader(param, w)

                else:
                    # Language model weights and any other weights
                    if name.startswith("language_model."):
                        # Qwen3ForCausalLM expects model.* prefix
                        name = "model." + name[len("language_model.") :]
                    yield (name, w)

        self.language_model.load_weights(llm_weights_generator())


EntryClass = LightOnOCRForConditionalGeneration
