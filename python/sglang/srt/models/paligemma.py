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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/paligemma.py

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple, TypedDict

import torch
from torch import nn
from transformers import PaliGemmaConfig, PreTrainedModel

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma2 import Gemma2ForCausalLM
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class PaliGemmaImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: (batch_size * num_images, num_channels, height, width)"""


class PaliGemmaMultiModalProjector(nn.Module):
    """Simple linear projector for PaliGemma2.

    Unlike Gemma3's pooling projector, PaliGemma2 uses a single linear layer
    that maps vision hidden states to the language model embedding dimension.
    """

    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(PreTrainedModel):
    """PaliGemma2 multimodal model for SGLang.

    Architecture:
        SiglipVisionModel → PaliGemmaMultiModalProjector → Gemma2ForCausalLM

    Supports LoRA on language model layers (qkv_proj, o_proj, gate_up_proj,
    down_proj). Vision tower and projector are excluded from LoRA.
    """

    config_class = PaliGemmaConfig

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    # Only apply LoRA to language model layers, not vision tower or projector
    lora_pattern = re.compile(
        r"^language_model\.model\.layers\.\d+\."
        r"(?:self_attn|mlp)\."
        r"(?:qkv_proj|o_proj|gate_up_proj|down_proj)"
    )

    def __init__(
        self,
        config: PaliGemmaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        # Expose text config attributes at top level for LoRA compatibility
        if not hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = config.text_config.num_hidden_layers
        if not hasattr(config, "hidden_size"):
            config.hidden_size = config.text_config.hidden_size

        # Vision encoder (SigLIP)
        self.vision_tower = SiglipVisionModel(
            config=config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )

        # Linear projector: vision_hidden → projection_dim (= text hidden_size)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            projection_dim=config.vision_config.projection_dim,
        )

        # Language model (Gemma2)
        self.language_model = Gemma2ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.vocab_size = config.text_config.vocab_size
        self.post_init()

    def pad_input_ids(
        self, input_ids: List[int], image_inputs: MultimodalInputs
    ) -> List[int]:
        """Replace image placeholder tokens with actual image feature tokens.

        PaliGemma2 uses repeated <image> tokens (no start/end pair),
        so we use the multimodal token pattern.
        """
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, image_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode images through vision tower and project to text space."""
        all_pixel_values = [item.feature for item in items]
        # Flatten list of tensors
        pixel_values = torch.cat(
            [pv if pv.dim() == 4 else pv.unsqueeze(0) for pv in all_pixel_values],
            dim=0,
        )

        pixel_values = pixel_values.to(
            device=self.vision_tower.device,
            dtype=self.dtype,
        )

        # Encode through SigLIP
        image_features = self.vision_tower(pixel_values=pixel_values)

        # Project to language model hidden size
        projected = self.multi_modal_projector(image_features)

        # Scale embeddings (matches HuggingFace implementation)
        projected = projected * (self.config.hidden_size**-0.5)

        return projected

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> LogitsProcessor:
        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )
        return hs

    def should_apply_lora(self, module_name: str) -> bool:
        """Skip vision tower and projector for LoRA, only apply to LM."""
        return bool(self.lora_pattern.match(module_name))

    def tie_weights(self, **kwargs):
        return self.language_model.tie_weights(**kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Weight key remapping (transformers v4.52+ uses different prefixes)
        hf_to_sglang = {
            "model.language_model.": "language_model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }

        lm_weights = []
        other_weights = []

        for name, loaded_weight in weights:
            # Apply prefix remapping
            for old_prefix, new_prefix in hf_to_sglang.items():
                if name.startswith(old_prefix):
                    name = new_prefix + name[len(old_prefix) :]
                    break

            if name.startswith("language_model."):
                # Strip the "language_model." prefix before delegating
                lm_name = name[len("language_model.") :]
                lm_weights.append((lm_name, loaded_weight))
            else:
                other_weights.append((name, loaded_weight))

        # Delegate language model weights to Gemma2ForCausalLM
        loaded_params: Set[str] = set()
        lm_loaded = self.language_model.load_weights(lm_weights)
        loaded_params.update(f"language_model.{n}" for n in lm_loaded)

        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        for name, loaded_weight in other_weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


EntryClass = PaliGemmaForConditionalGeneration
