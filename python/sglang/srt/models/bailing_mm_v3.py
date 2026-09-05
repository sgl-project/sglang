# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
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
"""SGLang implementation of Bailing/Ling 3 VL image and video inference."""

import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.srt.runtime_context import get_mm
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class BailingMoeV3VLForConditionalGeneration(nn.Module):
    """Bailing MoE V3 language model with Qwen3 vision encoding."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.norm_query_embeds = getattr(config, "norm_query_embeds", False)
        self.use_data_parallel = get_mm().mm_enable_dp_encoder

        text_config = config.text_config
        # Bailing V3 VL checkpoints carry image-specific router tensors even
        # though the public config does not expose the internal router name.
        text_config.multi_gate = True
        self.model = BailingMoeV3ForCausalLM(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.vision_config is None:
            raise ValueError("BailingMoeV3VL requires vision_config")
        self._build_mm_encoders = self.pp_group.is_first_rank
        if self._build_mm_encoders:
            self.visual = Qwen3VLMoeVisionModel(
                config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
                use_data_parallel=self.use_data_parallel,
            )
        else:
            self.visual = PPMissingLayer()

        self.disable_merger_proj = getattr(
            config.vision_config, "disable_merger_proj", False
        )
        self.vision_out_dim = (
            config.vision_config.hidden_size
            * config.vision_config.spatial_merge_size**2
            if self.disable_merger_proj
            else config.vision_config.out_hidden_size
        )
        if self._build_mm_encoders:
            self.linear_proj = nn.Sequential(
                nn.Linear(self.vision_out_dim, text_config.hidden_size),
                nn.GELU(),
                nn.Linear(text_config.hidden_size, text_config.hidden_size),
            )
        else:
            self.linear_proj = PPMissingLayer()

        self.is_mrope_enabled = "mrope_section" in text_config.rope_parameters
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _materialize_items(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return materialize_multimodal_features(
            [item.feature for item in items],
            device=self.visual.device,
            dtype=self.visual.dtype,
        )

    def _get_vision_feature(
        self, items: List[MultimodalDataItem], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = self._materialize_items(items)
        if self.use_data_parallel:
            from sglang.srt.multimodal.mm_utils import (
                run_dp_sharded_mrope_vision_model,
            )

            vision_embeds = run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )
        else:
            vision_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        if self.config.vision_config.deepstack_visual_indexes:
            expected_dim = (
                len(self.config.vision_config.deepstack_visual_indexes) + 1
            ) * self.vision_out_dim
            if vision_embeds.shape[-1] != expected_dim:
                raise ValueError(
                    "Unexpected Bailing vision embedding width: "
                    f"expected={expected_dim}, got={vision_embeds.shape[-1]}"
                )
            vision_embeds = vision_embeds[..., : self.vision_out_dim]

        vision_embeds = self.linear_proj(vision_embeds)
        if self.norm_query_embeds:
            vision_embeds = F.normalize(vision_embeds, dim=-1)
        return vision_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not self._build_mm_encoders:
            raise RuntimeError("Vision encoder is only available on the first PP stage")
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        return self._get_vision_feature(items, image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not self._build_mm_encoders:
            raise RuntimeError("Vision encoder is only available on the first PP stage")
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        return self._get_vision_feature(items, video_grid_thw)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        del input_embeds, get_embedding
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def _load_non_text_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
    ) -> Optional[str]:
        if name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)
        elif name.startswith("model.linear_proj."):
            name = name.replace("model.linear_proj.", "linear_proj.", 1)
        if name.startswith("visual."):
            name = name.replace("attn.qkv.", "attn.qkv_proj.")
        if name not in params_dict:
            return None
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_non_text: Set[str] = set()
        unexpected_non_text = []

        def dispatch_text_weights():
            for name, loaded_weight in weights:
                is_text = (
                    name.startswith("model.")
                    and not name.startswith("model.visual.")
                    and not name.startswith("model.linear_proj.")
                ) or name == "lm_head.weight"
                if is_text:
                    yield name.replace("model.model.", "model.", 1), loaded_weight
                    continue
                loaded_name = self._load_non_text_weight(
                    name, loaded_weight, params_dict
                )
                if loaded_name is None:
                    unexpected_non_text.append(name)
                else:
                    loaded_non_text.add(loaded_name)

        loaded_text = self.model.load_weights(dispatch_text_weights())
        required_non_text = {
            name
            for name in params_dict
            if self._build_mm_encoders
            and (name.startswith("visual.") or name.startswith("linear_proj."))
        }
        missing_non_text = required_non_text - loaded_non_text
        if missing_non_text:
            raise RuntimeError(
                f"Missing required Bailing VL weights: {sorted(missing_non_text)[:20]}"
            )

        required_image_gates = {
            name
            for name, _ in self.model.named_parameters()
            if ".image_gate." in name or name.endswith(".gate.expert_bias")
        }
        missing_image_gates = required_image_gates - loaded_text
        if missing_image_gates:
            raise RuntimeError(
                "Missing required Bailing VL router weights: "
                f"{sorted(missing_image_gates)[:20]}"
            )
        if unexpected_non_text:
            logger.warning(
                "Skipped %d non-text Bailing checkpoint tensors; examples: %s",
                len(unexpected_non_text),
                unexpected_non_text[:10],
            )
        logger.info(
            "Loaded Bailing VL weights: %d text tensors and %d vision/projection tensors",
            len(loaded_text),
            len(loaded_non_text),
        )
        return set(loaded_text) | loaded_non_text


EntryClass = [BailingMoeV3VLForConditionalGeneration]
