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
"""Legacy Bailing multimodal wrappers for image and video inference."""

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
from sglang.srt.models.bailing_moe import BailingMoeV2ForCausalLM
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.srt.runtime_context import get_mm
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class BailingMMNativeForConditionalGeneration(nn.Module):
    """Bailing MoE V2 wrapper with optional image/video encoding."""

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
        self.use_data_parallel = get_mm().mm_enable_dp_encoder
        self.model = BailingMoeV2ForCausalLM(
            config.llm_config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

        if getattr(config, "audio_config", None) is not None:
            raise ValueError(
                "Audio is not supported by the Bailing SGLang port; "
                "use an image/video-only checkpoint"
            )

        self._build_mm_encoders = self.pp_group.is_first_rank
        self.vision = None
        self.linear_proj = None
        if config.vision_config is not None:
            if self._build_mm_encoders:
                vision_config = config.vision_config
                architectures = getattr(vision_config, "architectures", None) or []
                arch = architectures[0] if architectures else vision_config.model_type
                if arch in {
                    "Qwen3MoeVisionTransformer",
                    "Qwen3_VisionTransformer",
                    "qwen3_vl_moe",
                }:
                    from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

                    vision_cls = Qwen3VLMoeVisionModel
                elif arch == "Qwen2_5_VisionTransformer":
                    vision_cls = Qwen2_5_VisionTransformer
                else:
                    raise ValueError(f"Unsupported Bailing vision architecture: {arch}")
                self.vision = vision_cls(
                    vision_config,
                    quant_config=quant_config,
                    prefix=add_prefix("vision", prefix),
                    use_data_parallel=self.use_data_parallel,
                )
                projection_layers = [
                    nn.Linear(
                        vision_config.out_hidden_size,
                        self.model.config.hidden_size,
                    )
                ]
                for _ in range(1, config.mlp_depth):
                    projection_layers.extend(
                        [
                            nn.GELU(),
                            nn.Linear(
                                self.model.config.hidden_size,
                                self.model.config.hidden_size,
                            ),
                        ]
                    )
                self.linear_proj = nn.Sequential(*projection_layers)
            else:
                self.vision = PPMissingLayer()
                self.linear_proj = PPMissingLayer()

        self.is_mrope_enabled = "mrope_section" in config.llm_config.rope_parameters
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _get_vision_feature(
        self, items: List[MultimodalDataItem], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if self.vision is None or not self._build_mm_encoders:
            raise RuntimeError("Vision encoder is unavailable on this PP stage")
        pixel_values = materialize_multimodal_features(
            [item.feature for item in items],
            device=self.vision.device,
            dtype=self.vision.dtype,
        )
        if self.use_data_parallel:
            from sglang.srt.multimodal.mm_utils import (
                run_dp_sharded_mrope_vision_model,
            )

            vision_embeds = run_dp_sharded_mrope_vision_model(
                self.vision,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )
        else:
            vision_embeds = self.vision(pixel_values, grid_thw=grid_thw)
        deepstack_indexes = getattr(
            self.config.vision_config, "deepstack_visual_indexes", []
        )
        if deepstack_indexes:
            expected_dim = (len(deepstack_indexes) + 1) * (
                self.config.vision_config.out_hidden_size
            )
            if vision_embeds.shape[-1] != expected_dim:
                raise ValueError(
                    "Unexpected Bailing vision embedding width: "
                    f"expected={expected_dim}, got={vision_embeds.shape[-1]}"
                )
            vision_embeds = vision_embeds[
                ..., : self.config.vision_config.out_hidden_size
            ]
        return F.normalize(self.linear_proj(vision_embeds).float(), dim=-1)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        return self._get_vision_feature(items, image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        return self._get_vision_feature(items, video_grid_thw)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        raise ValueError("Audio inputs are not supported by the Bailing SGLang port")

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers_dict = dict(self.named_buffers())
        loaded_non_text: Set[str] = set()
        unexpected_non_text = []

        def dispatch_text_weights():
            for name, loaded_weight in weights:
                is_multimodal = name.startswith(
                    ("model.visual.", "model.vision.", "model.linear_proj.")
                )
                if name.startswith("model.") and not is_multimodal:
                    yield name[len("model.") :], loaded_weight
                    continue
                if name.startswith(("model.visual.", "model.vision.")):
                    _, _, suffix = name.partition(".")
                    _, _, suffix = suffix.partition(".")
                    name = f"vision.{suffix}"
                elif name.startswith("model.linear_proj."):
                    name = name[len("model.") :]
                mapped_name = name.replace("attn.qkv.", "attn.qkv_proj.")
                target = params_dict.get(mapped_name)
                if target is not None:
                    weight_loader = getattr(
                        target, "weight_loader", default_weight_loader
                    )
                    weight_loader(target, loaded_weight)
                    loaded_non_text.add(mapped_name)
                elif mapped_name in buffers_dict:
                    buffers_dict[mapped_name].copy_(loaded_weight)
                    loaded_non_text.add(mapped_name)
                else:
                    unexpected_non_text.append(name)

        loaded_text = self.model.load_weights(dispatch_text_weights())
        required_non_text = {
            name
            for name in params_dict
            if self._build_mm_encoders
            and (name.startswith("vision.") or name.startswith("linear_proj."))
        }
        missing_non_text = required_non_text - loaded_non_text
        if missing_non_text:
            raise RuntimeError(
                "Missing required Bailing multimodal weights: "
                f"{sorted(missing_non_text)[:20]}"
            )
        if unexpected_non_text:
            logger.warning(
                "Skipped %d Bailing checkpoint tensors; examples: %s",
                len(unexpected_non_text),
                unexpected_non_text[:10],
            )
        logger.info(
            "Loaded Bailing weights: %d text tensors and %d multimodal tensors",
            len(loaded_text),
            len(loaded_non_text),
        )
        return set(loaded_text) | loaded_non_text


class BailingMM2NativeForConditionalGeneration(BailingMMNativeForConditionalGeneration):
    pass


EntryClass = [
    BailingMMNativeForConditionalGeneration,
    BailingMM2NativeForConditionalGeneration,
]
