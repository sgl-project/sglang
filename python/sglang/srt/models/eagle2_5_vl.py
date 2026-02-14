# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from the following projects:
#   - LLaVA-OneVision, part of the Hugging Face Transformers project
#     (https://github.com/huggingface/transformers),
#     licensed under the Apache License, Version 2.0.
#   - InternVL (https://github.com/OpenGVLab/InternVL),
#     licensed under the MIT License.
#
# Modifications © 2025 NVIDIA CORPORATION & AFFILIATES, licensed under
# the Apache License, Version 2.0.
#
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License
#
# Hugging Face Transformers / LLaVA-OneVision
# Copyright (c) 2024 Hugging Face Inc.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------

import copy
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.internlm2 import InternLM2ForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.models.siglip import SiglipVisionModel
from sglang.srt.server_args import get_global_server_args
from sglang.utils import logger

try:
    from sglang.srt.layers.attention import vision_utils
except Exception:
    vision_utils = None

from sglang.srt.configs.eagle2_5_vl import Eagle2_5_VLConfig
from sglang.srt.models.siglip import SiglipVisionModel  # type: ignore


class Eagle2_5_VLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Eagle2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # Unify config naming (Eagle config often uses text_config; InternVL uses llm_config)
        if not hasattr(config, "llm_config") and hasattr(config, "text_config"):
            config.llm_config = config.text_config  # type: ignore[attr-defined]

        if vision_utils is not None:
            # harmless if config doesn't need dummy heads
            try:
                vision_utils.update_vit_attn_dummy_heads_config(self.config)
            except Exception:
                pass

        image_size = (
            getattr(config, "force_image_size", None) or config.vision_config.image_size
        )
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size

        self.downsample_ratio = getattr(config, "downsample_ratio", 0.5)
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )

        # Eagle uses select_layer to decide how many vision layers to keep.
        # We'll truncate the vision stack so we can always read last_hidden_state.
        self.select_layer = getattr(config, "select_layer", -1)
        self._init_vision_model()

        # LLM side
        llm_arch = config.llm_config.architectures[0]
        if llm_arch == "Qwen2ForCausalLM":
            self.language_model = Qwen2ForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif llm_arch == "InternLM2ForCausalLM":
            self.language_model = InternLM2ForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif llm_arch == "Qwen3MoeForCausalLM":
            self.language_model = Qwen3MoeForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        elif llm_arch == "Qwen3ForCausalLM":
            self.language_model = Qwen3ForCausalLM(
                config=config.llm_config, quant_config=quant_config
            )
        else:
            raise NotImplementedError(
                f"{llm_arch} is not implemented in Eagle2_5_VLChatModel."
            )

        # Connector MLP
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        mult = int(1 / self.downsample_ratio) ** 2

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * mult),
            nn.Linear(vit_hidden_size * mult, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.external_mm_data_embedding_funcs = {
            Modality.IMAGE: self.get_image_feature,
            Modality.VIDEO: self.get_video_feature,
        }

        self.model = self.language_model.model

        logger.info(
            f"[Eagle2.5-VL] num_image_token: {self.num_image_token}, downsample_ratio: {self.downsample_ratio}"
        )
        logger.info(
            f"[Eagle2.5-VL] select_layer(original): {getattr(config, 'select_layer', -1)}"
        )

    def _init_vision_model(self) -> None:
        """
        Truncate SigLIP num_hidden_layers based on select_laye,
        so we can always use last_hidden_state without output_hidden_states.
        """
        vision_cfg = copy.deepcopy(self.config.vision_config)

        # Determine number of hidden layers based on select_layer
        vision_feature_layer = self.select_layer
        if vision_feature_layer < 0:
            num_hidden_layers = vision_cfg.num_hidden_layers + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1

        if hasattr(vision_cfg, "vision_use_head"):
            vision_cfg.vision_use_head = False

        self.vision_model = SiglipVisionModel(
            vision_cfg,
            quant_config=self.quant_config,
            num_hidden_layers_override=num_hidden_layers,
        )

    def _vision_device_dtype(self):
        try:
            p = next(self.vision_model.parameters())
            return p.device, p.dtype
        except StopIteration:
            p = next(self.mlp1.parameters())
            return p.device, p.dtype

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """
        Pixel shuffle downsample:
          x: (n, w, h, c) -> (n, w*scale, h*scale, c/(scale^2))
        """
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        SigLIP -> patch features -> pixel shuffle -> MLP to LLM hidden space.
        Returns: (B, L_img, hidden_size_llm)
        """
        vit_embeds = self.vision_model(pixel_values=pixel_values)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        L = int(self.num_image_token)
        if vit_embeds.size(1) > L:
            vit_embeds = vit_embeds[:, :L, :]
        elif vit_embeds.size(1) < L:
            vit_embeds = torch.cat(
                [
                    vit_embeds,
                    vit_embeds.new_zeros(
                        vit_embeds.shape[0], L - vit_embeds.size(1), vit_embeds.size(2)
                    ),
                ],
                dim=1,
            )
        return vit_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        items: one item per image (or per tile).
        item.feature: [1, 3, H, W] (or [num_tiles, 3, H, W]) depending on preprocessing.
        Return: [num_images_or_tiles, img_len, hidden]
        """
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        return self.extract_feature(pixel_values)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Treat video frames as a batch of images.
        item.feature: [num_frames, 3, H, W]
        Return: [num_frames_total, img_len, hidden]
        """
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        return self.extract_feature(pixel_values)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs=self.external_mm_data_embedding_funcs,
            positions=positions,
        )
        return hidden_states

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        has_vision_head = any(
            k.startswith("vision_model.") and ".head." in k for k in params_dict
        )
        loaded_params: Set[str] = set()

        def _pick_existing(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in params_dict:
                    return c
            return None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            merged = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                merged_name = name.replace(weight_name, param_name)
                if merged_name.endswith(".bias") and merged_name not in params_dict:
                    merged = True
                    break
                if merged_name in params_dict:
                    param = params_dict[merged_name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(merged_name)
                    merged = True
                    break
            if merged:
                continue

            if (
                name.startswith("vision_model.")
                and (".head." in name)
                and (not has_vision_head)
            ):
                continue

            if name.startswith("vision_model."):
                cands = [
                    name,
                    name.replace(".attention.", ".self_attn."),
                    name.replace(".attn.", ".self_attn."),
                    name.replace(
                        ".attention.in_proj_weight", ".self_attn.qkv_proj.weight"
                    ),
                    name.replace(".attention.in_proj_bias", ".self_attn.qkv_proj.bias"),
                    name.replace(".attention.out_proj.", ".self_attn.proj."),
                    name.replace(".attention.qkv_proj.", ".self_attn.qkv_proj."),
                    name.replace(".attention.proj.", ".self_attn.proj."),
                    name.replace("attn.qkv.", "attn.qkv_proj."),
                ]
                picked = _pick_existing(cands)
                if picked is None:
                    continue
                name = picked

            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict.get(name, None)
            if param is None:
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)


EntryClass = [Eagle2_5_VLForConditionalGeneration]
