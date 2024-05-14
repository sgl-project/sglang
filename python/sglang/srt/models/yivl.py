"""Inference-only Yi-VL model."""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlavaConfig

from sglang.srt.models.llava import (
    LlavaLlamaForCausalLM,
    clip_vision_embed_forward,
    monkey_path_clip_vision_embed_forward,
)
from sglang.srt.weight_utils import default_weight_loader, hf_model_weights_iterator


class YiVLForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        self.config = kwargs["config"]
        super().__init__(self.config)

        self.multi_modal_projector = YiVLMultiModalProjector(self.config)
        self.vision_tower_subfolder = self.config.mm_vision_tower.replace(
            "./", ""
        )  # Everything after "./"

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        # We have to use the subfolder of the main model directory (e.g. 01-ai/Yi-VL-6B)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            subfolder=self.vision_tower_subfolder,
        ).cuda()

        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        self.image_grid_pinpoints = getattr(self.config, "image_grid_pinpoints", None)

        self.image_feature_len = int((self.image_size / self.patch_size) ** 2)
        if self.vision_feature_select_strategy == "patch":
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # load mm_projector
        # TODO: support TP?
        projector_weights = {
            "model.mm_projector.0": "multi_modal_projector.linear_1",
            "model.mm_projector.1": "multi_modal_projector.ln_1",
            "model.mm_projector.3": "multi_modal_projector.linear_2",
            "model.mm_projector.4": "multi_modal_projector.ln_2",
            "model.vision_tower.vision_tower": "vision_tower",  # Update the vision tower weights if we find them in the checkpoint (it may be finetuned).
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "projector" in name or "vision_tower" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # load language model
        self.language_model.load_weights(
            model_name_or_path, cache_dir, load_format, revision
        )

        monkey_path_clip_vision_embed_forward()


class YiVLMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size
        )
        self.ln_1 = nn.LayerNorm(config.text_config.hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size
        )
        self.ln_2 = nn.LayerNorm(config.text_config.hidden_size)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_state = self.ln_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_2(hidden_states)
        return hidden_states


EntryClass = YiVLForCausalLM
