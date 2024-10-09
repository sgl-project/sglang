"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Inference-only NVLM-D model compatible with HuggingFace weights."""

import math
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlavaConfig,
    MistralConfig,
    Qwen2Config,
    SiglipVisionModel,
)
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from vllm.config import CacheConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


# config = AutoConfig.from_pretrained("nvidia/NVLM-D-72B")

# adapted from https://huggingface.co/nvidia/NVLM-D-72B/blob/main/modeling_nvlm_d.py#L76
class NVLMVisionEncoder(nn.Module):
    def __init__(self, config: NVLM_D_Config):
        super().__init__()

        self.config = config
        self.image_vit = AutoModel.from_pretrained(
            'OpenGVLab/InternViT-6B-224px',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda().eval()


        vit_hidden_size = config.vision_config.hidden_size
        llm_intermediate_size = config.llm_config.intermediate_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(llm_intermediate_size, llm_hidden_size, bias=False)
        )

    def encode_images(self, pixel_values: torch.Tensor):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds