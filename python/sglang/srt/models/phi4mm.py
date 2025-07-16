# Copyright 2024 SGLang Team
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
# Adapted from
# https://github.com/vllm-project/vllm/blob/6071e989df1531b59ef35568f83f7351afb0b51e/vllm/model_executor/models/phi4mm.py
# https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py

import logging
import math
import re
from collections.abc import Iterable
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig, SiglipVisionConfig

from sglang.srt.layers.quantization import QuantizationConfig
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
from sglang.srt.models.idefics2 import Idefics2VisionTransformer
from sglang.srt.models.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)

SIGLIP_NAME = "siglip-so400m-patch14-448"
VISION_ENCODER_TO_PROCESSING_CONFIG = {
    "siglip-so400m-patch14-448": {
        "vit_image_size": 448,
        "vit_patch_size": 14,
        "token_compression_factor": 2,
    },
}


def get_navit_vision_model():
    vision_config = {
        "hidden_size": 1152,
        "image_size": 448,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 26,  # Model is originally 27-layer, we only need the first 26 layers for feature extraction.
        "patch_size": 14,
    }
    model_config = SiglipVisionConfig(**vision_config)

    vision_model = Idefics2VisionTransformer(
        config=model_config, require_post_norm=False
    )

    return vision_model


class Phi4MMImageEncoder(nn.Module):
    """Image embedding."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
        model_dir: str = "",
    ) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
        self.type_feature = "patch"

        self.img_processor = get_navit_vision_model()

        pe_weight = self.img_processor.embeddings.position_embedding.weight
        L, D = pe_weight.size()
        H = int(math.sqrt(L))
        assert H**2 == L, f"position embedding size {L} is not square"
        if H % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            H += 1
        image_dim_out = D
        # ((448/14)//2)**2
        self.num_img_tokens = (H // 2) ** 2
        self.base_feat_height_target = H

        self.image_dim_out = image_dim_out
        self.img_sizes = None
        self.image_attention_mask = None

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = True
        self.with_learnable_separator = True
        self.hd_transform_order = "sub_glb"
        self.freeze_img_processor = False
        self.crop_size = 448

        # image token compression
        self.image_token_compression_cls = "avg_pool_2d"
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.base_feat_height_reduction = 1
        self.base_feat_height_target = self.base_feat_height_target // 2

        # with_hd_transform and with_learnable_separator should have same value
        assert (
            self.use_hd_transform == self.with_learnable_separator
        ), "use_hd_transform and with_learnable_separator should have same value"
        assert self.use_hd_transform, "learnable separator is only for hd transform"
        # 1024 * 4, merge spatial to channel dimension
        self.glb_GN = nn.Parameter(
            torch.zeros([1, 1, self.image_dim_out * self.base_feat_height_reduction**2])
        )
        self.sub_GN = nn.Parameter(
            torch.zeros(
                [1, 1, 1, self.image_dim_out * self.base_feat_height_reduction**2]
            )
        )

        dim_projection = hidden_size
        depth = 2
        layers = [
            nn.Linear(
                image_dim_out * self.base_feat_height_reduction**2, dim_projection
            )
        ]
        for _ in range(1, depth):
            layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
        self.img_projection = nn.Sequential(*layers)

        self.vocab_size = config.vocab_size
        self.img_features = None

        self.use_out_place_operations = False

    def get_img_features(
        self, img_embeds: torch.FloatTensor, attention_mask=None
    ) -> torch.FloatTensor:
        img_feature = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask
        )

        patch_feature = img_feature

        use_token_compression = self.image_token_compression is not None
        use_padding = getattr(self, "img_processor_padding", None) is not None
        if use_token_compression or use_padding:
            # reshape to 2D tensor
            width = int(math.sqrt(patch_feature.size(1)))
            patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
            # convert to NCHW
            patch_feature = patch_feature.permute(0, 3, 1, 2)

            if use_padding:
                patch_feature = self.img_processor_padding(patch_feature)
            if use_token_compression:
                patch_feature = self.image_token_compression(patch_feature)

            # convert to NHWC
            patch_feature = patch_feature.permute(0, 2, 3, 1)
            patch_feature = patch_feature.view(
                -1,
                patch_feature.size(1) * patch_feature.size(2),
                patch_feature.size(-1),
            )

        return patch_feature

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        image_attention_mask: torch.Tensor,
    ) -> list[torch.FloatTensor]:
        """
        process image and return vision embeddings.

        pixel_values: (num_images, num_crops, c, h, w)
        image_sizes: [[h1, w1], [h2, w2]]
        image_attention_mask: num_images x num_crops x 32 x 32
        output: (num_images, num_img_tokens, hidden_size)
        """

        # eg
        # pixel_values: torch.Size([1, 7, 3, 448, 448])
        # image_sizes: tensor([[ 896, 1344]], device='cuda:0')
        # output: torch.Size([1, 1841, 3072])

        img_projection_params = next(self.img_projection.parameters())
        target_device = img_projection_params.device
        target_dtype = img_projection_params.dtype

        img_sizes = image_sizes
        num_images, num_crops, c, h, w = pixel_values.shape
        bs = num_images
        pixel_values = pixel_values.flatten(0, 1)

        img_features = self.get_img_features(
            pixel_values,
            image_attention_mask.type(torch.BoolTensor).flatten(0, 1).to(target_device),
        )

        base_feat_height_target = self.base_feat_height_target
        base_resolution = self.crop_size
        base_feat_height_reduction = self.base_feat_height_reduction

        base_feat_height = base_feat_width = int(np.sqrt(img_features.shape[1]))
        assert (
            base_feat_height == base_feat_height_target
            and base_feat_width == base_feat_height_target
        ), f'base_feat_height: {base_feat_height},"\
                f" base_feat_width: {base_feat_width}, "\
                f"expect {base_feat_height_target} features for hd transform'

        # bs x max_num_crops x (24x24) x C
        img_features = img_features.view(
            bs, -1, base_feat_height * base_feat_width, self.image_dim_out
        )
        C = self.image_dim_out
        H = base_feat_height

        output_imgs = []
        output_len = []
        # training is tensor, inference is list
        if isinstance(img_sizes, torch.Tensor):
            img_sizes = img_sizes.view(-1, 2)
        for _bs in range(bs):
            h, w = img_sizes[_bs]
            h = h // base_resolution
            w = w // base_resolution
            B_ = h * w

            # 1 x (24x24) x 1024
            global_img_feature = img_features[_bs, :1]

            # 1 x 12 x 12 x 4096
            glb_img = (
                global_img_feature.reshape(1, H, H, C)
                .reshape(
                    1,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    C,
                )
                .contiguous()
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    1,
                    H // base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction * base_feat_height_reduction * C,
                )
                .contiguous()
            )
            temp_glb_GN = self.sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)

            # 1 x 156 x 4096
            glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                1, -1, base_feat_height_reduction * base_feat_height_reduction * C
            )

            # (max_num_crops-1) x (12x12) x C
            sub_img = img_features[_bs, 1:]
            # 16x574x1024
            # get rid of padding sub_img
            sub_img = sub_img[:B_]

            # (num_crops, 12, 2, 12, 2, 1024) ->
            # (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
            sub_img = (
                sub_img.reshape(B_, H, H, C)
                .reshape(
                    B_,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    H // base_feat_height_reduction,
                    base_feat_height_reduction,
                    C,
                )
                .contiguous()
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    B_, -1, base_feat_height_reduction * base_feat_height_reduction * C
                )
                .contiguous()
            )
            sub_img = (
                sub_img.reshape(
                    1,
                    h,
                    w,
                    base_feat_height // base_feat_height_reduction,
                    base_feat_width // base_feat_height_reduction,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(
                    1,
                    h * base_feat_height // base_feat_height_reduction,
                    w * base_feat_width // base_feat_height_reduction,
                    base_feat_height_reduction * base_feat_height_reduction * C,
                )
            )

            if image_attention_mask is not None and len(image_attention_mask) > 0:
                reshaped_image_attention_mask = (
                    image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]
                    .reshape(
                        1,
                        h,
                        w,
                        base_feat_height // base_feat_height_reduction,
                        base_feat_width // base_feat_height_reduction,
                    )
                    .permute(0, 1, 3, 2, 4)
                    .reshape(
                        1,
                        h * base_feat_height // base_feat_height_reduction,
                        w * base_feat_width // base_feat_height_reduction,
                    )
                )
                useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                temp_len = (
                    int(image_attention_mask[_bs, : B_ + 1, 0::2, 0::2].sum().item())
                    + (useful_height + 1)
                    + base_feat_height // base_feat_height_reduction
                )
            else:
                temp_sub_GN = self.sub_GN.repeat(
                    1, h * base_feat_height // base_feat_height_reduction, 1, 1
                )
                temp_len = int(
                    (h * w + 1) * self.num_img_tokens
                    + 1
                    + (h + 1) * base_feat_height // base_feat_height_reduction
                )

            sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                1, -1, base_feat_height_reduction * base_feat_height_reduction * C
            )
            # (1, num_img_tokens, 1024*4)

            # glb + sub
            if self.hd_transform_order == "glb_sub":
                output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
            elif self.hd_transform_order == "sub_glb":
                output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
            else:
                raise NotImplementedError(
                    f'hd_transform_order = {self.hd_transform_order}, "\
                        "not implemented'
                )

            # temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
            assert (
                temp_len == output_imgs[-1].shape[1]
            ), f'temp_len: {temp_len}, output_imgs[-1].shape[1]: "\
                    "{output_imgs[-1].shape[1]}'

            output_len.append(temp_len)

        img_set_tensor = []
        for _output_img in output_imgs:
            img_feature_proj = self.img_projection(
                _output_img.to(target_device).to(target_dtype)
            )
            img_set_tensor.append(img_feature_proj.squeeze(0))

        return img_set_tensor


class Phi4MMForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    lora_pattern = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(?:qkv_proj|o_proj|down_proj|gate_up_proj)"
    )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.language_model = LlamaForCausalLM(
            config=config, quant_config=quant_config, prefix=prefix
        )

        self.vision_encoder = Phi4MMImageEncoder(
            config,
            quant_config,
            prefix="model.vision_embed_tokens",
            model_dir=config._name_or_path,
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = torch.cat([item.pixel_values for item in items], dim=0).type(
            dtype
        )
        image_attention_mask = torch.cat([item.image_emb_mask for item in items], dim=0)
        image_sizes = torch.cat([item.image_sizes for item in items], dim=0)
        image_embeds = self.vision_encoder(
            pixel_values, image_sizes, image_attention_mask
        )
        return torch.cat(image_embeds).type(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hidden_states

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def should_apply_lora(self, module_name: str) -> bool:
        return bool(self.lora_pattern.match(module_name))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        prefix_mapping = {
            "model.embed_tokens_extend.image_embed.": "vision_encoder.",
            "model.": "language_model.model.",
        }

        skip_list = [
            "img_processor.encoder.layers.26",
            "img_processor.head",
            "img_processor.post_layernorm",
            "audio",
        ]

        def _should_skip(name: str) -> bool:
            return any(substr in name for substr in skip_list)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Skip the last layer
            if _should_skip(name):
                continue

            for old_name, new_name in prefix_mapping.items():
                if name.startswith(old_name):
                    name = name.replace(old_name, new_name)
                    break

            # Adapt to VisionAttention
            name = name.replace(r"self_attn.out_proj", r"self_attn.proj")
            name = name.replace(r"base_layer.", r"")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.get(name)
                if param is None:
                    if "lora" not in name:
                        logger.warning("Warning: {name} not found in model parameters")
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [Phi4MMForCausalLM]
