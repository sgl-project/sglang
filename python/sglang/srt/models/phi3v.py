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
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi3v.py

import logging
import math
import re
from collections.abc import Iterable
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPVisionConfig, PretrainedConfig

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
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
from sglang.srt.models.clip import CLIPVisionModel
from sglang.srt.models.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 32044

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
    dropout=0.0,
    hidden_act="quick_gelu",
    hidden_size=1024,
    image_size=336,
    intermediate_size=4096,
    num_attention_heads=16,
    num_channels=3,
    num_hidden_layers=24,
    patch_size=14,
    projection_dim=768,
)


def _init_img_processor(
    hf_config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
    prefix: str = "",
) -> CLIPVisionModel:
    clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
    layer_idx = hf_config.img_processor.get("layer_idx", -2)

    # Initialize the CLIP only up to the required feature layer
    if layer_idx < 0:
        num_hidden_layers = clip_config.num_hidden_layers + layer_idx + 1
    else:
        num_hidden_layers = layer_idx + 1

    # Override the num_hidden_layers in config
    clip_config.num_hidden_layers = num_hidden_layers
    
    img_processor = CLIPVisionModel(
        config=clip_config,
        quant_config=quant_config,
        prefix=prefix,
    )

    return img_processor


class Phi3ImageEmbeddingBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_idx: int
        self.type_feature: str
        self.img_processor: CLIPVisionModel

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        TYPE_FEATURE = self.type_feature

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the img_processor
        img_feature = self.img_processor(img_embeds)

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError


# adapted from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_embedding_phi3_v.py
class Phi3HDImageEmbedding(Phi3ImageEmbeddingBase):
    """Phi3 Image embedding with HD transform."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = (
            config.n_embd if hasattr(config, "n_embd") else config.hidden_size
        )

        self.img_processor = _init_img_processor(
            config, quant_config, prefix=f"{prefix}.img_processor"
        )

        image_dim_out = config.img_processor["image_dim_out"]
        self.num_img_tokens = config.img_processor["num_img_tokens"]

        self.image_dim_out = image_dim_out

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = config.embd_layer.get("use_hd_transform", False)
        self.with_learnable_separator = config.embd_layer.get(
            "with_learnable_separator", False
        )
        self.hd_transform_order = config.embd_layer.get(
            "hd_transform_order", "glb_sub"
        )
        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform and self.with_learnable_separator

        # 1024 * 4, merge spatial to channel dimension
        self.glb_GN = nn.Parameter(torch.empty([1, 1, self.image_dim_out * 4]))
        self.sub_GN = nn.Parameter(torch.empty([1, 1, 1, self.image_dim_out * 4]))

        dim_projection = hidden_size
        depth = 2
        layers = [nn.Linear(image_dim_out * 4, dim_projection)]
        for _ in range(1, depth):
            layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
        self.img_projection = nn.Sequential(*layers)

        self.type_feature = config.img_processor.get("type_feature", "patch")

    def forward(
        self, pixel_values: torch.FloatTensor, image_sizes: torch.Tensor
    ) -> List[torch.FloatTensor]:
        """
        process image and return vision embeddings.

        pixel_values: (num_images, num_crops, c, h, w)
        output: list of (num_img_tokens, hidden_size) tensors
        """
        num_images, num_crops, c, h, w = pixel_values.shape
        pixel_values = pixel_values.flatten(0, 1)
        img_features = self.get_img_features(pixel_values)
        img_features = img_features.reshape(
            num_images, num_crops, -1, self.image_dim_out
        )
        image_features_proj = self.hd_feature_transform(img_features, image_sizes)
        return image_features_proj

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """
        assert (
            self.hd_transform_order == "sub_glb"
        ), f"hd_transform_order `{self.hd_transform_order}` not implemented"
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(
            global_image_features, 1, 1
        )
        global_image_features_hd_newline = self.add_image_newline(
            global_image_features_hd
        )

        batch_image_features_proj = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1 : 1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                sub_image_features, h_crop, w_crop
            )
            sub_image_features_hd_newline = self.add_image_newline(
                sub_image_features_hd
            )

            # [sub features, separator, global features]
            image_embeddings = torch.cat(
                [
                    sub_image_features_hd_newline.squeeze(
                        0
                    ),  # (h_crop*12*(w_crop*12+1), 4096)
                    self.glb_GN.squeeze(0),
                    global_image_features_hd_newline[i],
                ]
            )
            img_proj = self.img_projection(
                image_embeddings.to(target_device, target_dtype)
            )
            batch_image_features_proj.append(img_proj)

        return batch_image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096)
        where h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 576 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(
                num_images, h_crop, w_crop, H // 2, H // 2, -1
            )  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(
                num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
            )  # n_img, h_crop*12, w_crop*12, 4096
        )
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(
            num_images, h, -1, -1
        )  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings], dim=2
        ).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline


class Phi3VForCausalLM(nn.Module):
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

        self.config = config
        self.multimodal_config = config
        self.image_token_id = _IMAGE_TOKEN_ID

        # Vision encoder
        self.vision_embed_tokens = Phi3HDImageEmbedding(
            config, quant_config, prefix="model.vision_embed_tokens"
        )

        # Language model
        self.language_model = LlamaForCausalLM(
            config=config, quant_config=quant_config, prefix=prefix
        )

        self.vocab_size = config.vocab_size

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        dtype = next(self.vision_embed_tokens.parameters()).dtype
        
        # Extract pixel values and image sizes from items
        pixel_values_list = []
        image_sizes_list = []
        
        for item in items:
            # item.feature should be (num_crops, c, h, w)
            pixel_values_list.append(item.feature.unsqueeze(0))  # Add batch dimension
            # item.image_sizes should be (h, w)
            image_sizes_list.append(item.image_sizes)
        
        pixel_values = torch.cat(pixel_values_list, dim=0).type(dtype)
        image_sizes = torch.stack(image_sizes_list, dim=0)
        
        # Get image embeddings - returns a list of tensors
        image_embeds = self.vision_embed_tokens(pixel_values, image_sizes)
        
        # Concatenate all image embeddings
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
            "model.vision_embed_tokens.wte": "language_model.model.embed_tokens",
            "model.vision_embed_tokens.": "vision_embed_tokens.",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }

        params_dict = dict(self.named_parameters())
        loaded_params = set()
        
        for name, loaded_weight in weights:
            # Apply prefix mapping
            for old_name, new_name in prefix_mapping.items():
                if name.startswith(old_name):
                    name = name.replace(old_name, new_name)
                    break

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                break
            else:
                # Regular loading
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                elif "lora" not in name:
                    logger.warning(f"Warning: {name} not found in model parameters")

        # Check if embeddings are tied
        if "language_model.model.embed_tokens.weight" not in loaded_params:
            # Share embeddings between vision and language models if needed
            if hasattr(self.vision_embed_tokens, "wte"):
                self.language_model.model.embed_tokens = self.vision_embed_tokens.wte
                loaded_params.add("language_model.model.embed_tokens.weight")


EntryClass = [Phi3VForCausalLM]