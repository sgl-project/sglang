# Copyright 2025 The SwissAI Initiative
# Copyright 2023-2024 SGLang Team
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
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only Apertus model compatible with HuggingFace weights."""
import concurrent
import concurrent.futures
import copy
import logging
from typing import Iterable, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.srt.compilation.compile import IntermediateTensors
from sglang.srt.configs.deepseek_ocr import PRINT_NUM_VIS_TOKENS, DeepseekVLV2Config
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV3ForCausalLM,
)
from sglang.srt.models.transformers import maybe_prefix

NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]

logger = logging.getLogger(__name__)


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


def isin_list(
    elements: torch.Tensor,
    test_elements_list: list[int],
) -> torch.Tensor:
    test_elements = torch.tensor(test_elements_list, pin_memory=True).to(
        device=elements.device, non_blocking=True
    )

    return torch.isin(elements, test_elements)


def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: int | list[int],
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    `placeholder_token_id` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the `input_ids` MUST MATCH the order of
    their embeddings in `multimodal_embeddings` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

    Note:
        This updates `inputs_embeds` in place.
    """
    if isinstance(placeholder_token_id, list):
        is_multimodal = isin_list(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id

    return _merge_multimodal_embeddings(
        inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


class MlpProjector(nn.Module):

    def __init__(
        self,
        projector_type,
        input_dim,
        n_embed,
        depth=1,
        mlp_ratio=1,
        downsample_ratio=4,
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth

        super().__init__()

        # self.cfg = cfg

        if projector_type == "identity":
            modules = nn.Identity()

        elif projector_type == "linear":
            modules = nn.Linear(input_dim, n_embed)

        elif projector_type == "mlp_gelu":
            mlp_depth = depth
            modules = [nn.Linear(input_dim, n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "normlayer_downsample_mlp_gelu":
            mlp_depth = depth
            mlp_ratio = mlp_ratio
            modules = [
                nn.LayerNorm(input_dim * downsample_ratio * downsample_ratio),
                nn.Linear(
                    input_dim * downsample_ratio * downsample_ratio,
                    n_embed * mlp_ratio,
                ),
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed * mlp_ratio, n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(n_embed * mlp_ratio, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "downsample_mlp_gelu":
            mlp_depth = depth
            mlp_ratio = mlp_ratio
            modules = [
                nn.Linear(
                    input_dim * downsample_ratio * downsample_ratio,
                    n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed * mlp_ratio, n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(n_embed * mlp_ratio, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = depth
            self.high_up_proj = nn.Linear(input_dim, n_embed // 2)
            self.low_up_proj = nn.Linear(input_dim, n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "hybrid_split_feature_mlp_gelu":
            mlp_depth = depth
            channel_div = 0.5
            self.high_up_proj = nn.Linear(input_dim[0], int(n_embed * channel_div))
            self.low_up_proj = nn.Linear(
                input_dim[1], n_embed - int(n_embed * channel_div)
            )

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*modules)

        elif projector_type == "low_high_split_mlp_gelu":
            mlp_depth = depth
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(n_embed // 2, n_embed // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)
            self.low_layers = copy.deepcopy(modules)

        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        token_pooling = None
        # if get("token_pooling", False):
        #     self.token_pooling_layer = nn.Linear(input_dim * 4, input_dim)

        # if get("conv_fusion_high_low_features", False):
        #     self.fusion_layer = nn.Linear(input_dim, input_dim)
        self.layers = modules

    def forward(self, x):
        if self.get("token_pooling", False):
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        if self.get("conv_fusion_high_low_features", False):
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        if self.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.projector_type == "hybrid_split_feature_mlp_gelu":
            high_x = x[..., : self.input_dim[0]]
            low_x = x[..., self.input_dim[0] :]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)

        if self.projector_type == "low_high_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
            return x

        if (
            self.projector_type == "downsample_mlp_gelu"
            or self.projector_type == "normlayer_downsample_mlp_gelu"
        ):
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.downsample_ratio:
                pad = self.downsample_ratio - h % self.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.downsample_ratio,
                stride=self.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class DeepseekOCRForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        config: DeepseekVLV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # multimodal_config = config.multimodal_config

        # config.model_type ='deepseek_vl_v2'

        self.config = config
        # self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        n_embed = 1280
        downsample_ratio: int = 4

        self.projector = MlpProjector(
            projector_type="linear",
            input_dim=2048,
            n_embed=n_embed,
            downsample_ratio=downsample_ratio,
        )
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
        # self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
        # self.projector = torch.compile(self.projector, mode="max-autotune")

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if self.text_config.topk_method == "noaux_tc":
            architectures = ["DeepseekV3ForCausalLM"]
            self.language_model = DeepseekV3ForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        elif not self.text_config.use_mla:
            architectures = ["DeepseekForCausalLM"]
            self.language_model = DeepseekForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        else:
            architectures = ["DeepseekV2ForCausalLM"]
            self.language_model = DeepseekV2ForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )

        print(f"{architectures=}")

        # self.make_empty_intermediate_tensors = (
        #     self.language_model.make_empty_intermediate_tensors
        # )

    def _parse_and_validate_image_input(self, **kwargs: object):

        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)

        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. "
                    f"Got type: {type(images_spatial_crop)}"
                )

            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image crop. " f"Got type: {type(images_crop)}"
                )

            return [pixel_values, images_crop, images_spatial_crop]

        raise AssertionError("This line should be unreachable.")

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:

        # Pixel_values (global view): [n_image, batch_size, 3, height, width]
        # images_spatial_crop: [n_image, batch_size, [num_tiles_w, num_tiles_h]]
        # images_crop (local view): [n_image, batch_size, num_pathes, 3, h, w]
        # split the pixel and image_crop, all batch_size = 1

        images_in_this_batch = []

        # print(type(images_crop))

        # print(pixel_values.shape)

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                # with torch.set_grad_enabled(False):
                patches = images_crop[jdx][0].to(torch.bfloat16)  # batch_size = 1
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:  # if all values = 0, no crop
                    # P, C, H, W = patches.shape
                    # crop_flag = 1
                    local_features_1 = self.sam_model(patches)
                    # TODO del patches
                    # torch.compiler.cudagraph_mark_step_begin()
                    local_features_2 = self.vision_model(patches, local_features_1)

                    local_features = torch.cat(
                        (
                            local_features_2[:, 1:],
                            local_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print("=====================")
                        print("BASE: ", global_features.shape)
                        print("PATCHES: ", local_features.shape)
                        print("=====================")

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)

                    width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [
                            global_features,
                            self.image_newline[None, None, :].expand(h, 1, n_dim),
                        ],
                        dim=1,
                    )

                    global_features = global_features.view(-1, n_dim)

                    local_features = (
                        local_features.view(
                            height_crop_num, width_crop_num, h2, w2, n_dim2
                        )
                        .permute(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = torch.cat(
                        [
                            local_features,
                            self.image_newline[None, None, :].expand(
                                height_crop_num * h2, 1, n_dim2
                            ),
                        ],
                        dim=1,
                    )
                    local_features = local_features.view(-1, n_dim2)

                    global_local_features = torch.cat(
                        [local_features, global_features, self.view_seperator[None, :]],
                        dim=0,
                    )

                else:
                    global_features_1 = self.sam_model(image_ori)
                    global_features_2 = self.vision_model(image_ori, global_features_1)
                    global_features = torch.cat(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(2).permute(0, 2, 1),
                        ),
                        dim=-1,
                    )
                    global_features = self.projector(global_features)

                    if PRINT_NUM_VIS_TOKENS:
                        print("=====================")
                        print("BASE: ", global_features.shape)
                        print("NO PATCHES")
                        print("=====================")

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)

                    global_features = global_features.view(h, w, n_dim)

                    global_features = torch.cat(
                        [
                            global_features,
                            self.image_newline[None, None, :].expand(h, 1, n_dim),
                        ],
                        dim=1,
                    )

                    global_features = global_features.view(-1, n_dim)

                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(self, image_input) -> torch.Tensor:

        # image_input: [pixel_values, images_crop, images_spatial_crop]

        pixel_values = image_input[0].to(torch.bfloat16)
        # print(image_input[1][0].shape)
        # print(type(image_input[1]))
        # exit()

        # images_crop = image_input[1].to(torch.bfloat16)
        images_crop = image_input[1]
        # images_crop = image_input[1]
        images_spatial_crop = image_input[2].to(dtype=torch.long)

        # local_start = time.time()
        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        # local_total_time = time.time() - local_start

        # print('encoder_time: ', local_total_time)
        # exit()
        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )
            # print(len(multimodal_embeddings))
            # print(input_ids.shape)
            # print(type(inputs_embeds))
            # print(inputs_embeds.shape)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    # def compute_logits(
    #     self,
    #     hidden_states: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[torch.Tensor]:
    #     return self.language_model.compute_logits(hidden_states,
    #                                               sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [DeepseekOCRForCausalLM]
