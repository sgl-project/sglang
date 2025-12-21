import itertools
import math
from collections.abc import Iterable
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel

import sglang.srt.managers.mm_utils as mm_utils
import sglang.srt.model_loader.weight_utils as weight_utils
import sglang.srt.utils as utils
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

MM_HIDDEN_SIZE = 3456


class NVILAConfig(PretrainedConfig):
    model_type = "nvila"
    sub_configs = {
        "text_config": Qwen2Config,
        "vision_config": SiglipVisionConfig,
    }
    _auto_class = "AutoConfig"

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        **kwargs,
    ):
        self.text_config = (
            Qwen2Config(**text_config) if text_config is not None else Qwen2Config()
        )
        self.vision_config = (
            SiglipVisionConfig(**vision_config)
            if vision_config is not None
            else SiglipVisionConfig()
        )

        self.image_token_id = image_token_id if image_token_id is not None else -1
        self.video_token_id = video_token_id if video_token_id is not None else -1

        super().__init__(**kwargs)


class NVILAMultiModalProjectorDownsampleBlock(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = x.shape

        feat_size = math.isqrt(sequence_length)

        features = x.reshape(batch_size, feat_size, feat_size, hidden_size)

        pad_after = feat_size % 2
        if pad_after > 0:
            features = F.pad(features, (0, 0, 0, pad_after, 0, pad_after))
            feat_size = feat_size + pad_after

        features = features.reshape(
            batch_size, feat_size // 2, 2, feat_size // 2, 2, hidden_size
        )
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        features = features.reshape(batch_size, -1, 4 * hidden_size)

        return features


class NVILAMultiModalProjector(nn.Module):
    def __init__(self, config: NVILAConfig):
        super().__init__()

        self.layers = nn.Sequential(
            NVILAMultiModalProjectorDownsampleBlock(),
            nn.LayerNorm(MM_HIDDEN_SIZE * 4),
            nn.Linear(MM_HIDDEN_SIZE * 4, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class NVILAForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: NVILAConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.mm_projector = NVILAMultiModalProjector(config)
        self.llm = Qwen2ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=utils.add_prefix("llm", prefix),
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        output = mm_utils.general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_image_feature,
            },
            get_embedding=get_embedding,
            positions=positions,
        )

        assert isinstance(output, LogitsProcessorOutput)

        return output

    def get_image_feature(self, mm_input: list[MultimodalDataItem]) -> Tensor:
        block_sizes = (
            list(
                itertools.chain.from_iterable(
                    x.block_sizes for x in mm_input if hasattr(x, "block_sizes")
                )
            )
            or None
        )
        pixel_values = torch.cat([torch.tensor(x.feature) for x in mm_input], dim=0)

        vision_tower_output: BaseModelOutputWithPooling = self.vision_tower(
            pixel_values.to(
                device=self.vision_tower.device, dtype=self.vision_tower.dtype
            ),
            output_hidden_states=True,
        )
        assert vision_tower_output.hidden_states is not None

        vision_features: Tensor = vision_tower_output.hidden_states[-2]

        vision_features_list, block_sizes = merge_features_for_dynamic_s2(
            vision_features,
            block_sizes=(
                block_sizes
                if block_sizes is not None
                else [None] * vision_features.shape[0]
            ),
            resize_output_to_scale_idx=-1,
            scales=[448, 896, 1344],
        )

        vision_features_list = [
            split_chessboard(x, block_size[0], block_size[1])
            for x, block_size in zip(vision_features_list, block_sizes)
        ]

        vision_features = torch.cat(
            [einops.rearrange(x, "b c h w -> b (h w) c") for x in vision_features_list]
        )

        vision_features = self.mm_projector(vision_features)

        vision_features_list = list(
            vision_features.split(
                [block_size[0] * block_size[1] for block_size in block_sizes], dim=0
            )
        )
        vision_features_list = [
            merge_chessboard(x, block_size[0], block_size[1])
            for x, block_size in zip(vision_features_list, block_sizes)
        ]

        vision_features = torch.stack(
            [einops.rearrange(x, "1 c h w -> (h w) c") for x in vision_features_list]
        )

        vision_features = einops.rearrange(vision_features, "n p d -> (n p) d")

        return vision_features

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.startswith("llm."):
                self.llm.load_weights([(name[len("llm.") :], loaded_weight)])
            else:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", weight_utils.default_weight_loader
                )
                weight_loader(param, loaded_weight)

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)


def merge_chessboard(x, num_split_h, num_split_w):
    """
    x: b * n * c or b * h * w * c
    out: b * c * h * w
    Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
    """
    B = x.shape[0]
    if x.dim() == 3:
        N = x.shape[1]
        x = einops.rearrange(
            x, "b (h w) c -> b c h w", h=math.isqrt(N), w=math.isqrt(N)
        )

    assert B % (num_split_h * num_split_w) == 0
    b = B // (num_split_h * num_split_w)

    x_merge = torch.cat(
        [
            torch.cat(
                [
                    x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b]
                    for j in range(num_split_w)
                ],
                dim=-1,
            )
            for i in range(num_split_h)
        ],
        dim=-2,
    )

    return x_merge


def merge_features_for_dynamic_s2(
    image_features, block_sizes, *, scales, resize_output_to_scale_idx
):
    image_features_each_image = []
    new_block_sizes = []
    block_cnt = 0
    for block_size_each_image in block_sizes:
        if block_size_each_image is None:
            cur_features = image_features[block_cnt : block_cnt + 1]
            cur_features = einops.rearrange(
                cur_features,
                "1 (h w) c -> 1 c h w",
                h=math.isqrt(cur_features.shape[1]),
            )
            cur_features = cur_features.repeat(1, len(scales), 1, 1)
            image_features_each_image.append(cur_features)
            new_block_sizes.append((1, 1))
            block_cnt += 1
        else:
            cur_features_each_scale = []
            for scale in scales[:-1]:
                num_blocks_this_scale = (scale // scales[0]) ** 2
                cur_features_each_scale.append(
                    merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_this_scale],
                        num_split_h=scale // scales[0],
                        num_split_w=scale // scales[0],
                    )
                )  # 1 * C * H * W
                block_cnt += num_blocks_this_scale
            num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]
            cur_features_each_scale.append(
                merge_chessboard(
                    image_features[block_cnt : block_cnt + num_blocks_last_scale],
                    num_split_h=block_size_each_image[0],
                    num_split_w=block_size_each_image[1],
                )
            )  # 1 * C * H * W
            block_cnt += num_blocks_last_scale

            # resize and concat features from different scales
            output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[-2:]
            cur_features = torch.cat(
                [
                    F.interpolate(
                        cur_features_each_scale[i].to(torch.float32),
                        size=output_size,
                        mode="area",
                    ).to(cur_features_each_scale[i].dtype)
                    for i in range(len(cur_features_each_scale))
                ],
                dim=1,
            )

            image_features_each_image.append(cur_features)

            if (
                resize_output_to_scale_idx == len(scales) - 1
                or resize_output_to_scale_idx == -1
            ):
                new_block_sizes.append(block_size_each_image)
            else:
                new_block_sizes.append(
                    (
                        scales[resize_output_to_scale_idx] // scales[0],
                        scales[resize_output_to_scale_idx] // scales[0],
                    )
                )

    assert block_cnt == len(
        image_features
    ), f"The number of blocks ({block_cnt}) does not match length of image_features ({len(image_features)})!"

    return image_features_each_image, new_block_sizes


def split_chessboard(x, num_split_h, num_split_w):
    """
    x: b * c * h * w
    out: b * c * h * w
    Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split_h == 0 and W % num_split_w == 0
    h, w = H // num_split_h, W // num_split_w
    x_split = torch.cat(
        [
            x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w]
            for i in range(num_split_h)
            for j in range(num_split_w)
        ],
        dim=0,
    )
    return x_split


EntryClass = [NVILAForConditionalGeneration]
