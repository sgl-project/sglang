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

MM_HIDDEN_SIZE = 1152


class NVILALiteConfig(PretrainedConfig):
    model_type = "nvila_lite"
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


class NVILALiteMultiModalProjectorDownsampleBlock(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = x.shape

        feat_size = math.isqrt(sequence_length)

        features = x.reshape(batch_size, feat_size, feat_size, hidden_size)

        pad_after = (3 - feat_size % 3) % 3
        if pad_after > 0:
            features = F.pad(features, (0, 0, 0, pad_after, 0, pad_after))
            feat_size = feat_size + pad_after

        features = features.reshape(
            batch_size, feat_size // 3, 3, feat_size // 3, 3, hidden_size
        )
        features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
        features = features.reshape(batch_size, -1, 9 * hidden_size)

        return features


class NVILALiteMultiModalProjector(nn.Module):
    def __init__(self, config: NVILALiteConfig):
        super().__init__()

        self.layers = nn.Sequential(
            NVILALiteMultiModalProjectorDownsampleBlock(),
            nn.LayerNorm(MM_HIDDEN_SIZE * 9),
            nn.Linear(MM_HIDDEN_SIZE * 9, MM_HIDDEN_SIZE * 3),
            nn.GELU(),
            nn.LayerNorm(MM_HIDDEN_SIZE * 3),
            nn.Linear(MM_HIDDEN_SIZE * 3, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class NVILALiteForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: NVILALiteConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.mm_projector = NVILALiteMultiModalProjector(config)
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
        pixel_values = torch.cat([torch.tensor(x.feature) for x in mm_input], dim=0)

        vision_tower_output: BaseModelOutputWithPooling = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
        )
        assert vision_tower_output.hidden_states is not None

        vision_features = vision_tower_output.hidden_states[-2]

        vision_features = self.mm_projector(vision_features)

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


EntryClass = [NVILALiteForConditionalGeneration]
