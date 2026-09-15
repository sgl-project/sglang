import math
from collections.abc import Iterable

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip import SiglipVisionModel

import sglang.srt.managers.mm_utils as mm_utils
import sglang.srt.model_loader.weight_utils as weight_utils
import sglang.srt.utils as utils
from sglang.srt.configs.jet_vlm import JetVLMConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.jet_nemotron import JetNemotronForCausalLM

MM_HIDDEN_SIZE = 1152


class JetVLMDownSample2x2BlockFix(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        _, seq_len, _ = x.shape

        feat_size = math.isqrt(seq_len)

        features = einops.rearrange(x, "b (h w) d -> b h w d", h=feat_size, w=feat_size)

        if feat_size % 2 == 1:
            features = F.pad(features, (0, 0, 0, 1, 0, 1))

        features = einops.rearrange(
            features, "b (h p1) (w p2) d -> b (h w) (p1 p2 d)", p1=2, p2=2
        )

        return features


class JetVLMMultiModalProjector(nn.Module):
    def __init__(self, config: JetVLMConfig) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            JetVLMDownSample2x2BlockFix(),
            nn.LayerNorm(MM_HIDDEN_SIZE * 4),
            nn.Linear(MM_HIDDEN_SIZE * 4, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class JetVLMForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: JetVLMConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.mm_projector = JetVLMMultiModalProjector(config)
        self.llm = JetNemotronForCausalLM(
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


EntryClass = [JetVLMForConditionalGeneration]
