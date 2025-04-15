import logging
from typing import Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoModelForImageTextToText,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2Config,
)

import sglang.srt.utils
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers import mm_utils
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternImageTokens
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

_IMAGE_TOKEN_IDS = [151649]

logger = logging.getLogger(__name__)


class VILAConfig(PretrainedConfig):
    text_config: Qwen2Config


class VILAForConditionalGenerationHF(PreTrainedModel, GenerationMixin):
    mm_projector: PreTrainedModel
    llm: PreTrainedModel
    vision_tower: PreTrainedModel


class VILAForConditionalGeneration(nn.Module):
    config: VILAConfig
    quant_config: Optional[QuantizationConfig]

    language_model: Qwen2ForCausalLM

    mm_projector: PreTrainedModel
    vision_tower: PreTrainedModel

    def __init__(
        self,
        config: VILAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        hf_model: VILAForConditionalGenerationHF = (
            AutoModelForImageTextToText.from_pretrained(
                config.name_or_path,
                trust_remote_code=True,
            )
        )
        self.mm_projector = hf_model.mm_projector
        self.vision_tower = hf_model.vision_tower

        self.language_model = Qwen2ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=sglang.srt.utils.add_prefix("llm", prefix),
        )
        self.language_model.load_weights(hf_model.llm.named_parameters())

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        input_embeds = mm_utils.general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            embed_tokens=cast(nn.Embedding, self.language_model.model.embed_tokens),
            mm_data_embedding_func=self._embed_mm_data,
        )

        return self.language_model.__call__(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        # VILA stores weights in sub-directories, so SGLang cannot load them directly.
        logger.warning(
            "VILA does not support SGLang weight loading. Its weights are directly loaded from HuggingFace on initialization."
        )

    def pad_input_ids(
        self,
        input_ids: List[int],
        image_inputs: MultimodalInputs,
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternImageTokens(
            image_token_id=torch.tensor(_IMAGE_TOKEN_IDS)
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)

    def _embed_mm_data(self, mm_input: MultimodalInputs) -> Tensor:
        assert isinstance(mm_input.pixel_values, Tensor)

        pixel_values = mm_input.pixel_values.to(
            device=self.vision_tower.device, dtype=self.vision_tower.dtype
        )

        image_features: Tensor = self.vision_tower.__call__(pixel_values)
        image_features = self.mm_projector.__call__(image_features)

        return image_features


EntryClass = [VILAForConditionalGeneration]
