import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

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
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


##### BEGIN COPY configuration.py #####


class VILAConfig(PretrainedConfig):
    # Class attributes.
    model_type: str = "vila"
    sub_configs: Dict[str, PretrainedConfig] = {
        "text_config": Qwen2Config(),
        "vision_config": SiglipVisionConfig(),
    }
    _auto_class: Optional[str] = "AutoConfig"

    # Configuration for sub-modules.
    text_config: Qwen2Config = Qwen2Config()
    vision_config: SiglipVisionConfig = SiglipVisionConfig()

    # Model configuration.
    hidden_size: int
    image_token_id: int
    mm_hidden_size: int
    mm_projector_type: str
    mm_vision_select_feature: str
    mm_vision_select_layer: int
    video_token_id: int

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        *,
        hidden_size: int = 1536,
        image_token_id: int = 151649,
        mm_hidden_size: int = 1152,
        mm_projector_type: str = "mlp_downsample_3x3_fix",
        mm_vision_select_feature: str = "cls_patch",
        mm_vision_select_layer: int = -2,
        video_token_id: int = 151650,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_config = Qwen2Config(**text_config) if text_config else Qwen2Config()
        self.vision_config = (
            SiglipVisionConfig(**vision_config)
            if vision_config
            else SiglipVisionConfig()
        )

        self.hidden_size = hidden_size
        self.image_token_id = image_token_id
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.video_token_id = video_token_id


##### END COPY configuration.py #####

##### BEGIN COPY modeling_vila.py #####


class DownSample3x3BlockFix(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: The input tensor of shape (batch_size, sequence_length, mm_hidden_size).

        Returns:
            The output tensor of shape (batch_size, image_pad_len, mm_hidden_size * 9).
        """

        batch_size, sequence_length, hidden_size = x.shape

        feat_size = int(sequence_length**0.5)
        if feat_size**2 != sequence_length:
            raise ValueError(
                f"Cannot take square root: sequence_length {sequence_length} is not a perfect square"
            )

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


class MultimodalProjector(nn.Module):
    layers: nn.Sequential

    def __init__(
        self,
        config: VILAConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if config.mm_projector_type == "mlp_downsample_3x3_fix":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(
                    config.mm_hidden_size * 9,
                    config.mm_hidden_size * 3,
                ),
                nn.GELU(),
                nn.LayerNorm(config.vision_config.hidden_size * 3),
                nn.Linear(config.vision_config.hidden_size * 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            raise NotImplementedError(
                f"Unsupported mm_projector_type: {config.mm_projector_type}"
            )

        self.layers.type(config.torch_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: The input tensor of shape (batch_size, sequence_length, mm_hidden_size).

        Returns:
            The output tensor of shape (batch_size, image_pad_len, hidden_size).
        """

        return self.layers(x.to(device=self.device, dtype=self.dtype))


##### END COPY modeling_vila.py #####


class VILAForConditionalGeneration(nn.Module):
    config: VILAConfig
    quant_config: Optional[QuantizationConfig]

    logits_processor: LogitsProcessor
    pooler: Pooler

    llm: Qwen2ForCausalLM
    mm_projector: MultimodalProjector
    vision_tower: SiglipVisionModel

    def __init__(
        self,
        config: VILAConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        self.llm = Qwen2ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=utils.add_prefix("llm", prefix),
        )
        self.mm_projector = MultimodalProjector(config)
        self.vision_tower = SiglipVisionModel(config.vision_config)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.torch_dtype

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
            },
            get_embedding=get_embedding,
            positions=positions,
        )

        return cast(LogitsProcessorOutput, output)

    def get_image_feature(self, mm_input: List[MultimodalDataItem]) -> Tensor:
        pixel_values = cast(Tensor, mm_input[0].feature)

        ##### BEGIN COPY modeling_vila.py #####

        vision_tower_output: BaseModelOutputWithPooling = self.vision_tower.__call__(
            pixel_values.to(
                device=self.vision_tower.device, dtype=self.vision_tower.dtype
            ),
            output_hidden_states=True,
        )

        mm_projector_input = self._vision_tower_output_to_mm_projector_input(
            vision_tower_output
        )

        image_embedding: Tensor = self.mm_projector.__call__(
            mm_projector_input.to(
                device=self.mm_projector.device, dtype=self.mm_projector.dtype
            )
        )

        ##### END COPY modeling_vila.py #####

        return image_embedding

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
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
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    ##### BEGIN COPY modeling_vila.py #####

    def _vision_tower_output_to_mm_projector_input(
        self,
        vision_tower_output: BaseModelOutputWithPooling,
    ) -> Tensor:
        assert vision_tower_output.hidden_states is not None

        selected_layer_hidden_states = vision_tower_output.hidden_states[
            self.config.mm_vision_select_layer
        ]

        if self.config.mm_vision_select_feature == "cls_patch":
            return selected_layer_hidden_states
        else:
            raise NotImplementedError(
                f"Unsupported mm_vision_select_feature: {self.config.mm_vision_select_feature}"
            )

    ##### END COPY modeling_vila.py #####


EntryClass = [VILAForConditionalGeneration]
