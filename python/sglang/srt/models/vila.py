import logging
from typing import Dict, Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as Qwen2ForCausalLMHF,
)
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel

import sglang.srt.managers.mm_utils as mm_utils
import sglang.srt.utils as utils
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


##### BEGIN COPY configuration.py #####


class VILAConfig(PretrainedConfig):
    # Configuration for sub-modules.
    text_config: Qwen2Config = Qwen2Config()
    vision_config: SiglipVisionConfig = SiglipVisionConfig()

    # Model configuration.
    hidden_size: int
    image_token_id: int
    image_end_token_id: int
    mm_hidden_size: int
    mm_projector_type: str
    mm_vision_select_feature: str
    mm_vision_select_layer: int
    video_token_id: int


##### END COPY configuration.py #####

##### BEGIN COPY modeling_vila.py #####


class DownSampleBlock(nn.Module):
    @staticmethod
    def flat_square(x: Tensor) -> Tensor:
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, 1, h, c), device=x.device, dtype=x.dtype)], dim=1
            ).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, w, 1, c), device=x.device, dtype=x.dtype)], dim=2
            ).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x: Tensor) -> Tensor:
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


class DownSample3x3BlockFix(nn.Module):
    @staticmethod
    def flat_square_3x3(x: Tensor) -> Tensor:
        n, w, h, c = x.size()
        if w % 3 != 0:
            x = torch.concat(
                [
                    x,
                    torch.zeros((n, 3 - (w % 3), h, c), device=x.device, dtype=x.dtype),
                ],
                dim=1,
            ).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        if h % 3 != 0:
            x = torch.concat(
                [
                    x,
                    torch.zeros((n, w, 3 - (h % 3), c), device=x.device, dtype=x.dtype),
                ],
                dim=2,
            ).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 3), int(c * 3))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 3), int(w / 3), int(c * 9))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x: Tensor) -> Tensor:
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square_3x3(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


class MultimodalProjector(nn.Module):
    layers: nn.Sequential

    def __init__(
        self,
        config: VILAConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        match config.mm_projector_type:
            case "linear":
                self.layers = nn.Sequential(
                    nn.Linear(config.vision_config.hidden_size, config.hidden_size),
                )
            case "mlp_downsample":
                self.layers = nn.Sequential(
                    DownSampleBlock(),
                    nn.LayerNorm(config.mm_hidden_size * 4),
                    nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            case "mlp_downsample_3x3_fix":
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
            case _:
                raise NotImplementedError(
                    f"mm_projector_type={config.mm_projector_type} not implemented."
                )

        self.layers.to(dtype=config.torch_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


##### END COPY modeling_vila.py #####


class VILAForConditionalGenerationHF(PreTrainedModel, GenerationMixin):
    config: VILAConfig

    llm: Qwen2ForCausalLMHF
    mm_projector: MultimodalProjector
    vision_tower: SiglipVisionModel


class VILAForConditionalGeneration(nn.Module):
    config: VILAConfig
    quant_config: Optional[QuantizationConfig]

    logits_processor: LogitsProcessor
    pooler: Pooler

    llm: Qwen2ForCausalLM
    mm_projector: MultimodalProjector
    vision_tower: SiglipVisionModel

    image_end_token_embedding: Optional[Tensor] = None

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
            image_data_embedding_func=self._embed_image_data,
            get_embedding=get_embedding,
            positions=positions,
        )

        return cast(LogitsProcessorOutput, output)

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> None:
        llm_state_dict: Dict[str, Tensor] = {}
        other_state_dict: Dict[str, Tensor] = {}

        for name, param in weights:
            if name.startswith("llm."):
                llm_state_dict[name[len("llm.") :]] = param
            else:
                other_state_dict[name] = param

        self.load_state_dict(other_state_dict, strict=False)
        self.llm.load_weights(llm_state_dict.items())

    def pad_input_ids(
        self,
        input_ids: List[int],
        image_inputs: MultimodalInputs,
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens(
            token_ids=[self.config.image_token_id],
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)

    def _embed_image_data(self, mm_input: List[MultimodalDataItem]) -> Tensor:
        pixel_values = torch.cat([mm_item.pixel_values for mm_item in mm_input], dim=0)

        ##### BEGIN COPY AND MODIFY modeling_vila.py #####

        image_features: BaseModelOutputWithPooling = self.vision_tower.__call__(
            pixel_values.to(
                device=self.vision_tower.device,
                dtype=self.vision_tower.dtype,
            ),
            output_hidden_states=True,
        )
        assert image_features.hidden_states is not None

        # Select image feature.
        selected_layer_output = image_features.hidden_states[
            self.config.mm_vision_select_layer
        ]
        match self.config.mm_vision_select_feature:
            case "cls_patch":
                selected_feature = selected_layer_output
            case _:
                raise NotImplementedError(
                    f"mm_vision_select_feature={self.config.mm_vision_select_feature} not implemented."
                )

        # TODO: Support dynamic_s2.

        image_embedding: Tensor = self.mm_projector.__call__(
            selected_feature.to(
                device=self.mm_projector.device,
                dtype=self.mm_projector.dtype,
            )
        )

        ##### END COPY AND MODIFY modeling_vila.py #####

        # Append the image end token to every image embedding.
        if self.image_end_token_embedding is None:
            image_end_token_embedding: (
                Tensor
            ) = self.llm.get_input_embeddings().__call__(
                torch.tensor(
                    self.config.image_end_token_id,
                    device=next(self.llm.parameters()).device,
                    dtype=torch.long,
                ).view(1, -1)
            )  # Shape: (1, 1, dim_feature)
            self.image_end_token_embedding = image_end_token_embedding
        else:
            image_end_token_embedding = self.image_end_token_embedding

        image_end_token_embedding = image_end_token_embedding.expand(
            image_embedding.shape[0], 1, -1
        )  # Shape: (n_images, 1, dim_feature)
        image_embedding = torch.concat(
            [
                image_embedding.to(device=image_end_token_embedding.device),
                image_end_token_embedding,
            ],
            dim=1,
        )

        return image_embedding


EntryClass = [VILAForConditionalGeneration]
