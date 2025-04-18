import logging
from typing import Dict, Iterable, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel, Qwen2Config
from transformers import Qwen2ForCausalLM as Qwen2ForCausalLMHF
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

import sglang.srt.managers.mm_utils as mm_utils
import sglang.srt.utils as utils
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternImageTokens
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

_IMAGE_TOKEN_IDS = [151649]

logger = logging.getLogger(__name__)


class VILAConfig(PretrainedConfig):
    text_config: Qwen2Config
    vision_config: SiglipVisionConfig

    hidden_size: int
    image_token_id: int
    mm_projector_type: str
    mm_vision_select_feature: str
    mm_vision_select_layer: int


##### Copy from remote code. #####


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
            case "mlp_downsample_3x3_fix":
                self.layers = nn.Sequential(
                    DownSample3x3BlockFix(),
                    nn.LayerNorm(config.vision_config.hidden_size * 9),
                    nn.Linear(
                        config.vision_config.hidden_size * 9,
                        config.vision_config.hidden_size * 3,
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

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


##### End of copy. #####


class VILAForConditionalGenerationHF(PreTrainedModel, GenerationMixin):
    config: VILAConfig

    llm: Qwen2ForCausalLMHF
    mm_projector: MultimodalProjector
    vision_tower: SiglipVisionModel


class VILAForConditionalGeneration(nn.Module):
    config: VILAConfig
    quant_config: Optional[QuantizationConfig]

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

        self.llm = Qwen2ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=utils.add_prefix("llm", prefix),
        )
        self.mm_projector = MultimodalProjector(config)
        self.vision_tower = SiglipVisionModel(config.vision_config)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        input_embeds = mm_utils.general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            embed_tokens=cast(nn.Embedding, self.llm.model.embed_tokens),
            mm_data_embedding_func=self._embed_mm_data,
        )

        return self.llm.__call__(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

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
        pattern = MultiModalityDataPaddingPatternImageTokens(
            image_token_id=torch.tensor(_IMAGE_TOKEN_IDS)
        )

        return pattern.pad_input_tokens(input_ids, image_inputs)

    def _embed_mm_data(self, mm_input: MultimodalInputs) -> Tensor:
        assert isinstance(mm_input.pixel_values, Tensor)
        pixel_values = mm_input.pixel_values.to(
            device=self.vision_tower.device, dtype=self.vision_tower.dtype
        )

        ##### Copy from remote code. #####

        image_features: BaseModelOutputWithPooling = self.vision_tower.__call__(
            pixel_values,
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

        image_embedding: Tensor = self.mm_projector.__call__(selected_feature)

        n_images, n_feature, dim_feature = image_embedding.shape
        image_embedding = image_embedding.view(n_images * n_feature, dim_feature)

        ##### End of copy. #####

        return image_embedding


EntryClass = [VILAForConditionalGeneration]
