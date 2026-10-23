"""NVIDIA Nemotron-3.5-Embed: a Pixtral vision tower and Mistral3 projector on
the bidirectional Nemotron-3-Embed text model, with normalized mean pooling."""

from array import array
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron3_embedding import EmbeddingNemotron3Model
from sglang.srt.models.pixtral import PixtralHFVisionModel
from sglang.srt.utils import add_prefix


class EmbeddingNemotron35Model(nn.Module):
    """Nemotron-3.5-Embed: image and text inputs share one bidirectional pass."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = PixtralHFVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = EmbeddingNemotron3Model(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return self.vision_tower.pad_input_ids(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        features = []
        for item in items:
            image_features = self.vision_tower(item.feature, item.image_sizes)
            features.append(
                self.multi_modal_projector(image_features.squeeze(0), item.image_sizes)
            )
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The checkpoint stores the submodules at the top level:
        # language_model.*, vision_tower.* and multi_modal_projector.*.
        params_dict = dict(self.multi_modal_projector.named_parameters())
        for name, loaded_weight in weights:
            part, _, rest = name.partition(".")
            if part in ("language_model", "vision_tower"):
                getattr(self, part).load_weights([(rest, loaded_weight)])
            elif part == "multi_modal_projector":
                param = params_dict[rest]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                raise ValueError(f"Unexpected Nemotron-3.5-Embed weight: {name}")


EntryClass = [EmbeddingNemotron35Model]
