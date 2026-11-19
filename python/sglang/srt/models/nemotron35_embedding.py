"""NVIDIA Nemotron-3.5-Embed: a Pixtral vision tower and the Mistral3 projector
on the bidirectional Nemotron-3-Embed text model, with normalized mean pooling."""

from typing import Optional

import torch
from transformers import AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.models.nemotron3_embedding import EmbeddingNemotron3Model
from sglang.srt.models.utils import WeightsMapper


class _QuantizableMistral3Projector(Mistral3MultiModalProjector):
    """The HF Mistral3 projector with SGLang linears, so quantized linear_1 and
    linear_2 (NVFP4 checkpoints) load; the norm and patch merger are unchanged."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__(config)
        bias = config.multimodal_projector_bias
        self.linear_1 = ReplicatedLinear(
            self.linear_1.in_features,
            self.linear_1.out_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
        )
        self.linear_2 = ReplicatedLinear(
            self.linear_2.in_features,
            self.linear_2.out_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
        )

    def forward(self, image_features: torch.Tensor, image_sizes) -> torch.Tensor:
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class EmbeddingNemotron35Model(Mistral3ForConditionalGeneration):
    """Nemotron-3.5-Embed: image and text tokens share one bidirectional pass."""

    # Quantization configs name the checkpoint's unfused projections; the vision
    # attention output projection is attention.proj here, attention.o_proj there.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "attention.proj": ["attention.o_proj"],
    }
    # NVFP4 checkpoints target text layers as language_model[.model].layers.N; the
    # nested text model names them model.layers.N.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            r"language_model(?:\.model)?\.layers": r"(?:language_model(?:\.model)?\.)?layers"
        }
    )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.multi_modal_projector = _QuantizableMistral3Projector(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("multi_modal_projector", prefix),
        )

    def _get_sgl_model_cls(self, config, auto_model_type=AutoModel):
        if auto_model_type is AutoModelForCausalLM:
            return EmbeddingNemotron3Model
        return super()._get_sgl_model_cls(
            config=config, auto_model_type=auto_model_type
        )


EntryClass = [EmbeddingNemotron35Model]
