from typing import Iterable, Optional, Tuple

import torch
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration as BaseClass,
)

from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.minicpmo import MiniCPMWhisperEncoder


class WhisperForConditionalGeneration(BaseClass):
    def __init__(
        self, quant_config: Optional[QuantizationConfig] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model.encoder = MiniCPMWhisperEncoder(self.model.encoder.config)

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
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [WhisperForConditionalGeneration]
