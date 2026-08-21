"""Qwen3-ASR model compatible with HuggingFace weights"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3ASRForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3ASRConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        thinker_config = config.thinker_config

        if getattr(thinker_config, "audio_config", None) is None:
            thinker_config.audio_config = Qwen3OmniMoeAudioEncoderConfig()

        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
        self.language_model = Qwen3ForCausalLM(
            thinker_config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        device = next(self.audio_tower.parameters()).device

        input_features = (
            torch.cat([item.feature for item in items])
            .type(self.audio_tower.dtype)
            .to(device)
        )

        has_mask = all(
            getattr(item, "feature_attention_mask", None) is not None for item in items
        )

        if has_mask:
            feature_attention_mask = (
                torch.cat([item.feature_attention_mask for item in items], dim=0)
                .type(torch.long)
                .to(device)
            )
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = torch.tensor(
                [input_features.shape[-1]] * input_features.shape[0],
                dtype=torch.long,
                device=device,
            )
            input_features = input_features.permute(0, 2, 1).reshape(
                -1, input_features.shape[1]
            )

        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
        )
        return audio_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # Audio tower has separate q/k/v in checkpoint → stack into qkv_proj
        audio_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            if (
                getattr(
                    self.config.thinker_config.text_config, "tie_word_embeddings", False
                )
                and "lm_head.weight" in name
            ):
                continue

            if "talker" in name or "code2wav" in name:
                continue

            if name.startswith("thinker.audio_tower."):
                name = name.replace("thinker.audio_tower.", "audio_tower.", 1)
            elif name.startswith("thinker.lm_head."):
                name = name.replace("thinker.lm_head.", "language_model.lm_head.", 1)
            elif name.startswith("thinker.model."):
                name = name.replace("thinker.model.", "language_model.model.", 1)

            is_audio = "audio_tower" in name

            # Audio tower: remap out_proj → proj for VisionAttention
            if is_audio and "out_proj" in name:
                name = name.replace("out_proj", "proj")

            stacked_params = audio_stacked_params if is_audio else llm_stacked_params

            for param_name, weight_name, shard_id in stacked_params:
                if weight_name not in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                if name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Qwen3ASRForConditionalGeneration
