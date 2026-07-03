from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.moss_transcribe_diarize import MossTranscribeDiarizeConfig
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
from sglang.srt.models.whisper import WhisperEncoder
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class VQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MossTranscribeDiarizeForConditionalGeneration(nn.Module):
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
        config: MossTranscribeDiarizeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
        self.vq_adaptor = VQAdaptor(
            input_dim=config.adaptor_input_dim,
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.rms_norm_eps,
        )
        self.language_model = Qwen3ForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("model.language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def time_merge(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = features.shape
        merge_size = int(self.config.audio_merge_size)
        trimmed_len = (seq_len // merge_size) * merge_size
        return features[:, :trimmed_len, :].reshape(
            batch_size, trimmed_len // merge_size, hidden_size * merge_size
        )

    def _encode_one_audio_item(
        self,
        item: MultimodalDataItem,
        forward_batch: ForwardBatch,
    ) -> list[torch.Tensor]:
        if item.feature is None:
            raise ValueError(
                "MOSS-Transcribe-Diarize audio item is missing input_features."
            )

        device = next(self.whisper_encoder.parameters()).device
        encoder_dtype = next(self.whisper_encoder.parameters()).dtype
        input_features = item.feature.to(device=device, dtype=encoder_dtype)

        audio_feature_lengths = getattr(item, "audio_feature_lengths", None)
        if audio_feature_lengths is None:
            raise ValueError(
                "MOSS-Transcribe-Diarize audio item is missing audio_feature_lengths."
            )
        audio_feature_lengths = audio_feature_lengths.to(device="cpu", dtype=torch.long)
        if audio_feature_lengths.numel() != input_features.shape[0]:
            raise ValueError(
                "audio_feature_lengths must contain one length per input_features chunk: "
                f"got {audio_feature_lengths.numel()} lengths for {input_features.shape[0]} chunks."
            )

        audio_chunk_mapping = getattr(item, "audio_chunk_mapping", None)
        if audio_chunk_mapping is None:
            audio_chunk_mapping = torch.zeros(
                input_features.shape[0], dtype=torch.long, device="cpu"
            )
        else:
            audio_chunk_mapping = audio_chunk_mapping.to(device="cpu", dtype=torch.long)
        if audio_chunk_mapping.numel() != input_features.shape[0]:
            raise ValueError(
                "audio_chunk_mapping must contain one sample index per input_features chunk: "
                f"got {audio_chunk_mapping.numel()} indices for {input_features.shape[0]} chunks."
            )

        encoder_len = (input_features.shape[-1] - 1) // 2 + 1
        encoder_position_ids = torch.arange(
            encoder_len,
            device=input_features.device,
            dtype=torch.long,
        )
        whisper_features = self.whisper_encoder(
            input_features,
            encoder_position_ids,
            forward_batch,
        )

        audio_feature_lengths_list = audio_feature_lengths.tolist()
        audio_chunk_mapping_list = audio_chunk_mapping.tolist()
        num_audios = 0
        if audio_chunk_mapping_list:
            num_audios = max(audio_chunk_mapping_list) + 1
        per_audio_chunks = [[] for _ in range(num_audios)]
        merge_size = int(self.config.audio_merge_size)
        for chunk_idx, token_len in enumerate(audio_feature_lengths_list):
            sample_idx = audio_chunk_mapping_list[chunk_idx]
            per_audio_chunks[sample_idx].append(
                whisper_features[
                    chunk_idx : chunk_idx + 1, : int(token_len) * merge_size
                ]
            )

        adapted = []
        adaptor_dtype = next(self.vq_adaptor.parameters()).dtype
        for parts in per_audio_chunks:
            if not parts:
                continue
            feat = torch.cat(parts, dim=1).to(dtype=adaptor_dtype)
            merged = self.time_merge(feat)
            adapted.append(self.vq_adaptor(merged).squeeze(0))
        return adapted

    def get_audio_feature(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        audio_embeds = []
        for item in items:
            audio_embeds.extend(self._encode_one_audio_item(item, forward_batch))
        if not audio_embeds:
            hidden_size = self.config.text_config.hidden_size
            device = next(self.vq_adaptor.parameters()).device
            dtype = next(self.vq_adaptor.parameters()).dtype
            return torch.empty((0, hidden_size), device=device, dtype=dtype)
        return torch.cat(audio_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: lambda items: self.get_audio_feature(
                    items,
                    forward_batch,
                ),
            },
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        whisper_stacked_params_mapping = [
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        def load_one(name: str, loaded_weight: torch.Tensor):
            original_name = name
            if "rotary_emb.inv_freq" in name:
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                return

            if name == "lm_head.weight":
                name = "language_model.lm_head.weight"
            elif name.startswith("model.language_model."):
                name = "language_model.model." + name[len("model.language_model.") :]
            elif name.startswith("model.whisper_encoder."):
                name = "whisper_encoder." + name[len("model.whisper_encoder.") :]
            elif name.startswith("model.vq_adaptor."):
                name = "vq_adaptor." + name[len("model.vq_adaptor.") :]

            if (
                name == "language_model.model.embed_tokens.weight"
                and self.config.text_config.tie_word_embeddings
                and "language_model.lm_head.weight" in params_dict
            ):
                param = params_dict["language_model.lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            handled = False
            if name.startswith("whisper_encoder."):
                for param_name, weight_name, shard_id in whisper_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if name.startswith("language_model."):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if handled:
                return

            if name.endswith(".bias") and name not in params_dict:
                return

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug("Skipping weight: %s -> %s", original_name, name)

        for name, loaded_weight in weights:
            load_one(name, loaded_weight)
            if (
                name.startswith("model.whisper_encoder.layers.")
                and ".self_attn.k_proj.weight" in name
            ):
                load_one(
                    name.replace(".k_proj.weight", ".k_proj.bias"),
                    torch.zeros(
                        loaded_weight.shape[0],
                        dtype=loaded_weight.dtype,
                        device=loaded_weight.device,
                    ),
                )


EntryClass = MossTranscribeDiarizeForConditionalGeneration
