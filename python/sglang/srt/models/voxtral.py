# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/voxtral.py
# https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
#
# Copyright 2025 Mistral AI and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Inference-only Voxtral (speech-to-text) model."""

import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
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
from sglang.srt.models.llama import LlamaForCausalLM


class AudioLanguageAdapter(nn.Module):
    """MLP projector: Linear -> GELU -> Linear (no bias)."""

    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


class VoxtralWhisperAttention(nn.Module):
    """Multi-headed self-attention using plain SDPA (no KV cache).

    Note: HF Voxtral has bias on q_proj, v_proj, out_proj but NOT on k_proj.
    We use QKVParallelLinear with bias=True and create a zero bias for k_proj
    during weight loading.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim, self.head_dim, num_heads, quant_config=quant_config
        )
        # After TP split, the local head count lives on the linear layer
        self.num_heads = self.qkv_proj.num_heads
        self.out_proj = RowParallelLinear(
            embed_dim, embed_dim, bias=True, quant_config=quant_config
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q * self.scaling

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1.0
        )
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class VoxtralWhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        embed_dim = config.d_model
        self.self_attn = VoxtralWhisperAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            quant_config=quant_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = get_act_fn(
            getattr(config, "activation_function", "gelu"),
            quant_config=quant_config,
        )
        self.fc1 = ColumnParallelLinear(embed_dim, config.encoder_ffn_dim)
        self.fc2 = RowParallelLinear(config.encoder_ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states


class VoxtralWhisperEncoder(nn.Module):
    """Whisper encoder (Conv1d + positional embed + transformer + layer norm)."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        embed_dim = config.d_model

        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)
        self.layers = nn.ModuleList(
            [
                VoxtralWhisperEncoderLayer(config, quant_config)
                for _ in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [batch, num_mel_bins, seq_len]
        Returns:
            [batch, seq_len // 2, d_model]
        """
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.embed_positions(position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class VoxtralForConditionalGeneration(nn.Module):
    """Voxtral: Whisper encoder + MLP projector + Llama decoder.

    HF weight prefixes:
        audio_tower.*           -> self.audio_tower (VoxtralWhisperEncoder)
        multi_modal_projector.* -> self.multi_modal_projector (AudioLanguageAdapter)
        language_model.*        -> self.language_model (LlamaForCausalLM)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        audio_config = config.audio_config
        text_config = config.text_config

        # Ensure text_config has rope_parameters (transformers v5 compatibility)
        if not hasattr(text_config, "rope_parameters"):
            text_config.rope_parameters = {
                "rope_type": getattr(text_config, "rope_type", "default"),
                "rope_theta": getattr(text_config, "rope_theta", 10000.0),
            }
            if getattr(text_config, "rope_scaling", None):
                text_config.rope_parameters.update(text_config.rope_scaling)

        # Infer downsample_factor: intermediate_size / hidden_size for HF format
        self.downsample_factor = getattr(
            audio_config,
            "downsample_factor",
            audio_config.intermediate_size // audio_config.hidden_size,
        )

        # Encoder (named audio_tower to match HF weight prefix directly)
        self.audio_tower = VoxtralWhisperEncoder(audio_config, quant_config)

        # Projector: input = d_model * downsample_factor, output = text_hidden_size
        adapter_input_dim = audio_config.d_model * self.downsample_factor
        self.multi_modal_projector = AudioLanguageAdapter(
            hidden_size=adapter_input_dim,
            dim=text_config.hidden_size,
        )

        # Language model
        self.language_model = LlamaForCausalLM(text_config, quant_config=quant_config)

        # Mel filter bank for raw waveform -> mel spectrogram
        self._init_mel_filters(audio_config)

        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def _init_mel_filters(self, audio_config: PretrainedConfig):
        """Initialize mel filter bank for mel spectrogram computation."""
        self._window_size = getattr(audio_config, "window_size", 400)
        self._hop_length = getattr(audio_config, "hop_length", 160)
        self._sampling_rate = getattr(audio_config, "sampling_rate", 16000)

        try:
            from mistral_common.audio import mel_filter_bank
        except ImportError:
            raise ImportError(
                "mistral_common is required for Voxtral. "
                "Install it with: pip install mistral_common"
            )

        mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self._window_size // 2,
            num_mel_bins=audio_config.num_mel_bins,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self._sampling_rate,
        )
        self.register_buffer(
            "mel_filters", torch.tensor(mel_filters, dtype=torch.float32)
        )

    @property
    def _conv_downsample_factor(self) -> int:
        return self.audio_tower.conv1.stride[0] * self.audio_tower.conv2.stride[0]

    @property
    def _chunk_size(self) -> int:
        return (
            self.config.audio_config.max_source_positions * self._conv_downsample_factor
        )

    def _compute_mel_spectrogram(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram from raw waveform using STFT."""
        window = torch.hann_window(self._window_size, device=audio_waveform.device)
        stft = torch.stft(
            audio_waveform,
            self._window_size,
            self._hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec_max = log_spec.max()
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _encode_audio(self, audio_waveforms: List[torch.Tensor]) -> List[torch.Tensor]:
        """Encode raw audio waveforms through mel spectrogram + whisper encoder."""
        dtype = self.audio_tower.conv1.weight.dtype
        device = self.audio_tower.conv1.weight.device

        chunked_features: List[torch.Tensor] = []
        chunks_per_example: List[int] = []
        chunk_size = self._chunk_size
        # Pad raw audio to a multiple of chunk_samples so that silence is
        # properly converted to mel features (matching HF VoxtralProcessor).
        chunk_samples = chunk_size * self._hop_length

        for waveform in audio_waveforms:
            waveform = waveform.to(device=device, dtype=torch.float32)
            n_samples = waveform.shape[-1]
            target_samples = chunk_samples * math.ceil(n_samples / chunk_samples)
            if target_samples > n_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, target_samples - n_samples)
                )
            mel = self._compute_mel_spectrogram(waveform)
            chunks = mel.split(chunk_size, dim=-1)
            chunked_features.extend(chunks)
            chunks_per_example.append(len(chunks))

        if not chunked_features:
            return []

        input_embeds = torch.stack(chunked_features).to(dtype)
        encoder_out = self.audio_tower(input_embeds)

        results = []
        chunk_idx = 0
        for n_chunks in chunks_per_example:
            result = encoder_out[chunk_idx : chunk_idx + n_chunks].flatten(0, 1)
            results.append(result)
            chunk_idx += n_chunks

        return results

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode audio waveforms -> downsample -> project."""
        audio_waveforms = [item.feature for item in items]
        audio_embeddings = self._encode_audio(audio_waveforms)

        # Downsample: reshape to merge adjacent frames
        for i, emb in enumerate(audio_embeddings):
            seq_len, dim = emb.shape
            audio_embeddings[i] = emb.reshape(
                seq_len // self.downsample_factor,
                dim * self.downsample_factor,
            )

        # Project through adapter
        packed = torch.cat(audio_embeddings, dim=0)
        packed = self.multi_modal_projector(packed)

        return packed

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

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        encoder_stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        encoder_dict = dict(self.audio_tower.named_parameters())
        projector_dict = dict(self.multi_modal_projector.named_parameters())

        # Collect all weights; synthesise missing k_proj bias as zeros.
        weights_list = list(weights)
        extra_weights = []
        for name, w in weights_list:
            if name.startswith("audio_tower.") and ".self_attn.k_proj.weight" in name:
                bias_name = name.replace(".weight", ".bias")
                if not any(n == bias_name for n, _ in weights_list):
                    extra_weights.append(
                        (bias_name, torch.zeros(w.shape[0], dtype=w.dtype))
                    )
        weights_list.extend(extra_weights)

        def llm_weights_generator():
            for name, w in weights_list:
                # Encoder weights
                if name.startswith("audio_tower."):
                    trimmed = name[len("audio_tower.") :]
                    loaded = False
                    for param_name, weight_name, shard_id in encoder_stacked:
                        if f".{weight_name}." in trimmed:
                            stacked_name = trimmed.replace(weight_name, param_name)
                            if stacked_name in encoder_dict:
                                param = encoder_dict[stacked_name]
                                param.weight_loader(param, w, shard_id)
                                loaded = True
                                break
                    if not loaded and trimmed in encoder_dict:
                        param = encoder_dict[trimmed]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, w)
                    continue

                # Projector weights
                if name.startswith("multi_modal_projector."):
                    trimmed = name[len("multi_modal_projector.") :]
                    trimmed = trimmed.replace("linear_1.", "w_in.").replace(
                        "linear_2.", "w_out."
                    )
                    if trimmed in projector_dict:
                        param = projector_dict[trimmed]
                        default_weight_loader(param, w)
                    continue

                # LLM weights
                if name.startswith("language_model."):
                    name = name[len("language_model.") :]
                yield (name, w)

        self.language_model.load_weights(llm_weights_generator())


EntryClass = [VoxtralForConditionalGeneration]
