# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Qwen3-TTS model support for SGLang.

Qwen3-TTS is a text-to-speech model from Alibaba/Qwen that generates audio
from text input. It uses a "Talker" architecture with multi-codebook audio
token generation.

Model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

Architecture:
- Talker: Main transformer that generates the first audio codebook tokens
- CodePredictor: Sub-transformer that predicts remaining codebook tokens
- SpeakerEncoder (optional): ECAPA-TDNN for voice cloning (Base model only)
- SpeechTokenizer: Encodes/decodes audio to/from discrete tokens

Note: This is an initial integration. Full TTS functionality requires:
1. Custom generation loop for multi-codebook prediction
2. Speech tokenizer integration for audio decoding
3. Specialized API endpoints for TTS inference

For now, this enables loading the model weights and provides the foundation
for TTS inference integration.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class Qwen3TTSConfig(PretrainedConfig):
    """Configuration class for Qwen3-TTS model."""

    model_type = "qwen3_tts"

    def __init__(
        self,
        talker_config: Optional[Dict] = None,
        speaker_encoder_config: Optional[Dict] = None,
        tokenizer_type: Optional[str] = None,
        tts_model_size: Optional[str] = None,
        tts_model_type: Optional[str] = None,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        tts_pad_token_id: int = 151671,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.talker_config = talker_config or {}
        self.speaker_encoder_config = speaker_encoder_config or {}
        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id


class Qwen3TTSTalkerMLP(nn.Module):
    """MLP layer for the Talker model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3TTSTalkerAttention(nn.Module):
    """Multi-headed attention for the Talker model with M-RoPE support."""

    def __init__(
        self,
        config: Dict,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=config.get("attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=config.get("attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Q/K normalization (Qwen3-style)
        self.q_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))
        self.k_norm = RMSNorm(self.head_dim, eps=config.get("rms_norm_eps", 1e-6))

        # RoPE for position encoding
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        # Apply Q/K normalization
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(-1, self.num_heads * self.head_dim)
        k = k.view(-1, self.num_kv_heads * self.head_dim)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Attention
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3TTSTalkerDecoderLayer(nn.Module):
    """A single transformer decoder layer for the Talker model."""

    def __init__(
        self,
        config: Dict,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        hidden_act = config.get("hidden_act", "silu")
        intermediate_size = config.get("intermediate_size", 2048)
        num_attention_heads = config.get("num_attention_heads", 16)
        num_kv_heads = config.get("num_key_value_heads", 2)
        rope_theta = config.get("rope_theta", 10000.0)
        rope_scaling = config.get("rope_scaling", None)
        max_position_embeddings = config.get("max_position_embeddings", 32768)
        rms_norm_eps = config.get("rms_norm_eps", 1e-6)

        self.self_attn = Qwen3TTSTalkerAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3TTSTalkerMLP(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3TTSTalkerModel(nn.Module):
    """The main Talker model for Qwen3-TTS."""

    def __init__(
        self,
        config: Dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.get("vocab_size", 3072)
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_hidden_layers = config.get("num_hidden_layers", 20)
        text_vocab_size = config.get("text_vocab_size", 151936)
        text_hidden_size = config.get("text_hidden_size", 2048)

        # Codec embedding for audio tokens
        self.codec_embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            prefix=add_prefix("codec_embedding", prefix),
        )

        # Text embedding (for processing input text)
        self.text_embedding = VocabParallelEmbedding(
            text_vocab_size,
            text_hidden_size,
            prefix=add_prefix("text_embedding", prefix),
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3TTSTalkerDecoderLayer(
                    config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(self.num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(
            self.hidden_size,
            eps=config.get("rms_norm_eps", 1e-6),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.codec_embedding(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3TTSCodePredictorLayer(nn.Module):
    """A decoder layer for the CodePredictor sub-model."""

    def __init__(
        self,
        config: Dict,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        intermediate_size = config.get("intermediate_size", 3072)
        num_attention_heads = config.get("num_attention_heads", 16)
        num_kv_heads = config.get("num_key_value_heads", 8)
        rms_norm_eps = config.get("rms_norm_eps", 1e-6)

        self.self_attn = Qwen3TTSTalkerAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3TTSTalkerMLP(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3TTSCodePredictor(nn.Module):
    """CodePredictor model that predicts remaining codebook tokens."""

    def __init__(
        self,
        config: Dict,
        talker_hidden_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 1024)
        self.vocab_size = config.get("vocab_size", 2048)
        self.num_hidden_layers = config.get("num_hidden_layers", 5)
        self.num_code_groups = config.get("num_code_groups", 32)

        # Codec embeddings for each codebook group (except first)
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(self.vocab_size, talker_hidden_size)
                for _ in range(self.num_code_groups - 1)
            ]
        )

        # Projection from talker hidden size to code predictor hidden size
        if talker_hidden_size != self.hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                talker_hidden_size, self.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3TTSCodePredictorLayer(
                    config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(self.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))

        # Output heads for each codebook group
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.vocab_size, bias=False)
                for _ in range(self.num_code_groups - 1)
            ]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.small_to_mtp_projection(hidden_states)

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3TTSTextProjection(nn.Module):
    """Projects text embeddings to talker hidden size."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    """Complete Talker model with codec head and code predictor."""

    def __init__(
        self,
        config: Dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        hidden_size = config.get("hidden_size", 1024)
        vocab_size = config.get("vocab_size", 3072)
        text_hidden_size = config.get("text_hidden_size", 2048)

        # Text to talker projection
        self.text_projection = Qwen3TTSTextProjection(
            input_size=text_hidden_size,
            intermediate_size=text_hidden_size,
            output_size=hidden_size,
            hidden_act=config.get("hidden_act", "silu"),
        )

        # Codec output head
        self.codec_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Code predictor for multi-codebook generation
        code_predictor_config = config.get("code_predictor_config", {})
        self.code_predictor = Qwen3TTSCodePredictor(
            code_predictor_config,
            talker_hidden_size=hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("code_predictor", prefix),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, inputs_embeds=inputs_embeds
        )
        return hidden_states


class Qwen3TTSSpeakerEncoder(nn.Module):
    """
    ECAPA-TDNN based speaker encoder for voice cloning.
    Extracts speaker embeddings from mel spectrograms.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.mel_dim = config.get("mel_dim", 128)
        self.enc_dim = config.get("enc_dim", 1024)
        enc_channels = config.get("enc_channels", [512, 512, 512, 512, 1536])
        enc_kernel_sizes = config.get("enc_kernel_sizes", [5, 3, 3, 3, 1])
        enc_dilations = config.get("enc_dilations", [1, 2, 3, 4, 1])

        # Build TDNN blocks
        self.blocks = nn.ModuleList()

        # Initial TDNN layer
        self.blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    self.mel_dim,
                    enc_channels[0],
                    enc_kernel_sizes[0],
                    dilation=enc_dilations[0],
                    padding="same",
                    padding_mode="reflect",
                ),
                nn.ReLU(),
            )
        )

        # SE-Res2Net layers (simplified - full implementation would need Res2Net)
        for i in range(1, len(enc_channels) - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        enc_channels[i - 1],
                        enc_channels[i],
                        enc_kernel_sizes[i],
                        dilation=enc_dilations[i],
                        padding="same",
                        padding_mode="reflect",
                    ),
                    nn.ReLU(),
                )
            )

        # MFA layer
        self.mfa = nn.Sequential(
            nn.Conv1d(
                enc_channels[-1],
                enc_channels[-1],
                enc_kernel_sizes[-1],
                dilation=enc_dilations[-1],
                padding="same",
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )

        # Final projection
        self.fc = nn.Conv1d(
            enc_channels[-1] * 2,  # Mean + std pooling
            self.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: (batch, time, mel_dim) mel spectrogram

        Returns:
            speaker_embedding: (batch, enc_dim) speaker embedding
        """
        # Transpose to (batch, mel_dim, time) for Conv1d
        x = mel_spectrogram.transpose(1, 2)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(features[1:], dim=1)
        x = self.mfa(x)

        # Statistics pooling
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        x = torch.cat([mean, std], dim=1)

        # Final projection
        x = self.fc(x.unsqueeze(-1)).squeeze(-1)
        return x


class Qwen3TTSForConditionalGeneration(nn.Module):
    """
    Full Qwen3-TTS model for text-to-speech generation.

    This model consists of:
    - Talker: Main transformer that generates audio codebook tokens
    - CodePredictor: Predicts remaining codebook tokens for multi-codebook audio
    - SpeakerEncoder: (Optional) Extracts speaker embeddings for voice cloning

    Note: Full TTS inference requires additional components:
    - Speech tokenizer for decoding audio codes to waveforms
    - Custom generation loop for multi-codebook prediction
    """

    def __init__(
        self,
        config: Qwen3TTSConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # Get talker config
        talker_config = config.talker_config
        if isinstance(talker_config, dict):
            pass
        else:
            talker_config = talker_config.__dict__ if hasattr(talker_config, "__dict__") else {}

        # Build Talker model
        self.talker = Qwen3TTSTalkerForConditionalGeneration(
            talker_config,
            quant_config=quant_config,
            prefix=add_prefix("talker", prefix),
        )

        # Build Speaker Encoder (only for base model)
        speaker_encoder_config = config.speaker_encoder_config
        if isinstance(speaker_encoder_config, dict):
            pass
        else:
            speaker_encoder_config = speaker_encoder_config.__dict__ if hasattr(speaker_encoder_config, "__dict__") else {}

        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_encoder_config)
        else:
            self.speaker_encoder = None

        # Logits processor for the codec head
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Talker model.

        Note: This is a simplified forward pass. Full TTS generation requires
        the custom multi-step generation loop implemented in the original model.
        """
        hidden_states = self.talker(
            input_ids, positions, forward_batch, inputs_embeds=inputs_embeds
        )

        # Get codec logits
        logits = self.talker.codec_head(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load model weights from checkpoint."""
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
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name_tmp = name.replace(weight_name, param_name)
                if name_tmp not in params_dict:
                    continue

                param = params_dict[name_tmp]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Standard weight loading
                if name not in params_dict:
                    # Try to find matching parameter with different prefix
                    logger.debug(f"Skipping weight: {name}")
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# Entry class for SGLang model registry
EntryClass = Qwen3TTSForConditionalGeneration
