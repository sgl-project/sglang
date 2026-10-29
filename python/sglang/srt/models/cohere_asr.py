"""Inference-only CohereLabs cohere-transcribe (``cohere_asr``) model.

A Fast Conformer audio encoder feeding a Transformer decoder that cross-attends
to the encoder output. The layout follows `models/whisper.py`: the encoder runs
outside the KV cache during prefill and its output is written to the
cross-attention KV slots that `pad_input_ids` reserved, while the decoder's
self- and cross-attention go through `RadixAttention`.
"""

from __future__ import annotations

import math
from array import array
from typing import Any, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.configs.cohere_asr import CohereAsrConfig, cohere_asr_encoder_seq_len
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel

# Feature-extractor buffers ride along in the checkpoint; they belong to the
# processor's mel frontend, not the model.
_PREPROCESSOR_WEIGHT_PREFIX = "preprocessor."


class CohereAsrConvSubsampling(nn.Module):
    """Depthwise-striding convolutional subsampling, 8x along the time axis."""

    def __init__(self, encoder_config: dict):
        super().__init__()
        feat_in = int(encoder_config["feat_in"])
        channels = int(encoder_config["subsampling_conv_channels"])
        subsampling_factor = int(encoder_config["subsampling_factor"])
        feat_out = int(encoder_config["feat_out"])
        if feat_out <= 0:
            feat_out = int(encoder_config["d_model"])

        # Indices are load-bearing: the checkpoint stores these as
        # `pre_encode.conv.{0,2,3,5,6}`, so the ReLUs must keep their slots.
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1, groups=channels
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=2, padding=1, groups=channels
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
        )
        self.out = nn.Linear(channels * (feat_in // subsampling_factor), feat_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [1, mel_bins, time] -> [1, 1, time, mel_bins]
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)
        batch, channels, time, freq = x.size()
        x = x.transpose(1, 2).reshape(batch, time, channels * freq)
        return self.out(x)


class CohereAsrRelPositionalEncoding(nn.Module):
    """Transformer-XL style relative sinusoidal positions, 2L-1 wide."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("pe", self._build(max_len), persistent=False)

    def _build(self, length: int) -> torch.Tensor:
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(positions.size(0), self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if self.pe.size(1) < 2 * length - 1:
            self.pe = self._build(length).to(device=x.device, dtype=self.pe.dtype)
        # Position 0 sits one past the midpoint; negative offsets address tokens
        # to the right, positive ones to the left.
        center = self.pe.size(1) // 2 + 1
        pos_emb = self.pe[:, center - length : center + length - 1]
        return pos_emb.to(device=x.device, dtype=x.dtype)


class CohereAsrConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class CohereAsrConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=(kernel_size - 1) // 2,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = nn.functional.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class CohereAsrRelPositionAttention(nn.Module):
    """Encoder self-attention with an untied relative-position bias.

    The `matrix_bd` term has no `RadixAttention` equivalent, so this runs as
    plain batched matmuls and claims no KV-cache layer id.
    """

    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.scaling = self.d_k**-0.5
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    def _heads(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        return x.view(batch, -1, self.h, self.d_k).transpose(1, 2)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        batch, heads, qlen, pos_len = x.size()
        x = nn.functional.pad(x, pad=(1, 0))
        x = x.view(batch, heads, -1, qlen)
        return x[:, :, 1:].view(batch, heads, qlen, pos_len)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        q = self._heads(self.linear_q(x), batch)
        k = self._heads(self.linear_k(x), batch)
        v = self._heads(self.linear_v(x), batch)
        p = self._heads(self.linear_pos(pos_emb), pos_emb.size(0))

        matrix_ac = torch.matmul(
            q + self.pos_bias_u[None, :, None, :], k.transpose(-1, -2)
        )
        matrix_bd = torch.matmul(
            q + self.pos_bias_v[None, :, None, :], p.transpose(-1, -2)
        )
        matrix_bd = self._rel_shift(matrix_bd)[..., : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) * self.scaling

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.linear_out(out)


class CohereAsrConformerLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel_size: int):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = CohereAsrConformerFeedForward(d_model, d_ff)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = CohereAsrRelPositionAttention(n_heads, d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = CohereAsrConformerConvolution(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = CohereAsrConformerFeedForward(d_model, d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class CohereAsrConformerEncoder(nn.Module):
    def __init__(self, config: CohereAsrConfig):
        super().__init__()
        encoder_config = config.encoder
        d_model = encoder_config["d_model"]
        self.pre_encode = CohereAsrConvSubsampling(encoder_config)
        self.pos_enc = CohereAsrRelPositionalEncoding(
            d_model, encoder_config["pos_emb_max_len"]
        )
        self.layers = nn.ModuleList(
            [
                CohereAsrConformerLayer(
                    d_model=d_model,
                    d_ff=d_model * encoder_config["ff_expansion_factor"],
                    n_heads=encoder_config["n_heads"],
                    conv_kernel_size=encoder_config["conv_kernel_size"],
                )
                for _ in range(encoder_config["n_layers"])
            ]
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode one clip, [1, mel_bins, time] -> [1, frames, d_model].

        Single-clip only: with no batch padding every attention/convolution mask
        in the reference degenerates to all-ones, so none is applied here.
        """
        hidden_states = self.pre_encode(input_features)
        pos_emb = self.pos_enc(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_emb)
        return hidden_states


class CohereAsrDecoderAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layer_id: int,
        is_cross_attention: bool,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        head_dim = hidden_size // num_heads
        tp_size = get_parallel().tp_size
        assert num_heads % tp_size == 0, (
            f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
        )
        self.num_heads = num_heads // tp_size
        self.kv_size = self.num_heads * head_dim
        self.is_cross_attention = is_cross_attention

        if is_cross_attention:
            self.q_proj = ColumnParallelLinear(
                hidden_size, hidden_size, quant_config=quant_config
            )
            self.kv_proj = QKVParallelLinear(
                hidden_size=hidden_size,
                head_size=head_dim,
                total_num_heads=0,
                total_num_kv_heads=num_heads,
                quant_config=quant_config,
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size, head_dim, num_heads, quant_config=quant_config
            )
        self.out_projection = RowParallelLinear(
            hidden_size, hidden_size, quant_config=quant_config
        )
        self.attn = RadixAttention(
            self.num_heads,
            head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            is_cross_attention=is_cross_attention,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_cross_attention:
            q, _ = self.q_proj(hidden_states)
            # None outside prefill: the encoder KV is already in the pool.
            if encoder_hidden_states is None:
                k = v = None
            else:
                kv, _ = self.kv_proj(encoder_hidden_states)
                k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            attn_output = self.attn(q, k, v, forward_batch)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
            attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=True)

        attn_output, _ = self.out_projection(attn_output)
        return attn_output


class CohereAsrDecoderFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.dense_in = ColumnParallelLinear(
            hidden_size, inner_size, quant_config=quant_config
        )
        # Resolved through ACT2FN, as the reference does: the checkpoint declares
        # "relu", which SGLang's own activation registry does not carry.
        self.activation = ACT2FN[hidden_act.lower().replace("swish", "silu")]
        self.dense_out = RowParallelLinear(
            inner_size, hidden_size, quant_config=quant_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.dense_in(x)
        x = self.activation(x)
        x, _ = self.dense_out(x)
        return x


class CohereAsrDecoderLayer(nn.Module):
    def __init__(
        self,
        config: CohereAsrConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads

        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = CohereAsrDecoderAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_id=layer_id,
            is_cross_attention=False,
            quant_config=quant_config,
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = CohereAsrDecoderAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_id=config.decoder_layers + layer_id,
            is_cross_attention=True,
            quant_config=quant_config,
        )
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = CohereAsrDecoderFeedForward(
            hidden_size=hidden_size,
            inner_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.first_sub_layer(
            self.layer_norm_1(hidden_states), forward_batch
        )
        hidden_states = hidden_states + self.second_sub_layer(
            self.layer_norm_2(hidden_states), forward_batch, encoder_hidden_states
        )
        hidden_states = hidden_states + self.third_sub_layer(
            self.layer_norm_3(hidden_states)
        )
        return hidden_states


class CohereAsrFixedPositionalEncoding(nn.Module):
    """Sinusoidal decoder positions; the table ships in the checkpoint."""

    def __init__(self, hidden_size: int, max_sequence_length: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.register_buffer(
            "pos_enc", torch.zeros(max_sequence_length, hidden_size), persistent=True
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.clamp(max=self.max_sequence_length - 1)
        return torch.index_select(self.pos_enc, 0, positions)


class CohereAsrDecoderEmbedding(nn.Module):
    def __init__(self, config: CohereAsrConfig):
        super().__init__()
        # The reference passes padding_idx=2 here, which only zeroes that row at
        # init; the checkpoint's own (non-zero) row 2 overwrites it, so carrying
        # the index through would change nothing at inference.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = CohereAsrFixedPositionalEncoding(
            config.hidden_size, config.max_target_positions
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        embeds = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.layer_norm(embeds)


class CohereAsrDecoderCore(nn.Module):
    def __init__(
        self,
        config: CohereAsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CohereAsrDecoderLayer(config, layer_id, quant_config)
                for layer_id in range(config.decoder_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, forward_batch)
        return self.final_layer_norm(hidden_states)


class CohereAsrDecoder(nn.Module):
    def __init__(
        self,
        config: CohereAsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self._embedding = CohereAsrDecoderEmbedding(config)
        self._decoder = CohereAsrDecoderCore(config, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self._embedding(input_ids, positions)
        return self._decoder(hidden_states, encoder_hidden_states, forward_batch)


class CohereAsrTokenClassifierHead(nn.Module):
    """Output head, named to match the checkpoint's ``log_softmax.mlp.layer0``.

    The reference applies ``log_softmax`` here. That is a per-row constant shift,
    so it leaves sampling, top-k/p and returned logprobs unchanged; the head is
    left as raw logits and `LogitsProcessor` normalizes once.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer0 = ParallelLMHead(
            num_classes, hidden_size, bias=True, quant_config=quant_config
        )


class CohereAsrForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: CohereAsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = CohereAsrConformerEncoder(config)
        self.transf_decoder = CohereAsrDecoder(config, quant_config)
        encoder_dim = config.encoder["d_model"]
        self.encoder_decoder_proj = (
            nn.Linear(encoder_dim, config.hidden_size)
            if encoder_dim != config.hidden_size
            else None
        )
        self.log_softmax = CohereAsrTokenClassifierHead(
            hidden_size=config.head["hidden_size"],
            num_classes=config.vocab_size,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    def pad_input_ids(
        self, input_ids: array[int], mm_inputs: MultimodalInputs
    ) -> array[int]:
        # Prepend placeholder encoder tokens so prepare_encoder_info_extend
        # reserves that many cross-attention KV slots. They are stripped again
        # before the model sees input_ids.
        mel_frames = mm_inputs.mm_items[0].feature.shape[-1]
        encoder_len = cohere_asr_encoder_seq_len(mel_frames)
        mm_inputs.num_image_tokens = encoder_len
        return array("q", [0]) * encoder_len + input_ids

    def _encode_audio(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Flattened encoder output for the requests still missing cached KV.

        Concatenation follows request order, matching how
        prepare_encoder_info_extend lays out `encoder_out_cache_loc`.
        """
        mm_inputs_list = forward_batch.mm_inputs or []
        encoder_cached = forward_batch.encoder_cached or []
        dtype = self.encoder.pre_encode.conv[0].weight.dtype
        device = self.encoder.pre_encode.conv[0].weight.device

        encoded = []
        for mm_input, cached in zip(mm_inputs_list, encoder_cached):
            if cached or mm_input is None or not mm_input.mm_items:
                continue
            features = mm_input.mm_items[0].feature
            if features.ndim == 2:
                features = features.unsqueeze(0)
            features = features.to(device=device, dtype=dtype)
            hidden_states = self.encoder(features)
            if self.encoder_decoder_proj is not None:
                hidden_states = self.encoder_decoder_proj(hidden_states)
            # Trim to the reserved slot count; the integer formula and the
            # strided convolutions can only disagree if one of them changes.
            encoder_len = cohere_asr_encoder_seq_len(features.shape[-1])
            encoded.append(hidden_states[0, :encoder_len])

        if not encoded:
            return None
        return torch.cat(encoded, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        encoder_hidden_states = None
        if not forward_batch.forward_mode.is_decode():
            encoder_hidden_states = self._encode_audio(forward_batch)

        hidden_states = self.transf_decoder(
            input_ids, positions, encoder_hidden_states, forward_batch
        )
        return self.logits_processor(
            input_ids=input_ids,
            lm_head=self.log_softmax.mlp.layer0,
            hidden_states=hidden_states,
            logits_metadata=forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".first_sub_layer.qkv_proj", ".first_sub_layer.query_net", "q"),
            (".first_sub_layer.qkv_proj", ".first_sub_layer.key_net", "k"),
            (".first_sub_layer.qkv_proj", ".first_sub_layer.value_net", "v"),
            (".second_sub_layer.kv_proj", ".second_sub_layer.key_net", "k"),
            (".second_sub_layer.kv_proj", ".second_sub_layer.value_net", "v"),
        ]
        # BatchNorm running stats and the decoder position table are buffers, so
        # named_parameters() alone would silently skip them.
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))

        for name, loaded_weight in weights:
            if name.startswith(_PREPROCESSOR_WEIGHT_PREFIX):
                continue
            name = name.replace(
                ".second_sub_layer.query_net", ".second_sub_layer.q_proj"
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [CohereAsrForConditionalGeneration]
