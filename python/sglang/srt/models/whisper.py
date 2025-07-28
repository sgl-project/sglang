from typing import Any, Iterable, List, Optional, Tuple

import torch
from transformers import WhisperConfig

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader


class WhisperAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        layer_id: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        is_cross_attention: bool = False,
        is_encoder=False,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.is_cross_attention = is_cross_attention
        self.is_encoder = is_encoder

        if (head_dim * num_heads) != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = head_dim**-0.5

        if is_cross_attention:
            self.q_proj = ColumnParallelLinear(
                embed_dim, embed_dim, quant_config=quant_config
            )
            self.kv_proj = ColumnParallelLinear(
                embed_dim, 2 * embed_dim, quant_config=quant_config
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                embed_dim, head_dim, num_heads, quant_config=quant_config
            )
        self.out_proj = RowParallelLinear(
            embed_dim, embed_dim, bias=bias, quant_config=quant_config
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            scaling=1.0,
            num_kv_heads=num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            is_cross_attention=is_cross_attention,
            attn_type=(
                AttentionType.ENCODER_ONLY if is_encoder else AttentionType.DECODER
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        cross_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        if self.is_cross_attention:
            q, _ = self.q_proj(hidden_states)
            kv, _ = self.kv_proj(cross_hidden_states)
            k, v = kv.chunk(chunks=2, dim=-1)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        q = q * self.scaling

        attn_output = self.attn(
            q, k, v, forward_batch, save_kv_cache=not self.is_encoder
        )

        attn_output, _ = self.out_proj(attn_output)

        return attn_output


class WhisperEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_id: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            is_encoder=True,
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)

        self.activation_fn = get_act_fn(
            config.activation_function, quant_config=quant_config
        )

        self.fc1 = ColumnParallelLinear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = RowParallelLinear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, forward_batch)

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


class WhisperDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        layer_id: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=layer_id,
            quant_config=quant_config,
        )

        self.activation_fn = get_act_fn(
            config.activation_function, quant_config=quant_config
        )

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.fc1 = ColumnParallelLinear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = RowParallelLinear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        residual = decoder_hidden_states
        decoder_hidden_states = self.self_attn_layer_norm(decoder_hidden_states)

        # Self Attention
        decoder_hidden_states = self.self_attn(decoder_hidden_states, forward_batch)
        decoder_hidden_states = residual + decoder_hidden_states

        # Cross-Attention Block
        residual = decoder_hidden_states
        decoder_hidden_states = self.encoder_attn_layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.encoder_attn(
            decoder_hidden_states, forward_batch, encoder_hidden_states
        )

        decoder_hidden_states = residual + decoder_hidden_states

        # Fully Connected
        residual = decoder_hidden_states
        decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)
        decoder_hidden_states, _ = self.fc1(decoder_hidden_states)
        decoder_hidden_states = self.activation_fn(decoder_hidden_states)
        decoder_hidden_states, _ = self.fc2(decoder_hidden_states)

        decoder_hidden_states = residual + decoder_hidden_states

        return decoder_hidden_states


class WhisperEncoder(torch.nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(
        self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()

        embed_dim = config.d_model
        self.embed_scale = embed_dim**-0.5 if config.scale_embedding else 1.0

        self.conv1 = torch.nn.Conv1d(
            config.num_mel_bins, embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )

        self.embed_positions = VocabParallelEmbedding(
            config.max_source_positions, embed_dim
        )

        self.layers = torch.nn.ModuleList(
            [
                WhisperEncoderLayer(config, id, quant_config)
                for id in range(config.encoder_layers)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_features: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.mT
        all_positions = torch.arange(
            self.embed_positions.num_embeddings, device=inputs_embeds.device
        )

        hidden_states = inputs_embeds + self.embed_positions(all_positions)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, forward_batch)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(torch.nn.Module):

    def __init__(
        self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = config.d_model**-0.5 if config.scale_embedding else 1.0

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.d_model)
        self.embed_positions = VocabParallelEmbedding(
            self.max_target_positions, config.d_model
        )

        self.layers = torch.nn.ModuleList(
            [
                WhisperDecoderLayer(config, layer_idx, quant_config)
                for layer_idx in range(config.decoder_layers)
            ]
        )

        self.layer_norm = torch.nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        position_ids=None,
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        positions = self.embed_positions(position_ids)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, encoder_hidden_states, forward_batch
            )

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class WhisperForConditionalGeneration(torch.nn.Module):

    def __init__(
        self, config: WhisperConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.encoder = WhisperEncoder(config, quant_config)
        self.decoder = WhisperDecoder(config, quant_config)
        self.proj_out = ParallelLMHead(
            config.vocab_size, config.d_model, quant_config=quant_config
        )
        self.logits_processor = LogitsProcessor(config)
        self.config = config

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        weights_dict = dict(weights)
        for layer_idx in range(self.config.decoder_layers):
            layer_prefix = f"model.decoder.layers.{layer_idx}.encoder_attn."

            v_proj_weight = weights_dict[layer_prefix + "v_proj.weight"]
            v_proj_bias = weights_dict[layer_prefix + "v_proj.bias"]

            k_proj_weight = weights_dict[layer_prefix + "k_proj.weight"]

            del (
                weights_dict[layer_prefix + "v_proj.weight"],
                weights_dict[layer_prefix + "v_proj.bias"],
                weights_dict[layer_prefix + "k_proj.weight"],
            )
            k_proj_bias = torch.zeros_like(v_proj_bias)

            weights_dict[f"decoder.layers.{layer_idx}.encoder_attn.kv_proj.weight"] = (
                torch.cat([k_proj_weight, v_proj_weight], dim=0)
            )
            weights_dict[f"decoder.layers.{layer_idx}.encoder_attn.kv_proj.bias"] = (
                torch.cat([k_proj_bias, v_proj_bias], dim=0)
            )

        for name, loaded_weight in weights_dict.items():
            name = name.replace("model.", "")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        # Whisper models handle text/audio separately, so we don't need to pad input ids
        return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:

        mm_inputs = forward_batch.merge_mm_inputs()
        assert mm_inputs is not None
        features = mm_inputs.mm_items[0].feature
        dtype = self.encoder.conv1.weight.dtype

        encoder_outputs = self.encoder(features.to(dtype), forward_batch)
        decoder_outputs = self.decoder(
            input_ids, encoder_outputs, forward_batch, positions
        )

        logits = self.logits_processor(
            input_ids=input_ids,
            lm_head=self.proj_out,
            hidden_states=decoder_outputs[:, -1, :],
            logits_metadata=forward_batch,
        )

        return logits


EntryClass = [WhisperForConditionalGeneration]
