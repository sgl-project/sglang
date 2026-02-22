from typing import Any, Iterable, List, Optional, Tuple

import torch
from transformers import WhisperConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
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
        self.total_num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.is_cross_attention = is_cross_attention
        self.is_encoder = is_encoder

        tp_size = get_tensor_model_parallel_world_size()
        assert (
            num_heads % tp_size == 0
        ), f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
        self.num_heads = num_heads // tp_size

        if (head_dim * num_heads) != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = head_dim**-0.5
        self.head_dim = head_dim
        self.kv_size = self.num_heads * head_dim

        if is_cross_attention:
            self.q_proj = ColumnParallelLinear(
                embed_dim, embed_dim, quant_config=quant_config
            )
            self.kv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=head_dim,
                total_num_heads=0,
                total_num_kv_heads=num_heads,
                bias=bias,
                quant_config=quant_config,
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                embed_dim, head_dim, num_heads, quant_config=quant_config
            )
        self.out_proj = RowParallelLinear(
            embed_dim, embed_dim, bias=bias, quant_config=quant_config
        )
        self.attn = RadixAttention(
            self.num_heads,
            head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads,
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
            if cross_hidden_states is not None:
                kv, _ = self.kv_proj(cross_hidden_states)
                k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            else:
                k = torch.zeros_like(q)
                v = torch.zeros_like(q)

            q = q * self.scaling
            num_heads = self.attn.tp_q_head_num
            head_dim = self.attn.head_dim

            q = q.view(-1, num_heads, head_dim)
            k = k.view(-1, num_heads, head_dim)
            v = v.view(-1, num_heads, head_dim)

            q_len = q.shape[0]
            kv_len = k.shape[0]

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

            attn_weights = torch.bmm(q, k.transpose(1, 2))

            # Apply block-diagonal mask for batched cross-attention
            batch_size = forward_batch.batch_size if forward_batch else 1
            if batch_size > 1 and kv_len > 0:
                encoder_len_per_request = kv_len // batch_size
                if encoder_len_per_request * batch_size == kv_len:
                    is_decode = forward_batch.forward_mode.is_decode()
                    if is_decode:
                        mask = torch.zeros(
                            (q_len, kv_len), device=q.device, dtype=torch.bool
                        )
                        for i in range(batch_size):
                            enc_start = i * encoder_len_per_request
                            enc_end = (i + 1) * encoder_len_per_request
                            mask[i, enc_start:enc_end] = True
                        attn_weights = attn_weights.masked_fill(
                            ~mask.unsqueeze(0), float("-inf")
                        )
                    else:
                        seq_lens = forward_batch.seq_lens
                        if seq_lens is not None and len(seq_lens) == batch_size:
                            seq_lens_list = seq_lens.tolist()
                            mask = torch.zeros(
                                (q_len, kv_len), device=q.device, dtype=torch.bool
                            )
                            q_start = 0
                            for i, dec_len in enumerate(seq_lens_list):
                                enc_start = i * encoder_len_per_request
                                enc_end = (i + 1) * encoder_len_per_request
                                q_end = q_start + dec_len
                                mask[q_start:q_end, enc_start:enc_end] = True
                                q_start = q_end
                            attn_weights = attn_weights.masked_fill(
                                ~mask.unsqueeze(0), float("-inf")
                            )

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.bmm(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(q_len, num_heads * head_dim)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
            q = q * self.scaling

            if self.is_encoder:
                num_heads = self.attn.tp_q_head_num
                head_dim = self.attn.head_dim
                batch_size, seq_len, _ = hidden_states.shape

                q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                v = v.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=1.0
                )
                attn_output = attn_output.permute(0, 2, 1, 3).reshape(
                    batch_size, seq_len, num_heads * head_dim
                )
            else:
                attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=True)

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
            num_heads=config.encoder_attention_heads,
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

        # Offset decoder layer IDs to avoid overlap with encoder layers
        decoder_self_attn_layer_id = config.encoder_layers + layer_id
        decoder_cross_attn_layer_id = (
            config.encoder_layers + config.decoder_layers + layer_id
        )

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=decoder_self_attn_layer_id,
            quant_config=quant_config,
        )

        self.activation_fn = get_act_fn(
            config.activation_function, quant_config=quant_config
        )

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            layer_id=decoder_cross_attn_layer_id,
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
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        residual = decoder_hidden_states
        decoder_hidden_states = self.self_attn_layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.self_attn(decoder_hidden_states, forward_batch)
        decoder_hidden_states = residual + decoder_hidden_states

        residual = decoder_hidden_states
        decoder_hidden_states = self.encoder_attn_layer_norm(decoder_hidden_states)
        decoder_hidden_states = self.encoder_attn(
            decoder_hidden_states, forward_batch, encoder_hidden_states
        )
        decoder_hidden_states = residual + decoder_hidden_states

        residual = decoder_hidden_states
        decoder_hidden_states = self.final_layer_norm(decoder_hidden_states)
        decoder_hidden_states, _ = self.fc1(decoder_hidden_states)
        decoder_hidden_states = self.activation_fn(decoder_hidden_states)
        decoder_hidden_states, _ = self.fc2(decoder_hidden_states)

        decoder_hidden_states = residual + decoder_hidden_states

        return decoder_hidden_states


class WhisperEncoder(torch.nn.Module):

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
        self.embed_positions = torch.nn.Embedding(
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
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.mT

        hidden_states = inputs_embeds + self.embed_positions(position_ids)

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

        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.embed_positions = torch.nn.Embedding(
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
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        position_ids=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
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
        self._encoder_cache = {}

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        weights_dict = dict(weights)

        # Whisper has no k_proj bias, create zeros
        for layer_idx in range(self.config.decoder_layers):
            layer_prefix = f"model.decoder.layers.{layer_idx}.encoder_attn."
            k_proj_key = layer_prefix + "k_proj.weight"
            if k_proj_key in weights_dict:
                k_proj_weight = weights_dict[k_proj_key]
                bias_key = layer_prefix + "k_proj.bias"
                if bias_key not in weights_dict:
                    weights_dict[bias_key] = torch.zeros(k_proj_weight.size(0))

        weights_dict["proj_out.weight"] = weights_dict[
            "model.decoder.embed_tokens.weight"
        ]

        for name, loaded_weight in weights_dict.items():
            name = name.replace("model.", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def pad_input_ids(self, input_ids: List[int], _mm_inputs: MultimodalInputs):
        return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        dtype = self.encoder.conv1.weight.dtype
        is_decode = forward_batch.forward_mode.is_decode()

        if is_decode:
            encoder_outputs = None
            if forward_batch.req_pool_indices is not None:
                req_indices = forward_batch.req_pool_indices.tolist()
                encoder_list = []
                for req_idx in req_indices:
                    if req_idx in self._encoder_cache:
                        encoder_list.append(self._encoder_cache[req_idx])
                if encoder_list:
                    encoder_outputs = torch.cat(encoder_list, dim=0)
        else:
            encoder_list = []
            mm_inputs_list = forward_batch.mm_inputs if forward_batch.mm_inputs else []
            req_indices = (
                forward_batch.req_pool_indices.tolist()
                if forward_batch.req_pool_indices is not None
                else []
            )

            for req_idx, mm_input in zip(req_indices, mm_inputs_list):
                if mm_input is None or not mm_input.mm_items:
                    continue

                features = mm_input.mm_items[0].feature
                if features.ndim == 2:
                    features = features.unsqueeze(0)

                encoder_len = features.shape[-1] // 2
                encoder_position_ids = torch.arange(encoder_len).to(
                    features.device, non_blocking=True
                )

                req_encoder_outputs = self.encoder(
                    features.to(dtype), encoder_position_ids, forward_batch
                )
                req_encoder_outputs = req_encoder_outputs.squeeze(0)

                self._encoder_cache[req_idx] = req_encoder_outputs
                encoder_list.append(req_encoder_outputs)

            if encoder_list:
                encoder_outputs = torch.cat(encoder_list, dim=0)
            else:
                encoder_outputs = None

        decoder_outputs = self.decoder(
            input_ids, encoder_outputs, forward_batch, positions
        )

        logits = self.logits_processor(
            input_ids=input_ids,
            lm_head=self.proj_out,
            hidden_states=decoder_outputs,
            logits_metadata=forward_batch,
        )

        return logits


EntryClass = [WhisperForConditionalGeneration]
