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
            # For cross-attention: compute K,V from encoder outputs
            # cross_hidden_states should always be provided (from encoder cache)
            if cross_hidden_states is not None:
                kv, _ = self.kv_proj(cross_hidden_states)
                k, v = kv.chunk(chunks=2, dim=-1)
            else:
                # Fallback for warmup or missing encoder outputs
                k = torch.zeros_like(q)
                v = torch.zeros_like(q)

            # Apply scaling to Q
            q = q * self.scaling

            # For cross-attention, use direct scaled dot product attention
            # because the attention backend's decode path reads from KV cache
            # which doesn't have our encoder K,V
            num_heads = self.attn.tp_q_head_num
            head_dim = self.attn.head_dim

            q = q.view(-1, num_heads, head_dim)
            k = k.view(-1, num_heads, head_dim)
            v = v.view(-1, num_heads, head_dim)

            q_len = q.shape[0]
            kv_len = k.shape[0]

            # Compute attention: Q @ K^T -> softmax -> @ V
            # Q: (q_len, num_heads, head_dim) -> (num_heads, q_len, head_dim)
            # K: (kv_len, num_heads, head_dim) -> (num_heads, kv_len, head_dim)
            # V: (kv_len, num_heads, head_dim) -> (num_heads, kv_len, head_dim)
            q = q.transpose(0, 1)  # (num_heads, q_len, head_dim)
            k = k.transpose(0, 1)  # (num_heads, kv_len, head_dim)
            v = v.transpose(0, 1)  # (num_heads, kv_len, head_dim)

            # Compute attention weights: (num_heads, q_len, head_dim) @ (num_heads, head_dim, kv_len)
            attn_weights = torch.bmm(q, k.transpose(1, 2))  # (num_heads, q_len, kv_len)

            # Apply cross-attention mask for batched requests
            # When multiple requests are batched, each decoder should only attend
            # to its own request's encoder outputs, not other requests' encoders
            batch_size = forward_batch.batch_size if forward_batch else 1
            if batch_size > 1 and kv_len > 0:
                # Each request has equal encoder output length (1500 for Whisper 30s audio)
                encoder_len_per_request = kv_len // batch_size
                if encoder_len_per_request * batch_size == kv_len:
                    # Build block-diagonal mask
                    # In decode mode: q_len = batch_size (1 token per request)
                    # In prefill mode: q_len = sum of prefill lengths
                    is_decode = forward_batch.forward_mode.is_decode()
                    if is_decode:
                        # Decode: each request contributes 1 decoder token
                        # q[i] should attend to k[i*enc_len:(i+1)*enc_len]
                        mask = torch.zeros((q_len, kv_len), device=q.device, dtype=torch.bool)
                        for i in range(batch_size):
                            enc_start = i * encoder_len_per_request
                            enc_end = (i + 1) * encoder_len_per_request
                            mask[i, enc_start:enc_end] = True
                        # Apply mask: positions with False get -inf
                        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(0), float('-inf'))
                    else:
                        # Prefill: need seq_lens to determine decoder token boundaries
                        seq_lens = forward_batch.seq_lens
                        if seq_lens is not None and len(seq_lens) == batch_size:
                            seq_lens_list = seq_lens.tolist()
                            mask = torch.zeros((q_len, kv_len), device=q.device, dtype=torch.bool)
                            q_start = 0
                            for i, dec_len in enumerate(seq_lens_list):
                                enc_start = i * encoder_len_per_request
                                enc_end = (i + 1) * encoder_len_per_request
                                q_end = q_start + dec_len
                                mask[q_start:q_end, enc_start:enc_end] = True
                                q_start = q_end
                            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(0), float('-inf'))

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

            # Compute output: (num_heads, q_len, kv_len) @ (num_heads, kv_len, head_dim)
            attn_output = torch.bmm(attn_weights, v)  # (num_heads, q_len, head_dim)

            # Transpose back: (num_heads, q_len, head_dim) -> (q_len, num_heads, head_dim)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(-1, num_heads * head_dim)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.chunk(chunks=3, dim=-1)
            # Apply scaling to Q for self-attention
            q = q * self.scaling

            if self.is_encoder:
                # For encoder: use direct attention (no KV cache, no RadixAttention)
                # Encoder hidden_states may have batch dim: [batch, seq_len, embed_dim]
                num_heads = self.attn.tp_q_head_num
                head_dim = self.attn.head_dim

                has_batch_dim = hidden_states.ndim == 3
                if has_batch_dim:
                    batch_size, seq_len, _ = hidden_states.shape
                    # Reshape: [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
                    q = q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
                    k = k.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                    v = v.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

                    # Use PyTorch's scaled_dot_product_attention for efficiency
                    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)  # scaling already applied to q

                    # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
                    attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)
                else:
                    seq_len = hidden_states.shape[0]
                    q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
                    k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
                    v = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

                    attn_weights = torch.bmm(q, k.transpose(1, 2))
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
                    attn_output = torch.bmm(attn_weights, v)

                    attn_output = attn_output.transpose(0, 1).reshape(seq_len, num_heads * head_dim)
            else:
                # For decoder self-attention: use RadixAttention with KV cache
                attn_output = self.attn(
                    q, k, v, forward_batch, save_kv_cache=True
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

        # Offset decoder layer IDs to avoid overlap with encoder layers
        # Encoder uses layer IDs 0 to encoder_layers-1
        # Decoder self-attention uses encoder_layers to encoder_layers + decoder_layers - 1
        # Decoder cross-attention uses encoder_layers + decoder_layers to encoder_layers + 2*decoder_layers - 1
        decoder_self_attn_layer_id = config.encoder_layers + layer_id
        decoder_cross_attn_layer_id = config.encoder_layers + config.decoder_layers + layer_id

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
        encoder_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        position_ids=None,
    ):

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
        # Cache encoder outputs per request for use during decode
        # Key: request_id, Value: encoder_outputs tensor
        self._encoder_cache = {}

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
        weights_dict["proj_out.weight"] = weights_dict[
            "model.decoder.embed_tokens.weight"
        ]

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

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # For Whisper, we manage encoder outputs at the model level (_encoder_cache)
        # rather than using the attention backend's encoder cache mechanism.
        # Therefore, we don't need to prepend placeholder tokens or set num_image_tokens.
        return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        dtype = self.encoder.conv1.weight.dtype
        device = input_ids.device

        # Check if we're in decode mode - use forward_mode instead of mm_inputs
        # because mm_inputs can persist across prefill and decode
        is_decode = forward_batch.forward_mode.is_decode()

        if get_is_capture_mode():
            # During CUDA graph capture, create dummy encoder outputs
            total_encoder_len = forward_batch.batch_size
            encoder_outputs = torch.zeros(
                total_encoder_len, self.config.d_model, dtype=dtype, device=device
            )
        elif is_decode:
            # Decode phase: retrieve cached encoder outputs for all requests in batch
            # Each request needs its own encoder outputs for cross-attention
            encoder_outputs = None
            if forward_batch.req_pool_indices is not None:
                req_indices = forward_batch.req_pool_indices.tolist()
                encoder_list = []
                for req_idx in req_indices:
                    if req_idx in self._encoder_cache:
                        encoder_list.append(self._encoder_cache[req_idx])

                if encoder_list:
                    if len(encoder_list) == 1:
                        encoder_outputs = encoder_list[0]
                    else:
                        # Concatenate encoder outputs for batched decode
                        # This aligns with how decoder hidden states are concatenated
                        encoder_outputs = torch.cat(encoder_list, dim=0)
        else:
            # Prefill (extend) phase: process each request's audio separately
            encoder_list = []

            # Access per-request multimodal inputs instead of merged
            mm_inputs_list = forward_batch.mm_inputs if forward_batch.mm_inputs else []
            req_indices = forward_batch.req_pool_indices.tolist() if forward_batch.req_pool_indices is not None else []

            # Process each request's audio separately and cache
            for i, (req_idx, mm_input) in enumerate(zip(req_indices, mm_inputs_list)):
                if mm_input is None or not mm_input.mm_items:
                    continue

                features = mm_input.mm_items[0].feature

                # Add batch dimension if needed
                if features.ndim == 2:
                    features = features.unsqueeze(0)

                # Compute encoder output length from features
                encoder_len = features.shape[-1] // 2
                encoder_position_ids = torch.arange(encoder_len).to(
                    features.device, non_blocking=True
                )

                req_encoder_outputs = self.encoder(
                    features.to(dtype), encoder_position_ids, forward_batch
                )

                # Squeeze batch dimension if present
                if req_encoder_outputs.ndim == 3 and req_encoder_outputs.shape[0] == 1:
                    req_encoder_outputs = req_encoder_outputs.squeeze(0)

                # Cache this request's encoder outputs
                self._encoder_cache[req_idx] = req_encoder_outputs
                encoder_list.append(req_encoder_outputs)

            # Concatenate encoder outputs for all requests in batch
            if encoder_list:
                if len(encoder_list) == 1:
                    encoder_outputs = encoder_list[0]
                else:
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
