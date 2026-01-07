"""
LFM2 (Liquid Foundation Model 2) implementation for SGLang.

This is a hybrid architecture with both attention and short conv layers.
- Attention layers use standard KV cache (RadixAttention)
- Conv layers use MambaPool for state caching (via HybridReqToTokenPool)

The model uses a gated 1D causal convolution (kernel=3) instead of attention
in some layers, providing linear memory complexity for those layers.
"""

import logging
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.lfm2 import Lfm2Config
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
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


class Lfm2RMSNorm(nn.Module):
    """LFM2-specific RMSNorm: weight * x (not (1 + weight) * x like Gemma)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Lfm2MLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        intermediate_size = config.intermediate_size

        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(
                    config.block_ffn_dim_multiplier * intermediate_size
                )
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1)
                    // config.block_multiple_of
                )

        self.w1 = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w1", prefix),
        )
        self.w3 = ColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w3", prefix),
        )
        self.w2 = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.w1(x)
        up, _ = self.w3(x)
        out, _ = self.w2(F.silu(gate) * up)
        return out


class Lfm2Attention(nn.Module):
    """Grouped-query attention with RoPE and Q/K layernorm."""

    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        attn_layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.total_num_heads
        )
        self.scaling = self.head_dim**-0.5

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        else:
            rope_theta = getattr(config, "rope_theta", 10000)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_scaling=getattr(config, "rope_scaling", None),
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        self.q_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)

        self.num_local_q_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads

        self.attn = RadixAttention(
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.reshape(T, self.num_local_q_heads, self.head_dim)
        k = k.reshape(T, self.num_local_kv_heads, self.head_dim)

        q = self.q_layernorm(q.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_q_heads, self.head_dim
        )
        k = self.k_layernorm(k.reshape(-1, self.head_dim)).reshape(
            T, self.num_local_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch)
        out, _ = self.out_proj(attn_out)
        return out


class Lfm2ShortConv(nn.Module):
    """
    Gated short convolution layer using SGLang's MambaPool for state management.

    Architecture: in_proj -> split(B, C, x) -> Bx -> conv1d -> C*conv_out -> out_proj
    - Uses double gating: B (before conv) and C (after conv)
    - Fixed-size cache: stores last (kernel_size - 1) tokens
    """

    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel = int(config.conv_L_cache)
        self.L_cache = self.conv_kernel - 1
        self.bias = bool(config.conv_bias)
        self.hidden_size = config.hidden_size

        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=self.conv_kernel,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        layer_cache = forward_batch.req_to_token_pool.mamba2_layer_cache(self.layer_idx)
        conv_state = layer_cache.conv[0]
        req_pool_indices = forward_batch.req_pool_indices

        if forward_batch.forward_mode.is_decode():
            return self._forward_decode(hidden_states, conv_state, req_pool_indices)

        seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if seq_lens is not None and len(seq_lens) > 1:
            return self._forward_prefill_multi(
                hidden_states, conv_state, req_pool_indices, seq_lens
            )
        return self._forward_prefill_single(hidden_states, conv_state, req_pool_indices)

    def _forward_prefill_single(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]

        proj = self.in_proj(hidden_states)
        proj_t = proj.transpose(0, 1).unsqueeze(0)
        B_gate, C_gate, x = proj_t.chunk(3, dim=1)
        Bx = B_gate * x
        conv_out = self.conv(Bx)[..., :T]
        y = C_gate * conv_out
        y = self.out_proj(y.squeeze(0).transpose(0, 1))

        # Store final conv state
        if T >= self.L_cache:
            final_state = Bx[0, :, -self.L_cache :]
        else:
            final_state = F.pad(Bx[0], (self.L_cache - T, 0), value=0.0)

        if req_pool_indices.numel() > 0:
            conv_state.index_copy_(
                0,
                req_pool_indices[:1].long(),
                final_state.unsqueeze(0).to(conv_state.dtype),
            )

        return y

    def _forward_prefill_multi(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        start_idx = 0
        seq_lens_list = (
            seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else list(seq_lens)
        )
        req_pool_indices_long = req_pool_indices.long()

        for i, seq_len in enumerate(seq_lens_list):
            seq_len = int(seq_len)
            end_idx = start_idx + seq_len
            seq_hidden = hidden_states[start_idx:end_idx]
            T = seq_hidden.shape[0]

            proj = self.in_proj(seq_hidden)
            proj_t = proj.transpose(0, 1).unsqueeze(0)
            B_gate, C_gate, x = proj_t.chunk(3, dim=1)
            Bx = B_gate * x
            conv_out = self.conv(Bx)[..., :T]
            y = C_gate * conv_out
            y = self.out_proj(y.squeeze(0).transpose(0, 1))
            outputs.append(y)

            if T >= self.L_cache:
                final_state = Bx[0, :, -self.L_cache :]
            else:
                final_state = F.pad(Bx[0], (self.L_cache - T, 0), value=0.0)

            conv_state.index_copy_(
                0,
                req_pool_indices_long[i : i + 1],
                final_state.unsqueeze(0).to(conv_state.dtype),
            )
            start_idx = end_idx

        return torch.cat(outputs, dim=0)

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        req_pool_indices_long = req_pool_indices.long()

        proj = self.in_proj(hidden_states)
        B_gate, C_gate, x = proj.chunk(3, dim=-1)
        Bx = B_gate * x

        conv_weights = self.conv.weight[:, 0, :]
        current_states = conv_state[req_pool_indices_long]

        # Update state: roll left, insert new value at end
        new_states = torch.cat(
            [current_states[:, :, 1:], Bx.unsqueeze(-1)], dim=-1
        )
        conv_state.index_copy_(
            0, req_pool_indices_long, new_states.to(conv_state.dtype)
        )

        # Apply conv: use last kernel_size values
        conv_input = torch.cat(
            [current_states[:, :, -(self.conv_kernel - 1) :], Bx.unsqueeze(-1)], dim=-1
        )
        conv_out = (conv_input * conv_weights.unsqueeze(0)).sum(dim=-1)

        if self.bias and self.conv.bias is not None:
            conv_out = conv_out + self.conv.bias

        y = C_gate * conv_out
        return self.out_proj(y.to(hidden_states.dtype))


class Lfm2DecoderLayer(nn.Module):
    """Decoder layer - either attention or conv based on config."""

    def __init__(
        self,
        config: Lfm2Config,
        layer_id: int,
        attn_layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type == "full_attention"

        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(
                config=config,
                layer_id=layer_id,
                attn_layer_id=attn_layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.conv = Lfm2ShortConv(
                config=config,
                layer_idx=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )

        self.feed_forward = Lfm2MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward(
        self,
        layer_id: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not forward_batch.forward_mode.is_idle():
            residual = hidden_states
            normed = self.operator_norm(hidden_states)

            if self.is_attention_layer:
                hidden_states = self.self_attn(positions, normed, forward_batch)
            else:
                hidden_states = self.conv(normed, forward_batch)

            hidden_states = hidden_states + residual
            hidden_states = hidden_states + self.feed_forward(
                self.ffn_norm(hidden_states)
            )

        return hidden_states, residual


class Lfm2Model(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        # Compute attention layer IDs for KV cache
        attn_layer_ids = []
        attn_count = 0
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                attn_layer_ids.append(attn_count)
                attn_count += 1
            else:
                attn_layer_ids.append(-1)

        self.num_attention_layers = attn_count

        def get_layer(idx: int, prefix: str, **kwargs):
            return Lfm2DecoderLayer(
                config=config,
                layer_id=idx,
                attn_layer_id=attn_layer_ids[idx],
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        )

        residual = None
        for i in range(len(self.layers)):
            hidden_states, residual = self.layers[i](
                layer_id=i,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        return self.embedding_norm(hidden_states)


class Lfm2ForCausalLM(nn.Module):
    """LFM2 for causal language modeling with hybrid attention/conv architecture."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Lfm2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank

        self.quant_config = quant_config
        self.model = Lfm2Model(config, quant_config, prefix=add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.num_attention_layers = self.model.num_attention_layers

    def get_num_kv_cache_layers(self) -> int:
        return self.num_attention_layers

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        embed_tokens_weight = None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "embed_tokens.weight" in name:
                embed_tokens_weight = loaded_weight

            # Handle QKV stacking
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Handle tied lm_head weight
        if "lm_head.weight" not in loaded_params and "lm_head.weight" in params_dict:
            if embed_tokens_weight is not None:
                param = params_dict["lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, embed_tokens_weight)
                loaded_params.add("lm_head.weight")

        return loaded_params


EntryClass = [Lfm2ForCausalLM]
