# SPDX-License-Identifier: Apache-2.0

"""SRT-native BAGEL Qwen2-MoT language model pieces.

This file intentionally keeps the first native BAGEL step narrow: it expresses
the MoT layer shape and runs the understanding branch through normal SRT
ForwardBatch/KV paths. The generation branch is exposed as a narrow
embed-level hook so the UG runtime can route text tokens and VAE latent tokens
without exposing KV cache internals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bagel_visual import BAGELVisualFeatureMixin
from sglang.srt.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_rope_config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BAGELMoTTokenRouting:
    """Token routing for BAGEL `mode="gen"` MoT branches.

    Text tokens stay on the normal Qwen2 branch. VAE latent tokens take the
    BAGEL generation branch. The object carries positions only, never KV slots.
    """

    text_token_indices: torch.Tensor
    vae_token_indices: torch.Tensor

    def to(self, device: torch.device) -> "BAGELMoTTokenRouting":
        return BAGELMoTTokenRouting(
            text_token_indices=self.text_token_indices.to(
                device=device, dtype=torch.long
            ),
            vae_token_indices=self.vae_token_indices.to(
                device=device, dtype=torch.long
            ),
        )

    def validate(self, total_tokens: int) -> None:
        text_indices = self.text_token_indices
        vae_indices = self.vae_token_indices
        if text_indices.dim() != 1 or vae_indices.dim() != 1:
            raise ValueError("BAGEL MoT token routing indices must be 1-D tensors")

        merged = torch.cat([text_indices, vae_indices])
        if merged.numel() != total_tokens:
            raise ValueError(
                "BAGEL MoT token routing must cover each input token exactly once"
            )
        if merged.numel() == 0:
            return
        if merged.min().item() < 0 or merged.max().item() >= total_tokens:
            raise ValueError("BAGEL MoT token routing index is out of range")
        if torch.unique(merged).numel() != merged.numel():
            raise ValueError("BAGEL MoT token routing indices must be disjoint")


class BAGELRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        weight_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        input_dtype = hidden_states.dtype
        if residual is not None:
            hidden_states = hidden_states + residual
            if post_residual_addition is not None:
                hidden_states = hidden_states + post_residual_addition

        residual_out = hidden_states.to(input_dtype) if residual is not None else None
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)
        if residual is None:
            return output
        return output, residual_out


def _normalize_bagel_rope_scaling(rope_scaling):
    if not isinstance(rope_scaling, dict):
        return rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    scaling_keys = set(rope_scaling) - {"rope_theta", "rope_type", "type"}
    if rope_type == "default" and not scaling_keys:
        return None
    return rope_scaling


class BAGELQwen2MLP(nn.Module):
    """BAGEL-compatible Qwen2 MLP without fused gate/up activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for BAGEL Qwen2-MoT."
            )
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class BAGELQwen2MoTAttention(nn.Module):
    """Qwen2 attention with BAGEL MoT branch parameters.

    The default forward is the BAGEL `mode="und"` branch: normal Qwen2 qkv
    projection, QK norm, RoPE, SRT RadixAttention, normal output projection.
    """

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        rope_theta, rope_scaling = get_rope_config(config)
        rope_scaling = _normalize_bagel_rope_scaling(rope_scaling)
        self.rope_scaling = rope_scaling
        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = _make_bagel_rms_norm(config, self.head_dim)
        self.k_norm = _make_bagel_rms_norm(config, self.head_dim)
        self.q_norm_moe_gen = _make_bagel_rms_norm(config, self.head_dim)
        self.k_norm_moe_gen = _make_bagel_rms_norm(config, self.head_dim)
        self.alt_stream = alt_stream
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=getattr(
                config, "dual_chunk_attention_config", None
            ),
        )
        official_inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float)
                / self.head_dim
            )
        )
        self.register_buffer(
            "official_inv_freq",
            official_inv_freq,
            persistent=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.q_proj_moe_gen = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("q_proj_moe_gen", prefix),
        )
        self.k_proj_moe_gen = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("k_proj_moe_gen", prefix),
        )
        self.v_proj_moe_gen = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("v_proj_moe_gen", prefix),
        )
        self.o_proj_moe_gen = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj_moe_gen", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        mode: str = "und",
        routing: Optional[BAGELMoTTokenRouting] = None,
    ) -> torch.Tensor:
        if mode == "gen":
            if routing is None:
                raise ValueError("BAGEL Qwen2-MoT gen forward requires token routing")
            return self.forward_gen(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                routing=routing,
            )
        if mode != "und":
            raise ValueError(f"Unsupported BAGEL Qwen2-MoT mode: {mode}")

        q, k, v = self._project_qkv(
            hidden_states=hidden_states,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
        )
        q, k = self._apply_official_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        if _can_use_bagel_official_non_causal_attention(forward_batch):
            q, k = self._apply_official_rotary_pos_emb(positions, q, k)
            attn_output = self._forward_official_non_causal_attention(
                q=q,
                k=k,
                v=v,
                forward_batch=forward_batch,
            )
        elif _can_use_bagel_official_causal_attention(forward_batch):
            q, k = self._apply_official_rotary_pos_emb(positions, q, k)
            attn_output = self._forward_official_causal_attention(
                q=q,
                k=k,
                v=v,
                forward_batch=forward_batch,
            )
        else:
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def _apply_official_rotary_pos_emb(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope_scaling is not None:
            return self.rotary_emb(positions, q, k)

        q_shape = q.shape
        k_shape = k.shape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)

        inv_freq = self.official_inv_freq.to(device=q.device)
        position_ids = positions.reshape(1, -1).to(device=q.device)
        inv_freq_expanded = inv_freq[None, :, None].float()
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = q.device.type
        if not isinstance(device_type, str) or device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().squeeze(0).to(dtype=q.dtype)
            sin = emb.sin().squeeze(0).to(dtype=q.dtype)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q.reshape(q_shape), k.reshape(k_shape)

    def _forward_official_non_causal_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        q = q.view(-1, self.num_heads, self.head_dim).to(torch.bfloat16)
        k = k.view(-1, self.num_kv_heads, self.head_dim).to(torch.bfloat16)
        v = v.view(-1, self.num_kv_heads, self.head_dim).to(torch.bfloat16)

        if forward_batch.out_cache_loc is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn,
                forward_batch.out_cache_loc,
                k,
                v,
                self.attn.k_scale,
                self.attn.v_scale,
            )

        query_lens = forward_batch.extend_seq_lens.to(torch.int32)
        if int(query_lens.sum().item()) != q.shape[0]:
            raise ValueError(
                "BAGEL official non-causal attention query_lens do not match Q "
                f"tokens: {int(query_lens.sum().item())} vs {q.shape[0]}"
            )

        if _is_bagel_zero_prefix_extend(forward_batch):
            merged_key_states = k
            merged_value_states = v
            key_values_lens = query_lens
        else:
            key_values_lens = forward_batch.seq_lens.to(torch.int32)
            key_buffer, value_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                self.attn.layer_id
            )
            kv_indices = _bagel_full_sequence_kv_indices(
                forward_batch=forward_batch,
                key_values_lens=key_values_lens,
            )
            merged_key_states = key_buffer.index_select(0, kv_indices).to(
                torch.bfloat16
            )
            merged_value_states = value_buffer.index_select(0, kv_indices).to(
                torch.bfloat16
            )

        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(query_lens, dim=0),
            (1, 0),
        ).to(torch.int32)
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0),
            (1, 0),
        ).to(torch.int32)
        max_seqlen = int(forward_batch.extend_seq_lens.max().item())
        attn_output = flash_attn_varlen_func(
            q=q,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=int(key_values_lens.max().item()),
            causal=False,
        )
        return attn_output.reshape(-1, self.q_size)

    def _forward_official_causal_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        q = q.view(-1, self.num_heads, self.head_dim).to(torch.bfloat16)
        k = k.view(-1, self.num_kv_heads, self.head_dim).to(torch.bfloat16)
        v = v.view(-1, self.num_kv_heads, self.head_dim).to(torch.bfloat16)

        if forward_batch.out_cache_loc is None:
            raise ValueError("BAGEL official causal attention requires KV cache locs")
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn,
            forward_batch.out_cache_loc,
            k,
            v,
            self.attn.k_scale,
            self.attn.v_scale,
        )

        query_lens = _bagel_query_lens(forward_batch)
        key_values_lens = forward_batch.seq_lens.to(torch.int32)
        if int(query_lens.sum().item()) != q.shape[0]:
            raise ValueError(
                "BAGEL official causal attention query_lens do not match Q tokens: "
                f"{int(query_lens.sum().item())} vs {q.shape[0]}"
            )

        if _is_bagel_zero_prefix_extend(forward_batch):
            merged_key_states = k
            merged_value_states = v
        else:
            key_buffer, value_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                self.attn.layer_id
            )
            kv_indices = _bagel_full_sequence_kv_indices(
                forward_batch=forward_batch,
                key_values_lens=key_values_lens,
            )
            merged_key_states = key_buffer.index_select(0, kv_indices).to(
                torch.bfloat16
            )
            merged_value_states = value_buffer.index_select(0, kv_indices).to(
                torch.bfloat16
            )

        cu_seqlens_q = torch.nn.functional.pad(
            torch.cumsum(query_lens, dim=0),
            (1, 0),
        ).to(torch.int32)
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(key_values_lens, dim=0),
            (1, 0),
        ).to(torch.int32)
        attn_output = flash_attn_varlen_func(
            q=q,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(query_lens.max().item()),
            max_seqlen_k=int(key_values_lens.max().item()),
            causal=True,
        )
        return attn_output.reshape(-1, self.q_size)

    def forward_gen(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        routing: BAGELMoTTokenRouting,
    ) -> torch.Tensor:
        routing = routing.to(hidden_states.device)

        q = hidden_states.new_empty(hidden_states.shape[0], self.q_size)
        k = hidden_states.new_empty(hidden_states.shape[0], self.kv_size)
        v = hidden_states.new_empty(hidden_states.shape[0], self.kv_size)

        self._project_qkv_branch(
            hidden_states=hidden_states,
            token_indices=routing.text_token_indices,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            q_out=q,
            k_out=k,
            v_out=v,
        )
        self._project_qkv_branch(
            hidden_states=hidden_states,
            token_indices=routing.vae_token_indices,
            q_proj=self.q_proj_moe_gen,
            k_proj=self.k_proj_moe_gen,
            v_proj=self.v_proj_moe_gen,
            q_norm=self.q_norm_moe_gen,
            k_norm=self.k_norm_moe_gen,
            q_out=q,
            k_out=k,
            v_out=v,
        )

        if _can_use_bagel_official_non_causal_attention(forward_batch):
            q, k = self._apply_official_rotary_pos_emb(positions, q, k)
            attn_output = self._forward_official_non_causal_attention(
                q=q,
                k=k,
                v=v,
                forward_batch=forward_batch,
            )
        else:
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, forward_batch)

        output = torch.empty_like(attn_output)
        _apply_indexed_module(
            module=self.o_proj,
            source=attn_output,
            token_indices=routing.text_token_indices,
            output=output,
        )
        _apply_indexed_module(
            module=self.o_proj_moe_gen,
            source=attn_output,
            token_indices=routing.vae_token_indices,
            output=output,
        )
        return output

    def _project_qkv(
        self,
        *,
        hidden_states: torch.Tensor,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, _ = q_proj(hidden_states)
        k, _ = k_proj(hidden_states)
        v, _ = v_proj(hidden_states)
        return q, k, v

    def _apply_official_qk_norm(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm: nn.Module,
        k_norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q = q_norm(q)
        k = k_norm(k)
        return q.contiguous().view(q_shape), k.contiguous().view(k_shape)

    def _project_qkv_branch(
        self,
        *,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        q_norm: nn.Module,
        k_norm: nn.Module,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> None:
        if token_indices.numel() == 0:
            return
        branch_hidden_states = hidden_states.index_select(0, token_indices)
        q, k, v = self._project_qkv(
            hidden_states=branch_hidden_states,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
        )
        q, k = self._apply_official_qk_norm(
            q=q,
            k=k,
            q_norm=q_norm,
            k_norm=k_norm,
        )
        q_out[token_indices] = q
        k_out[token_indices] = k
        v_out[token_indices] = v


class BAGELQwen2MoTDecoderLayer(nn.Module):
    """BAGEL Qwen2-MoT decoder layer with native und/gen branch routing."""

    def __init__(
        self,
        config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.self_attn = BAGELQwen2MoTAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = BAGELQwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.mlp_moe_gen = BAGELQwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp_moe_gen", prefix),
        )
        self.input_layernorm = _make_bagel_rms_norm(config, config.hidden_size)
        self.input_layernorm_moe_gen = _make_bagel_rms_norm(
            config, config.hidden_size
        )
        self.post_attention_layernorm = _make_bagel_rms_norm(
            config, config.hidden_size
        )
        self.post_attention_layernorm_moe_gen = _make_bagel_rms_norm(
            config, config.hidden_size
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(forward_batch, "ug_g_non_causal_query_attention", False):
            return self.forward_und_official_residual(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_und_official_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        if residual is not None:
            hidden_states = hidden_states + residual

        residual_states = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual_states + hidden_states

        residual_states = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_states + hidden_states
        return hidden_states, None

    def forward_gen(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        routing: BAGELMoTTokenRouting,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        routing = routing.to(hidden_states.device)

        hidden_states, residual = _apply_indexed_norm_with_residual(
            source=hidden_states,
            residual=residual,
            routing=routing,
            text_norm=self.input_layernorm,
            vae_norm=self.input_layernorm_moe_gen,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            mode="gen",
            routing=routing,
        )
        hidden_states, residual = _apply_indexed_norm_with_residual(
            source=hidden_states,
            residual=residual,
            routing=routing,
            text_norm=self.post_attention_layernorm,
            vae_norm=self.post_attention_layernorm_moe_gen,
        )

        mlp_output = torch.empty_like(hidden_states)
        _apply_indexed_module(
            module=self.mlp,
            source=hidden_states,
            token_indices=routing.text_token_indices,
            output=mlp_output,
        )
        _apply_indexed_module(
            module=self.mlp_moe_gen,
            source=hidden_states,
            token_indices=routing.vae_token_indices,
            output=mlp_output,
        )
        return mlp_output, residual


class BAGELQwen2MoTModel(Qwen2Model):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=BAGELQwen2MoTDecoderLayer,
            alt_stream=alt_stream,
        )
        self.use_moe = "Mo" in getattr(config, "layer_module", "Qwen2MoTDecoderLayer")
        if self.pp_group.is_last_rank:
            self.norm = _make_bagel_rms_norm(config, config.hidden_size)
        if self.use_moe and self.pp_group.is_last_rank:
            self.norm_moe_gen = _make_bagel_rms_norm(config, config.hidden_size)

    def forward_gen_embeds(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        text_token_indices: torch.Tensor,
        vae_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.pp_group.is_first_rank or not self.pp_group.is_last_rank:
            raise NotImplementedError(
                "BAGEL Qwen2-MoT gen forward is currently single-stage PP only"
            )

        routing = BAGELMoTTokenRouting(
            text_token_indices=text_token_indices,
            vae_token_indices=vae_token_indices,
        ).to(input_embeds.device)
        routing.validate(input_embeds.shape[0])

        hidden_states = input_embeds
        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer.forward_gen(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                routing=routing,
            )

        if hidden_states.shape[0] != 0:
            hidden_states, _ = _apply_indexed_norm_with_residual(
                source=hidden_states,
                residual=residual,
                routing=routing,
                text_norm=self.norm,
                vae_norm=self.norm_moe_gen,
            )
        return hidden_states


class BAGELQwen2MoTForCausalLM(BAGELVisualFeatureMixin, Qwen2ForCausalLM):
    """SRT-native BAGEL language model shell for U-forward bring-up."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.latent_patch_size = int(getattr(config, "bagel_latent_patch_size", 2))
        self.max_latent_size = int(getattr(config, "bagel_max_latent_size", 64))
        self.latent_channel = int(getattr(config, "bagel_latent_channel", 16))
        self.latent_downsample = int(getattr(config, "bagel_latent_downsample", 16))
        self.timestep_shift = float(getattr(config, "bagel_timestep_shift", 1.0))
        self.model = BAGELQwen2MoTModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        patch_latent_dim = int(
            getattr(
                config,
                "bagel_patch_latent_dim",
                self.latent_patch_size**2 * self.latent_channel,
            )
        )
        max_latent_tokens = int(
            getattr(config, "bagel_max_latent_tokens", self.max_latent_size**2)
        )
        self.time_embedder = BAGELTimestepEmbedder(config.hidden_size)
        self.vae2llm = nn.Linear(patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, patch_latent_dim)
        self.latent_pos_embed = BAGELLatentPositionEmbedding(
            max_latent_tokens,
            config.hidden_size,
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False
        self.bagel_visual_feature_extractors_loaded = False
        self._init_visual_feature_extractors(config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())

        def iter_language_weights():
            for name, loaded_weight in weights:
                if self._load_visual_feature_weight(
                    name,
                    loaded_weight,
                    params_dict=params_dict,
                ):
                    continue
                yield from _iter_bagel_language_model_weights([(name, loaded_weight)])

        for name, loaded_weight in iter_language_weights():
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if name == "model.embed_tokens.weight":
                if (
                    not hasattr(self, "pp_group") or self.pp_group.is_last_rank
                ) and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param,
                            "weight_loader",
                            default_weight_loader,
                        )
                        weight_loader(param, loaded_weight)

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
            else:
                logger.warning("Parameter %s not found in params_dict", name)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> torch.Tensor:
        text_token_indices = getattr(
            forward_batch,
            "bagel_mot_text_token_indices",
            None,
        )
        vae_token_indices = getattr(
            forward_batch,
            "bagel_mot_vae_token_indices",
            None,
        )
        if text_token_indices is None or vae_token_indices is None:
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        hidden_states = self.model.forward_gen_embeds(
            input_embeds=input_embeds,
            positions=positions,
            forward_batch=forward_batch,
            text_token_indices=text_token_indices,
            vae_token_indices=vae_token_indices,
        )
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            None,
        )

    def forward_gen_embeds(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        text_token_indices: torch.Tensor,
        vae_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_gen_embeds(
            input_embeds=input_embeds,
            positions=positions,
            forward_batch=forward_batch,
            text_token_indices=text_token_indices,
            vae_token_indices=vae_token_indices,
        )

    def predict_velocity_from_packed_gen(
        self,
        *,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        packed_vae_token_indexes: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        device = self.vae2llm.weight.device
        latent_tokens = latent_tokens.to(device=device)
        packed_text_ids = packed_text_ids.to(device=device, dtype=torch.long)
        packed_text_indexes = packed_text_indexes.to(device=device, dtype=torch.long)
        packed_vae_token_indexes = packed_vae_token_indexes.to(
            device=device,
            dtype=torch.long,
        )
        packed_vae_position_ids = packed_vae_position_ids.to(
            device=device,
            dtype=torch.long,
        )
        packed_position_ids = packed_position_ids.to(device=device, dtype=torch.long)

        sequence_length = int(packed_seqlens.sum().item())
        text_embeds = self.model.embed_tokens(packed_text_ids)
        packed_sequence = text_embeds.new_zeros(
            sequence_length, self.config.hidden_size
        )
        packed_sequence[packed_text_indexes] = text_embeds
        packed_sequence[packed_vae_token_indexes] = self._embed_bagel_latents(
            latent_tokens=latent_tokens,
            timestep=timestep,
            latent_position_ids=packed_vae_position_ids,
        )

        hidden_states = self.model.forward_gen_embeds(
            input_embeds=packed_sequence,
            positions=packed_position_ids,
            forward_batch=forward_batch,
            text_token_indices=packed_text_indexes,
            vae_token_indices=packed_vae_token_indexes,
        )
        return self.llm2vae(hidden_states.index_select(0, packed_vae_token_indexes))

    def _embed_bagel_latents(
        self,
        *,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        latent_tokens = latent_tokens.to(
            device=self.vae2llm.weight.device,
            dtype=self.vae2llm.weight.dtype,
        )
        timestep = _expand_bagel_timestep(timestep, latent_tokens).to(
            device=latent_tokens.device
        )
        latent_position_ids = latent_position_ids.to(
            device=latent_tokens.device,
            dtype=torch.long,
        )
        return (
            self.vae2llm(latent_tokens)
            + self.time_embedder(timestep)
            + self.latent_pos_embed(latent_position_ids)
        )


class BAGELTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = _bagel_timestep_embedding(
            timestep,
            self.frequency_embedding_size,
        )
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class BAGELLatentPositionEmbedding(nn.Module):
    def __init__(self, max_latent_tokens: int, hidden_size: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(max_latent_tokens, hidden_size),
            requires_grad=False,
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


def _apply_indexed_norm_with_residual(
    *,
    source: torch.Tensor,
    residual: Optional[torch.Tensor],
    routing: BAGELMoTTokenRouting,
    text_norm: nn.Module,
    vae_norm: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        residual = source
        norm_source = source
    else:
        norm_source = source + residual
        residual = norm_source

    output = torch.empty_like(source)
    _apply_indexed_module(
        module=text_norm,
        source=norm_source,
        token_indices=routing.text_token_indices,
        output=output,
    )
    _apply_indexed_module(
        module=vae_norm,
        source=norm_source,
        token_indices=routing.vae_token_indices,
        output=output,
    )
    return output, residual


def _apply_indexed_module(
    *,
    module: nn.Module,
    source: torch.Tensor,
    token_indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if token_indices.numel() == 0:
        return
    branch_output = module(source.index_select(0, token_indices))
    if isinstance(branch_output, tuple):
        branch_output = branch_output[0]
    output[token_indices] = branch_output


def _expand_bagel_timestep(
    timestep: torch.Tensor,
    latent_tokens: torch.Tensor,
) -> torch.Tensor:
    timestep = timestep.to(device=latent_tokens.device)
    if timestep.numel() == 1:
        return timestep.reshape(1).expand(latent_tokens.shape[0])
    if timestep.shape[0] != latent_tokens.shape[0]:
        raise ValueError(
            "BAGEL timestep must be scalar or match latent token count: "
            f"{tuple(timestep.shape)} vs {tuple(latent_tokens.shape)}"
        )
    return timestep


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _can_use_bagel_official_non_causal_attention(forward_batch: ForwardBatch) -> bool:
    if not getattr(forward_batch, "ug_g_non_causal_query_attention", False):
        return False
    if getattr(forward_batch, "extend_seq_lens", None) is None:
        return False
    return True


def _can_use_bagel_official_causal_attention(forward_batch: ForwardBatch) -> bool:
    if getattr(forward_batch, "ug_g_non_causal_query_attention", False):
        return False
    if getattr(forward_batch, "out_cache_loc", None) is None:
        return False
    ug_metadata = getattr(forward_batch, "ug_u_forward_metadata", None)
    if ug_metadata and any(metadata is not None for metadata in ug_metadata):
        return True
    # BAGEL's Qwen2-MoT uses the official Qwen2 RoPE/attention semantics for
    # understanding/text forwards. Keep this model on that path even when a
    # caller reaches it outside the experimental UG session metadata bridge.
    return True


def _bagel_query_lens(forward_batch: ForwardBatch) -> torch.Tensor:
    if forward_batch.forward_mode.is_decode():
        return torch.ones(
            forward_batch.batch_size,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )
    if forward_batch.extend_seq_lens is None:
        raise ValueError(
            "BAGEL official causal attention requires extend_seq_lens outside decode"
        )
    return forward_batch.extend_seq_lens.to(torch.int32)


def _is_bagel_zero_prefix_extend(forward_batch: ForwardBatch) -> bool:
    if not forward_batch.forward_mode.is_extend():
        return False
    if forward_batch.extend_prefix_lens is None:
        return False
    return bool(torch.all(forward_batch.extend_prefix_lens == 0).item())


def _bagel_full_sequence_kv_indices(
    *,
    forward_batch: ForwardBatch,
    key_values_lens: torch.Tensor,
) -> torch.Tensor:
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    req_pool_indices = forward_batch.req_pool_indices.tolist()
    seq_lens = key_values_lens.tolist()

    kv_indices = []
    for req_pool_idx, seq_len in zip(req_pool_indices, seq_lens):
        if seq_len == 0:
            continue
        kv_indices.append(req_to_token[req_pool_idx, :seq_len])
    if not kv_indices:
        return torch.empty(0, dtype=torch.long, device=req_to_token.device)
    return torch.cat(kv_indices, dim=0).to(torch.long)


def _bagel_timestep_embedding(
    timestep: torch.Tensor,
    embedding_size: int,
    *,
    max_period: int = 10000,
) -> torch.Tensor:
    half = embedding_size // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
        / half
    )
    args = timestep[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_size % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=-1,
        )
    return embedding


def _iter_bagel_language_model_weights(weights: Iterable[Tuple[str, torch.Tensor]]):
    for name, loaded_weight in weights:
        if name.startswith("language_model."):
            yield name[len("language_model.") :], loaded_weight
        elif _is_bagel_visual_gen_key(name):
            yield name, loaded_weight
        elif _is_qwen2_language_model_key(name):
            yield name, loaded_weight


def _is_qwen2_language_model_key(name: str) -> bool:
    return name.startswith(
        (
            "model.",
            "lm_head.",
        )
    )


def _is_bagel_visual_gen_key(name: str) -> bool:
    return name.startswith(
        (
            "time_embedder.",
            "vae2llm.",
            "llm2vae.",
            "latent_pos_embed.",
        )
    )


def _make_bagel_rms_norm(config, hidden_size: int) -> BAGELRMSNorm:
    weight_dtype = (
        torch.float32
        if get_global_server_args().rl_on_policy_target is not None
        else None
    )
    return BAGELRMSNorm(
        hidden_size,
        eps=config.rms_norm_eps,
        weight_dtype=weight_dtype,
    )


EntryClass = [BAGELQwen2MoTForCausalLM]
