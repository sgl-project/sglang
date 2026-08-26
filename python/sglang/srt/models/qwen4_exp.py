# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Native text inference path for Qwen4-Exp / Qwen3.8-Flash-Next."""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.srt.configs.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    _find_nth_prime_after,
)
from sglang.srt.distributed import divide
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import has_forward_context
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _activation(name: str):
    if name == "silu":
        return F.silu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    raise ValueError(f"Unsupported Qwen4-Exp activation: {name}")


def _parallel_value(primary: str, fallback: str, default: int) -> int:
    parallel = get_parallel()
    for name in (primary, fallback):
        try:
            return getattr(parallel, name)
        except AssertionError:
            continue
    return default


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    x_rope, x_nope = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
    return torch.cat([x_rope, x_nope], dim=-1)


def _rms_norm(x: torch.Tensor, eps: float, group_size: Optional[int] = None):
    if group_size is not None:
        x = x.reshape(*x.shape[:-1], -1, group_size)
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return out.flatten(-2) if group_size is not None else out


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by {group_size=}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = _rms_norm(x, self.eps, self.group_size)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen4ExpTextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        hidden_states = _rms_norm(hidden_states, self.variance_epsilon)
        hidden_states = self.weight.float() * hidden_states
        hidden_states = hidden_states * _activation(self.activation)(gate.float())
        return hidden_states.to(gate.dtype)


class Qwen4ExpTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        rope_parameters = config.rope_parameters
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        freqs = torch.einsum(
            "d,bs->bsd", self.inv_freq.to(x.device).float(), position_ids.float()
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen4ExpTextGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpTextRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(
            hc_hidden_size, config.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            config.hc_lowrank, hc_hidden_size, bias=False
        )
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, self.hc_count, bias=False)
            if use_combine
            else None
        )

    def forward(self, hyper_input: torch.Tensor):
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection "
                f"features, got {hyper_input.shape[-1]}."
            )
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(
            self.input_mix_weight_down(hyper_input_normed) / self.hc_count
        )
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        mixed_input = (
            input_mix_weight
            * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight(hyper_input_normed) / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights


class Qwen4ExpTextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.attn_tp_rank = _parallel_value("attn_tp_rank", "tp_rank", 0)
        self.attn_tp_size = _parallel_value("attn_tp_size", "tp_size", 1)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.local_num_v_heads = divide(self.num_v_heads, self.attn_tp_size)
        self.local_num_k_heads = divide(self.num_k_heads, self.attn_tp_size)
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.local_key_dim = self.head_k_dim * self.local_num_k_heads
        self.local_value_dim = self.head_v_dim * self.local_num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.local_conv_dim = self.local_key_dim * 2 + self.local_value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        self._override_weight_loader(
            self.conv1d.weight,
            mamba_v2_sharded_weight_loader(
                [
                    (self.key_dim, 0, False),
                    (self.key_dim, 0, False),
                    (self.value_dim, 0, False),
                ],
                self.attn_tp_size,
                self.attn_tp_rank,
            ),
        )
        self.dt_bias = nn.Parameter(torch.ones(self.local_num_v_heads))
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=torch.float32)
        )
        with torch.no_grad():
            self.A_log.copy_(
                torch.empty(self.local_num_v_heads).uniform_(0.01, 16).log_()
            )
        state_weight_loader = self._make_sharded_weight_loader(
            self.attn_tp_rank, self.attn_tp_size
        )
        set_weight_attrs(self.A_log, {"weight_loader": state_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": state_weight_loader})
        self.norm = Qwen4ExpTextRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            activation=config.output_gate_type or config.hidden_act,
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        self.in_proj_qkv = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        nn.init.kaiming_uniform_(self.in_proj_qkv.weight, a=math.sqrt(5))
        self._override_weight_loader(
            self.in_proj_qkv.weight,
            mamba_v2_sharded_weight_loader(
                [
                    (self.key_dim, 0, False),
                    (self.key_dim, 0, False),
                    (self.value_dim, 0, False),
                ],
                self.attn_tp_size,
                self.attn_tp_rank,
            ),
        )
        self.in_proj_z = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.value_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        self.in_proj_b = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        self.in_proj_a = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        for linear in (self.in_proj_z, self.in_proj_b, self.in_proj_a):
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        self.attn = RadixLinearAttention(
            layer_id=layer_idx,
            num_q_heads=self.local_num_k_heads,
            num_k_heads=self.local_num_k_heads,
            num_v_heads=self.local_num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    @staticmethod
    def _override_weight_loader(param: torch.Tensor, loader) -> None:
        if hasattr(param, "_weight_loader"):
            param._weight_loader = loader
        elif hasattr(param, "weight_loader"):
            delattr(param, "weight_loader")
            set_weight_attrs(param, {"weight_loader": loader})
        else:
            set_weight_attrs(param, {"weight_loader": loader})

    @staticmethod
    def _make_sharded_weight_loader(tp_rank: int, tp_size: int):
        def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if tp_size > 1:
                shard_size = param.data.shape[0]
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size
                )
            assert param.data.shape == loaded_weight.shape, (
                f"param.data.shape={param.data.shape} != "
                f"loaded_weight.shape={loaded_weight.shape}"
            )
            param.data.copy_(loaded_weight)

        return loader

    @staticmethod
    def _linear(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(module, RowParallelLinear) and module.tp_size == 1:
            return F.linear(hidden_states, module.weight, module.bias)
        output = module(hidden_states)
        return output[0] if isinstance(output, tuple) else output

    def _sync_radix_parameters(self) -> None:
        self.attn.conv_weights = self.conv1d.weight.squeeze(1)
        self.attn.bias = self.conv1d.bias
        self.attn.A_log = self.A_log
        self.attn.dt_bias = self.dt_bias

    def _project_qkv_after_conv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self._linear(self.in_proj_qkv, hidden_states).transpose(1, 2)
        mixed_qkv = F.conv1d(
            mixed_qkv,
            self.conv1d.weight,
            self.conv1d.bias,
            padding=self.conv_kernel_size - 1,
            groups=self.local_conv_dim,
        )[:, :, :seq_len]
        mixed_qkv = _activation(self.activation)(mixed_qkv).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.local_key_dim, self.local_key_dim, self.local_value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch_size, seq_len, self.local_num_k_heads, self.head_k_dim
        )
        key = key.reshape(batch_size, seq_len, self.local_num_k_heads, self.head_k_dim)
        value = value.reshape(
            batch_size, seq_len, self.local_num_v_heads, self.head_v_dim
        )
        return query, key, value

    def _forward_reference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self._project_qkv_after_conv(hidden_states)
        z = self._linear(self.in_proj_z, hidden_states).reshape(
            batch_size, seq_len, self.local_num_v_heads, self.head_v_dim
        )
        beta = torch.sigmoid(self._linear(self.in_proj_b, hidden_states))
        g = -self.A_log.float().exp() * F.softplus(
            self._linear(self.in_proj_a, hidden_states).float() + self.dt_bias
        )
        if self.local_num_v_heads // self.local_num_k_heads > 1:
            repeat = self.local_num_v_heads // self.local_num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)
        scale = self.head_k_dim**-0.5
        query = query * scale
        state = value.new_zeros(
            batch_size, self.local_num_v_heads, self.head_k_dim, self.head_v_dim
        ).float()
        outputs = []
        for token_idx in range(seq_len):
            q_t = query[:, token_idx].float()
            k_t = key[:, token_idx].float()
            v_t = value[:, token_idx].float()
            g_t = g[:, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, token_idx].float().unsqueeze(-1)
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        core_attn_out = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.local_value_dim)
        return self._linear(self.out_proj, core_attn_out)

    def _forward_chunked_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self._project_qkv_after_conv(hidden_states)
        z = self._linear(self.in_proj_z, hidden_states).reshape(
            batch_size, seq_len, self.local_num_v_heads, self.head_v_dim
        )
        beta = torch.sigmoid(self._linear(self.in_proj_b, hidden_states)).float()
        g = -self.A_log.float().exp() * F.softplus(
            self._linear(self.in_proj_a, hidden_states).float() + self.dt_bias
        )
        initial_state = torch.zeros(
            batch_size,
            self.local_num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        initial_state_indices = torch.arange(
            batch_size, device=hidden_states.device, dtype=torch.int32
        )
        core_attn_out, _, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.local_value_dim)
        return self._linear(self.out_proj, core_attn_out)

    def _forward_radix(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        mixed_qkv = self._linear(self.in_proj_qkv, hidden_states)
        z = self._linear(self.in_proj_z, hidden_states).reshape(
            -1, self.local_num_v_heads, self.head_v_dim
        )
        b = self._linear(self.in_proj_b, hidden_states)
        a = self._linear(self.in_proj_a, hidden_states)
        self._sync_radix_parameters()
        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )
        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        output = self._linear(self.out_proj, core_attn_out)
        return output.reshape(original_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if (
            forward_batch is not None
            and has_forward_context()
            and hidden_states.dim() == 2
        ):
            return self._forward_radix(hidden_states, forward_batch)

        if (
            forward_batch is None
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        ):
            return self._forward_chunked_prefill(hidden_states)

        return self._forward_reference(hidden_states)


class Qwen4ExpTextQSAIndexer(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = (
            self.token_budget // self.compress_ratio if self.token_budget else 0
        )
        if all(
            value is not None
            for value in (
                self.index_n_heads,
                self.index_kv_heads,
                self.index_head_dim,
                self.token_budget,
                self.compress_ratio,
            )
        ):
            self.index_qk_proj = nn.Linear(
                config.hidden_size,
                (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
                bias=False,
            )
            self.q_layernorm = Qwen4ExpTextRMSNorm(
                self.index_head_dim, eps=config.rms_norm_eps
            )
            self.k_layernorm = Qwen4ExpTextRMSNorm(
                self.index_head_dim, eps=config.rms_norm_eps
            )
        else:
            self.index_qk_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.index_qk_proj is None:
            return attention_mask
        batch_size, seq_length, _ = hidden_states.shape
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ],
            dim=-1,
        )
        q = q.reshape(batch_size, seq_length, self.index_n_heads, self.index_head_dim)
        raw_keys = token_k.reshape(
            batch_size, seq_length, self.index_kv_heads, self.index_head_dim
        ).squeeze(2)
        q = self.q_layernorm(q)
        q = _apply_rotary(q, cos.unsqueeze(2), sin.unsqueeze(2))

        selected = torch.zeros_like(attention_mask, dtype=torch.bool)
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                visible = torch.nonzero(
                    attention_mask[batch_idx, query_idx], as_tuple=False
                ).flatten()
                if visible.numel() <= self.token_budget:
                    selected[batch_idx, query_idx, visible] = True
                    continue
                num_complete_blocks = visible.numel() // self.compress_ratio
                selected_tokens = visible.new_empty((0,))
                if num_complete_blocks > 0:
                    block_token_indices = visible[
                        : num_complete_blocks * self.compress_ratio
                    ].view(num_complete_blocks, self.compress_ratio)
                    key_groups = raw_keys[batch_idx].index_select(
                        0, block_token_indices.flatten()
                    )
                    key_groups = key_groups.view(
                        *block_token_indices.shape, self.index_head_dim
                    )
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_keys = _apply_rotary(
                        pooled_keys,
                        cos[batch_idx].index_select(0, group_starts),
                        sin[batch_idx].index_select(0, group_starts),
                    )
                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_keys.float().T
                    ).T
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(
                        self.index_head_dim
                    )
                    selected_blocks = scores.topk(
                        min(self.block_topk, num_complete_blocks), dim=0
                    ).indices
                    selected_tokens = block_token_indices.index_select(
                        0, selected_blocks
                    ).flatten()
                tail = visible[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail])
                selected[batch_idx, query_idx, selected_tokens] = True
        return attention_mask & selected


class Qwen4ExpTextAttention(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        attention_mask = self.indexer(hidden_states, cos, sin, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                batch_size, seq_len, self.num_heads, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(
                batch_size, seq_len, self.num_key_value_heads, self.head_dim
            )
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        query_states = _apply_rotary(query_states, cos.unsqueeze(2), sin.unsqueeze(2))
        key_states = _apply_rotary(key_states, cos.unsqueeze(2), sin.unsqueeze(2))
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)
        attn_weights = (
            torch.einsum("bqhd,bkhd->bhqk", query_states, key_states) * self.scaling
        )
        attn_weights = attn_weights.masked_fill(
            ~attention_mask[:, None, :, :], torch.finfo(attn_weights.dtype).min
        )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        if self.training and self.attention_dropout:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value_states)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen4ExpTextMLP(nn.Module):
    def __init__(
        self, config: Qwen4ExpTextConfig, intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.moe_intermediate_size
            if intermediate_size is None
            else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _activation(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen4ExpTextExperts(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.activation = config.hidden_act
        self.force_reference = False
        default_device = torch.empty(0).device
        expert_device = (
            torch.device("cpu") if default_device.type == "cuda" else default_device
        )
        expert_numel = self.num_experts * self.intermediate_dim * self.hidden_dim * 3
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
                device=expert_device,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
                device=expert_device,
            )
        )
        if expert_numel <= 10_000_000:
            nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
            nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)
        self.act_fn = _activation(config.hidden_act)
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size_per_partition=self.intermediate_dim,
            top_k=self.top_k,
            activation=self.activation,
            is_gated=True,
            inplace=False,
            gate_up_interleaved=False,
        )

    def _can_use_fused(self, hidden_states: torch.Tensor) -> bool:
        return (
            not self.force_reference
            and hidden_states.is_cuda
            and self.gate_up_proj.device == hidden_states.device
            and self.down_proj.device == hidden_states.device
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self.activation == "silu"
            and self.intermediate_dim % 8 == 0
        )

    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_weight = self.gate_up_proj[expert_idx].to(current_state.device)
            down_weight = self.down_proj[expert_idx].to(current_state.device)
            gate, up = F.linear(current_state, gate_up_weight).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_weight)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self._can_use_fused(hidden_states):
            return self.forward_reference(hidden_states, top_k_index, top_k_weights)

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        return fused_experts(
            hidden_states.contiguous(),
            self.gate_up_proj,
            self.down_proj,
            StandardTopKOutput(
                topk_weights=top_k_weights,
                topk_ids=top_k_index,
                router_logits=None,
            ),
            moe_runner_config=self.moe_runner_config,
        )


class Qwen4ExpTextTopKRouter(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value.to(router_logits.dtype), router_indices


class Qwen4ExpTextSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.gate = Qwen4ExpTextTopKRouter(config)
        self.experts = Qwen4ExpTextExperts(config)
        self.shared_expert = Qwen4ExpTextMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.force_reference_moe = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        self.experts.force_reference = self.force_reference_moe
        expert_output = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )
        if (
            not self.force_reference_moe
            and expert_output.is_cuda
            and expert_output.dtype in (torch.float16, torch.bfloat16)
        ):
            fused_gate_sigmoid_mul_add(
                hidden_states_reshaped,
                self.shared_expert_gate.weight.squeeze(),
                shared_expert_output,
                expert_output,
            )
        else:
            shared_expert_output = (
                torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
                * shared_expert_output
            )
            expert_output = expert_output + shared_expert_output
        return expert_output.reshape(original_shape)


class Qwen4ExpTextNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        self.seed = config.seed
        self.eos_token_id = (
            config.eos_token_id[0]
            if isinstance(config.eos_token_id, list)
            else config.eos_token_id
        )
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.head_vocab_sizes = []
        self.head_offsets = []
        total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(
                self.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(total_vocab_size)
            total_vocab_size += size
        self.register_buffer(
            "layer_multipliers",
            _build_layer_multipliers(
                self.unigram_vocab_size,
                self.ngram_size,
                self.ple_layer_index,
                self.seed,
            ),
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(self.head_vocab_sizes, dtype=torch.long),
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.tensor(self.head_offsets, dtype=torch.long)
        )
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(total_vocab_size / divisor) * divisor
        self.ngram_embedding = Qwen4ExpTextShardedNGramEmbedding(
            padded_vocab_size,
            head_dim_per_ngram,
            max(1, config.split_ngram_parts),
        )

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int):
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = (
            source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        )
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def _ngram_ids_from_shifted_tokens(
        self, shifted_tokens: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        blocks = []
        layer_multipliers = self.layer_multipliers.to(shifted_tokens[0].device)
        ngram_heads_vocab_sizes = self.ngram_heads_vocab_sizes.to(
            shifted_tokens[0].device
        )
        ngram_heads_offsets = self.ngram_heads_offsets.to(shifted_tokens[0].device)
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * layer_multipliers[position],
                )
            head_vocab_sizes = ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = ngram_heads_offsets[start_idx:end_idx]
            view_shape = (1,) * mixed_ids.dim() + (-1,)
            ngram_ids = torch.remainder(
                mixed_ids.unsqueeze(-1), head_vocab_sizes.view(*view_shape)
            )
            blocks.append(ngram_ids + head_offsets.view(*view_shape))
        return torch.cat(blocks, dim=-1)

    def _shift_right_ignore_eos_flat(
        self, token_ids: torch.Tensor, seq_lens: Sequence[int], shift: int
    ) -> torch.Tensor:
        if shift == 0:
            return token_ids
        seq_lens_tensor = torch.tensor(
            seq_lens, device=token_ids.device, dtype=torch.long
        )
        seq_offsets = torch.cumsum(seq_lens_tensor, dim=0) - seq_lens_tensor
        seq_starts = torch.repeat_interleave(seq_offsets, seq_lens_tensor)
        positions = torch.arange(
            token_ids.numel(), device=token_ids.device, dtype=torch.long
        )
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
        previous_eos = torch.cat(
            [eos_positions.new_full((1,), -1), previous_eos_inclusive[:-1]]
        )
        previous_reset = torch.maximum(previous_eos, seq_starts - 1)
        source_positions = positions - shift
        shifted = token_ids.gather(dim=0, index=source_positions.clamp_min(0))
        valid = (positions - previous_reset - 1 >= shift) & (
            source_positions >= seq_starts
        )
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward_flat(
        self, input_ids: torch.Tensor, seq_lens: Sequence[int]
    ) -> torch.Tensor:
        input_ids = input_ids.long().reshape(-1)
        if input_ids.numel() != sum(seq_lens):
            raise ValueError(
                f"PLE flat input has {input_ids.numel()} tokens, "
                f"but seq_lens sum to {sum(seq_lens)}."
            )
        shifted_tokens = [
            self._shift_right_ignore_eos_flat(input_ids, seq_lens, shift)
            for shift in range(self.ngram_size)
        ]
        ngram_ids = self._ngram_ids_from_shifted_tokens(shifted_tokens)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.long()
        previous_context = input_ids.new_full(
            (input_ids.shape[0], self.context_len), self.eos_token_id
        )
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        ngram_ids = self._ngram_ids_from_shifted_tokens(shifted_tokens)[
            :, -input_ids.shape[1] :
        ]
        return self.ngram_embedding(ngram_ids).flatten(-2)


class Qwen4ExpTextShardedNGramEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_shards: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_shards = min(max(1, num_shards), num_embeddings)
        base_rows = num_embeddings // self.num_shards
        remainder = num_embeddings % self.num_shards
        offset = 0
        self.shard_names = []
        self.shard_offsets = []
        self.shard_sizes = []
        default_device = torch.empty(0).device
        shard_device = (
            torch.device("cpu") if default_device.type == "cuda" else default_device
        )
        for shard_idx in range(self.num_shards):
            rows = base_rows + (1 if shard_idx < remainder else 0)
            shard_name = f"shard_{shard_idx}"
            weight = torch.empty((rows, embedding_dim), device=shard_device)
            if num_embeddings * embedding_dim <= 10_000_000:
                nn.init.normal_(weight)
            self.add_module(
                shard_name,
                nn.Embedding(rows, embedding_dim, _weight=weight),
            )
            self.shard_names.append(shard_name)
            self.shard_offsets.append(offset)
            self.shard_sizes.append(rows)
            offset += rows

    def _check_ids_cpu(self, flat_ids: torch.Tensor) -> None:
        valid_ids = (flat_ids >= 0) & (flat_ids < self.num_embeddings)
        if not valid_ids.all().item():
            raise IndexError("Qwen4-Exp n-gram embedding index out of range")

    def _forward_cuda_from_cpu_shards(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids_cpu = input_ids.reshape(-1).to("cpu")
        self._check_ids_cpu(flat_ids_cpu)
        first_weight = getattr(self, self.shard_names[0]).weight
        output = torch.empty(
            (flat_ids_cpu.numel(), self.embedding_dim),
            device=first_weight.device,
            dtype=first_weight.dtype,
        )
        boundaries = torch.tensor(self.shard_offsets[1:], dtype=torch.long)
        shard_indices = torch.bucketize(flat_ids_cpu, boundaries, right=True)
        order = torch.argsort(shard_indices, stable=True)
        sorted_shards = shard_indices.index_select(0, order)
        hit_shards, hit_counts = torch.unique_consecutive(
            sorted_shards, return_counts=True
        )
        group_start = 0
        for shard_idx_tensor, count_tensor in zip(hit_shards, hit_counts):
            shard_idx = int(shard_idx_tensor.item())
            count = int(count_tensor.item())
            positions = order[group_start : group_start + count]
            group_start += count
            shard = getattr(self, self.shard_names[shard_idx])
            local_ids = (
                flat_ids_cpu.index_select(0, positions) - self.shard_offsets[shard_idx]
            )
            output[positions] = shard(local_ids)
        return output.reshape(*input_ids.shape, self.embedding_dim).to(
            input_ids.device, non_blocking=True
        )

    def forward_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        self._check_ids_cpu(flat_ids.cpu() if flat_ids.is_cuda else flat_ids)
        first_weight = getattr(self, self.shard_names[0]).weight
        output = torch.empty(
            (flat_ids.numel(), self.embedding_dim),
            device=input_ids.device,
            dtype=first_weight.dtype,
        )
        for shard_name, offset, size in zip(
            self.shard_names, self.shard_offsets, self.shard_sizes
        ):
            mask = (flat_ids >= offset) & (flat_ids < offset + size)
            if not mask.any().item():
                continue
            shard = getattr(self, shard_name)
            local_ids = (flat_ids[mask] - offset).to(shard.weight.device)
            output[mask] = shard(local_ids).to(input_ids.device)
        return output.reshape(*input_ids.shape, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        first_weight = getattr(self, self.shard_names[0]).weight
        if input_ids.is_cuda and first_weight.device.type == "cpu":
            return self._forward_cuda_from_cpu_shards(input_ids)
        return self.forward_reference(input_ids)


class Qwen4ExpTextPLELayer(nn.Module):
    def __init__(
        self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(
            config, config.ple_embed_dim, layer_idx, ple_layer_index
        )
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(config.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(config.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpTextRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_query = Qwen4ExpTextRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_conv = Qwen4ExpTextRMSNorm(
            hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps
        )
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)

    def _short_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]
        hidden_states = F.silu(self.conv1d(hidden_states))
        return hidden_states.transpose(1, 2)

    def _short_conv_flat(
        self, hidden_states: torch.Tensor, seq_lens: Sequence[int]
    ) -> torch.Tensor:
        outputs = []
        token_offset = 0
        for seq_len in seq_lens:
            seq_slice = slice(token_offset, token_offset + seq_len)
            outputs.append(
                self._short_conv(hidden_states[seq_slice].unsqueeze(0)).squeeze(0)
            )
            token_offset += seq_len
        return torch.cat(outputs, dim=0)

    def _forward_with_embeddings(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        seq_lens: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
            self.hidden_size
        )
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if hidden_states.dim() == 2:
            if seq_lens is None:
                raise ValueError("seq_lens is required for flat PLE convolution.")
            conv_out = self._short_conv_flat(gated_value_normed, seq_lens)
        else:
            conv_out = self._short_conv(gated_value_normed)
        return gated_value + conv_out

    def forward_flat(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        seq_lens: Sequence[int],
    ) -> torch.Tensor:
        embeddings = self.ple_embedding.forward_flat(input_ids, seq_lens)
        return self._forward_with_embeddings(hidden_states, embeddings, seq_lens)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids)
        return self._forward_with_embeddings(hidden_states, embeddings)


class Qwen4ExpTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpTextGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen4ExpTextAttention(config, layer_idx)
        self.mlp = Qwen4ExpTextSparseMoeBlock(config)
        ple_layer_index = (
            config.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in config.ple_layer_ids
            else None
        )
        self.ple = (
            Qwen4ExpTextPLELayer(config, layer_idx, ple_layer_index)
            if ple_layer_index is not None
            else None
        )
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids)
        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(
            hidden_states
        )
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, forward_batch=forward_batch)
        else:
            hidden_states = self.self_attn(
                hidden_states, position_embeddings, attention_mask
            )
        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(
            hidden_states
        )
        hidden_states = self.mlp(hidden_states)
        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return hyper_input + injection.flatten(-2)


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen4ExpTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(
            config, use_combine=False
        )
        self.force_sequence_ple = (
            os.environ.get("SGLANG_QWEN4_EXP_FORCE_SEQUENCE_PLE", "0") == "1"
        )

    @property
    def start_layer(self) -> int:
        return 0

    @property
    def end_layer(self) -> int:
        return self.config.num_hidden_layers

    def _sequence_lengths(self, num_tokens: int, logits_metadata) -> list[int]:
        extend_lens = getattr(logits_metadata, "extend_seq_lens_cpu", None)
        if extend_lens is not None and sum(extend_lens) == num_tokens:
            return list(extend_lens)
        extend_lens_gpu = getattr(logits_metadata, "extend_seq_lens", None)
        if (
            extend_lens_gpu is not None
            and int(extend_lens_gpu.sum().item()) == num_tokens
        ):
            return [int(x) for x in extend_lens_gpu.cpu().tolist()]
        return [num_tokens]

    def _can_use_flat_hybrid_path(self, logits_metadata) -> bool:
        forward_mode = getattr(logits_metadata, "forward_mode", None)
        return (
            not self.force_sequence_ple
            and forward_mode is not None
            and not forward_mode.is_idle()
            and has_forward_context()
        )

    def _apply_full_attention_flat(
        self,
        decoder_layer: Qwen4ExpTextDecoderLayer,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: list[int],
    ) -> torch.Tensor:
        outputs = []
        token_offset = 0
        for seq_len in seq_lens:
            seq_slice = slice(token_offset, token_offset + seq_len)
            seq_hidden_states = hidden_states[seq_slice].unsqueeze(0)
            seq_positions = positions[seq_slice].unsqueeze(0)
            position_embeddings = self.rotary_emb(seq_hidden_states, seq_positions)
            attention_mask = torch.ones(
                (1, seq_len, seq_len),
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            outputs.append(
                decoder_layer.self_attn(
                    seq_hidden_states, position_embeddings, attention_mask
                ).squeeze(0)
            )
            token_offset += seq_len
        return torch.cat(outputs, dim=0)

    def _forward_flat_hybrid(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_metadata,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        seq_lens = self._sequence_lengths(input_ids.numel(), logits_metadata)
        hidden_states = input_embeds.repeat(1, self.config.hc_count)
        for decoder_layer in self.layers:
            if decoder_layer.ple is not None:
                hidden_states = hidden_states + decoder_layer.ple.forward_flat(
                    hidden_states, input_ids, seq_lens
                )
            hidden_states, hyper_input, injection_weights = (
                decoder_layer.attn_hyper_connection(hidden_states)
            )
            if decoder_layer.layer_type == "linear_attention":
                hidden_states = decoder_layer.linear_attn(
                    hidden_states, forward_batch=logits_metadata
                )
            else:
                hidden_states = self._apply_full_attention_flat(
                    decoder_layer, hidden_states, positions, seq_lens
                )
            injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
            hidden_states = hyper_input + injection.flatten(-2)

            hidden_states, hyper_input, injection_weights = (
                decoder_layer.mlp_hyper_connection(hidden_states)
            )
            hidden_states = decoder_layer.mlp(hidden_states)
            injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
            hidden_states = hyper_input + injection.flatten(-2)
        return self.hyper_connection_mixer(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor],
        logits_metadata,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 1:
            input_ids = input_ids.reshape(-1)
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        if positions is None:
            positions = torch.arange(input_ids.numel(), device=input_ids.device)
        if positions.dim() != 1:
            positions = positions.reshape(-1)

        if self._can_use_flat_hybrid_path(logits_metadata):
            return self._forward_flat_hybrid(
                input_ids, positions, logits_metadata, input_embeds
            )

        outputs = []
        token_offset = 0
        for seq_len in self._sequence_lengths(input_ids.numel(), logits_metadata):
            seq_slice = slice(token_offset, token_offset + seq_len)
            seq_input_ids = input_ids[seq_slice].unsqueeze(0)
            hidden_states = input_embeds[seq_slice].unsqueeze(0)
            seq_positions = positions[seq_slice].unsqueeze(0)
            hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)
            position_embeddings = self.rotary_emb(hidden_states, seq_positions)
            attention_mask = torch.ones(
                (1, seq_len, seq_len), dtype=torch.bool, device=input_ids.device
            ).tril()
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    input_ids=seq_input_ids,
                    forward_batch=logits_metadata,
                )
            hidden_states = self.hyper_connection_mixer(hidden_states)
            outputs.append(hidden_states.squeeze(0))
            token_offset += seq_len
        return torch.cat(outputs, dim=0)


class Qwen4ExpForCausalLM(nn.Module):
    packed_modules_mapping = {}
    supported_lora_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "gate_up_proj",
        "down_proj",
        "lm_head",
    ]

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        del quant_config, prefix
        super().__init__()
        self.config = config
        self.model = Qwen4ExpTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_processor = LogitsProcessor(config)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Union[ForwardBatch, object],
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if (
                "rotary_emb.inv_freq" in name
                or name.startswith("mtp.")
                or ".ple_embedding.layer_multipliers" in name
                or ".ple_embedding.ngram_heads_offsets" in name
                or ".ple_embedding.ngram_heads_vocab_sizes" in name
            ):
                continue
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]
            if name.startswith("language_model."):
                name = "model." + name[len("language_model.") :]
            if name.startswith("model.visual.") or name.startswith("visual."):
                continue
            if name not in params_dict:
                logger.warning("Parameter %s not found in params_dict", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen4ExpForConditionalGeneration(Qwen4ExpForCausalLM):
    packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping
    supported_lora_modules = Qwen4ExpForCausalLM.supported_lora_modules

    def __init__(
        self,
        config: Qwen4ExpConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.vision_config = getattr(config, "vision_config", None)
        super().__init__(
            config=config.text_config, quant_config=quant_config, prefix=prefix
        )
        self.root_config = config
        self.language_model_only = getattr(config, "language_model_only", False)


EntryClass = [Qwen4ExpForConditionalGeneration, Qwen4ExpForCausalLM]
