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

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
"""Inference-only Grok1 model."""
import functools
import logging
import math
import os
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.elementwise import (
    fused_dual_residual_rmsnorm,
    fused_rmsnorm,
    gelu_and_mul_triton,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.router import fused_moe_router_shim
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import (
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_get_mscale,
    get_rope,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, dispose_tensor, dump_to_file

logger = logging.getLogger(__name__)


# Dump tensors for debugging
debug_tensor_dump_output_folder = None
debug_tensor_dump_inject = False
debug_tensor_dump_layers = None
debug_tensor_dump_test = False


class Grok1MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results=True,
        use_presharded_weights: bool = False,
        split_gate_up: bool = False,
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            use_presharded_weights=use_presharded_weights,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            use_presharded_weights=use_presharded_weights,
        )
        self.act_fn = GeluAndMul(approximate="tanh")
        self.layer_id = layer_id

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x, _ = gelu_and_mul_triton(gate_up)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nn.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        reduce_results: bool = True,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at full precision for stability (see https://arxiv.org/pdf/2101.03961)
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
        )

        self.router_logit_softcapping = getattr(
            config, "router_logit_softcapping", 30.0
        )
        custom_routing_function = functools.partial(
            fused_moe_router_shim, self.router_logit_softcapping
        )

        self.topk = TopK(
            top_k=top_k,
            renormalize=False,
            custom_routing_function=custom_routing_function,
        )

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            layer_id=layer_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            activation="gelu",
            reduce_results=reduce_results,
            use_presharded_weights=use_presharded_weights,
            inplace=inplace,
            no_combine=no_combine,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # need to assert self.gate.quant_method is unquantized
        topk_output = self.topk(hidden_states, self.gate.weight)
        return self.experts(hidden_states, topk_output)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        low -= 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_rope_scaling(config):
    rope_type = getattr(config, "rope_type", None)
    if rope_type:
        original_max_position_embeddings = getattr(
            config, "original_max_position_embeddings", None
        )
        scaling_factor = getattr(config, "scaling_factor", None)
        extrapolation_factor = getattr(config, "extrapolation_factor", 1.0)
        attn_factor = getattr(config, "attn_factor", 1.0)
        beta_fast = getattr(config, "beta_fast", 32)
        beta_slow = getattr(config, "beta_slow", 1)
        rope_scaling = {
            "extra_method": rope_type,
            "max_position_embeddings": original_max_position_embeddings,
            "scaling_factor": scaling_factor,
            "extrapolation_factor": extrapolation_factor,
            "attn_factor": attn_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "dtype": torch.bfloat16,
        }
        return rope_scaling
    else:
        return None


class ScalingRotaryEmbedding(RotaryEmbedding):
    """Scale the RotaryEmbedding in a way similar to YaRN method. https://arxiv.org/pdf/2309.00071."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extra_method: str = "yarn_log",
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extra_method = extra_method
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        if self.extra_method in ["original"]:
            inv_freq = inv_freq_extrapolation
        elif self.extra_method in ["yarn", "yarn_linear"]:
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
        elif self.extra_method == "yarn_log":
            inv_freq = torch.exp(
                torch.log(inv_freq_extrapolation) * inv_freq_mask
                + torch.log(inv_freq_interpolation) * (1.0 - inv_freq_mask)
            )
        elif self.extra_method == "theta_scale":
            exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
            theta_scale_exponent = self.base ** (
                math.log(
                    self.max_position_embeddings * self.scaling_factor / (2 * math.pi)
                )
                / math.log(self.max_position_embeddings / (2 * math.pi))
            )
            inv_freq = torch.tensor(
                1.0 / (theta_scale_exponent ** (exponents / self.rotary_dim)),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extra_method}")
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # cos = freqs.cos() * self.mscale
        # sin = freqs.sin() * self.mscale
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class Grok1Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        alt_stream: Optional[torch.cuda.Stream] = None,
        load_presharded_attn: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        attn_tp_rank = get_tensor_model_parallel_rank()
        attn_tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = getattr(config, "head_dim", 128)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        rope_scaling = get_rope_scaling(config)
        self.load_presharded_attn = load_presharded_attn
        self.alt_stream = alt_stream or torch.cuda.Stream()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            load_presharded_attn=self.load_presharded_attn,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            use_presharded_weights=self.load_presharded_attn,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims", False)

        if rope_scaling is not None:
            self.rotary_emb = ScalingRotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                base=int(self.rope_theta),
                is_neox_style=True,
                **rope_scaling,
            )
            pos_encoding_mode = "NONE"
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )
            pos_encoding_mode = "NONE"

        logit_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)
        logit_capping_method = getattr(config, "attn_logit_softcapping_method", "tanh")

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            quant_config=quant_config,
            pos_encoding_mode=pos_encoding_mode,
            logit_capping_method=logit_capping_method,
            prefix=add_prefix("attn", prefix),
        )
        self.attn.xai_temperature_len = getattr(self.config, "attn_temperature_len", -1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states
        if debug_tensor_dump_output_folder:
            dump_to_file(
                debug_tensor_dump_output_folder,
                f"attn_input_{self.layer_id}",
                hidden_states,
            )

            if debug_tensor_dump_inject:
                name = os.path.join(
                    debug_tensor_dump_output_folder,
                    f"jax_dump_attn_input_{self.layer_id}.npy",
                )
                logger.info(f"Load {name} from jax.")
                x = np.load(name)
                hidden_states = torch.tensor(x[0, : hidden_states.shape[0]]).to(
                    hidden_states
                )

        qkv, _ = self.qkv_proj(hidden_states)
        dispose_tensor(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if debug_tensor_dump_output_folder:
            num_tokens = q.shape[0]
            num_heads_q = self.num_heads
            head_dim = self.head_dim
            num_heads_kv = k.numel() // (num_tokens * head_dim)

            dump_to_file(
                debug_tensor_dump_output_folder,
                f"q_{self.layer_id}",
                tensor_model_parallel_all_gather(
                    q.reshape(num_tokens, num_heads_q, head_dim).contiguous(), dim=1
                ).contiguous(),
            )
            dump_to_file(
                debug_tensor_dump_output_folder,
                f"k_{self.layer_id}",
                tensor_model_parallel_all_gather(
                    k.reshape(num_tokens, num_heads_kv, head_dim).contiguous(), dim=1
                ).contiguous(),
            )
            dump_to_file(
                debug_tensor_dump_output_folder,
                f"v_{self.layer_id}",
                tensor_model_parallel_all_gather(
                    v.reshape(num_tokens, num_heads_kv, head_dim).contiguous(), dim=1
                ).contiguous(),
            )

        attn_output = self.attn(q, k, v, forward_batch)
        del q, k, v, qkv

        if debug_tensor_dump_output_folder:
            dump_to_file(
                debug_tensor_dump_output_folder,
                f"attn_output_{self.layer_id}",
                tensor_model_parallel_all_gather(
                    attn_output.reshape(num_tokens, num_heads_q, head_dim).contiguous(),
                    dim=1,
                ).contiguous(),
            )

        output, _ = self.o_proj(attn_output)
        return output


class Grok1DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        load_presharded_moe: bool = False,
        load_presharded_attn: bool = False,
        load_presharded_mlp: bool = False,
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_moe: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.residual_moe = getattr(config, "residual_moe", False)
        self.layer_id = layer_id
        self.alt_stream = alt_stream or torch.cuda.Stream()

        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = Grok1Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=(
                config.context_len
                if hasattr(config, "context_len")
                else config.max_position_embeddings
            ),
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
            reduce_results=False,
            alt_stream=self.alt_stream,
            load_presharded_attn=load_presharded_attn,
            prefix=add_prefix("attn", prefix),
        )

        split_gate_up = not getattr(config, "merge_gate_up", True)
        if self.num_experts > 0:
            self.block_sparse_moe = Grok1MoE(
                config=config,
                layer_id=layer_id,
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=getattr(
                    config,
                    "moe_intermediate_size",
                    getattr(config, "intermediate_size", None),
                ),
                quant_config=quant_config,
                reduce_results=not self.residual_moe,
                use_presharded_weights=load_presharded_moe,
                inplace=False,  # not self.residual_moe,
                no_combine=False,  # self.residual_moe,  # just a suggestion to not combine topk
                prefix=add_prefix("block_sparse_moe", prefix),
            )
            if self.residual_moe:
                self.mlp = Grok1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    quant_config=quant_config,
                    reduce_results=False,
                    use_presharded_weights=load_presharded_mlp,
                    layer_id=layer_id,
                    split_gate_up=split_gate_up,
                )
        else:
            raise NotImplementedError()

        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.num_experts > 0:
            if self.residual_moe:
                # NOTE: self.block_sparse_moe modifies the input in-place,
                # so we have to call it later. Be aware of any possible related errors.
                if get_tensor_model_parallel_world_size() > 1:
                    self.ffn = lambda x: tensor_model_parallel_all_reduce(
                        self.moe_with_rmoe(x)
                    )
                else:
                    self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.block_sparse_moe
        else:
            raise NotImplementedError()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        deferred_norm: Optional[RMSNorm] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, RMSNorm]:

        hidden_states_original = hidden_states
        residual_original = residual

        # Self Attention
        if deferred_norm is not None:
            assert residual is not None
            # here hidden_states is output of ffn, residual is residual from after previous attn layer
            hidden_states, residual = fused_dual_residual_rmsnorm(
                hidden_states,
                residual,
                deferred_norm.weight,
                self.pre_attn_norm.weight,
                deferred_norm.variance_epsilon,
            )
        else:
            # here hidden_states is the residual
            hidden_states, residual = (
                fused_rmsnorm(
                    hidden_states,
                    self.pre_attn_norm.weight,
                    self.pre_attn_norm.variance_epsilon,
                ),
                hidden_states,
            )

        if residual_original is not None:
            dispose_tensor(residual_original)

        dispose_flag = False
        if residual is not hidden_states_original:
            dispose_flag = True
            dispose_tensor(hidden_states_original)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        hidden_states, residual = fused_dual_residual_rmsnorm(
            hidden_states,
            residual,
            self.post_attn_norm.weight,
            self.pre_moe_norm.weight,
            self.post_attn_norm.variance_epsilon,
        )

        if not dispose_flag:
            dispose_tensor(hidden_states_original)

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        return hidden_states, residual, self.post_moe_norm  # defer layernorm

    def moe_with_rmoe(self, x):
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        mlp_result = self.mlp(x)
        with torch.cuda.stream(self.alt_stream):
            # moe should not be inplace because of stream race condition
            moe_result = self.block_sparse_moe(x)
        current_stream.wait_stream(self.alt_stream)
        return (mlp_result + moe_result) / 1.4142135623730951


class Grok1Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        load_presharded_moe: bool = False,
        load_presharded_embedding: bool = False,
        load_presharded_attn: bool = False,
        load_presharded_mlp: bool = False,
        replicate_embedding: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            use_presharded_weights=load_presharded_embedding,
            enable_tp=not replicate_embedding,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                Grok1DecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    load_presharded_moe=load_presharded_moe,
                    load_presharded_attn=load_presharded_attn,
                    load_presharded_mlp=load_presharded_mlp,
                    alt_stream=self.alt_stream,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states.mul_(self.config.embedding_multiplier_scale)
        else:
            hidden_states = input_embeds

        residual, deferred_norm = None, None
        for i in range(len(self.layers)):
            hidden_states, residual, deferred_norm = self.layers[i](
                positions, hidden_states, forward_batch, residual, deferred_norm
            )

        if debug_tensor_dump_output_folder:
            hidden_states = (
                fused_rmsnorm(
                    hidden_states,
                    deferred_norm.weight,
                    deferred_norm.variance_epsilon,
                )
                + residual
            )

            dump_to_file(
                debug_tensor_dump_output_folder,
                "last_hidden_before_norm",
                hidden_states,
            )

            hidden_states = fused_rmsnorm(
                hidden_states,
                self.norm.weight,
                self.norm.variance_epsilon,
            )

            dump_to_file(
                debug_tensor_dump_output_folder,
                "last_hidden_after_norm",
                hidden_states,
            )
        else:
            hidden_states, _ = fused_dual_residual_rmsnorm(
                hidden_states,
                residual,
                deferred_norm.weight,
                self.norm.weight,
                deferred_norm.variance_epsilon,
            )

        return hidden_states


class Grok1ForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Get presharded weights.
        self.load_presharded_mlp = getattr(config, "load_presharded_mlp", False)
        self.load_presharded_moe = (
            getattr(config, "load_presharded_moe", True)
            and self.config.num_local_experts > 0
            and get_tensor_model_parallel_world_size() > 1
        )
        self.load_presharded_attn = getattr(config, "load_presharded_attn", False)
        self.load_presharded_embedding = getattr(
            config, "load_presharded_embedding", False
        )

        self.is_weights_presharded = (
            self.load_presharded_mlp
            or self.load_presharded_moe
            or self.load_presharded_attn
            or self.load_presharded_embedding
        )

        default_replicate_lm_head = False
        self.replicate_lm_head = getattr(
            config, "replicate_lm_head", default_replicate_lm_head
        )

        if self.is_weights_presharded:
            setattr(DefaultModelLoader, "_prepare_weights", _prepare_presharded_weights)

        self.replicate_embedding = getattr(config, "replicate_embedding", False)

        self.model = Grok1Model(
            config,
            quant_config=quant_config,
            load_presharded_moe=self.load_presharded_moe,
            load_presharded_embedding=self.load_presharded_embedding,
            load_presharded_attn=self.load_presharded_attn,
            load_presharded_mlp=self.load_presharded_mlp,
            replicate_embedding=self.replicate_embedding,
            prefix=add_prefix("model", prefix),
        )

        lm_head_params_dtype = None
        if self.replicate_lm_head:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=lm_head_params_dtype,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                use_presharded_weights=self.load_presharded_embedding,
                params_dtype=lm_head_params_dtype,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config)

        # Dump tensors for debugging
        global debug_tensor_dump_output_folder, debug_tensor_dump_inject
        debug_tensor_dump_output_folder = (
            get_global_server_args().debug_tensor_dump_output_folder
        )
        debug_tensor_dump_inject = get_global_server_args().debug_tensor_dump_inject
        warnings.filterwarnings("ignore", category=FutureWarning)

        if get_tensor_model_parallel_rank() == 0:
            logger.info(
                f"#parameters (analytical): {self.get_num_params_analytical() / 1e9:.2f} B, "
                f"#parameters (actual): {self.get_num_params_torch() / 1e9:.2f} B"
            )
        self.loaded_param_names = set()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if debug_tensor_dump_output_folder:
            dump_to_file(debug_tensor_dump_output_folder, "input_ids", input_ids)

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        ignore_parent_name: bool = False,
        check_hit_names: bool = True,
        model_config: PretrainedConfig | None = None,
    ) -> dict[str, torch.Tensor]:
        if model_config is None:
            model_config = self.config

        stacked_params_mapping = []
        stacked_params_mapping += [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        stacked_params_mapping += [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        num_experts = model_config.num_local_experts
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
        )

        params_dict = dict(self.named_parameters())
        all_names = set(params_dict.keys())
        hit_names = set()

        def load_weight_wrapper(
            name: str, loaded_weight: torch.Tensor, *args, **kwargs
        ):
            # Fuse constant multipliers into the weights
            if "lm_head" in name:
                loaded_weight = (
                    loaded_weight.to(torch.float32)
                    * model_config.output_multiplier_scale
                )

            original_name = name
            if ignore_parent_name:
                name = name.split(".")[-1]

            if name not in params_dict:
                logger.info(f"Skipping {name=} in load_weights_wrapper")
                return

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight, *args, **kwargs)
            hit_names.add(name)
            self.loaded_param_names.add(original_name)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                load_weight_wrapper(name, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    load_weight_wrapper(
                        name,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name is None:
                        continue

                    load_weight_wrapper(name=name, loaded_weight=loaded_weight)

        if check_hit_names:
            if len(hit_names) > 5:
                missing = all_names - hit_names
                missing_exclude_scales = {x for x in missing if "scale" not in x}
                logger.info(
                    f"#all_names: {len(all_names)}, #hit_names: {len(hit_names)}, #missing_exclude_scales: {len(missing_exclude_scales)}",
                )
                if len(missing_exclude_scales) > 0:
                    raise ValueError(
                        f"load_weights failed because some weights are missing: {missing_exclude_scales=}."
                    )

            elif len(hit_names) == 0:
                raise ValueError(
                    f"load_weights failed because it did not hit any names. {all_names=} {hit_names=}"
                )

        return hit_names

    def get_num_params_analytical(self):
        cfg = self.config
        moe_intermediate_size = getattr(
            cfg,
            "moe_intermediate_size",
            getattr(cfg, "intermediate_size", None),
        )
        residual_moe = getattr(cfg, "residual_moe", False)
        if cfg.num_local_experts > 0:
            num_experts = cfg.num_local_experts + (1 if residual_moe else 0)
        else:
            num_experts = 1

        wq = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        wkv = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_key_value_heads
            * cfg.head_dim
            * 2
        )
        out = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        ffn1 = (
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
            * 2
        )
        ffn2 = (
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
        )
        embed = cfg.hidden_size * cfg.vocab_size * 2
        return wq + wkv + out + ffn1 + ffn2 + embed

    def get_num_params_torch(self):
        return (
            sum(p.numel() for p in self.parameters())
            * get_tensor_model_parallel_world_size()
        )


old_prepare_weights = getattr(DefaultModelLoader, "_prepare_weights")


def _prepare_presharded_weights(
    self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
) -> Tuple[str, list[str], bool]:
    import glob
    import os

    if get_tensor_model_parallel_world_size() == 1:
        return old_prepare_weights(self, model_name_or_path, revision, fall_back_to_pt)

    if not os.path.isdir(model_name_or_path):
        from sglang.srt.model_loader.weight_utils import download_weights_from_hf

        allow_patterns = ["*.safetensors", "*.bin"]
        hf_folder = download_weights_from_hf(
            model_name_or_path,
            self.load_config.download_dir,
            allow_patterns,
            revision,
            ignore_patterns=self.load_config.ignore_patterns,
        )
    else:
        hf_folder = model_name_or_path

    tp_rank = get_tensor_model_parallel_rank()

    # The old format
    allow_patterns = [f"*-{tp_rank:03d}.bin"]

    # The new format
    allow_patterns += [f"*-TP-{tp_rank:03d}.safetensors", "*-TP-common.safetensors"]

    hf_weights_files = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))

    if hf_weights_files[0].endswith("safetensors"):
        use_safetensors = True
    else:
        use_safetensors = False

    return hf_folder, hf_weights_files, use_safetensors


class Grok1ModelForCausalLM(Grok1ForCausalLM):
    """An alias for backward-compatbility."""

    pass


EntryClass = [Grok1ForCausalLM, Grok1ModelForCausalLM]
