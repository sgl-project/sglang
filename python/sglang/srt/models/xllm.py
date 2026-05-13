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

# Adapted from qwen2_moe.py for xllm K2MoE architecture.
# Key differences from Qwen2Moe:
#   - Sigmoid routing (not softmax)
#   - Gate bias used for expert selection only (correction_bias pattern)
#   - Router scaling factor applied after renormalization
#   - No shared_expert_gate (shared expert output added directly)
#   - Dense layers specified via mlp_only_layers config
#   - Partial RoPE (rope_head_dim < head_dim)
"""Inference-only Xllm K2MoE model compatible with HuggingFace weights."""

import logging
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)

try:
    from sglang.srt.layers.communicator import enable_moe_dense_fully_dp
except ImportError:
    def enable_moe_dense_fully_dp():
        return getattr(get_global_server_args(), "moe_dense_tp_size", -1) == 1

from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm


class XllmGroupRMSNorm(nn.Module):
    """RMSNorm that normalizes each of `n_groups` groups independently.

    When n_groups=1 this is mathematically equivalent to standard RMSNorm.
    Matches the HF XllmRMSNorm implementation used by the xllm model family.
    """

    def __init__(self, hidden_size: int, n_groups: int = 1, eps: float = 1e-6):
        super().__init__()
        self.n_groups = n_groups
        self.hidden_size = hidden_size
        assert hidden_size % n_groups == 0
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states
        if post_residual_addition is not None:
            hidden_states = hidden_states + post_residual_addition
            residual = hidden_states
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states.reshape(
            *hidden_states.shape[:-1], self.n_groups, -1
        )
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
        hidden_states = (self.weight * hidden_states).to(orig_dtype)
        if residual is not None:
            return hidden_states, residual
        return hidden_states


def _make_norm(config):
    """Create the appropriate RMSNorm for this config."""
    n_groups = getattr(config, "layernorm_num_groups", 1)
    if n_groups is None or n_groups <= 1:
        return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    return XllmGroupRMSNorm(
        config.hidden_size, n_groups=n_groups, eps=config.rms_norm_eps
    )


from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
try:
    from sglang.srt.layers.moe.utils import RoutingMethodType
except ImportError:
    from enum import IntEnum

    class RoutingMethodType(IntEnum):
        Default = 0
        Renormalize = 1
        DeepSeekV3 = 2
        Llama4 = 3
        RenormalizeNaive = 4
        TopK = 5
        Unspecified = 6

try:
    from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
except ImportError:

    def filter_moe_weight_param_global_expert(name, x, num_local_experts):
        return (
            not getattr(x, "_sglang_require_global_experts", False)
            and x.data.ndim > 0
            and x.data.shape[0] == num_local_experts
        )
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    make_layers,
    use_intel_amx_backend,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()


def permute_to_xllm(x):
    """Interleave first half and second half: [0,1,...,63,64,...,127] -> [0,64,1,65,...,63,127]"""
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def permute_to_hf(x):
    """Inverse of permute_to_xllm: [0,64,1,65,...,63,127] -> [0,1,...,63,64,...,127]"""
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


class XllmMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class XllmMoEGate(nn.Module):
    """Router gate for xllm.

    Stores weight and bias separately. The bias is used as correction_bias
    for expert selection (added to sigmoid scores) but not in the linear
    computation of router logits.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.num_experts, config.hidden_size))
        )
        if getattr(config, "moe_gate_bias", False):
            # topk_sigmoid kernel requires correction_bias in float32
            self.bias = nn.Parameter(
                torch.empty(config.num_experts, dtype=torch.float32)
            )
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor):
        return F.linear(hidden_states, self.weight)


class XllmSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)

        self.gate = XllmMoEGate(config)

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
            scoring_func=getattr(config, "router_score_func", "sigmoid"),
            correction_bias=self.gate.bias,
        )

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
        )

        # Shared expert (no gating — output added directly)
        num_shared_experts = getattr(config, "num_shared_experts", 0)
        if num_shared_experts > 0:
            shared_intermediate_size = config.moe_intermediate_size * num_shared_experts
            self.shared_experts = XllmMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        else:
            self.shared_experts = None

        if get_moe_a2a_backend().is_deepep():
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.num_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        if self.shared_experts is not None:
            return self.shared_experts(hidden_states)
        return None

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        shared_output = None
        if hidden_states.shape[0] > 0:
            router_logits = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
            # Match SGLang's native MoE contract. The generic TopK.__call__ path
            # is numerically unsafe for long 375B MoE decodes.
            if hasattr(self.topk, "forward_native"):
                topk_output = self.topk.forward_native(hidden_states, router_logits)
            else:
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    num_token_non_padded=forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=(
                        ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
                    ),
                )
            # Apply router scaling factor after renormalization
            if self.router_scaling_factor != 1.0:
                scaled_weights = topk_output.topk_weights * self.router_scaling_factor
                if hasattr(topk_output, "_replace"):
                    topk_output = topk_output._replace(topk_weights=scaled_weights)
                else:
                    topk_output.topk_weights = scaled_weights
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            final_hidden_states.add_(shared_output)

        return final_hidden_states

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        router_logits = self.gate(hidden_states)
        if hasattr(self.topk, "forward_native"):
            topk_output = self.topk.forward_native(hidden_states, router_logits)
        else:
            topk_output = self.topk(hidden_states, router_logits)
        # Apply router scaling factor after renormalization
        # TopK output is a NamedTuple (immutable), so we must replace it
        if self.router_scaling_factor != 1.0:
            topk_output = topk_output._replace(
                topk_weights=topk_output.topk_weights * self.router_scaling_factor
            )
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)

        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)

        if shared_output is not None:
            final_hidden_states += shared_output
        if self.tp_size > 1 and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class XllmAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # Partial RoPE: xLLM/HF stores each head in neox ordering, where the
        # rotary dimensions are not contiguous when rope_head_dim < head_dim.
        # Mirror HF exactly: permute to interleaved, split rope/nope, apply RoPE
        # on the rope slice, then recombine and permute back.
        self.rope_head_dim = rope_head_dim
        self.use_xllm_partial_rope = rope_head_dim < head_dim
        self.rotary_emb = get_rope(
            self.rope_head_dim if self.use_xllm_partial_rope else self.head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
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

    def _apply_partial_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_heads = q.reshape(-1, self.num_heads, self.head_dim)
        k_heads = k.reshape(-1, self.num_kv_heads, self.head_dim)

        q_interleaved = permute_to_xllm(q_heads)
        k_interleaved = permute_to_xllm(k_heads)

        nope_dim = self.head_dim - self.rope_head_dim
        q_rope, q_nope = q_interleaved.split([self.rope_head_dim, nope_dim], dim=-1)
        k_rope, k_nope = k_interleaved.split([self.rope_head_dim, nope_dim], dim=-1)

        q_rope_flat = permute_to_hf(q_rope).reshape(
            -1, self.num_heads * self.rope_head_dim
        )
        k_rope_flat = permute_to_hf(k_rope).reshape(
            -1, self.num_kv_heads * self.rope_head_dim
        )
        q_rope_flat, k_rope_flat = self.rotary_emb(
            positions, q_rope_flat, k_rope_flat
        )

        q_rope = permute_to_xllm(
            q_rope_flat.reshape(-1, self.num_heads, self.rope_head_dim)
        )
        k_rope = permute_to_xllm(
            k_rope_flat.reshape(-1, self.num_kv_heads, self.rope_head_dim)
        )

        q = permute_to_hf(torch.cat([q_rope, q_nope], dim=-1)).reshape(
            -1, self.num_heads * self.head_dim
        )
        k = permute_to_hf(torch.cat([k_rope, k_nope], dim=-1)).reshape(
            -1, self.num_kv_heads * self.head_dim
        )
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_xllm_partial_rope:
            q, k = self._apply_partial_rope(positions, q, k)
        else:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class XllmDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        qkv_bias = getattr(config, "attention_bias", False)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_head_dim = getattr(config, "rope_head_dim", head_dim)

        self.self_attn = XllmAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            qkv_bias=qkv_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Determine if this layer is sparse (MoE) or dense
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        if (layer_id not in mlp_only_layers) and (
            config.num_experts > 0
            and (layer_id + 1) % decoder_sparse_step == 0
        ):
            self.is_layer_sparse = True
        else:
            self.is_layer_sparse = False

        # Check neighbors for scatter modes
        def _is_sparse(lid):
            if lid < 0 or lid >= config.num_hidden_layers:
                return False
            return (lid not in mlp_only_layers) and (
                config.num_experts > 0
                and (lid + 1) % decoder_sparse_step == 0
            )

        is_previous_layer_sparse = _is_sparse(layer_id - 1)
        is_next_layer_sparse = _is_sparse(layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = XllmSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = XllmMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = _make_norm(config)
        self.post_attention_layernorm = _make_norm(config)
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                **kwargs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if isinstance(self.mlp, XllmMLP):
            hidden_states = self.mlp(hidden_states, use_reduce_scatter=use_reduce_scatter)
        else:
            hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class XllmModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: XllmDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = _make_norm(config)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                _server_args = get_global_server_args()
                _disable_pcg = getattr(
                    _server_args, "disable_piecewise_cuda_graph",
                    not getattr(_server_args, "enable_piecewise_cuda_graph", True),
                )
                ctx = (
                    nullcontext()
                    if not _disable_pcg
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class XllmForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        # Keep the 375B MoE path single-stream. CUDA graph capture is still
        # allowed, but the capture-mode dual-stream path adds async
        # shared-expert/router risk without being required for correctness.
        alt_stream = None
        self.model = XllmModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if self.pp_group.is_last_rank:
            logits_output = self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
            if logits_output.next_token_logits is not None:
                logits_output.next_token_logits = torch.nan_to_num(
                    logits_output.next_token_logits, nan=0.0, posinf=65504.0, neginf=-65504.0
                )
            return logits_output
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
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
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip experts (handled below in expert_params_mapping)
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        if getattr(config, "num_experts", 0) <= 0:
            return None
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = XllmForCausalLM
