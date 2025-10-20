# Copyright 2025-2026 SGLang Team
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

"""Inference-only GLM-4.5, GLM-4.6 model compatible with HuggingFace weights"""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_hip,
    log_info_on_rank0,
    use_intel_amx_backend,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()

if _is_cuda:
    from sgl_kernel import dsv3_router_gemm
elif _is_cpu and _is_cpu_amx_available:
    pass

logger = logging.getLogger(__name__)


class Glm4MoeMLP(nn.Module):
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
        self.tp_size = tp_size

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
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion=False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, skip_all_reduce=should_allreduce_fusion)
        return x


class Glm4MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        partial_rotary_factor: float = 0.5,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-05,
        attention_bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        use_qk_norm: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
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
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.use_qk_norm = use_qk_norm
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class Glm4MoeGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((config.n_routed_experts), dtype=torch.float32)
        )
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])

    def forward(self, hidden_states):
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states,
                self.weight,
                None,  # bias
                True,  # is_vnni
            )

        # NOTE: For some unknown reason, router_gemm seems degrade accept length.
        if (
            _is_cuda
            and not self.is_nextn
            and hidden_states.shape[0] < 4
            and hidden_states.shape[1] == 7168
            and self.weight.shape[0] == 256
            and _device_sm >= 90
        ):
            logits = dsv3_router_gemm(hidden_states, self.weight).to(
                hidden_states.dtype
            )
        else:
            logits = F.linear(hidden_states, self.weight, None)

        return logits


class Glm4MoeSparseMoeBlock(DeepseekV2MoE):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = Glm4MoeGate(
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        # self.shared_experts_weight_block_size = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(dict(tp_rank=0, tp_size=1) if self.ep_size > 1 else {}),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            )
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )

        self.top_k = config.num_experts_per_tok

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
        )

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states)

        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(hidden_states, topk_output)
            if not _is_cuda:
                final_hidden_states *= self.routed_scaling_factor
        current_stream.wait_stream(self.alt_stream)

        if self.ep_size > 1:
            if (
                self.tp_size > 1
                and not should_allreduce_fusion
                and not use_reduce_scatter
            ):
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
            final_hidden_states += shared_output
        else:
            final_hidden_states += shared_output
            if (
                self.tp_size > 1
                and not should_allreduce_fusion
                and not use_reduce_scatter
            ):
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
        return final_hidden_states

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states, should_allreduce_fusion)

        shared_output = self._forward_shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        if not _is_cuda and not _use_aiter:
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor
        if self.ep_size > 1:
            if self.tp_size > 1 and not should_allreduce_fusion:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
            if shared_output is not None:
                final_hidden_states += shared_output
        else:
            if shared_output is not None:
                final_hidden_states += shared_output
            if self.tp_size > 1 and not should_allreduce_fusion:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
        return final_hidden_states


class Glm4MoeDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.layer_id = layer_id
        self.self_attn = Glm4MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            use_qk_norm=config.use_qk_norm,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)

        num_layers = 1 if is_nextn else config.num_hidden_layers
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=num_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Glm4MoeSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Glm4MoeModel(DeepseekV2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
        )
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.layers = nn.ModuleList(
            [
                Glm4MoeDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.pp_group = get_pp_group()
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm4MoeForCausalLM(DeepseekV2ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config.moe_layer_freq = 1
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.determine_num_fused_shared_experts("Glm4MoeForCausalLM")
        self.model = Glm4MoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )

    def determine_num_fused_shared_experts(
        self, architecture: str = "Glm4MoeForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if (
            not _is_cuda
            or torch.cuda.get_device_capability("cuda") < (8, 0)
            or self.config.architectures[0] != architecture
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Only GLM-4.5 or GLM-4.6 on NV-platform with capability >= 80 can use shared experts fusion optimization."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = "Deepseek and GLM-4.5 or GLM-4.6 can not use shared experts fusion optimization under expert parallelism."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            weights_list = list(weights)
            weights_dict = dict(weights_list)
            if self.quant_config is not None:
                if self.quant_config.get_name() == "w8a8_int8":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif (
                    self.quant_config.get_name() == "fp8"
                    or self.quant_config.get_name() == "blockwise_int8"
                    or self.quant_config.get_name() == "compressed_tensors"
                ):
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                    ]
                elif self.quant_config.get_name() == "awq":
                    suffix_list = [
                        "down_proj.qweight",
                        "down_proj.qzeros",
                        "down_proj.scales",
                        "gate_proj.qweight",
                        "gate_proj.qzeros",
                        "gate_proj.scales",
                        "up_proj.qweight",
                        "up_proj.qzeros",
                        "up_proj.scales",
                    ]
                elif self.quant_config.get_name() == "modelopt_fp4":
                    suffix_list = [
                        "down_proj.weight",
                        "down_proj.weight_scale",
                        "down_proj.weight_scale_2",
                        "down_proj.input_scale",
                        "gate_proj.weight",
                        "gate_proj.weight_scale",
                        "gate_proj.weight_scale_2",
                        "gate_proj.input_scale",
                        "up_proj.weight",
                        "up_proj.weight_scale",
                        "up_proj.weight_scale_2",
                        "up_proj.input_scale",
                    ]
                else:
                    raise ValueError(
                        f"Unsupported shared expert fusion for quantization: {self.quant_config.get_name()}."
                    )
            else:
                suffix_list = [
                    "down_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                ]
            names_to_remove = []

            moe_layers = (
                range(
                    self.config.first_k_dense_replace,
                    self.config.num_hidden_layers,
                    self.config.moe_layer_freq,
                )
                if not is_nextn
                else [nextn_layer_id]
            )

            for moe_layer in moe_layers:
                for suffix in suffix_list:
                    shared_expert_weight_name = (
                        f"model.layers.{moe_layer}.mlp.shared_experts.{suffix}"
                    )
                    # online fp8 quantization does not load weight_scale
                    if shared_expert_weight_name not in weights_dict:
                        continue
                    weights_list.append(
                        (
                            f"model.layers.{moe_layer}."
                            f"mlp.experts."
                            f"{self.config.n_routed_experts + 0}"
                            f".{suffix}",
                            weights_dict[shared_expert_weight_name],
                        )
                    )
                    names_to_remove += [shared_expert_weight_name]
            weights = [w for w in weights_list if w[0] not in names_to_remove]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        params_dict = dict(self.named_parameters())
        weight_names = []
        for name, loaded_weight in weights:
            weight_names.append(name)

            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
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
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        # When both q_a_proj and kv_a_proj_with_mqa has been cached, load the fused weight to parameter
                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=0
                            )
                            param_name = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                                )
                            )
                            param = params_dict[param_name]

                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        if (
                            "k_scale" in name or "v_scale" in name
                        ) and name not in params_dict:
                            # modelopt attn kv scale is named differently
                            if any(scale in name for scale in ["k_scale", "v_scale"]):
                                name = name.replace("_proj", "attn_mqa")
                            else:
                                logger.warning(
                                    f"Unknown scale found in checkpoint: {name}"
                                )
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)


EntryClass = [Glm4MoeForCausalLM]
