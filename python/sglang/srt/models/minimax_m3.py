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

# Adapted from DeepSeek and Mixtral implementation
"""Inference-only MiniMax M3 model compatible with HuggingFace weights."""

import logging
from contextlib import nullcontext
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.configs.model_config import (
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
)
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.minimax_m2 import MiniMaxM2RMSNormTP
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    get_device_sm,
    is_cuda,
    is_hip,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_cuda = is_cuda()
_is_hip = is_hip()
_device_sm = get_device_sm()

_has_rocm_qk_norm_rope = False
if _is_hip:
    try:
        from sglang.jit_kernel.minimax_m3.qk_norm_rope import qk_gemma_rmsnorm_rope

        _has_rocm_qk_norm_rope = True
    except ImportError:
        _has_rocm_qk_norm_rope = False

logger = logging.getLogger(__name__)


class MultiHeadRMSNorm(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        apply_layernorm_1p: bool = False,
    ) -> None:
        super().__init__()
        self.tp_world = get_attention_tp_size()
        self.tp_rank = get_attention_tp_rank()
        self.num_heads = num_heads
        self.num_heads_per_tp = num_heads // self.tp_world
        self.head_dim = head_dim
        self.weight = nn.Parameter(
            torch.ones(self.num_heads_per_tp, self.head_dim, dtype=torch.float32)
        )
        self.weight.weight_loader = self.weight_loader
        self.apply_layernorm_1p = apply_layernorm_1p
        self.variance_epsilon = eps

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_attention_tp_size()
        tp_rank = get_attention_tp_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard].reshape_as(param))

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.view(-1, self.num_heads_per_tp, self.head_dim).to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.apply_layernorm_1p:
            x = x * (self.weight + 1)[None, ...]
        else:
            x = x * self.weight[None, ...]
        x = x.view(-1, self.num_heads_per_tp * self.head_dim)
        return x.to(orig_dtype)


class MiniMaxM3MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        intermediate_size: int = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        hidden_act = config.hidden_act

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
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "swigluoai":
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                swiglu_no_interleaved_with_alpha_and_limit,
            )

            self.act_fn = lambda x: swiglu_no_interleaved_with_alpha_and_limit(
                x, config.swiglu_alpha, config.swiglu_limit
            )
        else:
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


class MiniMaxM3MoE(nn.Module):
    """MiniMax MoE implementation using DeepEP for Expert Parallel support."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )

        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_local_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            activation="silu",
            is_gated=True,
            gemm1_alpha=config.swiglu_alpha,
            gemm1_clamp_limit=config.swiglu_limit,
            prefix=add_prefix("experts", prefix),
            interleaved=False,
        )
        # use sigmoid_topk, instead of grouped_topk
        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            renormalize=True,
            layer_id=layer_id,
            scoring_func=config.scoring_func,
            correction_bias=self.e_score_correction_bias,
            num_fused_shared_experts=self.num_fused_shared_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
        )

        if self.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.intermediate_size * self.n_shared_experts
            # Under DeepEP the layer output is all-gathered, not all-reduced, so a
            # TP-sharded shared MLP (reduce_results=False) would leave an unreduced
            # partial. Replicate it (tp_size=1) for a complete output, like GLM4 / DSV2.
            shared_experts_tp1 = get_moe_a2a_backend().is_deepep()
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                reduce_results=False,
                intermediate_size=intermediate_size,
                **(dict(tp_rank=0, tp_size=1) if shared_experts_tp1 else {}),
            )
        else:
            self.shared_experts = None

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.layer_id = layer_id

        if get_moe_a2a_backend().is_deepep():
            self.ep_size = get_moe_expert_parallel_world_size()
            self.top_k = config.num_experts_per_tok

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if get_moe_a2a_backend().is_deepep():
            return self.forward_deepep(hidden_states, forward_batch)
        else:
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            shared_output = self._forward_shared_experts(hidden_states)
            router_logits, _ = self.gate(hidden_states.to(torch.float32))
            topk_output = self.topk(hidden_states, router_logits)
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1 and not should_allreduce_fusion and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        if hidden_states.shape[0] > 0:
            shared_output = self._forward_shared_experts(hidden_states)
            router_logits, _ = self.gate(hidden_states.to(torch.float32))
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        # DeepEPMoE returns the complete per-token routed result (no TP all-reduce
        # here, unlike forward_normal), and the shared experts are replicated
        # (tp_size=1, see __init__), so both are complete per token and add directly.
        final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(hidden_states)
        else:
            return None


class MiniMaxM3Attention(nn.Module):
    """MiniMax Attention implementation with QK normalization and partial RoPE.

    Supports two modes selected by ``is_sparse_attention_layer``:

    * Dense (default): standard QKV attention.
    * Sparse: extra "index" branch (``index_q/k/v_proj`` + ``index_o_proj``)
      whose outputs flow through the MiniMax sparse attention backend and
      are summed into the dense output. Sparse layers must run under
      ``MiniMaxHybridAttnBackend``, which dispatches per-layer based on
      ``get_minimax_sparse_layer_ids``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_sparse_attention_layer: bool = False,
        disable_index_value: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_sparse_attention_layer = is_sparse_attention_layer
        self.disable_index_value = is_sparse_attention_layer and disable_index_value

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.attn_tp_size = attn_tp_size
        self.attn_tp_rank = attn_tp_rank

        # Get dimensions from config
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        # Use head_dim from config if available, otherwise calculate
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # RoPE settings - support partial RoPE
        self.rope_theta, self.rope_scaling = get_rope_config(config)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rotary_dim = getattr(
            config, "rotary_dim", self.head_dim
        )  # MiniMax uses rotary_dim=64

        # QK Normalization settings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.qk_norm_type = getattr(config, "qk_norm_type", "per_layer")
        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)
        self.attention_output_gate = getattr(config, "attention_output_gate", False)

        # Sparse-attention-specific config (read from sparse_attention_config).
        # Only the index-branch dimensions are needed at module level; all
        # block/topk/local/init knobs live in the sparse attention backend.
        if self.is_sparse_attention_layer:
            assert self.qk_norm_type == "per_head", (
                f"sparse attention only supports qk_norm_type='per_head', "
                f"got {self.qk_norm_type!r}"
            )
            assert (
                not self.attention_output_gate
            ), "sparse attention does not support attention_output_gate"
            sparse_cfg = config.sparse_attention_config
            self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
            self.idx_head_dim = sparse_cfg["sparse_index_dim"]
            # Index heads use a GQA-style sharding (mirrors KV head replication
            # for num_kv_heads < tp_size). When tp_size > total_idx_heads, each
            # rank holds one head and `idx_replica_size` ranks share the same
            # head; idx_o is divided by idx_replica_size in forward_core so the
            # layer-level all-reduce sums to the correct value (done on the
            # activation rather than the weight to remain quantization-safe).
            if self.total_idx_heads >= attn_tp_size:
                assert self.total_idx_heads % attn_tp_size == 0
            else:
                assert attn_tp_size % self.total_idx_heads == 0
            self.idx_head_tp_size = min(attn_tp_size, self.total_idx_heads)
            self.idx_replica_size = attn_tp_size // self.idx_head_tp_size
            self.idx_head_rank = attn_tp_rank // self.idx_replica_size
            self.num_idx_heads = self.total_idx_heads // self.idx_head_tp_size

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attention_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        # Setup RoPE with partial rotary dimension
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,  # Use partial rotary dimension
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

        # Index branch (sparse-only): an additional QKV+O projection that
        # feeds the sparse attention backend's index path. Constructed only
        # when this layer is sparse so dense layers / dense models stay
        # parameter-identical to the original implementation.
        if self.is_sparse_attention_layer:
            # Always use Column/RowParallel for index_q_proj / index_o_proj
            # along the idx-head TP group. When idx_replica_size > 1 (i.e.
            # total_idx_heads < attn_tp_size), multiple ranks in a replica
            # group load the same per-head weight slice (GQA-style); the
            # double-count introduced by the layer-level all-reduce is
            # compensated by dividing idx_o by idx_replica_size in
            # forward_core (quantization-safe runtime scaling).
            self.index_q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_idx_heads * self.idx_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("index_q_proj", prefix),
                tp_rank=self.idx_head_rank,
                tp_size=self.idx_head_tp_size,
            )
            self.index_k_proj = ReplicatedLinear(
                self.hidden_size,
                self.idx_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("index_k_proj", prefix),
            )

            if self.disable_index_value:
                self.index_v_proj = None
                self.index_o_proj = None
            else:
                self.index_v_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.idx_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("index_v_proj", prefix),
                )
                self.index_o_proj = RowParallelLinear(
                    self.total_idx_heads * self.idx_head_dim,
                    self.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    reduce_results=False,
                    quant_config=quant_config,
                    prefix=add_prefix("index_o_proj", prefix),
                    tp_rank=self.idx_head_rank,
                    tp_size=self.idx_head_tp_size,
                )
            self.index_rotary_emb = self.rotary_emb

        # QK Normalization layers
        # Use RMSNormTP for proper tensor parallel support
        # Use total dimensions (before TP sharding) for correct normalization
        if self.qk_norm_type == "per_layer":
            if attn_tp_size > 1:
                self.q_norm = MiniMaxM2RMSNormTP(
                    self.total_num_heads * self.head_dim, eps=config.rms_norm_eps
                )
                self.k_norm = MiniMaxM2RMSNormTP(
                    self.total_num_kv_heads * self.head_dim, eps=config.rms_norm_eps
                )
            else:
                self.q_norm = RMSNorm(
                    self.total_num_heads * self.head_dim, eps=config.rms_norm_eps
                )
                self.k_norm = RMSNorm(
                    self.total_num_kv_heads * self.head_dim, eps=config.rms_norm_eps
                )
        elif self.qk_norm_type == "per_head":
            if self.use_gemma_norm:
                self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            else:
                self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if self.is_sparse_attention_layer:
                if self.use_gemma_norm:
                    self.index_q_norm = GemmaRMSNorm(
                        self.idx_head_dim, eps=config.rms_norm_eps
                    )
                    self.index_k_norm = GemmaRMSNorm(
                        self.idx_head_dim, eps=config.rms_norm_eps
                    )
                else:
                    self.index_q_norm = RMSNorm(
                        self.idx_head_dim, eps=config.rms_norm_eps
                    )
                    self.index_k_norm = RMSNorm(
                        self.idx_head_dim, eps=config.rms_norm_eps
                    )
        elif self.qk_norm_type == "multi_head":
            self.q_norm = MultiHeadRMSNorm(
                self.total_num_heads,
                self.head_dim,
                eps=config.rms_norm_eps,
                apply_layernorm_1p=self.use_gemma_norm,
            )
            self.k_norm = MultiHeadRMSNorm(
                self.total_num_kv_heads,
                self.head_dim,
                eps=config.rms_norm_eps,
                apply_layernorm_1p=self.use_gemma_norm,
            )
        else:
            raise ValueError(f"Invalid qk_norm_type: {self.qk_norm_type}")

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _can_use_rocm_qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> bool:
        return (
            _has_rocm_qk_norm_rope
            and self.qk_norm_type == "per_head"
            and self.use_gemma_norm
            and not self.attention_output_gate
            and positions.dim() == 1
            and q.dim() == 2
            and k.dim() == 2
            and q.dtype in (torch.bfloat16, torch.float16)
            and k.dtype == q.dtype
            and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
            and hasattr(self.rotary_emb, "cos_sin_cache")
            and self.rotary_emb.rotary_dim == self.rotary_dim
            and self.rotary_dim <= self.head_dim
        )

    def _qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._can_use_rocm_qk_norm_rope(positions, q, k):
            return qk_gemma_rmsnorm_rope(
                q,
                k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.q_norm.variance_epsilon,
                self.head_dim,
                self.rotary_dim,
                self.rotary_emb.is_neox_style,
            )
        q, k = self._qk_norm(q, k)
        return self.rotary_emb(positions, q, k)

    def _qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm_type == "per_layer":
            if self.attn_tp_size > 1:
                q, k = MiniMaxM2RMSNormTP.forward_qk(
                    self.q_norm, self.k_norm, q.contiguous(), k.contiguous()
                )
            else:
                q = self.q_norm(q.contiguous())
                k = self.k_norm(k.contiguous())
        elif self.qk_norm_type == "per_head":
            q_shape = q.shape
            k_shape = k.shape
            q = q.reshape(-1, self.head_dim).contiguous()
            k = k.reshape(-1, self.head_dim).contiguous()
            q = self.q_norm(q).reshape(q_shape)
            k = self.k_norm(k).reshape(k_shape)
        elif self.qk_norm_type == "multi_head":
            q = self.q_norm(q.contiguous())
            k = self.k_norm(k.contiguous())
        else:
            raise ValueError(f"Invalid qk_norm_type: {self.qk_norm_type}")
        return q, k

    def _index_qk_norm_rope(
        self, positions: torch.Tensor, idx_q: torch.Tensor, idx_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            _has_rocm_qk_norm_rope
            and self.use_gemma_norm
            and positions.dim() == 1
            and idx_q.dim() == 2
            and idx_k.dim() == 2
            and idx_q.dtype in (torch.bfloat16, torch.float16)
            and idx_k.dtype == idx_q.dtype
            and self.index_q_norm.variance_epsilon == self.index_k_norm.variance_epsilon
            and hasattr(self.index_rotary_emb, "cos_sin_cache")
            and self.index_rotary_emb.rotary_dim == self.rotary_dim
            and self.rotary_dim <= self.idx_head_dim
        ):
            return qk_gemma_rmsnorm_rope(
                idx_q,
                idx_k,
                self.index_q_norm.weight.data,
                self.index_k_norm.weight.data,
                positions,
                self.index_rotary_emb.cos_sin_cache,
                self.index_q_norm.variance_epsilon,
                self.idx_head_dim,
                self.rotary_dim,
                self.index_rotary_emb.is_neox_style,
            )
        idx_q, idx_k = self._index_qk_norm(idx_q, idx_k)
        return self.index_rotary_emb(positions, idx_q, idx_k)

    def _index_qk_norm(
        self, idx_q: torch.Tensor, idx_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # sparse index branch only supports per_head norm; init asserts this.
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape
        idx_q = idx_q.reshape(-1, self.idx_head_dim)
        idx_k = idx_k.reshape(-1, self.idx_head_dim)
        idx_q = self.index_q_norm(idx_q).reshape(idx_q_shape)
        idx_k = self.index_k_norm(idx_k).reshape(idx_k_shape)
        return idx_q, idx_k

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attention_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(
                *orig_shape, -1, self.num_heads_per_group, self.head_dim
            )
            q = q_gate[..., ::2, :, :]
            gate = q_gate[..., 1::2, :, :]
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q, k = self._qk_norm_rope(positions, q, k)

        if self.is_sparse_attention_layer:
            idx_q, _ = self.index_q_proj(hidden_states)
            idx_k, _ = self.index_k_proj(hidden_states)
            if self.disable_index_value:
                idx_v = None
            else:
                idx_v, _ = self.index_v_proj(hidden_states)
            idx_q, idx_k = self._index_qk_norm_rope(positions, idx_q, idx_k)
            inner_state = (q, k, v, gate, idx_q, idx_k, idx_v, forward_batch)
        else:
            inner_state = (q, k, v, gate, forward_batch)
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        _, _, inner_state = intermediate_state

        if self.is_sparse_attention_layer:
            q, k, v, gate, idx_q, idx_k, idx_v, forward_batch = inner_state
            # The sparse attention backend expects 3D shapes; the dense
            # backend accepts 2D q (it reshapes k/v internally). The shapes
            # are equivalent flat-vs-grouped views, see RadixAttention.forward.
            q = q.view(q.shape[0], self.num_heads, self.head_dim)
            k = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
            v = v.view(v.shape[0], self.num_kv_heads, self.head_dim)
            idx_q = idx_q.view(idx_q.shape[0], self.num_idx_heads, self.idx_head_dim)
            idx_k = idx_k.view(idx_k.shape[0], 1, self.idx_head_dim)
            if idx_v is not None:
                idx_v = idx_v.view(idx_v.shape[0], 1, self.idx_head_dim)
            idx_o, attn_output = self.attn(
                q, k, v, forward_batch, idx_q=idx_q, idx_k=idx_k, idx_v=idx_v
            )
            output, _ = self.o_proj(attn_output)
            if self.disable_index_value:
                return output
            # When idx_replica_size > 1, `idx_replica_size` ranks share the
            # same idx head and produce identical idx_o. After the layer-level
            # all-reduce sums those duplicates, each head's contribution would
            # be multiplied by idx_replica_size. Pre-dividing idx_o here keeps
            # the post-reduce sum correct and is quantization-agnostic
            # (unlike scaling the o_proj weight, which would re-quantize FP8).
            if self.idx_replica_size > 1:
                idx_o = idx_o / self.idx_replica_size
            idx_output, _ = self.index_o_proj(idx_o)
            return output + idx_output

        q, k, v, gate, forward_batch = inner_state
        attn_output = self.attn(q, k, v, forward_batch)
        if self.attention_output_gate:
            gate = torch.sigmoid(gate.float())
            attn_output = (attn_output * gate).to(attn_output.dtype)
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


class MiniMaxM3DecoderLayer(nn.Module):
    """MiniMax Decoder Layer implementation with MoE support.

    The attention block can be either dense or sparse depending on
    ``config.sparse_attention_config``:

    * If ``sparse_attention_config`` is None (or absent), all layers run
      dense attention -- behavior identical to the original M3 model.
    * If present, the per-layer dense/sparse split is read from
      ``sparse_attention_config['sparse_attention_freq']`` (the same
      source consulted by ``MiniMaxHybridAttnBackend``), so the model and
      the attention backend always agree on which layers are sparse.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        sparse_attention_config = getattr(config, "sparse_attention_config", None)
        if sparse_attention_config is not None:
            _, sparse_layer_ids = get_minimax_sparse_layer_ids(sparse_attention_config)
            is_sparse_attention_layer = layer_id in sparse_layer_ids
            disable_value_layer_ids = set(
                get_minimax_sparse_disable_value_layer_ids(sparse_attention_config)
            )
            disable_index_value = layer_id in disable_value_layer_ids
        else:
            is_sparse_attention_layer = False
            disable_index_value = False

        self.self_attn = MiniMaxM3Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            is_sparse_attention_layer=is_sparse_attention_layer,
            disable_index_value=disable_index_value,
        )

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        # ``is_layer_sparse`` here means "this layer's MLP is a sparse MoE",
        # not anything about attention sparsity. The name is kept (instead of
        # the clearer ``is_layer_moe``) to match the convention used by the
        # rest of sglang -- ``OperationsStrategy``, ``LayerScatterModes``,
        # ``LayerCommunicator``, ``gpt_oss``, ``falcon_h1`` etc all access
        # ``layer.is_layer_sparse``.
        self.is_layer_sparse = (
            moe_layer_freq[layer_id] != 0 if moe_layer_freq is not None else True
        )

        if self.is_layer_sparse:
            self.mlp = MiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = MiniMaxM3MLP(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                intermediate_size=config.dense_intermediate_size,
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)
        if self.use_gemma_norm:
            self.input_layernorm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        def _is_layer_sparse(lid):
            if moe_layer_freq is None:
                return True
            if lid < 0 or lid >= config.num_hidden_layers:
                return True
            return moe_layer_freq[lid] != 0

        is_previous_layer_sparse = _is_layer_sparse(layer_id - 1)
        is_next_layer_sparse = _is_layer_sparse(layer_id + 1)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
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

        # Fully Connected (MLP or MoE)
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if self.is_layer_sparse or hidden_states.shape[0] != 0:
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                should_allreduce_fusion,
                use_reduce_scatter,
            )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class MiniMaxM3Model(nn.Module):
    """MiniMax Model implementation."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.use_gemma_norm = getattr(config, "use_gemma_norm", False)

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def layer_fn(idx, prefix: str) -> nn.Module:
            return MiniMaxM3DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            layer_fn,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            if self.use_gemma_norm:
                self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            else:
                self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors, Tuple[torch.Tensor, list[torch.Tensor]]]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                embeds = self.get_input_embeddings()
                hidden_states = embeds(input_ids)
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
                ctx = (
                    nullcontext()
                    if not get_global_server_args().disable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions=positions,
                        forward_batch=forward_batch,
                        hidden_states=hidden_states,
                        residual=residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if hidden_states.shape[0] != 0:
            if residual is not None:
                hidden_states, _ = self.norm(hidden_states, residual)
            else:
                hidden_states = self.norm(hidden_states)

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class MiniMaxM3SparseForCausalLM(nn.Module):
    """MiniMax M3 model for causal language modeling.

    Always loaded as the mixed sparse/dense backbone: which layers are sparse
    vs dense is decided by ``config.sparse_attention_config`` (see
    ``MiniMaxM3DecoderLayer`` and ``MiniMaxHybridAttnBackend``). A checkpoint
    that omits ``sparse_attention_config`` will produce a pure-dense model.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.num_fused_shared_experts = 0
        self.determine_num_fused_shared_experts()

        self.model = MiniMaxM3Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )

            self.logits_processor = LogitsProcessor(config)
        else:
            self.lm_head = PPMissingLayer()

        # For EAGLE3
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def determine_num_fused_shared_experts(self):
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if not getattr(self.config, "n_shared_experts", None):
            disable_reason = "No shared experts are defined in the config."
        elif not _is_cuda:
            disable_reason = "Shared experts fusion currently requires CUDA devices."
        elif _is_cuda and (_device_sm is not None) and (_device_sm < 80):
            disable_reason = "Shared experts fusion requires SM80 or newer GPUs."
        elif get_moe_expert_parallel_world_size() > 1:
            disable_reason = "Shared experts fusion is not supported together with expert parallelism yet."
        elif get_moe_a2a_backend().is_deepep():
            disable_reason = "Shared experts fusion is not supported when Deepep MoE backend is enabled."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts
        assert (
            self.num_fused_shared_experts == 1
        ), "Only 1 fused shared expert is supported for Glm4MoeForCausalLM"
        log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # _print_tensor_info(input_ids, "input_ids")
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights with proper mapping for MiniMax architecture."""

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # Leading "." prevents falsely matching the sparse-attention
            # ``index_q_proj`` / ``index_k_proj`` / ``index_v_proj`` weights,
            # which contain ``q_proj`` / ``k_proj`` / ``v_proj`` as substrings.
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts + self.num_fused_shared_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.model.start_layer or layer_id >= self.model.end_layer
            ):
                continue

            # MiniMax-M3 checkpoints name MoE blocks "block_sparse_moe", while
            # this implementation exposes the same module as layer.mlp to match
            # SGLang's sparse-layer conventions.
            name = name.replace(".block_sparse_moe", ".mlp")

            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                # Map shared expert weights to the last expert slot
                # Shared expert becomes expert ID = n_routed_experts
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.num_local_experts}",
                )
                name = name.replace("gate_proj", "w1")
                name = name.replace("down_proj", "w2")
                name = name.replace("up_proj", "w3")

            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

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
                if "mlp.experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        continue

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
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        try:
                            weight_loader(param, loaded_weight)
                        except Exception as e:
                            logger.warning(f"Error loading weight {name}: {e}")
                            continue
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> Optional[int]:
    # M3 checkpoints emit MTP weights as ``model.mtp.layers.{i}.*``. Treat
    # them as spec-layer so the weight loader skips them cleanly when no
    # NextN module is built. Names not covered here fall through to the
    # catch-all and stay visible as warnings.
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.mtp.layers.{i}."):
                return config.num_hidden_layers + i
    return None


EntryClass = [MiniMaxM3SparseForCausalLM]
