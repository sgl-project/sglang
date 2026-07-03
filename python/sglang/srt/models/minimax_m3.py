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
from sglang.srt.environ import envs
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
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
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
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
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
    is_npu,
    log_info_on_rank0,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_device_sm = get_device_sm()

# fp8 main-K/V cache dtypes (index cache always stays bf16). When the sparse
# pool is one of these, the bf16-only qknorm+rope+kv-insert fusion is skipped so
# the backend's set_kv_buffer performs the bf16->fp8 cache write instead.
_FP8_KV_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
)

# rotary_dim required by the fused qknorm+rope JIT kernel: rotary_dim/2 must
# equal the CUDA warp size (32) so each warp norms+ropes one head in one pass.
_M3_FUSED_QKNORM_ROPE_ROTARY_DIM = 64

_has_rocm_qk_norm_rope = False
if _is_hip:
    try:
        from sglang.jit_kernel.minimax_m3.qk_norm_rope import (
            qk_gemma_rmsnorm_rope,
            sparse_qk_index_gemma_rmsnorm_rope,
            sparse_qk_index_gemma_rmsnorm_rope_cache,
        )

        _has_rocm_qk_norm_rope = True
    except ImportError:
        _has_rocm_qk_norm_rope = False

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope import split_qkv_tp_rmsnorm_rope

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


class _FusedQKVIndexProj(nn.Module):
    """One GEMM for the main ``qkv_proj`` and the sparse ``index_qkv_proj``.

    Both projections consume the same hidden input, so their weights are
    concatenated along the output dim (and, for mxfp8, their raw UE8M0 scales)
    and run through the *shared* quant_method once: the activation is quantized
    a single time and one matmul produces ``[q | k | v | idx_q | idx_k (| idx_v)]``.

    This is built after weight loading from the two already-loaded linears. The
    raw fp8 weight + uint8 scale are final right after ``load_weights`` (the
    mxfp8 ``process_weights_after_loading`` only *derives* the packed deep_gemm
    scale, it does not mutate the raw tensors), so the build needs no separate
    post-process hook; the backend scale layout is derived here once.

    Only the unquantized bf16 path and mxfp8 are supported as a single concat
    GEMM; other quant methods make the caller fall back to two GEMMs.
    """

    def __init__(
        self,
        quant_method,
        weight: torch.Tensor,
        weight_scale_inv: Optional[torch.Tensor],
        input_size_per_partition: int,
        logical_widths: List[int],
        orig_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        # Stored as ``_qm`` (not ``quant_method``) so the model loader's
        # post-process loop -- which keys off a ``quant_method`` attribute --
        # skips this module: its scale layout is already finalized below.
        self._qm = quant_method
        self.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        self.input_size_per_partition = input_size_per_partition
        self.output_size_per_partition = weight.shape[0]
        self.logical_widths = logical_widths
        self.orig_dtype = orig_dtype
        self.input_scale = None
        if weight_scale_inv is not None:
            self.register_parameter(
                "weight_scale_inv", nn.Parameter(weight_scale_inv, requires_grad=False)
            )
            self.weight_scale_inv.format_ue8m0 = True
            # Derive the backend scale layout (deep_gemm packed / swizzled) once,
            # exactly as process_weights_after_loading would for a real linear.
            quant_method._process_mxfp8_linear_weight_scale(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._qm.apply(self, x, None)


def build_minimax_fused_qkv_index(model: nn.Module) -> None:
    """Build the fused qkv+index GEMM for every sparse MiniMax-M3 attention.

    Called at the end of ``load_weights`` (before the loader's per-module
    ``process_weights_after_loading`` pass and before CUDA graph capture). A
    no-op for layers where the fusion is disabled or the quant method is
    unsupported (those keep the two separate projections).
    """
    for module in model.modules():
        if isinstance(module, MiniMaxM3Attention):
            module.maybe_build_fused_qkv_index()


class MiniMaxM3MLP(nn.Module):
    @staticmethod
    def _swigluoai_torch(
        x: torch.Tensor, gemm1_alpha: float, gemm1_limit: float
    ) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=gemm1_limit)
        up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
        return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)

    @staticmethod
    def _swigluoai_fused(x: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
        """swiglu_oai using fused Triton kernel (sgl_kernel_npu), no quant."""
        from sglang.srt.layers.triton_ops.npu_swiglu_oai_quant import swiglu_oai_quant

        out, _ = swiglu_oai_quant(x, alpha, limit, need_quant=False)
        return out

    def __init__(
        self,
        config: PretrainedConfig,
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
            if _is_npu:
                self.act_fn = lambda x: self._swigluoai_fused(
                    x, config.swiglu_alpha, config.swiglu_limit
                )
            else:
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
        forward_batch: Optional[ForwardBatch] = None,
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
            gate_up_interleaved=False,
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
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                reduce_results=False,
                intermediate_size=intermediate_size,
                **(dict(tp_rank=0, tp_size=1) if shared_experts_tp1 else {}),
            )
        else:
            self.shared_experts = None

        self.bf16_router_gemm = envs.SGLANG_OPT_USE_BF16_ROUTER_GEMM.get()
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.bfloat16 if self.bf16_router_gemm else torch.float32,
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
            router_logits = self._compute_router_logits(hidden_states)
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
            router_logits = self._compute_router_logits(hidden_states)
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

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.bf16_router_gemm and not _is_npu:
            return torch.mm(
                hidden_states, self.gate.weight.t(), out_dtype=torch.float32
            )
        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        return router_logits

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

        # Sparse-attention-specific config (read from sparse_attention_config).
        # Only the index-branch dimensions are needed at module level; all
        # block/topk/local/init knobs live in the sparse attention backend.
        if self.is_sparse_attention_layer:
            assert self.qk_norm_type == "per_head", (
                f"sparse attention only supports qk_norm_type='per_head', "
                f"got {self.qk_norm_type!r}"
            )
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
            self.total_num_heads,
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
            # Index q (total_idx_heads heads) + a single replicated k/v head have
            # GQA-with-one-kv-head structure, so they merge into one
            # QKVParallelLinear: q sharded across the idx-head TP group, k/v
            # replicated. The shard placement matches a separate ColumnParallel(q)
            # + Replicated(k/v), so it stays correct under TP / DP-attn / PP / EP.
            # Checkpoints store the three separately; restacked at load (see
            # load_weights). Value-disabled layers have no v (v_head_size == 0).
            self.index_qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.idx_head_dim,
                self.total_idx_heads,
                total_num_kv_heads=1,
                bias=False,
                quant_config=quant_config,
                v_head_size=(0 if self.disable_index_value else self.idx_head_dim),
                tp_rank=self.idx_head_rank,
                tp_size=self.idx_head_tp_size,
                prefix=add_prefix("index_qkv_proj", prefix),
            )

            # Output projection for the index value path; independent of the
            # fused vs unfused input projection above.
            if self.disable_index_value:
                self.index_o_proj = None
            else:
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
                    self.total_num_heads * self.head_dim,
                    num_heads=self.total_num_heads,
                    eps=config.rms_norm_eps,
                )
                self.k_norm = MiniMaxM2RMSNormTP(
                    self.total_num_kv_heads * self.head_dim,
                    num_heads=self.total_num_kv_heads,
                    eps=config.rms_norm_eps,
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

        # Fused GemmaRMSNorm + partial NeoX RoPE (minimax_qknorm_rope) is a CUDA
        # JIT kernel, valid only for the exact verified config: per-head gemma
        # norm, head_dim=128, rotary_dim=64 (so rotary_dim/2 == warpSize), NeoX
        # style. Everything else (incl. ROCm) falls back to the
        # _qk_norm_rope path. The fp32 cos_sin_cache requirement is re-checked at
        # call time.
        self._use_fused_qknorm_rope = (
            _is_cuda
            and envs.SGLANG_OPT_USE_MINIMAX_FUSED_QKNORM_ROPE.get()
            and self.qk_norm_type == "per_head"
            and self.use_gemma_norm
            and self.head_dim == 128
            and self.rotary_dim == _M3_FUSED_QKNORM_ROPE_ROTARY_DIM
            and getattr(self.rotary_emb, "is_neox_style", False)
        )

        # Fuse the main qkv_proj and the sparse index_qkv_proj into one GEMM
        # (both project the same hidden input). Built after weight load via
        # maybe_build_fused_qkv_index(); falls back to two GEMMs when the quant
        # method does not support a safe output-dim concat (only unquantized
        # bf16 and mxfp8 are fused; anything else keeps the two projections).
        self._fuse_qkv_index_enabled = self.is_sparse_attention_layer and (
            _is_cuda or _is_hip
        )
        self._fused_qkv_index = None
        # Per-token main width (q | k | v), in elements; index columns follow it
        # in the fused output.
        self._fused_main_size = self.q_size + 2 * self.kv_size

        # A single combined GemmaRMSNorm+RoPE launch over the fused output (main
        # Q/K + index Q/K) is valid under the same conditions as the main fused
        # qknorm, plus a 128-dim index head and an fp32 rotary cache (the index
        # branch reuses the main rotary_emb). The V / index-V heads are skipped.
        self._combined_qknorm_ok = (
            self.is_sparse_attention_layer
            and self._use_fused_qknorm_rope
            and self.idx_head_dim == 128
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
        )
        if self.is_sparse_attention_layer:
            # (head_offset, head_count) of each normed group in the fused output,
            # in head units (head_dim == idx_head_dim == 128 -> a uniform grid):
            # q | k | v | idx_q | idx_k (| idx_v). v and idx_v are not groups.
            off_iq = self.num_heads + 2 * self.num_kv_heads
            off_ik = off_iq + self.num_idx_heads
            self._qknorm_group_meta = (
                (0, self.num_heads),
                (self.num_heads, self.num_kv_heads),
                (off_iq, self.num_idx_heads),
                (off_ik, 1),
            )

        # Static (init-time) ROCm fast-path eligibility caches. These avoid
        # repeating the same attribute checks on every forward; per-call shape /
        # dtype guards are applied where the cached value is consulted.
        self._can_use_rocm_qk_norm_rope_static = (
            _has_rocm_qk_norm_rope
            and self.qk_norm_type == "per_head"
            and self.use_gemma_norm
            and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
            and hasattr(self.rotary_emb, "cos_sin_cache")
            and self.rotary_emb.rotary_dim == self.rotary_dim
            and self.rotary_dim <= self.head_dim
        )
        self._can_use_rocm_index_qk_norm_rope_static = (
            self.is_sparse_attention_layer
            and _has_rocm_qk_norm_rope
            and self.use_gemma_norm
            and self.index_q_norm.variance_epsilon == self.index_k_norm.variance_epsilon
            and hasattr(self.index_rotary_emb, "cos_sin_cache")
            and self.index_rotary_emb.rotary_dim == self.rotary_dim
            and self.rotary_dim <= self.idx_head_dim
        )
        self._can_use_rocm_sparse_qk_index_norm_rope_static = (
            self.is_sparse_attention_layer
            and self._can_use_rocm_qk_norm_rope_static
            and self._can_use_rocm_index_qk_norm_rope_static
            and self.idx_head_dim == self.head_dim
            and self.index_q_norm.variance_epsilon == self.q_norm.variance_epsilon
            and self.index_k_norm.variance_epsilon == self.q_norm.variance_epsilon
            and self.index_rotary_emb is self.rotary_emb
        )

    def _can_use_rocm_qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> bool:
        return (
            self._can_use_rocm_qk_norm_rope_static
            and positions.dim() == 1
            and q.dim() == 2
            and k.dim() == 2
            and q.dtype in (torch.bfloat16, torch.float16)
            and k.dtype == q.dtype
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
            self._can_use_rocm_index_qk_norm_rope_static
            and positions.dim() == 1
            and idx_q.dim() == 2
            and idx_k.dim() == 2
            and idx_q.dtype in (torch.bfloat16, torch.float16)
            and idx_k.dtype == idx_q.dtype
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

    def _split_index_qkv(
        self, idx_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Split the fused index projection output into per-rank q (num_idx_heads
        # heads), the single replicated k head, and (when enabled) the single v
        # head. The slices are views; downstream norm/rope and the reshape in
        # forward_core handle the (non-)contiguity.
        q_size = self.num_idx_heads * self.idx_head_dim
        if self.disable_index_value:
            idx_q, idx_k = idx_qkv.split([q_size, self.idx_head_dim], dim=-1)
            idx_v = None
        else:
            idx_q, idx_k, idx_v = idx_qkv.split(
                [q_size, self.idx_head_dim, self.idx_head_dim], dim=-1
            )
        return idx_q, idx_k, idx_v

    def maybe_build_fused_qkv_index(self) -> None:
        """Build the single-GEMM fused qkv+index projection (idempotent).

        Concatenates the main ``qkv_proj`` and the sparse ``index_qkv_proj``
        weights (and, for mxfp8, raw UE8M0 scales) along the output dim and
        wraps them in a :class:`_FusedQKVIndexProj`. The two source projections'
        large tensors are freed and their ``quant_method`` dropped, so the
        loader's post-process loop skips them and no extra weight memory is
        held. A no-op (keeps the two separate GEMMs) when disabled or when the
        quant method is not a supported output-dim concat.
        """
        if not self._fuse_qkv_index_enabled or self._fused_qkv_index is not None:
            return
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        qp, ip = self.qkv_proj, self.index_qkv_proj
        qm = qp.quant_method
        # Both projections must share the same quant method/layout to concat.
        if type(ip.quant_method) is not type(qm):
            return

        # gfx942 converts MXFP8->block-fp8 in process_weights_after_loading; the
        # fused module skips that pass, so keep two separate (converted) GEMMs.
        if getattr(qm, "convert_mxfp8_to_block", False):
            return

        weight = torch.cat([qp.weight.data, ip.weight.data], dim=0).contiguous()
        if isinstance(qm, UnquantizedLinearMethod):
            scale = None
        elif getattr(qm, "use_mxfp8", False) and hasattr(qp, "weight_scale_inv"):
            scale = torch.cat(
                [qp.weight_scale_inv.data, ip.weight_scale_inv.data], dim=0
            ).contiguous()
        else:
            # Unsupported quant (e.g. non-mxfp8 fp8 block) -> keep two GEMMs.
            return

        # input_size_per_partition / orig_dtype are set by the quant method's
        # create_weights (fp8) but not by UnquantizedLinearMethod; fall back to
        # the always-present linear attrs (input isn't sharded for column TP).
        holder = _FusedQKVIndexProj(
            qm,
            weight,
            scale,
            getattr(qp, "input_size_per_partition", qp.input_size),
            [qp.output_size_per_partition, ip.output_size_per_partition],
            getattr(qp, "orig_dtype", qp.params_dtype),
        )
        self.add_module("fused_qkv_index_proj", holder)
        self._fused_qkv_index = holder

        # Reclaim the originals: free their data and drop quant_method so the
        # loader's post-process loop ignores them (the separate GEMMs are dead).
        for m in (qp, ip):
            m.quant_method = None
            for attr in ("weight", "weight_scale_inv"):
                p = getattr(m, attr, None)
                if isinstance(p, nn.Parameter):
                    p.data = torch.empty(0, dtype=p.dtype, device=p.data.device)

    def _qknorm_groups(self):
        # (norm weight, head offset, head count) for the combined qknorm+rope
        # over the fused output: main Q, main K, index Q, index K.
        weights = (
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
        )
        return [
            (w, off, cnt) for w, (off, cnt) in zip(weights, self._qknorm_group_meta)
        ]

    def _can_use_rocm_sparse_qk_index_norm_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
    ) -> bool:
        return (
            self._can_use_rocm_sparse_qk_index_norm_rope_static
            and positions.dim() == 1
            and q.dim() == 2
            and k.dim() == 2
            and idx_q.dim() == 2
            and idx_k.dim() == 2
            and q.dtype in (torch.bfloat16, torch.float16)
            and k.dtype == q.dtype
            and idx_q.dtype == q.dtype
            and idx_k.dtype == q.dtype
        )

    def _sparse_qk_index_norm_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._can_use_rocm_sparse_qk_index_norm_rope(positions, q, k, idx_q, idx_k):
            return sparse_qk_index_gemma_rmsnorm_rope(
                q,
                k,
                idx_q,
                idx_k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.index_q_norm.weight.data,
                self.index_k_norm.weight.data,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.q_norm.variance_epsilon,
                self.head_dim,
                self.rotary_dim,
                self.rotary_emb.is_neox_style,
            )
        q, k = self._qk_norm_rope(positions, q, k)
        idx_q, idx_k = self._index_qk_norm_rope(positions, idx_q, idx_k)
        return q, k, idx_q, idx_k

    @staticmethod
    def _mark_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> None:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        if layer_ids is None:
            layer_ids = set()
            forward_batch.minimax_m3_precached_sparse_layers = layer_ids
        layer_ids.add(layer_id)

    @staticmethod
    def _get_sparse_kv_pool():
        if not has_forward_context():
            return None
        attn_backend = get_forward_context().attn_backend
        sparse_backend = getattr(attn_backend, "sparse", None)
        return getattr(sparse_backend, "kv_pool", None)

    def _sparse_qk_index_norm_rope_cache(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kv_pool = self._get_sparse_kv_pool()
        # The fused qknorm+rope+kv-insert kernel writes the (normed/roped) bf16
        # K and raw bf16 V straight into the paged cache buffer. When the main
        # K/V cache is fp8 (--kv-cache-dtype fp8_*) that buffer is fp8, so the
        # fusion cannot do the write. Fall back to the norm+rope-only path here
        # and let the sparse backend's set_kv_buffer do the bf16->fp8 cache
        # write (the index cache stays bf16, so its fusion is unaffected but is
        # bundled in the same kernel, hence the whole fusion is skipped).
        main_kv_is_fp8 = kv_pool is not None and kv_pool.dtype in _FP8_KV_DTYPES
        can_use_cache_fusion = (
            not main_kv_is_fp8
            and idx_v is None
            and self._can_use_rocm_sparse_qk_index_norm_rope(
                positions, q, k, idx_q, idx_k
            )
            and getattr(forward_batch, "out_cache_loc", None) is not None
            and v.dim() == 2
            and v.dtype == q.dtype
            and v.shape == k.shape
        )
        if can_use_cache_fusion and kv_pool is not None:
            layer_id = self.attn.layer_id
            k_cache, v_cache = kv_pool.get_kv_buffer(layer_id)
            idx_k_cache = kv_pool.get_index_k_buffer(layer_id)
            q, k, idx_q, idx_k = sparse_qk_index_gemma_rmsnorm_rope_cache(
                q,
                k,
                v,
                idx_q,
                idx_k,
                k_cache,
                v_cache,
                idx_k_cache,
                forward_batch.out_cache_loc,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.index_q_norm.weight.data,
                self.index_k_norm.weight.data,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.q_norm.variance_epsilon,
                self.head_dim,
                self.rotary_dim,
                self.rotary_emb.is_neox_style,
            )
            self._mark_sparse_kv_cached_by_fusion(forward_batch, layer_id)
            return q, k, idx_q, idx_k
        return self._sparse_qk_index_norm_rope(positions, q, k, idx_q, idx_k)

    def _can_use_npu_split_qkv_tp_rmsnorm_rope(self) -> bool:
        return (
            _is_npu
            and not self.is_sparse_attention_layer
            and self.use_qk_norm
            and self.qk_norm_type == "per_layer"
            and not self.attention_output_gate
        )

    def forward_prepare_npu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states, forward_batch, None

        if not self._can_use_npu_split_qkv_tp_rmsnorm_rope():
            return self.forward_prepare(positions, hidden_states, forward_batch)

        qkv, _ = self.qkv_proj(hidden_states)
        cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, positions.flatten())
        cos, sin = cos_sin.chunk(2, dim=-1)
        q, k, v = split_qkv_tp_rmsnorm_rope(
            input=qkv,
            cos=cos,
            sin=sin,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_hidden_size=self.q_size,
            kv_hidden_size=self.kv_size,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim,
            eps=self.q_norm.variance_epsilon,
            tp_world=getattr(self.q_norm, "attn_tp_size", self.attn_tp_size),
            tp_group=get_attention_tp_group().device_group,
        )
        inner_state = (q, k, v, None, forward_batch)
        return None, forward_batch, inner_state

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # Single fused GEMM for main qkv + sparse index qkv when available;
        # otherwise the standalone main qkv_proj (index is projected below).
        fused_out = None
        if self._fused_qkv_index is not None:
            fused_out = self.fused_qkv_index_proj(hidden_states)
            qkv = fused_out[:, : self._fused_main_size]

            # Combined main+index GemmaRMSNorm + partial NeoX RoPE in one launch
            # over the whole fused tensor (q, k, idx_q, idx_k groups; v / idx_v
            # left untouched), then split both branches out of the fused buffer.
            if self._combined_qknorm_ok:
                from sglang.jit_kernel.minimax_qknorm_rope import (
                    minimax_qknorm_rope_grouped,
                )

                minimax_qknorm_rope_grouped(
                    fused_out,
                    self._qknorm_groups(),
                    self.rotary_emb.cos_sin_cache,
                    positions,
                    self.q_norm.variance_epsilon,
                )
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                idx_qkv = fused_out[:, self._fused_main_size :]
                idx_q, idx_k, idx_v = self._split_index_qkv(idx_qkv)
                inner_state = (q, k, v, idx_q, idx_k, idx_v, forward_batch)
                return None, forward_batch, inner_state
        else:
            qkv, _ = self.qkv_proj(hidden_states)

        if self._use_fused_qknorm_rope:
            # Fused per-head GemmaRMSNorm + partial NeoX RoPE, in place on qkv.
            from sglang.jit_kernel.minimax_qknorm_rope import minimax_qknorm_rope

            minimax_qknorm_rope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.q_norm.variance_epsilon,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            main_qk_already_normed = True
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            main_qk_already_normed = False

        if self.is_sparse_attention_layer:
            if fused_out is not None:
                idx_qkv = fused_out[:, self._fused_main_size :]
            else:
                idx_qkv, _ = self.index_qkv_proj(hidden_states)

            if main_qk_already_normed:
                use_fused_index_norm_rope = (
                    self._use_fused_qknorm_rope
                    and self.idx_head_dim == 128
                    and self.index_rotary_emb.cos_sin_cache.dtype == torch.float32
                )
                if use_fused_index_norm_rope:
                    # Preserve the existing CUDA sparse-index fast path when
                    # main q/k were already normed+roped by the fused base
                    # kernel. This runs only over idx_q/idx_k; idx_v is left
                    # untouched.
                    from sglang.jit_kernel.minimax_qknorm_rope import (
                        minimax_qknorm_rope,
                    )

                    minimax_qknorm_rope(
                        idx_qkv,
                        self.index_q_norm.weight,
                        self.index_k_norm.weight,
                        self.index_rotary_emb.cos_sin_cache,
                        positions,
                        self.num_idx_heads,
                        1,
                        0 if self.disable_index_value else 1,
                        self.index_q_norm.variance_epsilon,
                    )
                    idx_q, idx_k, idx_v = self._split_index_qkv(idx_qkv)
                else:
                    # Main q/k were normed+roped by the fused base kernel
                    # above; only the index branch still needs norm+rope.
                    idx_q, idx_k, idx_v = self._split_index_qkv(idx_qkv)
                    idx_q, idx_k = self._index_qk_norm_rope(positions, idx_q, idx_k)
            else:
                idx_q, idx_k, idx_v = self._split_index_qkv(idx_qkv)
                # Prefer the PR's ROCm sparse cache-store + norm/rope fusion
                # when it applies; it handles q/k/idx_q/idx_k norm+rope AND
                # the sparse KV cache store in a single kernel. The helper
                # falls back to _qk_norm_rope + _index_qk_norm_rope when
                # preconditions fail.
                q, k, idx_q, idx_k = self._sparse_qk_index_norm_rope_cache(
                    positions, q, k, v, idx_q, idx_k, idx_v, forward_batch
                )

            inner_state = (q, k, v, idx_q, idx_k, idx_v, forward_batch)
        else:
            if not main_qk_already_normed:
                q, k = self._qk_norm_rope(positions, q, k)
            inner_state = (q, k, v, forward_batch)
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        _, _, inner_state = intermediate_state

        if self.is_sparse_attention_layer:
            q, k, v, idx_q, idx_k, idx_v, forward_batch = inner_state
            # The sparse attention backend expects 3D shapes; the dense
            # backend accepts 2D q (it reshapes k/v internally). The shapes
            # are equivalent flat-vs-grouped views, see RadixAttention.forward.
            q = q.view(q.shape[0], self.num_heads, self.head_dim)
            k = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
            v = v.view(v.shape[0], self.num_kv_heads, self.head_dim)
            # reshape (not view): the index q/k/v are non-contiguous slices of
            # the single fused index_qkv_proj output tensor.
            idx_q = idx_q.reshape(idx_q.shape[0], self.num_idx_heads, self.idx_head_dim)
            idx_k = idx_k.reshape(idx_k.shape[0], 1, self.idx_head_dim)
            if idx_v is not None:
                idx_v = idx_v.reshape(idx_v.shape[0], 1, self.idx_head_dim)
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

        q, k, v, forward_batch = inner_state
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if _is_npu:
            s = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
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
        if self.is_layer_sparse and get_tensor_model_parallel_world_size() > 1:
            # Sparse MoE produces partial expert outputs per rank; deferring the
            # all-reduce into the next layer's fusion corrupts those partials and
            # re-triggers the M3 no-EOS runaway. Force the immediate all-reduce in
            # MiniMaxM3MoE.forward_normal (aligns with vLLM). Dense MLP keeps fusion.
            should_allreduce_fusion = False

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if self.is_layer_sparse or hidden_states.shape[0] != 0:
            hidden_states = self.mlp(
                hidden_states,
                forward_batch=forward_batch,
                should_allreduce_fusion=should_allreduce_fusion,
                use_reduce_scatter=use_reduce_scatter,
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
                # NOTE: torch dynamo does not support graph break in context manager
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
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
        ), "Only 1 fused shared expert is supported for MiniMax-M3"
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

        # MiniMaxM3Model.forward checks each layer's ``_is_layer_to_capture``
        # attribute (not ``i in layers_to_capture``), so the per-layer flag must
        # be set explicitly -- mirroring qwen3_next/qwen2_moe. Without this the
        # aux list stays empty and the (hidden, aux) tuple is never returned.
        for layer_id in self.model.layers_to_capture:
            if 0 <= layer_id < len(self.model.layers):
                setattr(
                    self.model.layers[layer_id], "_is_layer_to_capture", True
                )

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
            # Leading "." prevents the main qkv mapping from falsely matching
            # the sparse-attention ``index_q_proj`` / ``index_k_proj`` /
            # ``index_v_proj`` weights, which contain ``q_proj`` / ``k_proj`` /
            # ``v_proj`` as substrings (those are remapped separately below).
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Sparse models merge the index branch q/k/v into one index_qkv_proj
        # (see MiniMaxM3Attention.__init__), so the checkpoint's separate
        # index_q/k/v projections are restacked here. Value-disabled layers have
        # no ".index_v_proj" weight, so that entry never matches.
        if getattr(self.config, "sparse_attention_config", None) is not None:
            stacked_params_mapping += [
                (".index_qkv_proj", ".index_q_proj", "q"),
                (".index_qkv_proj", ".index_k_proj", "k"),
                (".index_qkv_proj", ".index_v_proj", "v"),
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

        # Fuse main qkv_proj + sparse index_qkv_proj into one GEMM. The raw fp8
        # weight + uint8 scale are final at this point (the mxfp8 post-process
        # only derives the packed scale), so this runs deterministically before
        # the loader's process pass and CUDA graph capture.
        build_minimax_fused_qkv_index(self)
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
