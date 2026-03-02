"""Inference-only Sarvam MoE models for SGLang.
- SarvamMLAForCausalLM (105B)
- SarvamMoEForCausalLM (30B)
"""

import gc
import math
from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import enable_moe_dense_fully_dp
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.bailing_moe import BailingMoEForCausalLM
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    bind_or_assign,
    is_cuda,
    is_nvidia_cublas_version_ge_12_9,
)

_is_cuda = is_cuda()
_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()

if _is_cuda:
    try:
        from sgl_kernel import bmm_fp8, merge_state_v2

        from sglang.srt.layers.quantization.fp8_kernel import per_tensor_quant_mla_fp8

        _has_fp8_support = True
    except ImportError:
        _has_fp8_support = False
        bmm_fp8 = None
        merge_state_v2 = None
        per_tensor_quant_mla_fp8 = None
else:
    _has_fp8_support = False
    bmm_fp8 = None
    merge_state_v2 = None
    per_tensor_quant_mla_fp8 = None


class AttnForwardMethod(IntEnum):
    MLA_SEPARATE_ROPE = auto()
    MLA_CONCAT_ROPE = auto()
    MHA_PREFILL = auto()


SEPARATE_ROPE_BACKENDS = frozenset(
    ["fa3", "flashinfer", "nsa", "cutlass_mla", "trtllm_mla"]
)
CONCAT_ROPE_BACKENDS = frozenset(["flashmla", "triton"])


class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name: str, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name: str):
        return cls._handlers.get(backend_name, cls._default_handler)

    @classmethod
    def _default_handler(cls, attn, forward_batch) -> AttnForwardMethod:
        return AttnForwardMethod.MLA_CONCAT_ROPE

    @classmethod
    def get_forward_method(
        cls, backend_name: str, attn, forward_batch
    ) -> AttnForwardMethod:
        handler = cls.get_handler(backend_name)
        return handler(attn, forward_batch)


def _handle_separate_rope_backend(attn, forward_batch) -> AttnForwardMethod:
    return AttnForwardMethod.MLA_SEPARATE_ROPE


def _handle_concat_rope_backend(attn, forward_batch) -> AttnForwardMethod:
    return AttnForwardMethod.MLA_CONCAT_ROPE


for backend in SEPARATE_ROPE_BACKENDS:
    AttentionBackendRegistry.register(backend, _handle_separate_rope_backend)
for backend in CONCAT_ROPE_BACKENDS:
    AttentionBackendRegistry.register(backend, _handle_concat_rope_backend)


def get_attn_forward_method(server_args, forward_batch) -> AttnForwardMethod:

    is_decode = forward_batch.forward_mode.is_decode_or_idle()
    if is_decode:
        backend = server_args.decode_attention_backend or server_args.attention_backend
    else:
        backend = server_args.prefill_attention_backend or server_args.attention_backend
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and backend == "fa3"
        ):
            return AttnForwardMethod.MHA_PREFILL
    return AttentionBackendRegistry.get_forward_method(backend, None, forward_batch)


class SarvamMoEMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
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
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if x.shape[0] == 0:
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class SarvamMoESparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 2.5)
        self.score_function = getattr(config, "score_function", "sigmoid")
        self.n_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)
        self.alt_stream = alt_stream

        router_dtype_cfg = getattr(config, "router_dtype", "fp32")
        if router_dtype_cfg is None:
            self.router_dtype = None
        elif router_dtype_cfg == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=self.n_group is not None and self.topk_group is not None,
            num_expert_group=self.n_group,
            topk_group=self.topk_group,
            renormalize=True,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
            scoring_func=self.score_function,
            correction_bias=self.e_score_correction_bias,
            quant_config=quant_config,
            layer_id=layer_id,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if (
            getattr(config, "num_shared_experts", None)
            and config.num_shared_experts > 0
        ):
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            if enable_moe_dense_fully_dp():
                shared_tp_rank, shared_tp_size = 0, 1
            else:
                shared_tp_rank, shared_tp_size = None, None
            self.shared_experts = SarvamMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                reduce_results=False,
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
            )
        else:
            self.shared_experts = None

        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.num_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: Optional[BumpAllocator] = None,
    ) -> torch.Tensor:
        del gemm_output_zero_allocator
        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            return self.forward_deepep(hidden_states, forward_batch)

        if (
            self.shared_experts is not None
            and self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            return self.forward_normal_dual_stream(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        else:
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)

    def _forward_router_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.router_dtype is not None:
            router_logits = F.linear(
                hidden_states.to(self.router_dtype),
                self.gate.weight.to(self.router_dtype),
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_out = self._forward_shared_experts(hidden_states)
        with torch.cuda.stream(self.alt_stream):
            final_hidden_states = self._forward_router_experts(hidden_states)
            if self.routed_scaling_factor != 1.0:
                final_hidden_states = final_hidden_states * self.routed_scaling_factor
        current_stream.wait_stream(self.alt_stream)
        final_hidden_states = final_hidden_states + shared_out
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        num_tokens, hidden_dim = hidden_states.shape
        identity = (
            hidden_states.clone() if self.shared_experts is not None else hidden_states
        )

        if self.router_dtype is not None:
            router_logits = F.linear(
                hidden_states.to(self.router_dtype),
                self.gate.weight.to(self.router_dtype),
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if self.shared_experts is not None:
            shared_out = self.shared_experts(identity)
            if self.routed_scaling_factor != 1.0:
                shared_out.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            else:
                shared_out.add_(final_hidden_states)
            final_hidden_states = shared_out
        elif self.routed_scaling_factor != 1.0:
            final_hidden_states = final_hidden_states * self.routed_scaling_factor

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class SarvamMoEMLAAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.quant_config = quant_config

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype

        self._server_args = None
        self.current_attention_backend = None

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )
        else:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_a_proj", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
        )

        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if rope_scaling and rope_scaling["type"] == "deepseek_yarn":
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)
            scaling_factor = rope_scaling.get("factor", 1.0)
            mscale = self.yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

    def yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _concat_and_cast_mha_k(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return DeepseekMHAForwardMixin._concat_and_cast_mha_k(
            self, k_nope, k_pe, forward_batch
        )

    def _set_current_attention_backend(self, forward_batch: ForwardBatch) -> None:
        if self._server_args is None:
            self._server_args = get_global_server_args()
        if forward_batch.forward_mode.is_decode_or_idle():
            self.current_attention_backend = (
                self._server_args.decode_attention_backend
                or self._server_args.attention_backend
            )
        else:
            self.current_attention_backend = (
                self._server_args.prefill_attention_backend
                or self._server_args.attention_backend
            )

    def _maybe_fp8_bmm(
        self,
        x_bmk: torch.Tensor,
        w_bkn: torch.Tensor,
        zero_allocator: Optional[BumpAllocator] = None,
    ) -> torch.Tensor:
        if (
            _has_fp8_support
            and w_bkn is not None
            and w_bkn.dtype == torch.float8_e4m3fn
        ):
            x_val, x_scale = per_tensor_quant_mla_fp8(
                x_bmk,
                (
                    torch.zeros((1,), dtype=torch.float32, device=x_bmk.device)
                    if _is_cublas_ge_129
                    else (
                        zero_allocator.allocate(1)
                        if zero_allocator
                        else torch.zeros((1,), dtype=torch.float32, device=x_bmk.device)
                    )
                ),
            )
            w_scale = self.w_scale if self.w_scale is not None else 1.0
            return bmm_fp8(x_val, w_bkn, x_scale, w_scale, torch.bfloat16)

        return torch.bmm(x_bmk, w_bkn)

    def _run_mha_prefill(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
            self.attn_mha,
            forward_batch.out_cache_loc,
            k_nope,
            k_pe,
        )

        kv_a = k_nope.squeeze(1)
        kv_expanded, _ = self.kv_b_proj(kv_a)
        kv_expanded = kv_expanded.view(
            -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_expanded = kv_expanded[..., : self.qk_nope_head_dim]
        v = kv_expanded[..., self.qk_nope_head_dim :]

        k = self._concat_and_cast_mha_k(k_nope_expanded, k_pe, forward_batch)

        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )

        self._set_current_attention_backend(forward_batch)
        can_use_prefix_cache = not self._server_args.disable_radix_cache
        do_prefix_merge = has_extend_prefix and can_use_prefix_cache

        if do_prefix_merge and forward_batch.num_prefix_chunks is None:
            if hasattr(forward_batch, "prepare_chunked_prefix_cache_info"):
                forward_batch.prepare_chunked_prefix_cache_info(q.device)
            else:
                forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.set_attn_attend_prefix_cache(False)
        forward_batch.mha_return_lse = do_prefix_merge
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        if do_prefix_merge and merge_state_v2 is not None:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        forward_batch.set_attn_attend_prefix_cache(None)

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return DeepseekMHAForwardMixin._chunked_prefix_attn_mha(
            self, q, accum_output, accum_lse, forward_batch
        )

    def _get_mla_kv_buffer(
        self,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        return DeepseekMHAForwardMixin._get_mla_kv_buffer(
            self, kv_indices, dst_dtype, forward_batch
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: Optional[BumpAllocator] = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        if hidden_states.shape[0] == 0:
            return hidden_states

        if self.q_lora_rank is None:
            q, _ = self.q_proj(hidden_states)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        else:
            q_a, _ = self.q_a_proj(hidden_states)
            q_a = self.q_a_layernorm(q_a)
            q, _ = self.q_b_proj(q_a)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self._server_args is None:
            self._server_args = get_global_server_args()
        self._set_current_attention_backend(forward_batch)

        forward_method = get_attn_forward_method(self._server_args, forward_batch)

        if forward_method == AttnForwardMethod.MHA_PREFILL:
            return self._run_mha_prefill(
                positions=positions,
                q=q,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                forward_batch=forward_batch,
            )

        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with torch.cuda.stream(self.alt_stream):
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            current_stream.wait_stream(self.alt_stream)
        else:
            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if forward_method == AttnForwardMethod.MLA_SEPARATE_ROPE:
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
            )
        elif forward_method == AttnForwardMethod.MLA_CONCAT_ROPE:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
            )
        else:
            raise ValueError(f"Unknown forward method: {forward_method}")
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = self._maybe_fp8_bmm(
            attn_output.transpose(0, 1), self.w_vc, zero_allocator
        )
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        output, _ = self.o_proj(attn_bmm_output)
        return output

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: Optional[BumpAllocator] = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], ForwardBatch, Optional[Tuple]]:
        del llama_4_scaling
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None

        if self.q_lora_rank is None:
            # Dual-stream parallel Q and KV projections
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
                q, _ = self.q_proj(hidden_states)
                current_stream.wait_stream(self.alt_stream)
            else:
                q, _ = self.q_proj(hidden_states)
                latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
        else:
            # For q_lora_rank path, overlap q_a_proj with kv_a_proj
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
                q_a, _ = self.q_a_proj(hidden_states)
                current_stream.wait_stream(self.alt_stream)
            else:
                q_a, _ = self.q_a_proj(hidden_states)
                latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
            q_a = self.q_a_layernorm(q_a)
            q, _ = self.q_b_proj(q_a)
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self._server_args is None:
            self._server_args = get_global_server_args()
        self._set_current_attention_backend(forward_batch)
        forward_method = get_attn_forward_method(self._server_args, forward_batch)

        if forward_method == AttnForwardMethod.MHA_PREFILL:
            output = self._run_mha_prefill(
                positions=positions,
                q=q,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                forward_batch=forward_batch,
            )
            return output, forward_batch, None

        # Parallel Absorption + RoPE on separate streams
        # - Stream 1 (main): Absorption (q_nope @ w_kc)
        # - Stream 2 (alt): RoPE (q_pe, k_pe)
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            # RoPE on alt stream
            with torch.cuda.stream(self.alt_stream):
                q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

            # Absorption on main stream (runs in parallel with RoPE)
            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            current_stream.wait_stream(self.alt_stream)
        else:
            q_nope_out = self._maybe_fp8_bmm(
                q_nope.transpose(0, 1), self.w_kc, zero_allocator
            )
            q_nope_out = q_nope_out.transpose(0, 1)

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        inner_state = (q_nope_out, k_nope, q_pe, k_pe, forward_batch, zero_allocator)
        return None, forward_batch, inner_state

    def forward_core(
        self,
        intermediate_state: Tuple[
            Optional[torch.Tensor], ForwardBatch, Optional[Tuple]
        ],
    ) -> torch.Tensor:
        hidden_states, forward_batch, inner_state = intermediate_state

        if inner_state is None:
            return hidden_states

        q_nope_out, k_nope, q_pe, k_pe, forward_batch, zero_allocator = inner_state

        if self._server_args is None:
            self._server_args = get_global_server_args()
        self._set_current_attention_backend(forward_batch)

        forward_method = get_attn_forward_method(self._server_args, forward_batch)

        if forward_method == AttnForwardMethod.MLA_SEPARATE_ROPE:
            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
            )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = self._maybe_fp8_bmm(
            attn_output.transpose(0, 1), self.w_vc, zero_allocator
        )
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        output, _ = self.o_proj(attn_bmm_output)
        return output

    def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        del forward_batch
        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        return latent_cache


class SarvamMLAForCausalLM(DeepseekV2ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self._remap_config(config)
        super().__init__(config, quant_config, prefix)
        self._swap_sarvam_blocks(quant_config)

    @staticmethod
    def _remap_config(config: PretrainedConfig) -> None:
        if not hasattr(config, "n_routed_experts"):
            config.n_routed_experts = config.num_experts
        if not hasattr(config, "n_shared_experts"):
            config.n_shared_experts = getattr(config, "num_shared_experts", None)

        config.n_shared_experts = None
        if not hasattr(config, "num_experts"):
            config.num_experts = config.n_routed_experts
        if not hasattr(config, "norm_topk_prob"):
            config.norm_topk_prob = True
        if not hasattr(config, "topk_method"):
            config.topk_method = "noaux_tc"

        defaults = {
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "n_group": 1,
            "topk_group": 1,
            "router_dtype": "fp32",
            "routed_scaling_factor": 2.5,
            "score_function": "sigmoid",
        }
        for attr, default in defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, default)

    def _swap_sarvam_blocks(self, quant_config: Optional[QuantizationConfig]) -> None:
        if hasattr(self.model, "layers"):
            for layer_id in range(self.start_layer, self.end_layer):
                layer = self.model.layers[layer_id]
                if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
                    continue

                if hasattr(self.config, "rope_parameters"):
                    rope_theta = self.config.rope_parameters.get("rope_theta")
                    rope_type = self.config.rope_parameters.get("rope_type")
                    rope_scaling = (
                        self.config.rope_parameters if rope_type != "default" else None
                    )
                else:
                    rope_theta = self.config.rope_theta
                    rope_scaling = self.config.rope_scaling

                layer_prefix = f"model.layers.{layer_id}"
                old_self_attn = layer.self_attn
                layer.self_attn = None
                del old_self_attn

                try:
                    layer.self_attn = SarvamMoEMLAAttention(
                        config=self.config,
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_attention_heads,
                        layer_id=layer_id,
                        rope_theta=rope_theta,
                        rope_scaling=rope_scaling,
                        max_position_embeddings=self.config.max_position_embeddings,
                        quant_config=quant_config,
                        prefix=add_prefix("self_attn", layer_prefix),
                        alt_stream=getattr(layer, "alt_stream", None),
                    )
                except torch.OutOfMemoryError:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    layer.self_attn = SarvamMoEMLAAttention(
                        config=self.config,
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_attention_heads,
                        layer_id=layer_id,
                        rope_theta=rope_theta,
                        rope_scaling=rope_scaling,
                        max_position_embeddings=self.config.max_position_embeddings,
                        quant_config=quant_config,
                        prefix=add_prefix("self_attn", layer_prefix),
                        alt_stream=getattr(layer, "alt_stream", None),
                    )
                if hasattr(layer, "layer_communicator") and hasattr(
                    layer.layer_communicator, "qkv_latent_func"
                ):
                    layer.layer_communicator.qkv_latent_func = (
                        layer.self_attn.prepare_qkv_latent
                    )

                if getattr(layer, "is_layer_sparse", False):
                    old_mlp = layer.mlp
                    layer.mlp = None
                    del old_mlp

                    try:
                        layer.mlp = SarvamMoESparseMoeBlock(
                            config=self.config,
                            layer_id=layer_id,
                            quant_config=quant_config,
                            prefix=add_prefix("mlp", layer_prefix),
                            alt_stream=getattr(layer, "alt_stream", None),
                        )
                    except torch.OutOfMemoryError:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        layer.mlp = SarvamMoESparseMoeBlock(
                            config=self.config,
                            layer_id=layer_id,
                            quant_config=quant_config,
                            prefix=add_prefix("mlp", layer_prefix),
                            alt_stream=getattr(layer, "alt_stream", None),
                        )

    def determine_num_fused_shared_experts(
        self, architecture: str = "SarvamMLAForCausalLM"
    ):
        del architecture
        self.num_fused_shared_experts = 0
        get_global_server_args().disable_shared_experts_fusion = True

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=getattr(
                config, "n_routed_experts", getattr(config, "num_experts", None)
            ),
            num_groups=getattr(config, "n_group", None),
        )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ) -> None:
        def _remap_bias_name(
            ws: Iterable[Tuple[str, torch.Tensor]],
        ) -> Iterable[Tuple[str, torch.Tensor]]:
            for name, w in ws:
                if ".mlp.gate.e_score_correction_bias" in name:
                    name = name.replace(
                        ".mlp.gate.e_score_correction_bias",
                        ".mlp.e_score_correction_bias",
                    )
                yield name, w

        super().load_weights(_remap_bias_name(weights), is_nextn)
        self._set_mla_wkc_wvc()

    def _set_mla_wkc_wvc(self) -> None:
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if not isinstance(self_attn, SarvamMoEMLAAttention):
                continue
            if not hasattr(self_attn, "kv_b_proj") or self_attn.kv_b_proj is None:
                continue

            w = self_attn.kv_b_proj.weight.data
            weight_scale = None
            if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.kv_b_proj.weight_scale is not None
                ):
                    weight_scale = self_attn.kv_b_proj.weight_scale
                elif (
                    hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    and self_attn.kv_b_proj.weight_scale_inv is not None
                ):
                    weight_scale = self_attn.kv_b_proj.weight_scale_inv
                elif (
                    hasattr(self_attn.kv_b_proj, "scale")
                    and self_attn.kv_b_proj.scale is not None
                ):
                    weight_scale = self_attn.kv_b_proj.scale

            w_reshaped = w.unflatten(
                0,
                (
                    self_attn.num_local_heads,
                    self_attn.qk_nope_head_dim + self_attn.v_head_dim,
                ),
            )
            w_kc, w_vc = w_reshaped.split(
                [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1
            )
            self_attn.w_kc = bind_or_assign(
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )
            self_attn.w_vc = bind_or_assign(
                self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
            )
            if weight_scale is not None:
                self_attn.w_scale = weight_scale


class SarvamMoEForCausalLM(BailingMoEForCausalLM):

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval

        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.word_embeddings(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            forward_batch.residual = None

        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            if forward_batch.residual is None:
                hidden_states = self.model.norm(forward_batch.hidden_states)
            else:
                hidden_states, _ = self.model.norm(
                    forward_batch.hidden_states, forward_batch.residual
                )
            forward_batch.hidden_states = hidden_states

            return self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )

        return None


EntryClass = [SarvamMLAForCausalLM, SarvamMoEForCausalLM]
