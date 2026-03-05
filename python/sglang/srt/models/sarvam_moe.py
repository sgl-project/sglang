"""Inference-only Sarvam MoE models for SGLang.
- SarvamMLAForCausalLM (105B)
- SarvamMoEForCausalLM (30B)
"""

import math
from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton
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
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe import should_use_flashinfer_cutlass_moe_fp4_allgather
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe import BailingMoEForCausalLM
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    DeepseekMHAForwardMixin,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    bind_or_assign,
    is_cuda,
    is_nvidia_cublas_version_ge_12_9,
    make_layers,
    next_power_of_2,
)

_is_cuda = is_cuda()
_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()

if _is_cuda:
    try:
        from sgl_kernel import bmm_fp8, concat_mla_k, merge_state_v2

        from sglang.srt.layers.quantization.fp8_kernel import per_tensor_quant_mla_fp8

        _has_fp8_support = True
        _has_concat_mla_k = True
    except ImportError:
        _has_fp8_support = False
        _has_concat_mla_k = False
        bmm_fp8 = None
        concat_mla_k = None
        merge_state_v2 = None
        per_tensor_quant_mla_fp8 = None
else:
    _has_fp8_support = False
    _has_concat_mla_k = False
    bmm_fp8 = None
    concat_mla_k = None
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

        dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        router_dtype_cfg = getattr(config, "router_dtype", "fp32")
        self.router_dtype = dtype_map.get(router_dtype_cfg, None)

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
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)

        if (
            _is_cuda
            and _has_concat_mla_k
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
            return k

        if (
            _is_cuda
            and next_power_of_2(self.num_local_heads) == self.num_local_heads
            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim
            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim
        ):
            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = forward_batch.token_to_kv_pool.dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
            return k

        k = k_nope.new_empty(*k_shape)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        return k

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


class SarvamMoEMLADecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        if hasattr(config, "rope_parameters"):
            rope_theta = config.rope_parameters.get("rope_theta")
            rope_type = config.rope_parameters.get("rope_type")
            rope_scaling = config.rope_parameters if rope_type != "default" else None
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = SarvamMoEMLAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        has_moe = getattr(config, "num_experts", None) is not None
        self.is_layer_sparse = (
            has_moe
            and layer_id >= first_k_dense
            and (layer_id - first_k_dense) % moe_layer_freq == 0
        )
        is_previous_layer_sparse = (
            has_moe
            and layer_id > 0
            and (layer_id - 1) >= first_k_dense
            and (layer_id - 1 - first_k_dense) % moe_layer_freq == 0
        )
        is_next_layer_sparse = (
            has_moe
            and layer_id < config.num_hidden_layers - 1
            and (layer_id + 1) >= first_k_dense
            and (layer_id + 1 - first_k_dense) % moe_layer_freq == 0
        )

        if self.is_layer_sparse:
            self.mlp = SarvamMoESparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = SarvamMoEMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=False,
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_tp_size = get_attention_tp_size()
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
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
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
        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )
        if (
            not self.is_layer_sparse
            and self.attn_tp_size > 1
            and not use_reduce_scatter
            and not should_allreduce_fusion
        ):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual


class SarvamMLAModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = nn.Identity()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SarvamMoEMLADecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
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

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class SarvamMLAForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._remap_config(config)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = SarvamMLAModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @staticmethod
    def _remap_config(config: PretrainedConfig) -> None:
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
            "norm_topk_prob": True,
            "topk_method": "noaux_tc",
        }
        for attr, default in defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, default)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

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
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
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

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=getattr(config, "n_group", None),
        )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ) -> None:
        del is_nextn
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
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
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            if ".mlp.gate.e_score_correction_bias" in name:
                name = name.replace(
                    ".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias"
                )

            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                is_stacked = True
                break
            if is_stacked:
                continue

            is_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                is_expert = True
                break
            if is_expert:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._set_mla_wkc_wvc()
        if not hasattr(self, "routed_experts_weights_of_layer"):
            self.routed_experts_weights_of_layer = {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, SarvamMoESparseMoeBlock)
            }

    def _set_mla_wkc_wvc(self) -> None:
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
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
