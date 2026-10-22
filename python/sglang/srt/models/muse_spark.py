"""Torch-native Muse 1.2 / Muse Spark text model for SGLang.

BF16 checkpoints run a plain ``torch.nn.functional.linear`` expert loop.
Serialized ModelOpt NVFP4 expert weights stay packed and are executed by the
public FlashInfer TRT-LLM MoE kernel via the public ModelOpt quant method. Muse
applies its post-expert norm between GEMM2 and the top-k combine, which that
kernel fuses together, so the combine is finished separately by the CuTeDSL op
in ``sglang_meta_common/ops`` (with a torch fallback).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store import FusedQKNorm
from sglang.kernels.ops.communication.muse_spark_all_gather import (
    gemm_ag_supported,
)
from sglang.kernels.ops.communication.muse_spark_all_gather_norm_residual import (
    all_gather_norm_residual,
    all_gather_norm_residual_supported,
)
from sglang.kernels.ops.gemm.muse_spark_router_gemm import (
    MAX_TOKENS as ROUTER_GEMM_MAX_TOKENS,
)
from sglang.kernels.ops.layernorm.muse_spark_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.layernorm.muse_spark_norm_residual import (
    FusedPostNormResidualPreNorm,
)
from sglang.kernels.ops.moe.muse_spark_routing import (
    FusedMuseSparkRouting,
)
from sglang.srt.configs.muse_spark import (
    MuseSparkTextConfig,
    should_skip_weight_loading,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.communication.muse_spark_gemm_ag import (
    gemm_ag,
    gemm_local,
    reserve_projection_workspace,
)
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor as OSSLogitsProcessor
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.muse_spark.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    QKVOGateParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_forward, get_memory, get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import add_prefix, is_cpu, use_intel_amx_backend
from sglang.srt.utils.common import get_device_capability, is_cuda

_is_cpu = is_cpu()

logger = logging.getLogger(__name__)

# FusedMoE shard ids that expert weight loaders expect per checkpoint projection.
_EXPERT_SHARD_IDS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


class MuseSpark1xLogitsProcessor(OSSLogitsProcessor):
    """Public logits pipeline with the checkpoint-requested accumulator dtype."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.use_fp32_output = True

    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ):
        if not hasattr(lm_head, "weight"):
            return super()._compute_lm_head(
                hidden_states, lm_head, embedding_bias, out=out
            )
        output_dtype = torch.float32 if self.use_fp32_output else lm_head.weight.dtype
        if use_intel_amx_backend(lm_head):
            # ParallelLMHead swaps in PackWeightMethod on CPU+AMX, which repacks
            # .weight into a VNNI blocked layout *without changing its shape*, so
            # a plain mm silently returns garbage instead of failing.
            logits = torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states.to(lm_head.weight.dtype),
                lm_head.weight,
                None,
                True,
            ).to(output_dtype)
        else:
            if out is not None and out.shape != (
                hidden_states.shape[0],
                lm_head.weight.shape[0],
            ):
                out = None
            logits = torch.mm(
                hidden_states.to(lm_head.weight.dtype),
                lm_head.weight.T,
                out_dtype=output_dtype,
                out=out,
            )
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits


def _quantization_name(quant_config: Optional[QuantizationConfig]) -> Optional[str]:
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    return get_name() if get_name is not None else type(quant_config).__name__


def _uses_nvfp4_experts(
    quant_config: Optional[QuantizationConfig], layer_prefix: str
) -> bool:
    if _quantization_name(quant_config) != "modelopt_fp4":
        return False
    excluded = getattr(quant_config, "exclude_modules", None) or []
    return not is_layer_skipped(add_prefix("mlp.experts", layer_prefix), excluded)


def _uses_gptq_int4_experts(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether the CPU quant config selects the GPTQ-packed expert path."""
    if not _is_cpu or quant_config is None:
        return False
    quant_name = _quantization_name(quant_config)
    if quant_name == "gptq":
        return True
    if quant_name != "auto-round":
        return False
    return "gptq" in getattr(quant_config, "packing_format", "") or "gptq" in getattr(
        quant_config, "backend", ""
    )


@functools.lru_cache(maxsize=1)
def _split_bf16_router_supported() -> bool:
    return is_cuda() and get_device_capability()[0] == 9


def _use_split_bf16_router(num_tokens: int) -> bool:
    if num_tokens <= ROUTER_GEMM_MAX_TOKENS or not _split_bf16_router_supported():
        return False
    return True


def _fused_norm_boundary(
    post_norm: nn.Module, residual_add: nn.Module, pre_norm: nn.Module
) -> FusedPostNormResidualPreNorm:
    return FusedPostNormResidualPreNorm(
        post_norm,
        residual_add,
        pre_norm,
        post_norm_is_input_scale=isinstance(
            post_norm, MuseSpark1xRMSNormWithInputScale
        ),
        use_residual_gate=isinstance(
            residual_add, MuseSpark1xNormPreservingResidualConnection
        ),
    )


class MuseSpark1xRouter(nn.Module):
    """Router stored in HF linear layout: ``[experts, hidden]``."""

    def __init__(self, hidden_size: int, num_experts: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=dtype), requires_grad=False
        )

        self.register_buffer(
            "e_score_correction_bias", torch.zeros(num_experts, dtype=torch.float32)
        )
        self._split2_hi = None
        self._split2_lo = None
        self._split2_key = None

    @torch.no_grad()
    def _build_split2(self) -> None:
        w = self.weight.data
        key = (w.data_ptr(), w.shape, w._version)
        if self._split2_key == key:
            return
        hi = w.to(torch.bfloat16)
        lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
        self._split2_hi = hi.t().contiguous()
        self._split2_lo = lo.t().contiguous()
        self._split2_key = key

    def forward_split2(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._build_split2()
        x = hidden_states.to(torch.bfloat16)
        acc = torch.mm(x, self._split2_hi, out_dtype=torch.float32)
        acc += torch.mm(x, self._split2_lo, out_dtype=torch.float32)
        return acc

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class MuseSpark1xAttention(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.use_o_gate = True
        self.use_o_norm = True
        self.use_qk_norm = True
        self.use_rope = config.layer_rope_theta[layer_idx] > 0
        sliding_window = (
            config.sliding_window - 1
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by attention TP")
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP"
                )
        elif tp_size % config.num_key_value_heads:
            raise ValueError("attention TP must divide replicated KV heads")
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.scale_query_by = config.qk_scale_factor
        if self.use_qk_norm:
            # RoPE is a rotation, so this post-QK-norm scalar commutes into the
            # softmax scale instead of being multiplied onto q.
            self.scaling *= self.scale_query_by

        if self.use_o_gate:
            self.qkv_proj = QKVOGateParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                quant_config=None,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            parallel_group="attn_tp",
            use_dp_attention_reduce=True,
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=getattr(
                    config, "context_len", config.max_position_embeddings
                ),
                base=config.layer_rope_theta[layer_idx],
                is_neox_style=False,
                dtype=torch.float32,
            )
            if self.use_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
            sliding_window_size=sliding_window or -1,
        )
        self.norm_eps = config.rms_norm_eps
        # The output norm and the sigmoid gate are one fused kernel. The norm is
        # weightless here, so this module carries no parameters -- it exists to
        # hand the op its eps.
        self._o_norm_module = (
            MuseSpark1xRMSNorm(self.head_dim, eps=self.norm_eps, has_weight=False)
            if self.use_o_norm
            else None
        )
        self._fused_o_norm_gate = FusedRMSNormGate(
            self._o_norm_module, self.use_o_gate, self.head_dim
        )
        self._fused_qk_norm = (
            FusedQKNorm(
                head_dim=self.head_dim,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                eps=self.norm_eps,
            )
            if self.use_qk_norm
            else None
        )

    def _fused_kv_store(
        self, forward_batch: ForwardBatch
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Unified pools interleave layers in each token row. Reuse the prefill
        # path, whose attention-backend cache writer supports these strides.
        if get_memory().enable_unified_memory:
            return None
        backend = get_attn_backend()
        pool = get_token_to_kv_pool()
        if not (
            isinstance(backend, TRTLLMHAAttnBackend)
            and backend.data_type in (torch.bfloat16, torch.float8_e4m3fn)
            and pool.dtype == backend.data_type
            and (pool.dtype == torch.bfloat16 or not backend.is_xqa_impl)
            and self.attn.k_scale is None
            and self.attn.v_scale is None
            and forward_batch.dcp_kv_mask is None
            # Prefill excluded: the fused store segfaults at breakable prefill replay.
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
            and not is_cp_active(forward_batch)
        ):
            return None
        cache_loc = backend._get_layer_cache_loc(self.attn, forward_batch)
        k_cache = pool.get_key_buffer(self.layer_idx)
        v_cache = pool.get_value_buffer(self.layer_idx)
        return (
            k_cache.view(k_cache.shape[0], -1),
            v_cache.view(v_cache.shape[0], -1),
            cache_loc,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.qkv_proj(hidden_states)
        split_sizes = [self.q_size, self.kv_size, self.kv_size]
        if self.use_o_gate:
            q, k, v, o_gate = qkv.split(split_sizes + [self.q_size], dim=-1)
        else:
            q, k, v = qkv.split(split_sizes, dim=-1)
            o_gate = None
        fuse_rope = self._fused_qk_norm is not None and self.rotary_emb is not None
        kv_store = (
            self._fused_kv_store(forward_batch)
            if self._fused_qk_norm is not None
            and (self.rotary_emb is None or fuse_rope)
            else None
        )
        if fuse_rope:
            q, k = self._fused_qk_norm(
                qkv, positions, self.rotary_emb.cos_sin_cache, kv_store=kv_store
            )
        elif self._fused_qk_norm is not None:
            q, k = self._fused_qk_norm(qkv, kv_store=kv_store)
        if self.rotary_emb is not None and not fuse_rope:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch, save_kv_cache=kv_store is None)
        output = self._fused_o_norm_gate(output, o_gate)
        output, _ = self.o_proj(output)
        return output


class MuseSpark1xMLP(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MuseSpark1xMoE(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        if config.num_shared_experts:
            raise ValueError("Muse Spark OSS supports routed experts only")
        parallel = get_parallel()
        int4_checkpoint = _uses_gptq_int4_experts(quant_config)
        if parallel.moe_tp_size != 1:
            if not int4_checkpoint:
                raise ValueError(
                    "The torch-native Muse MoE requires expert TP1; use EP equal "
                    "to TP. Expert TP is implemented for int4 checkpoints only."
                )
            if parallel.moe_ep_size != 1:
                raise ValueError(
                    "Muse int4 experts support either expert TP or EP, not both"
                )
        if config.num_local_experts % parallel.moe_ep_size:
            raise ValueError("num_experts must be divisible by expert parallel size")
        self.layer_idx = layer_idx
        self.tp_size = parallel.tp_size
        self.ep_size = parallel.moe_ep_size
        self.ep_rank = parallel.moe_ep_rank
        self.moe_tp_size = parallel.moe_tp_size
        # Reading the rank resolves the MoE-TP process group, which need not
        # exist when expert TP is off.
        self.moe_tp_rank = parallel.moe_tp_rank if self.moe_tp_size != 1 else 0
        self.num_experts = config.num_local_experts
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = 1e-15
        self.router_multiplier = config.output_multiplier
        self.router_in_fp32 = True
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.moe_hidden_size
        self.intermediate_size = config.moe_intermediate_size
        router_dtype = (
            torch.float32 if self.router_in_fp32 else torch.get_default_dtype()
        )
        self.gate = MuseSpark1xRouter(
            self.hidden_size, self.num_experts, dtype=router_dtype
        )

        if self.expert_hidden_size != self.hidden_size:
            if _is_cpu:
                self.pre_expert_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.expert_hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("pre_expert_proj", prefix),
                )
                self.post_expert_proj = ReplicatedLinear(
                    self.expert_hidden_size,
                    self.hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("post_expert_proj", prefix),
                )
            else:
                self.pre_expert_proj = nn.Linear(
                    self.hidden_size, self.expert_hidden_size, bias=False
                )
                self.post_expert_proj = nn.Linear(
                    self.expert_hidden_size, self.hidden_size, bias=False
                )
                for linear in (self.pre_expert_proj, self.post_expert_proj):
                    # The CuTe-DSL GEMM exports weights via DLPack, which rejects requires_grad.
                    linear.weight.requires_grad_(False)
        else:
            self.pre_expert_proj = None
            self.post_expert_proj = None
        if (
            self.pre_expert_proj is not None
            and self.tp_size in (4, 8)
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (10, 3)
        ):
            ca_comm = get_parallel().tp_group.ca_comm
            reserve_projection_workspace(
                ca_comm, max(self.expert_hidden_size, self.hidden_size)
            )
        self.pre_expert_norm = MuseSpark1xRMSNorm(
            self.expert_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        norm_cls = MuseSpark1xRMSNormWithInputScale
        post_expert_norm = norm_cls(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.nvfp4_checkpoint = nvfp4_checkpoint
        self.use_megamoe = getattr(parallel, "enable_megamoe_for_layernorm_sp", False)
        if int4_checkpoint:
            from sglang.srt.layers.muse_spark.muse_spark_int4_moe import (
                MuseSpark1xInt4Experts,
            )

            self.experts = MuseSpark1xInt4Experts(
                num_experts=self.num_experts,
                expert_hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                layer_id=layer_idx,
                quant_config=quant_config,
                prefix=add_prefix("experts", prefix),
            )
        elif nvfp4_checkpoint:
            # Packed NVFP4: the FlashInfer TRT-LLM kernel runs the experts and we
            # finalize in torch so the post-expert norm lands before the top-k sum.
            self.experts = MuseSpark1xFp4Experts(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_start,
                hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                quant_config=quant_config,
            )
        else:
            self.experts = MuseSpark1xBf16Experts(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_start,
                expert_hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
            )
        self.experts.post_expert_norm = post_expert_norm
        self.experts.use_megamoe = self.use_megamoe
        self.sp_dispatcher = None
        self.finalize_ar_workspace = None
        self.unpool_gather_boundary: Optional[FusedPostNormResidualPreNorm] = None
        self.side_stream = None

    def set_sp_dispatcher(self, dispatcher) -> None:
        """Bind the model-scoped FlashInfer A2A workspace to this MoE layer."""
        self.sp_dispatcher = dispatcher

    def set_finalize_ar_workspace(self, workspace) -> None:
        """Bind the model-scoped decode finalize/all-reduce workspace."""
        self.finalize_ar_workspace = workspace

    def set_unpool_gather_boundary(
        self, boundary: FusedPostNormResidualPreNorm
    ) -> None:
        """Bind the FFN boundary that can absorb the unpooler all-gather."""
        self.unpool_gather_boundary = boundary

    def _side_stream(self) -> torch.cuda.Stream:
        if self.side_stream is None:
            self.side_stream = torch.cuda.Stream()
        return self.side_stream

    def _routing_op(self) -> FusedMuseSparkRouting:
        """The fused router: sigmoid, biased top-k, gather, renormalise, scale.

        Built lazily and rebound if ``expert_bias`` is replaced, because
        ``Module._apply`` (``.to()``, ``.cuda()``) swaps the buffer for a new
        tensor and the op holds a direct reference to it.
        """
        op = getattr(self, "_fused_routing_op", None)
        if op is None or op.expert_bias is not self.gate.e_score_correction_bias:
            op = FusedMuseSparkRouting(
                expert_bias=self.gate.e_score_correction_bias,
                route_scale=self.route_scale,
                router_output_multiplier=self.router_multiplier,
                router_score_division_eps=self.route_eps,
            )
            self._fused_routing_op = op
        return op

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_input: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if (
            forward_batch is not None
            and is_dp_attention_enabled()
            and get_parallel().attn_dp_size > 1
        ):
            return self._forward_dp_attention(hidden_states, forward_batch)
        return self._forward_tp(hidden_states, router_input, forward_batch)

    def _forward_tp(
        self,
        hidden_states: torch.Tensor,
        router_input: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        ar_workspace = self.finalize_ar_workspace
        # All TP ranks must select the same reduction owner.
        if ar_workspace is not None and not (
            forward_batch is not None
            and (
                forward_batch.forward_mode.is_decode()
                or forward_batch.forward_mode.is_target_verify()
            )
            and self.moe_tp_size == 1
            and not get_forward().sp_active
            and hidden_states.dtype == torch.bfloat16
            and 0 < hidden_states.shape[0] <= ar_workspace.max_tokens
            and ar_workspace.can_fuse(get_parallel().tp_group, hidden_states.shape[0])
        ):
            if (
                forward_batch is not None
                and (
                    forward_batch.forward_mode.is_decode()
                    or forward_batch.forward_mode.is_target_verify()
                )
                and not ar_workspace.fallback_logged
            ):
                logger.warning(
                    "Finalize AR fusion unavailable for this decode/verify batch "
                    "(SP=%s, dtype=%s, tokens=%s, capacity=%s); "
                    "requires non-SP BF16 decode/verify within capacity and custom "
                    "AR with a compatible push workspace. Using separate finalize and AR.",
                    get_forward().sp_active,
                    hidden_states.dtype,
                    hidden_states.shape[0],
                    ar_workspace.max_tokens,
                )
                ar_workspace.fallback_logged = True
            ar_workspace = None
        output = self._forward_experts(
            hidden_states,
            router_input,
            ar_workspace=ar_workspace,
        )
        return self._reduce_and_unpool(output, already_reduced=ar_workspace is not None)

    def _reduce_and_unpool(
        self, output: torch.Tensor, *, already_reduced: bool = False
    ) -> torch.Tensor:
        # Expert TP already reduced across ranks, before its post-expert norm.
        if (
            self.tp_size > 1
            and self.moe_tp_size == 1
            and not getattr(get_forward(), "sp_active", False)
            and not already_reduced
        ):
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:
            if _is_cpu:
                output, _ = self.post_expert_proj(output)
                return output
            comm = get_parallel().tp_group.ca_comm if self.tp_size > 1 else None
            if not getattr(get_forward(), "sp_active", False) and gemm_ag_supported(
                comm, output, self.post_expert_proj.weight
            ):
                # Returns this rank's [M, H / world] slice; the FFN boundary gathers it.
                if self._defers_unpool_gather(comm, output.shape[0]):
                    return gemm_local(comm, output, self.post_expert_proj.weight)
                return gemm_ag(comm, output, self.post_expert_proj.weight)
            output = bf16_gemm_dispatch(output, self.post_expert_proj.weight, None)
        return output

    def _defers_unpool_gather(self, comm, num_tokens: int) -> bool:
        boundary = self.unpool_gather_boundary
        return (
            boundary is not None
            and boundary.jit_selected()
            and all_gather_norm_residual_supported(
                comm=comm, num_tokens=num_tokens, hidden_size=self.hidden_size
            )
        )

    def _forward_dp_attention(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # Every rank runs the experts on the same DP-replicated global batch,
        # so the full-TP all-reduce below stays correct.
        tp_group = get_parallel().tp_group
        global_hidden = get_global_dp_buffer(tp_group)
        dp_gather_replicate(global_hidden, hidden_states, forward_batch)
        output = self._forward_experts(global_hidden)
        if self.tp_size > 1 and (not _is_cpu or self.moe_tp_size == 1):
            output = tensor_model_parallel_all_reduce(output)
        if self.post_expert_proj is not None:
            if _is_cpu:
                output, _ = self.post_expert_proj(output)
            else:
                output = self.post_expert_proj(output)
        local_output = get_local_dp_buffer(tp_group)
        dp_scatter(local_output, output, forward_batch)
        return local_output

    def _forward_experts(
        self,
        hidden_states: torch.Tensor,
        router_input: Optional[torch.Tensor] = None,
        *,
        ar_workspace=None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros((0, self.expert_hidden_size))
        use_split2 = (
            self.router_in_fp32
            and _use_split_bf16_router(hidden_states.shape[0])
            and self.gate.weight.dtype == torch.float32
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
        )
        if not use_split2 and router_input is None:
            router_input = (
                hidden_states.float() if self.router_in_fp32 else hidden_states
            )
        sp_active = getattr(get_forward(), "sp_active", False)
        prepare_early = (
            self.nvfp4_checkpoint
            # Measured to help up to 128 tokens (decode c<=128); untested above.
            and hidden_states.shape[0] <= 128
            and not sp_active
            and not getattr(self.experts, "use_marlin", False)
        )
        prepared_input = None
        use_side_stream = not _is_cpu and torch.cuda.is_available()
        main_stream = torch.cuda.current_stream() if use_side_stream else None
        side_stream = self._side_stream() if use_side_stream else None
        if use_side_stream:
            side_stream.wait_stream(main_stream)
        # nn.Linear bypasses the bf16 GEMM dispatcher.
        if self.pre_expert_proj is None:
            expert_input = hidden_states
        elif _is_cpu:
            expert_input, _ = self.pre_expert_proj(hidden_states)
        else:
            comm = get_parallel().tp_group.ca_comm if self.tp_size > 1 else None
            if (
                self.tp_size in (4, 8)
                and not sp_active
                and torch.cuda.get_device_capability(hidden_states.device) == (10, 3)
                and gemm_ag_supported(comm, hidden_states, self.pre_expert_proj.weight)
            ):
                expert_input = gemm_ag(comm, hidden_states, self.pre_expert_proj.weight)
            else:
                expert_input = bf16_gemm_dispatch(
                    hidden_states, self.pre_expert_proj.weight, None
                )
        if prepare_early:
            prepared_input = self.experts.prepare_input(
                expert_input, self.pre_expert_norm
            )
        with (
            torch.cuda.stream(side_stream)
            if use_side_stream
            else contextlib.nullcontext()
        ):
            router_logits = (
                self.gate.forward_split2(hidden_states)
                if use_split2
                else self.gate(router_input)
            )
            topk_scores, topk_ids, packed_topk_ids = self._routing_op()(
                None, router_logits, self.top_k, True, want_packed=True
            )
        if use_side_stream:
            main_stream.wait_stream(side_stream)
        if use_side_stream and not torch.cuda.is_current_stream_capturing():
            cross_stream_tensors = (
                (router_logits, topk_scores, topk_ids)
                if getattr(get_forward(), "sp_active", False)
                else (topk_scores, topk_ids, packed_topk_ids)
            )
            for tensor in cross_stream_tensors:
                if tensor is not None:
                    tensor.record_stream(main_stream)

        if (capturer := get_global_experts_capturer()) is not None:
            # Capture on main after the router join, in full unpadded token order.
            # SP experts below still consume the local ids.
            capture_topk_ids = topk_ids
            if sp_active:
                capture_topk_ids = layernorm_sp.sp_exit_gather(
                    topk_ids, num_tokens=layernorm_sp.sp_num_tokens()
                )
            capturer.capture(layer_id=self.layer_idx, topk_indices=capture_topk_ids)
        if getattr(get_forward(), "sp_active", False):
            return self._forward_sp_experts(
                expert_input, topk_scores, topk_ids, router_logits
            )
        if self.nvfp4_checkpoint:
            output = self.experts(
                expert_input,
                topk_scores,
                topk_ids,
                self.experts.post_expert_norm,
                pre_expert_norm=self.pre_expert_norm,
                prepared_input=prepared_input,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
            )
        else:
            if self.pre_expert_norm is not None:
                expert_input = self.pre_expert_norm(expert_input)
            output = self.experts(
                expert_input,
                topk_scores,
                topk_ids,
                self.experts.post_expert_norm,
                packed_topk_ids=packed_topk_ids,
                ar_workspace=ar_workspace,
            )
        return output

    def _forward_sp_experts(
        self,
        expert_input: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return combined source-token shards, before the nonlinear FFN tail."""
        if self.pre_expert_norm is not None:
            expert_input = self.pre_expert_norm(expert_input)
        if self.use_megamoe:
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import forward_megamoe

            return forward_megamoe(
                self.experts,
                expert_input,
                topk_scores,
                topk_ids,
                self.experts.post_expert_norm,
            )
        if self.sp_dispatcher is None:
            raise RuntimeError(
                "Muse Spark SP prefill requires an initialized FlashInfer A2A dispatcher."
            )
        dispatch_output = self.sp_dispatcher.dispatch(
            expert_input, StandardTopKOutput(topk_scores, topk_ids, router_logits)
        )
        combine_input = self.experts.forward_dispatched(
            dispatch_output, self.experts.post_expert_norm
        )
        return self.sp_dispatcher.combine(combine_input)


class MuseSpark1xDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.operator_dtype = config.dtype
        self.self_attn = MuseSpark1xAttention(
            config, layer_idx, add_prefix("self_attn", prefix)
        )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=False,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 1,
        )
        self.pre_feedforward_layernorm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        ffn_norm_cls = MuseSpark1xRMSNormWithInputScale
        self.post_feedforward_layernorm = ffn_norm_cls(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.is_moe_layer = self._is_moe_layer(config, layer_idx)
        self._router_dtype = torch.float32 if self.is_moe_layer else None
        self.mlp = (
            MuseSpark1xMoE(
                config,
                layer_idx,
                nvfp4_checkpoint,
                add_prefix("mlp", prefix),
                quant_config,
            )
            if self.is_moe_layer
            else MuseSpark1xMLP(config, add_prefix("mlp", prefix))
        )
        residual_cls = MuseSpark1xNormPreservingResidualConnection
        self.post_attention_residual_gate = residual_cls(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = residual_cls(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        # Each boundary -- post-norm, residual add, next pre-norm -- is one fused
        # kernel. The FFN boundary's pre-norm belongs to the next layer, so the
        # model binds it in bind_next_pre_attn_norm once every layer exists.
        self._fused_attn_boundary = _fused_norm_boundary(
            self.post_attn_norm,
            self.post_attention_residual_gate,
            self.pre_feedforward_layernorm,
        )
        self._fused_ffn_boundary = None

    @staticmethod
    def _is_moe_layer(config: MuseSparkTextConfig, layer_idx: int) -> bool:
        return config.mlp_layer_types[layer_idx] == "sparse"

    def bind_next_pre_attn_norm(self, next_pre_attn_norm: MuseSpark1xRMSNorm) -> None:
        self._fused_ffn_boundary = _fused_norm_boundary(
            self.post_feedforward_layernorm,
            self.post_feedforward_residual_gate,
            next_pre_attn_norm,
        )
        if (
            self.is_moe_layer
            and self.operator_dtype == torch.bfloat16
            and envs.SGLANG_ENABLE_MUSE_SPARK_AG_NORM_FUSION.get()
        ):
            self.mlp.set_unpool_gather_boundary(self._fused_ffn_boundary)

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns the updated residual stream and the next layer's attention input."""
        attn_output = self.self_attn(positions, attn_input, forward_batch)
        router_dtype = self._router_dtype
        if router_dtype is not None and _use_split_bf16_router(attn_output.shape[0]):
            router_dtype = None
        ffn_input, stream, router_input = self._fused_attn_boundary(
            stream,
            attn_output,
            branch_dtype=self.operator_dtype,
            router_dtype=router_dtype,
        )
        if self.is_moe_layer:
            ffn_output = self.mlp(ffn_input, router_input, forward_batch)
        else:
            ffn_output = self.mlp(ffn_input)
        # A TP-sliced branch means the unpooler gather was deferred to the FFN boundary.
        if ffn_output.shape[-1] != stream.shape[-1]:
            next_attn_input, stream = all_gather_norm_residual(
                get_parallel().tp_group.ca_comm,
                ffn_output,
                stream,
                **self._fused_ffn_boundary.jit_kernel_kwargs(),
            )
            return stream, next_attn_input
        if self._fused_ffn_boundary is None:
            return self.post_feedforward_residual_gate(
                stream, self.post_feedforward_layernorm(ffn_output)
            ), None
        next_attn_input, stream, _ = self._fused_ffn_boundary(
            stream, ffn_output, branch_dtype=self.operator_dtype
        )
        return stream, next_attn_input

    def self_attn_input_norm(self, stream: torch.Tensor) -> torch.Tensor:
        return self.input_layernorm(stream).to(self.operator_dtype)


# Register the input norm at the decoder layer, as in the HF checkpoint.
def _install_pre_attn_norm(
    config: MuseSparkTextConfig, layer: MuseSpark1xDecoderLayer
) -> None:
    layer.input_layernorm = MuseSpark1xRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        has_weight=True,
        zero_centered_gamma=True,
        residual_id=2 * layer.layer_idx + 1,
    )


class MuseSpark1xModel(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.operator_dtype = config.dtype
        self.residual_dtype = torch.float32
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
            **get_embedding_tp_kwargs(),
        )
        self.embed_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.layers = nn.ModuleList()
        # Safe to share: each MoE joins its side work before returning.
        # torch.cuda.Stream() raises outright in a CPU-only build, so the CPU
        # path never builds one; _forward_experts skips the overlap there.
        side_stream = None if _is_cpu else torch.cuda.Stream()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = add_prefix(f"layers.{layer_idx}", prefix)
            layer = MuseSpark1xDecoderLayer(
                config,
                layer_idx,
                _uses_nvfp4_experts(quant_config, layer_prefix),
                layer_prefix,
                quant_config,
            )
            if layer.is_moe_layer:
                layer.mlp.side_stream = side_stream
            _install_pre_attn_norm(config, layer)
            self.layers.append(layer)
        for layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            layer.bind_next_pre_attn_norm(next_layer.input_layernorm)
        self._sp_moe_dispatcher = None
        self._finalize_ar_workspace = None
        if is_cuda() and envs.SGLANG_ENABLE_MUSE_SPARK_FINALIZE_AR.get():
            from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
                CustomAllReduceV2,
            )
            from sglang.srt.layers.communication.muse_spark_finalize_workspace import (
                FinalizePushWorkspace,
            )

            tp_group = get_parallel().tp_group
            if (
                get_parallel().tp_size not in (4, 8)
                or get_parallel().moe_ep_size != get_parallel().tp_size
                or is_dp_attention_enabled()
                or torch.cuda.get_device_capability()[0] != 10
            ):
                logger.warning(
                    "Finalize AR fusion unavailable: requires Blackwell TP4/EP4 or TP8/EP8 "
                    "without DP attention. Using separate finalize and AR."
                )
            else:
                moe_layers = [layer.mlp for layer in self.layers if layer.is_moe_layer]
                comm = tp_group.ca_comm
                if not moe_layers:
                    logger.warning("Finalize AR fusion unavailable: no MoE layers")
                elif not isinstance(comm, CustomAllReduceV2) or comm.disabled:
                    logger.warning(
                        "Finalize AR fusion unavailable: custom AR v2 is required. "
                        "Using separate finalize and AR."
                    )
                elif (
                    moe_layers[0].expert_hidden_size <= 0
                    or moe_layers[0].expert_hidden_size % 256
                    or comm.max_push_size < moe_layers[0].expert_hidden_size * 2
                    or not 0
                    < comm._push_counter.numel()
                    <= torch.cuda.get_device_properties(
                        comm.device
                    ).multi_processor_count
                ):
                    logger.warning(
                        "Finalize AR fusion unavailable: unsupported hidden size "
                        "or push workspace capacity. Using separate finalize and AR."
                    )
                else:
                    self._finalize_ar_workspace = FinalizePushWorkspace(
                        comm,
                        tp_group.device_group,
                        hidden_dim=moe_layers[0].expert_hidden_size,
                        multicast=comm.world_size > 4 and comm.has_multicast,
                    )
                for moe_layer in moe_layers:
                    moe_layer.set_finalize_ar_workspace(self._finalize_ar_workspace)
        if layernorm_sp.layernorm_sp_enabled():
            # Scope CUTLASS collectives to Muse; leave the generic Qwen SP
            # path unchanged. Decode never calls the SP participant hooks.
            for layer in self.layers:
                linears = [layer.self_attn.qkv_proj, layer.self_attn.o_proj]
                if not layer.is_moe_layer:
                    linears.extend([layer.mlp.gate_up_proj, layer.mlp.down_proj])
                for linear in linears:
                    linear._sp_use_cutlass = True
            moe_layers = [layer.mlp for layer in self.layers if layer.is_moe_layer]
            if moe_layers and not getattr(
                get_parallel(), "enable_megamoe_for_layernorm_sp", False
            ):
                from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
                    FlashinferDispatcher,
                )

                first_moe = moe_layers[0]
                self._sp_moe_dispatcher = FlashinferDispatcher(
                    group=get_parallel().tp_group.device_group,
                    router_topk=first_moe.top_k,
                    num_experts=first_moe.num_experts,
                    num_local_experts=first_moe.num_local_experts,
                    hidden_size=first_moe.expert_hidden_size,
                    moe_runner_config=first_moe.experts.moe_runner_config,
                    enable_sp_prefill_a2a=True,
                )
                for moe_layer in moe_layers:
                    moe_layer.set_sp_dispatcher(self._sp_moe_dispatcher)
        self.norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            output_norm_gain_center_type="zero",
            residual_id=-1,
        )
        # Layer indices whose output residual stream feeds the DFLASH draft context.
        self.layers_to_capture: List[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if input_embeds is None and self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)
        sp_active = layernorm_sp.runs_sp(forward_batch.forward_mode)
        get_forward().set("sp_active", sp_active)
        if sp_active:
            hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)
        stream = hidden_states.to(self.residual_dtype)
        attn_input = self.layers[0].self_attn_input_norm(stream)
        aux_hidden_states = []
        for layer_idx, layer in enumerate(self.layers):
            stream, attn_input = layer(positions, stream, attn_input, forward_batch)
            if layer_idx in self.layers_to_capture:
                # Draft context projection reads operator dtype, not residual_dtype.
                aux_hidden_states.append(stream.to(self.operator_dtype))
        norm_input = stream.to(self.operator_dtype)
        hidden_states = self.norm(norm_input).to(self.operator_dtype)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class MuseSparkModel(nn.Module):
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.language_model = MuseSpark1xModel(
            config, quant_config, add_prefix("language_model", prefix)
        )

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return self.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )

    @property
    def layers(self):
        return self.language_model.layers


class MuseSparkTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if getattr(config, "inference_moe_w13_interleaved", False):
            raise ValueError("Interleaved expert checkpoints are not supported")
        quant_name = _quantization_name(quant_config)
        int4_checkpoint = _uses_gptq_int4_experts(quant_config)
        if quant_name not in (None, "modelopt_fp4") and not int4_checkpoint:
            raise ValueError(
                "Only BF16, serialized ModelOpt NVFP4 and AutoGPTQ int4 are "
                f"supported, got {quant_name}"
            )
        self.config = config
        self.quant_config = quant_config
        self.nvfp4_checkpoint = quant_name == "modelopt_fp4"
        self.int4_checkpoint = int4_checkpoint
        self.model = MuseSparkModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.language_model.embed_tokens)
        self.logits_processor = MuseSpark1xLogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.capture_aux_hidden_states = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if aux_hidden_states is not None and layernorm_sp.runs_sp(
            forward_batch.forward_mode
        ):
            # DFLASH consumes every prompt token in replicated token order.
            # Gather the taps before logits pruning applies full-token indices;
            # FULL capture also needs the entry-scatter padding removed.
            num_tokens = (
                input_ids.shape[0] if input_ids is not None else input_embeds.shape[0]
            )
            # Pack while sequence-sharded so the logits processor does not
            # duplicate the full replicated draft context with torch.cat.
            # Release the tap list before allocating the gathered output.
            aux_hidden_states = pack_aux_hidden_states(aux_hidden_states)
            aux_hidden_states = layernorm_sp.sp_exit_gather(
                aux_hidden_states, num_tokens=num_tokens
            )
        # Omni prefill calls the language model with materialized embeddings and
        # ``input_ids=None``. The generic logits processor cannot infer the full
        # token count in that form, so close the SP region here using the
        # replicated embedding length before it indexes last-token positions.
        if (
            input_ids is None
            and input_embeds is not None
            and layernorm_sp.runs_sp(forward_batch.forward_mode)
        ):
            hidden_states = layernorm_sp.sp_exit_gather(
                hidden_states, num_tokens=input_embeds.shape[0]
            )
            get_forward().set("sp_active", False)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        """Capture the residual stream after each listed decoder layer.

        DFLASH drafts consume the target hidden state produced by layer
        ``layer_ids[k]`` (HF ``hidden_states[layer_ids[k] + 1]``).
        """
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        num_layers = len(self.model.language_model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(
                f"DFLASH target layer ids {bad} are out of range for a "
                f"{num_layers}-layer Muse Spark target."
            )
        self.capture_aux_hidden_states = True
        self.model.language_model.layers_to_capture = list(layer_ids)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_layers(self):
        return self.model.language_model.layers

    def get_attention_sliding_window_size(self) -> Optional[int]:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    def _copy_expert_stack(
        self, param: torch.Tensor, loaded_weight: torch.Tensor, name: str
    ) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return
        parallel = get_parallel()
        if loaded_weight.shape[0] == self.config.num_local_experts:
            local = loaded_weight.shape[0] // parallel.moe_ep_size
            start = parallel.moe_ep_rank * local
            loaded_weight = loaded_weight.narrow(0, start, local)
        if param.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(param.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    @staticmethod
    def _load_int4_expert_tensor(
        param: torch.Tensor,
        param_name: str,
        loaded_weight: torch.Tensor,
        projection: str,
        suffix: str,
        expert_id: int,
    ) -> None:
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            raise ValueError(f"Parameter '{param_name}' has no expert weight loader")

        if projection == "w13" and suffix != "g_idx":
            gate, up = loaded_weight.chunk(2, dim=-1)
            loader(param, gate, param_name, shard_id="w1", expert_id=expert_id)
            loader(param, up, param_name, shard_id="w3", expert_id=expert_id)
        else:
            loader(
                param,
                loaded_weight,
                param_name,
                shard_id="w1" if projection == "w13" else "w2",
                expert_id=expert_id,
            )

    def _load_packed_int4_expert(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Load fused per-expert or native stacked GPTQ tensors."""
        import re

        suffix_pattern = r"(qweight|qzeros|scales|g_idx)"
        per_expert = re.match(
            rf"^(.*\.experts)\.(\d+)\."
            rf"(w13_weight|w2_weight|gate_up_proj)\.{suffix_pattern}$",
            name,
        )
        if per_expert is not None:
            prefix, expert_raw, checkpoint_projection, suffix = per_expert.groups()
            projection = (
                "w13"
                if checkpoint_projection in ("w13_weight", "gate_up_proj")
                else "w2"
            )
            param_name = f"{prefix}.{projection}_{suffix}"
            param = params.get(param_name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{name}' has no parameter '{param_name}'"
                )
            self._load_int4_expert_tensor(
                param,
                param_name,
                loaded_weight,
                projection,
                suffix,
                int(expert_raw),
            )
            return True

        stacked = re.match(
            rf"^(.*\.experts)\.(gate_up_proj|down_proj)\.{suffix_pattern}$",
            name,
        )
        if stacked is not None:
            prefix, checkpoint_projection, suffix = stacked.groups()
            projection = "w13" if checkpoint_projection == "gate_up_proj" else "w2"
        else:
            stacked = re.match(
                rf"^(.*\.experts)\.(w13|w2)_{suffix_pattern}$",
                name,
            )
            if stacked is None:
                return False
            prefix, projection, suffix = stacked.groups()
        param_name = f"{prefix}.{projection}_{suffix}"
        param = params.get(param_name)
        if param is None:
            raise ValueError(
                f"Checkpoint weight '{name}' has no parameter '{param_name}'"
            )
        if loaded_weight.shape[0] != self.config.num_local_experts:
            raise RuntimeError(
                f"Expert count mismatch for {name}: "
                f"checkpoint={loaded_weight.shape[0]}, "
                f"config={self.config.num_local_experts}"
            )
        for expert_id, expert_weight in enumerate(loaded_weight):
            self._load_int4_expert_tensor(
                param,
                param_name,
                expert_weight,
                projection,
                suffix,
                expert_id,
            )
        return True

    def _load_expert_projection(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        import re

        match = re.match(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)$", name
        )
        if match is None:
            return False
        prefix, expert_raw, projection, suffix = match.groups()
        expert_id = int(expert_raw)
        if expert_id >= self.config.num_local_experts:
            return False
        is_down = projection == "down_proj"
        if self.int4_checkpoint:
            param_name = f"{prefix}.{'w2' if is_down else 'w13'}_{suffix}"
        elif suffix == "weight":
            param_name = f"{prefix}.{'down_proj' if is_down else 'gate_up_proj'}"
        else:
            return False
        param = params.get(param_name)
        if param is None:
            return False
        param.weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=_EXPERT_SHARD_IDS[projection],
            expert_id=expert_id,
        )
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        params.update(
            {name: value for name, value in self.named_buffers() if name not in params}
        )
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        stacked.append((".qkv_proj", ".gate_proj", "g"))

        skipped = 0
        for raw_name, loaded_weight in weights:
            name = raw_name
            if name.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
                # For below, using padding logic will have this "original_",
                # and currently it only exists under CPU path. Thus no impacts on others paths.
                heads = (
                    getattr(
                        self.config,
                        "original_num_attention_heads",
                        self.config.num_attention_heads,
                    )
                    if name.endswith(".q_proj.weight")
                    else getattr(
                        self.config,
                        "original_total_num_kv_heads",
                        self.config.num_key_value_heads,
                    )
                )
                shape = loaded_weight.shape
                loaded_weight = (
                    loaded_weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:])
                    .transpose(1, 2)
                    .reshape(shape)
                    .contiguous()
                )
            if should_skip_weight_loading(name, self.config):
                skipped += 1
                continue
            if name.endswith(
                (".self_attn.q_scale", ".self_attn.k_scale", ".self_attn.v_scale")
            ):
                skipped += 1
                continue
            # Expert activation scales are only meaningful when the experts stay
            # packed: the FP4 quant method turns them into w13/w2_input_scale_quant
            # and g1_scale_c. A BF16 expert loop has no use for them.
            if name.endswith((".gate_up_proj_input_scale", ".down_proj_input_scale")):
                if not self.nvfp4_checkpoint:
                    skipped += 1
                    continue
            # Packed NVFP4 experts reuse the BF16 parameter names without an
            # expert weight loader.
            if not self.nvfp4_checkpoint and self._load_expert_projection(
                params, name, loaded_weight
            ):
                continue
            if self.int4_checkpoint and self._load_packed_int4_expert(
                params, name, loaded_weight
            ):
                continue
            handled = False
            original_name = name
            for param_pattern, weight_pattern, shard_id in stacked:
                if weight_pattern not in name:
                    continue
                candidate = name.replace(weight_pattern, param_pattern)
                param = params.get(candidate)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue
            name = original_name
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{raw_name}' is outside the Muse Spark text model"
                )
            if ".experts." in name and ".post_expert_norm." not in name:
                self._copy_expert_stack(param, loaded_weight, name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        self.post_load_weights()

    @torch.no_grad()
    def post_load_weights(self) -> None:
        for layer in self.model.language_model.layers:
            for residual in (
                layer.post_attention_residual_gate,
                layer.post_feedforward_residual_gate,
            ):
                if isinstance(residual, MuseSpark1xNormPreservingResidualConnection):
                    residual.precompute_coefficients()
        for module in self.modules():
            if isinstance(module, MuseSpark1xRMSNorm):
                module.precompute_effective_weight()
        # Must follow precompute_effective_weight: the finalize op binds the weight.
        for module in self.modules():
            if (
                isinstance(module, MuseSpark1xMoE)
                and not module.nvfp4_checkpoint
                and not self.int4_checkpoint
            ):
                module.experts.prepare_trtllm_weights(module.experts.post_expert_norm)


__all__ = [
    "MuseSparkTextForCausalLM",
    "MuseSparkTextConfig",
]
