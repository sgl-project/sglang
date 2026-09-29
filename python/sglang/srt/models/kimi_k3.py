# Kimi-K3 multimodal model: KimiLinear text backbone + MoonViT3d vision tower.
# Based on kimi_linear.py with K3-specific features:
#   - Attention Residual (attn_res_block_size)
#   - Latent MoE (routed_expert_hidden_size)
#   - SiTU activation
#   - MLA output gate (mla_use_output_gate)
#   - Full-rank KDA gate (use_full_rank_gate)

import logging
import os
import re
from array import array
from collections.abc import Iterable
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    divide,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.activation import SiluAndMul, SituAndMul
from sglang.srt.layers.attn_residual import aggregate_stream, get_cw
from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStateAccumulator,
    AuxHiddenStateList,
    AuxHiddenStatePacker,
)
from sglang.srt.layers.communication import k3_ar_fusion, k3_sp_collective
from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layer_boundary import (
    ReadoutFusion,
    SumGroup,
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    REPLACE_AT_EXIT,
    NormReadout,
)
from sglang.srt.layers.layer_boundary.residual.attn_bank import (
    AttnBank,
    AttnBankOutputRead,
    AttnBankState,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    LinearParallelGroup,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import route_quant_handoff
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import (
    TopK,
    TopKOutputFormat,
    build_precomputed_topk_output,
    precomputed_topk_postprocess_is_noop,
)
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, MoEGate
from sglang.srt.models.kimi_k3_vl import (
    KimiK3MultiModalProjector,
    KimiK3VisionTower,
)
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.encoder_preprocessing import EncoderMediaProcessorConfig
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    fill_transparent_bg,
    normalization_tensors,
    to_chw_uint8,
)
from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_platform,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import is_hip, is_npu, make_pp_layers
from sglang.srt.utils.common import (
    BumpAllocator,
    add_prefix,
    get_bool_env_var,
    rank0_log,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)

# `experts.<expert_id>.<w1|w2|w3>.` fragment of a checkpoint tensor name, the
# key FusedMoE.make_expert_params_mapping entries match on.
_EXPERT_WEIGHT_NAME = re.compile(r"experts\.\d+\.w[123]\.")
_is_hip = is_hip()
_is_npu = is_npu()
_aiter_k3_opt = get_bool_env_var("SGLANG_AITER_K3_OPT")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _uses_modelopt_fp8_pb_wo(
    quant_config: Optional[QuantizationConfig], prefix: str
) -> bool:
    resolver = getattr(quant_config, "_resolve_quant_algo", None)
    return resolver is not None and resolver(prefix) == "FP8_PB_WO"


def _uses_split_gguf_kv_b(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether a K3 checkpoint stores MLA K/V as separate GGUF tensors."""
    return bool(getattr(quant_config, "supports_kimi_k3_split_gguf_kv_b", False))


def _maybe_map_fp8_pb_scale_name(name: str, params_dict: dict) -> str:
    """Map ModelOpt FP8_PB_WO scale keys to SGLang block-FP8 params."""
    if name.endswith(".weight_scale"):
        candidate = name.removesuffix(".weight_scale") + ".weight_scale_inv"
        if candidate in params_dict:
            return candidate
    return name


def _get_k3_dense_weight(module: nn.Module) -> torch.Tensor:
    """Return a dense weight with serialized block-FP8 scales applied."""
    weight = module.weight.data
    if not hasattr(module, "weight_scale_inv"):
        return weight
    return block_quant_dequant(
        weight,
        module.weight_scale_inv,
        module.quant_method.weight_block_size,
        module.params_dtype,
    )


def _k3_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """F.linear / torch.mm with the same TGV dispatch module-level GEMMs get
    through UnquantizedLinearMethod. The fused MoE front and the deferred
    shared down GEMM call torch directly on raw merged weights, so the
    --bf16-gemm-backend cutedsl selection would silently skip them."""
    if out is None and out_dtype is not None and out_dtype != x.dtype:
        out = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=out_dtype, device=x.device
        )
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend

        if get_bf16_gemm_backend().is_cutedsl():
            from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
                cutedsl_bf16_gemm,
                cutedsl_bf16_gemm_out,
                use_cutedsl_bf16_gemm,
            )

            if use_cutedsl_bf16_gemm(x.shape[0], weight.shape[0], weight.shape[1]):
                if out is None:
                    return cutedsl_bf16_gemm(x, weight)
                return cutedsl_bf16_gemm_out(x, weight, out)
    if out is None:
        return torch.nn.functional.linear(x, weight)
    if out.dtype != x.dtype:
        return torch.mm(x, weight.t(), out=out, out_dtype=out.dtype)
    return torch.mm(x, weight.t(), out=out)


# Fully fused KDA decode step (kernels/ops/attention/kda_fused_decode). The
# model hands the output-norm gate to the KDA backend via an attempt-and-verify
# stash on the attention layer; unconsumed stashes fall back to the unfused
# chain + o_norm here.


def _is_unquantized_mergeable(weights: list[torch.Tensor]) -> bool:
    """Return whether these weights may be concatenated into one fused buffer.

    _merge_weights_as_views cats .weight alone, so anything carrying a separate
    scale tensor (per-channel FP8, packed MXFP4) must stay unfused."""
    dtypes = {weight.dtype for weight in weights}
    return len(dtypes) == 1 and dtypes.pop() in (torch.bfloat16, torch.float16)


def _merge_weights_as_views(
    mods: list, pad_rows_to: int = 1
) -> tuple[torch.Tensor, list[int]]:
    """Cat module weights along dim 0; re-point each module's weight to a view
    of the merged buffer so the original storage is freed (net extra memory ~0).

    With pad_rows_to > 1 the merged buffer gets zero rows appended up to the
    next multiple, so every row of the fused GEMM output stays 16-byte aligned
    for vectorized consumers."""
    ws = [m.weight.data for m in mods]
    sizes = [w.shape[0] for w in ws]
    pad = (-sum(sizes)) % pad_rows_to
    if pad:
        ws = ws + [ws[0].new_zeros((pad, ws[0].shape[1]))]
    merged = torch.cat(ws, dim=0).contiguous()
    off = 0
    for m, n in zip(mods, sizes):
        m.weight.data = merged[off : off + n]
        off += n
    return merged, sizes


class KimiK3MLP(nn.Module):
    """K3 MLP; SiLU or SiTU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        parallel_group: Optional[LinearParallelGroup] = None,
    ) -> None:
        super().__init__()
        # The Ascend path shards the dense MLP inside each attention-TP
        # replica.  The GPU K3 refactor instead gathers all DP rows and shards
        # this one dense layer over the full TP group.  Keep the GPU default,
        # but allow the NPU launcher to retain the proven attention-TP layout
        # without a device-type branch in shared model code.
        self._dense_attn_tp = (
            get_parallel().enable_dense_mlp_attn_tp
            and is_dp_attention_enabled()
            and parallel_group is None
        )
        if parallel_group is None:
            parallel_group = "attn_tp" if self._dense_attn_tp else "tp"
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            parallel_group=parallel_group,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            use_dp_attention_reduce=self._dense_attn_tp,
            prefix=f"{prefix}.down_proj",
            parallel_group=parallel_group,
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=activation_situ_beta or 1.0,
                linear_beta=activation_situ_linear_beta,
            )
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


def _add3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
    *,
    prefetch_bc: bool = False,
) -> torch.Tensor:
    """bf16(a + b) [+ c]. A pending c (the attn-res delayed +prefix_sum)
    collapses the two elementwise adds into the 3-way JIT kernel — one
    launch and one memory pass; its double rounding matches the unfused
    pair bit-for-bit. prefetch_bc loads b/c before the PDL wait: only pass
    True when their producers are at least two kernels back."""
    if c is None:
        return a + b
    from sglang.kernels.ops.elementwise import add3

    if not add3.covered(a, b, c):
        return (a + b) + c
    return add3.add3(a, b, c, prefetch_bc=prefetch_bc)


# One-shot log guard: proves the merged front is live (see _ep_front).
_EP_FRONT_LOGGED = False


def _o_proj_takes_output(o_proj: RowParallelLinear) -> bool:
    """Whether o_proj can write into caller-owned storage. ``apply_into`` is an
    optional quant-method capability; only the unquantized method has it."""
    return getattr(o_proj.quant_method, "apply_into", None) is not None


def _k3_all_reduce_add(hidden_states, residual, forward_batch):
    """o_proj's attention-TP sum with the pending residual add in K3's fused
    all-reduce, for an output o_proj wrote into its multicast buffer."""
    if k3_ar_fusion.find_mc_ptr(hidden_states) is None:
        return None
    return k3_ar_fusion.all_reduce(hidden_states, residual)


def _k3_reduce_scatter_add(hidden_states, residual, forward_batch):
    """o_proj's attention-TP sum onto this rank's slice with the pending
    residual add, in K3's tuned reduce-scatter when it takes the batch."""
    return k3_sp_collective.reduce_scatter_res(hidden_states, residual)


def _route_sp_o_proj_output(o_proj: RowParallelLinear) -> None:
    """Under SP-MoE, write o_proj's TP-partial output into K3's persistent
    symmetric buffer whenever K3's tuned reduce-scatter for that row count
    reads through it (the bank's FFN read); otherwise the stage boundary
    reduce-scatters it from regular memory."""
    if k3_sp_collective.enabled():
        # The table selects NVLS pull RS for larger token buckets.
        # Only those o_proj outputs come from the persistent symmetric
        # buffer; small push RS keeps the regular graph allocator.
        _sp_inner_o_proj_forward = o_proj.forward

        def _sp_o_proj_forward(x, *args, **kwargs):
            output_rows = x.shape[0]
            if k3_sp_collective.requires_symmetric_rs(
                output_rows, x.device, x.element_size()
            ):
                output = k3_sp_collective.get_o_proj_output_buffer(
                    output_rows, x.dtype, o_proj.output_size
                )
                result = _sp_inner_o_proj_forward(
                    x, *args, output_tensor=output[: x.shape[0]], **kwargs
                )
                k3_sp_collective.register_o_proj_output(result[0], output)
                return result
            return _sp_inner_o_proj_forward(x, *args, **kwargs)

        o_proj.forward = _sp_o_proj_forward


def _k3_symm_o_proj_out(o_proj: RowParallelLinear, x: torch.Tensor) -> torch.Tensor:
    """Symmetric storage for o_proj's TP-partial output; the fused attention
    all-reduce reduces it in place."""
    return k3_ar_fusion.symm_buffer(
        k3_ar_fusion.ATTN_O_PROJ, x.shape[0], o_proj.weight.shape[0], x.dtype
    )


class KimiK3MoE(nn.Module):
    """K3 MoE with Latent MoE (experts run in moe_hidden_size space)."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        moe_renormalize = config.moe_renormalize
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream

        self.use_latent_moe = config.routed_expert_hidden_size is not None
        # Merged front weight ([H, gate_up + E + latent]), built after weight
        # loading by _merge_front_weights().
        self._front_w: Optional[torch.Tensor] = None
        self._front_sizes: Optional[List[int]] = None
        # True when _front_w merges only [gate, routed_expert_down_proj] (the EP
        # a2a pair) rather than the three-way fused-front weight.
        self._front_is_ep_pair = False
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else hidden_size
        )

        # Gate — fp32 output so routing (sigmoid, bias add, top-k) runs in
        # full precision (matches GateLinear in mke). codespell:ignore mke
        self.gate = MoEGate(config, quant_config=None, prefix=f"{prefix}.gate")

        # For MXFP4 compressed-tensors on non-NPU, replace quant_config with
        # Mxfp4Config so FusedMoE's weight_loader uses the MXFP4 fast path. On
        # NPU the compressed-tensors config is kept so the scheme-based path
        # selects NPUCompressedTensorsW4A8mxfp4MoE (see get_moe_scheme).
        moe_quant_config = quant_config
        if (
            quant_config is not None
            and getattr(quant_config, "quant_format", None)
            and "mxfp4" in quant_config.quant_format
            and not _is_npu
        ):
            from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config

            moe_quant_config = Mxfp4Config(is_checkpoint_mxfp4_serialized=True)

        # Routed experts (operate in moe_hidden_size space)
        # gate_up_interleaved=False: K3 loads per-expert w1/w3 into non-interleaved layout
        self.experts = get_moe_impl_class(moe_quant_config)(
            num_experts=getattr(config, "n_routed_experts", config.num_experts),
            top_k=config.num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_idx,
            quant_config=moe_quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=config.hidden_act,
            gemm1_alpha=config.activation_situ_beta,
            gemm1_clamp_limit=config.activation_situ_linear_beta,
            gate_up_interleaved=False,
            # trtllm fused-routing MoE backends (e.g. nvfp4 w4a4) route inside
            # the kernel and require the routing method; K3 uses DSv3-style
            # grouped topk with e_score_correction_bias.
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_token,
            layer_id=self.layer_idx,
            renormalize=moe_renormalize,
            use_grouped_topk=True,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            scoring_func=config.moe_router_activation_func,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            # TRT-LLM cannot consume fused-front's row-strided router logits;
            # keep K3's FP32 router and pass precomputed top-k instead.
            output_format=(
                TopKOutputFormat.STANDARD
                if quant_config is None
                or (
                    config.hidden_act == "situ"
                    and (
                        get_moe_runner_backend().is_flashinfer_mxfp4()
                        or get_moe_runner_backend().is_flashinfer_trtllm()
                    )
                )
                # mega pre-dispatch consumes raw topk_ids/topk_weights
                or get_moe_a2a_backend().is_megamoe()
                or get_moe_a2a_backend().is_flashinfer_megamoe()
                else None
            ),
        )

        # MegaMoE (deep_gemm fused a2a+GEMM over the EP symm buffer) replaces
        # the routed experts call below. K3 routes ALL batches through it when
        # enabled -- its non-mega fallback is a StandardDispatcher without a2a,
        # wrong for scattered tokens -- so
        # SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK must cover the
        # per-rank prefill chunk.
        self._use_mega_moe = get_moe_a2a_backend().is_megamoe()
        self._mega_intermediate_size = moe_intermediate_size
        self._mega_top_k = config.num_experts_per_token
        if self._use_mega_moe:
            assert self.use_latent_moe and config.hidden_act == "situ"
            assert (
                config.activation_situ_beta,
                config.activation_situ_linear_beta,
            ) == (4.0, 25.0), (
                "MegaMoE SiTU kernel bakes beta=4.0/linear_beta=25.0; "
                "got a checkpoint with different constants"
            )

        # EP a2a backends move each row to its experts directly, so the MoE
        # region consumes whatever rows this rank holds (SP-MoE shard or
        # DP-local batch). No DP gather and no TP reduce anywhere in the
        # region.
        _a2a_backend = get_moe_a2a_backend()
        self._ep_a2a = (
            _a2a_backend.is_megamoe()
            or _a2a_backend.is_flashinfer_megamoe()
            or _a2a_backend.is_deepep()
            or _a2a_backend.is_mooncake()
            or _a2a_backend.is_ascend_fuseep()
            or _a2a_backend.is_mori()
        )

        # Defer the trtllm-gen finalize (top-k weighted unpermute) into the
        # push all-reduce's staging pass (k3_ar_fusion.finalize_all_reduce_push_norm)
        # so the rank-local latent never materializes. Sizes beyond the push
        # window fall back to the in-op finalize at runtime.
        self._defer_moe_finalize = (
            get_moe_runner_backend().is_flashinfer_mxfp4()
            and config.hidden_act == "situ"
        )

        # Shared experts operate on original hidden states. EP a2a gives each
        # rank a token shard: either replicate the weights, or gather within
        # the shared-expert TP subgroup and reduce-scatter back to those rows.
        parallel = get_parallel()
        requested_shared_tp = parallel.shared_experts_tp_size
        shared_tp = requested_shared_tp
        if shared_tp is None and parallel.enable_shared_experts_attn_tp:
            shared_tp = parallel.attn_tp_size
        if requested_shared_tp is not None and not self._ep_a2a:
            raise ValueError("Independent shared-expert TP requires an EP a2a backend.")
        self._shared_experts_tp1 = self._ep_a2a and shared_tp in (None, 1)
        self._shared_experts_tp_comm = (
            self._ep_a2a and shared_tp is not None and shared_tp > 1
        )
        self._shared_experts_tp_group = None
        shared_experts_parallel_group = None
        if self._shared_experts_tp1:
            shared_experts_parallel_group = "replicated"
        elif self._shared_experts_tp_comm:
            group = (
                parallel.shared_experts_tp_group
                if requested_shared_tp is not None
                else parallel.attn_tp_group
            )
            assert group.world_size == shared_tp
            self._shared_experts_tp_group = group
            shared_experts_parallel_group = (
                "shared_experts_tp" if requested_shared_tp is not None else "attn_tp"
            )
        else:
            # The full TP group, which _reduce_shared sums over, whatever the
            # dense MLP's sharding.
            shared_experts_parallel_group = "tp"
        if self.num_shared_experts is not None and self.num_shared_experts > 0:
            shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
            if shared_tp is not None and shared_intermediate_size % shared_tp != 0:
                raise ValueError(
                    f"Shared-expert intermediate size ({shared_intermediate_size}) "
                    f"must be divisible by shared-expert TP size ({shared_tp})."
                )
            self.shared_experts = KimiK3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
                parallel_group=shared_experts_parallel_group,
            )
        else:
            self.shared_experts = None

        # SBO (single batch overlap): shared experts are bf16 + tp1-replicated
        # (~264 MB/layer/rank), the routed path is a2a-latency bound in decode
        # with HBM idle; issue shared on the side stream, join before the tail
        # add. NPU shared-expert TP can instead overlap the shared collectives
        # with SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM.
        self._sbo_shared_overlap = (
            self._ep_a2a
            and self.shared_experts is not None
            and self.alt_stream is not None
        )

        if self.use_latent_moe:
            latent_quant_config = (
                quant_config
                if getattr(
                    quant_config,
                    "supports_kimi_k3_quantized_latent_projections",
                    False,
                )
                else None
            )
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_norm = (
                RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)
                if config.latent_moe_use_norm
                else None
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_norm = None
            self.routed_expert_up_proj = None

        # Static eligibility for fusing the latent all-reduce with the RMSNorm
        # epilogue (SGLANG_K3_AR_FUSION): kernel needs latent == NORM_DIM and
        # shared == 2*NORM_DIM (3584 / 7168). Decided once so the hot path
        # reads a bool.
        self.fuse_ar_norm = (
            self.routed_expert_norm is not None
            and self.moe_hidden_size == k3_ar_fusion.NORM_DIM
            and hidden_size == 2 * k3_ar_fusion.NORM_DIM
        )
        # Static eligibility for the column-parallel up_proj tail (gemm_ag):
        # 1/8-column GEMV + multicast all-gather + spin-add3 replaces the
        # replicated [3584, 7168] GEMM (~1.5-2x at decode). Dims fixed to
        # fuse_ar_norm's over TP8; per-batch checks in k3_ar_fusion.
        self._gemm_ag_up_eligible = (
            self.fuse_ar_norm
            and self.tp_size == 8
            and self.routed_expert_up_proj is not None
            and isinstance(
                getattr(self.routed_expert_up_proj, "weight", None), torch.Tensor
            )
            and self.routed_expert_up_proj.weight.dtype == torch.bfloat16
            and self.routed_expert_up_proj.weight.is_contiguous()
        )

    def _merge_front_weights(self) -> None:
        """Merge shared gate_up + router gate + latent down_proj weights.

        All three GEMMs consume the same hidden_states; at decode each one is a
        skinny memory-bound GEMV with its own splitK epilogue. One merged
        [H, gu+E+latent] GEMM reads the input once and drops 2 GEMM launches
        plus their splitK-reduce tails per MoE layer.

        Called once from load_weights (after all weights are loaded, before
        cuda graph capture); only plain bf16/fp16 dense weights are merged —
        quantized or mixed-dtype checkpoints keep the unfused path.
        """
        if not self.use_latent_moe:
            return
        # These merged layouts feed CUDA and ROCm fused front kernels. Keeping the
        # regular parameters on other devices avoids a large transient copy
        # during post-load processing and leaves their native kernels in
        # control of weight layout.
        if not (get_platform().is_cuda or get_platform().is_hip):
            return
        if self.shared_experts is not None and get_moe_a2a_backend().is_none():
            mods = [
                self.shared_experts.gate_up_proj,
                self.gate,
                self.routed_expert_down_proj,
            ]
        elif envs.SGLANG_K3_FUSED_FRONT.get():
            # Merge the router gate into the latent down-proj so one GEMM reads
            # hidden_states once: the 896-row gate alone is too few to use the
            # machine well; folded into the 3584-row down-proj it is near-free.
            mods = [self.gate, self.routed_expert_down_proj]
        else:
            return
        if any(getattr(module, "weight", None) is None for module in mods):
            return
        dtypes = {m.weight.dtype for m in mods}
        if len(dtypes) != 1 or dtypes.pop() not in (torch.bfloat16, torch.float16):
            return
        self._front_w, self._front_sizes = _merge_weights_as_views(mods)
        self._front_is_ep_pair = len(mods) == 2
        # Invalidate the cached properties.
        for prop in (
            "_eligible_for_fused_front",
            "_front_fp32",
            "_routing_contract_ok",
            "_ep_front_eligible",
        ):
            self.__dict__.pop(prop, None)

    @cached_property
    def _routed_needs_reduce(self):
        return self.tp_size > 1 and get_moe_a2a_backend().is_none()

    @cached_property
    def _eligible_for_fused_front(self) -> bool:
        """The fused front commits to the single-collective tail (both
        partial sums in one symmetric buffer), so beyond the merged front
        weight it requires plain-TP routed sums (an a2a combine already
        returns the complete sum — all-reducing it again would multiply by
        tp_size) and a dense shared down weight for the direct out= GEMM."""
        return (
            self.use_latent_moe
            and self.shared_experts is not None
            and self._front_w is not None
            and not self._front_is_ep_pair
            and get_moe_a2a_backend().is_none()
            and self.shared_experts.down_proj.weight.dtype
            in (torch.bfloat16, torch.float16)
        )

    @cached_property
    def _front_fp32(self) -> bool:
        """Emit the merged front in fp32 so the router reads exact logits.

        The situ activation and the flashinfer_mxfp4 quantizer read the fp32
        slices directly. Every other runner takes routed_input rounded back to
        bf16 in _forward_fused, which is bit-identical to the bf16 front."""
        return (
            not _is_hip
            and self._eligible_for_fused_front
            and self._front_w.dtype == torch.bfloat16
        )

    def _forward_mega_experts(
        self, routed_input: torch.Tensor, topk_output
    ) -> torch.Tensor:
        """Routed experts via deep_gemm MegaMoE: fused a2a dispatch + grouped
        GEMMs + SiTU + combine over the EP-group symmetric buffer. Semantically
        equivalent to `self.experts(routed_input, topk_output)` on an a2a
        backend (combine returns fully-summed rows; `_reduce_latent` then only
        applies the norm)."""
        import deep_gemm

        from sglang.kernels.ops.moe.dsv4 import mega_moe_pre_dispatch
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.mega_moe import (
            _configure_mega_moe_deep_gemm_num_sms,
            _get_mega_moe_symm_buffer,
            _mega_moe_mma_type,
        )
        from sglang.srt.runtime_context import get_parallel

        # Under SP-MoE these rows are this rank's shard of the batch; an unpadded
        # batch stays on every row.
        num_tokens = routed_input.shape[0]
        num_max_tokens_per_rank = (
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
        )
        assert num_tokens <= num_max_tokens_per_rank, (
            f"mega MoE: num_tokens={num_tokens} exceeds "
            f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{num_max_tokens_per_rank}; K3 has no non-mega fallback — raise "
            f"the env var to cover the per-rank rows"
        )
        buf = _get_mega_moe_symm_buffer(
            get_parallel().moe_ep_group.device_group,
            num_experts=self.experts.num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=self._mega_top_k,
            hidden=self.moe_hidden_size,
            intermediate_hidden=self._mega_intermediate_size,
            mma_type=_mega_moe_mma_type(self.experts),
        )

        if num_tokens > 0:
            topk_ids_in = topk_output.topk_ids.to(torch.int32)
            topk_weights_in = topk_output.topk_weights.to(torch.float32)
        else:
            topk_ids_in = routed_input.new_empty(
                (0, self._mega_top_k), dtype=torch.int32
            )
            topk_weights_in = routed_input.new_empty(
                (0, self._mega_top_k), dtype=torch.float32
            )

        mega_moe_pre_dispatch(
            routed_input,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )
        # At least one row so the tvm-ffi binding sees a non-null data_ptr.
        y = torch.empty(
            (max(num_tokens, 1), self.moe_hidden_size),
            dtype=torch.bfloat16,
            device=routed_input.device,
        )
        with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
            deep_gemm.fp8_fp4_mega_moe(
                y,
                self.experts.mega_l1_weights,
                self.experts.mega_l2_weights,
                buf,
                recipe=(1, 1, 32),
                activation="situ",
                fast_math=True,
            )
        y = y[:num_tokens]
        if not self.experts.should_fuse_routed_scaling_factor_in_topk:
            if (
                self.routed_scaling_factor is not None
                and self.routed_scaling_factor != 1.0
            ):
                y.mul_(self.routed_scaling_factor)
        return y

    def _latent_norm(self, latent: torch.Tensor) -> torch.Tensor:
        if self.routed_expert_norm is None:
            return latent
        return self.routed_expert_norm(latent)

    @cached_property
    def _routing_contract_ok(self) -> bool:
        """Whether a kernel may emit (weights, ids) itself and bypass
        select_experts. Shared by the fused router and the merged front."""
        if self._eligible_for_fused_front:
            return False
        cfg = self.topk.topk_config
        if cfg.output_format is not TopKOutputFormat.STANDARD:
            return False
        # The kernel implements sigmoid scoring with bias-ranked ungrouped top-k.
        # K3 passes moe_router_activation_func explicitly to TopK. The legacy
        # GPU biased_grouped_topk path also hardwires sigmoid, but other platform
        # implementations consume cfg.scoring_func directly.
        if cfg.scoring_func != "sigmoid":
            return False
        if not (cfg.use_grouped_topk and cfg.correction_bias is not None):
            return False
        if (cfg.num_expert_group or 1) > 1 or (cfg.topk_group or 1) > 1:
            return False
        # A waterfill balancer rewrites the routing after the top-k; leave it on
        # the layer path that supports it.
        if self.topk.waterfill_balancer is not None or self.topk.enable_waterfill:
            return False
        if self.gate.e_score_correction_bias is None:
            return False
        # K3 calls self.topk() without a padding mask or EPLB dispatch info, so
        # select_experts' post-processing collapses to the capture hook and the
        # recorder -- both of which build_precomputed_topk_output runs. Bail out
        # if that ever stops holding rather than silently dropping the remap.
        if not precomputed_topk_postprocess_is_noop(cfg):
            return False
        if get_exec().deterministic.enable_deterministic_inference:
            return False
        try:
            from sglang.kernels.ops.moe import moe_front
        except Exception:
            return False
        return moe_front.available()

    @cached_property
    def _ep_front_eligible(self) -> bool:
        """Static eligibility for the merged EP front (gate + latent down-proj in
        one GEMM). Requires the two-module merge from _merge_front_weights and the
        same routing contract the single-kernel router needs."""
        return (
            envs.SGLANG_K3_FUSED_FRONT.get()
            and self._front_w is not None
            and self._front_is_ep_pair
            and self.use_latent_moe
            and self.routed_expert_down_proj is not None
            and self._routing_contract_ok
        )

    def _ep_front(self, hidden_states: torch.Tensor):
        """Merged front: returns ``(topk_output, routed_input)``, or None when the
        shape is not covered and the caller should run the unmerged path."""
        if not self._ep_front_eligible:
            return None
        from sglang.kernels.ops.moe import moe_front

        cfg = self.topk.topk_config
        bias = self.gate.e_score_correction_bias
        if (
            moe_front.get_front_strategy(hidden_states.shape[0], hidden_states.device)
            != "merged_fp32"
        ):
            return None
        if not moe_front.fused_front_covered(
            hidden_states, self._front_w, bias, cfg.top_k, self.moe_hidden_size
        ):
            return None

        w, i, routed = moe_front.fused_front(
            hidden_states,
            self._front_w,
            bias,
            latent=self.moe_hidden_size,
            topk=cfg.top_k,
            renormalize=cfg.renormalize,
            routed_scaling_factor=cfg.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=cfg.apply_routed_scaling_factor_on_output,
        )

        global _EP_FRONT_LOGGED
        if not _EP_FRONT_LOGGED:
            # An absence of fallback warnings does not prove a fast path ran.
            _EP_FRONT_LOGGED = True
            logger.info(
                "K3 merged MoE front active (layer %d, %d tokens)",
                self.layer_idx,
                hidden_states.shape[0],
            )
        return build_precomputed_topk_output(w, i, cfg, self.layer_idx), routed

    def _ep_front_overlap(self, hidden_states: torch.Tensor):
        """Overlap the exact fp32 gate+top-k with the latent down projection.

        The side stream is joined before returning. It is then free for the
        existing shared-expert overlap, which is deliberately issued later.
        """
        if (
            not self._ep_front_eligible
            or self.alt_stream is None
            or hidden_states.shape[0] == 0
        ):
            return None
        from sglang.kernels.ops.moe import moe_front

        if (
            moe_front.get_front_strategy(hidden_states.shape[0], hidden_states.device)
            != "overlap"
        ):
            return None

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)

        routed_input, _ = self.routed_expert_down_proj(hidden_states)
        current_stream.wait_stream(self.alt_stream)
        # Top-k tensors were allocated on alt_stream but are consumed by the
        # routed experts on current_stream. Tell the caching allocator about
        # that lifetime before alt_stream is reused for the shared experts.
        for value in topk_output:
            if isinstance(value, torch.Tensor):
                value.record_stream(current_stream)
        return topk_output, routed_input

    def _reduce_shared(self, shared_output: torch.Tensor) -> torch.Tensor:
        """The TP-sharded shared experts' sum, completed beside a routed
        output the latent reduction already completed."""
        return tensor_model_parallel_all_reduce(shared_output)

    def _reduce_latent_and_shared(self, buf: torch.Tensor) -> torch.Tensor:
        """The fused front's [latent | shared] pair in one collective."""
        return tensor_model_parallel_all_reduce(buf)

    def _reduce_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Unfused-front latent tail: TP-partial routed sums must be reduced
        in latent space BEFORE the RMSNorm (sum(norm(x_i)) != norm(sum(x_i)))."""
        if not self._routed_needs_reduce:
            return self._latent_norm(latent)
        return self._latent_norm(tensor_model_parallel_all_reduce(latent))

    def _gather_shared_expert_inputs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        group = self._shared_experts_tp_group
        # The attention DP buffer spans the entire attention-TP replica.
        # Size this buffer from the subgroup's actual token shards instead.
        with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
            gathered = hidden_states.new_empty(
                (hidden_states.shape[0] * group.world_size, *hidden_states.shape[1:])
            )
        group.all_gather_into_tensor(gathered, hidden_states)
        return gathered

    def _reduce_scatter_shared_experts(
        self, shared_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        self._shared_experts_tp_group.reduce_scatter_tensor(output, shared_output)
        return output

    def _forward_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run TP-sharded shared experts while DeepEP tokens stay scattered."""
        if not self._shared_experts_tp_comm:
            return self.shared_experts(hidden_states)

        gathered_hidden_states = self._gather_shared_expert_inputs(hidden_states)
        gathered_shared_output = self.shared_experts(gathered_hidden_states)
        return self._reduce_scatter_shared_experts(
            gathered_shared_output, hidden_states
        )

    def _can_overlap_shared_experts_npu(self, hidden_states: torch.Tensor) -> bool:
        if not (
            _is_npu
            and envs.SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM.get()
            and self._sbo_shared_overlap
            and self._shared_experts_tp_comm
            and self.use_latent_moe
            and hidden_states.shape[0] > 0
            and get_moe_a2a_backend().is_deepep()
        ):
            return False

        from sglang.srt.batch_overlap.two_batch_overlap import (
            MaybeTboDeepEPDispatcher,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            is_in_tc_piecewise_cuda_graph,
        )

        # The hooks must surround the complete dispatch, including its receive
        # wait. Fused EP bypasses these hooks. An eager/piecewise graph break
        # must not split the side-stream event record from its wait.
        return (
            isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            and not is_in_breakable_cuda_graph()
            and not is_in_tc_piecewise_cuda_graph()
        )

    def _forward_unfused(
        self,
        hidden_states: torch.Tensor,
        *,
        prefix_sum: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Front section with three separate GEMMs, each reading
        hidden_states: shared-expert MLP, router gate, latent down-proj."""
        # Shared experts on original hidden_states; under SBO they go to the
        # side stream, joined at the tail. CUDA issues this after the front so
        # they overlap the routed a2a rather than the front GEMMs; NPU starts
        # before the front. Fine-grained NPU overlap splits at the dispatch
        # boundaries:
        #   current: front ---------- dispatch ---------- routed GEMMs -- tail
        #   alt:     all-gather ----- shared MLP -------- reduce-scatter
        fine_grained_overlap = self._can_overlap_shared_experts_npu(hidden_states)
        shared_input = None
        shared_output = None
        shared_event = None
        shared_compute_event = None

        def issue_shared():
            nonlocal shared_input, shared_output, shared_event
            if self.shared_experts is None or hidden_states.shape[0] == 0:
                return
            if fine_grained_overlap:
                # Fork before the routed front so HCCL's completion wait is
                # queued on the side stream, leaving the front free to run.
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                hidden_states.record_stream(self.alt_stream)
                with torch.cuda.stream(self.alt_stream):
                    shared_input = self._gather_shared_expert_inputs(hidden_states)
                    shared_input.record_stream(self.alt_stream)
                return
            if self._sbo_shared_overlap:
                current_stream = torch.cuda.current_stream()
                # Keep HCCL collectives on the current stream. The alternate
                # stream only executes the shared-expert MLP.
                shared_input = hidden_states
                if self._shared_experts_tp_comm:
                    shared_input = self._gather_shared_expert_inputs(hidden_states)
                shared_input.record_stream(self.alt_stream)
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self.shared_experts(shared_input)
                    shared_event = self.alt_stream.record_event()
            else:
                shared_output = self._forward_shared_experts(hidden_states)

        def run_experts(expert_input, topk_output):
            if not fine_grained_overlap:
                return (
                    self._forward_mega_experts(expert_input, topk_output)
                    if self._use_mega_moe
                    else self.experts(expert_input, topk_output)
                )

            def pre_dispatch(dispatcher, dispatch_input, dispatch_topk):
                nonlocal shared_output, shared_compute_event
                # AllGather is already queued. Delay shared GEMMs until the
                # gate, TopK and latent down projection finish on current.
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self.shared_experts(shared_input)
                    shared_compute_event = self.alt_stream.record_event()

            def post_dispatch(dispatcher, dispatch_output):
                nonlocal shared_output, shared_event
                current_stream = torch.cuda.current_stream()
                # Dispatch has queued its receive wait. RS waits for that
                # communication and the shared MLP, while routed GEMMs wait
                # only for the MLP (not for RS).
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self._reduce_scatter_shared_experts(
                        shared_output, hidden_states
                    )
                    shared_event = self.alt_stream.record_event()
                current_stream.wait_event(shared_compute_event)

            dispatcher = self.experts.dispatcher
            pre_handle = dispatcher.register_pre_dispatch_hook(pre_dispatch)
            try:
                post_handle = dispatcher.register_post_dispatch_hook(post_dispatch)
                try:
                    return self.experts(expert_input, topk_output)
                finally:
                    post_handle.remove()
            finally:
                # Remove outside hook iteration, including on dispatch/GEMM
                # failures, so closures cannot leak into the next forward.
                pre_handle.remove()

        def wait_and_finalize_shared_experts():
            nonlocal shared_output
            if shared_event is None:
                return
            # Join just before consuming the shared result. The legacy path
            # still needs to reduce-scatter its TP-partial MLP output here.
            current_stream = torch.cuda.current_stream()
            current_stream.wait_event(shared_event)
            shared_output.record_stream(current_stream)
            if self._shared_experts_tp_comm and not fine_grained_overlap:
                shared_output = self._reduce_scatter_shared_experts(
                    shared_output, hidden_states
                )

        # Give the NPU shared-expert branch a head start. At this point
        # hidden_states is the decoder layer's post-attention RMSNorm output.
        if _is_npu and self._sbo_shared_overlap:
            issue_shared()

        # Front: gate + TopK (+ latent down-proj when merged). Strategy table:
        # kernels/ops/moe/moe_front.py.
        routed_input = self._ep_front(hidden_states)
        if routed_input is None and not fine_grained_overlap:
            routed_input = self._ep_front_overlap(hidden_states)
        topk_output = None
        if routed_input is not None:
            topk_output, routed_input = routed_input
        else:
            # MoEGate produces fp32 router logits on CUDA (via linear_bf16_fp32
            # or tiny_gemm_bf16); non-CUDA falls back to F.linear (bf16). The
            # fp32 logits reach the radix router from moe_fused_gate.
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        if not (_is_npu and self._sbo_shared_overlap):
            issue_shared()

        if not self.use_latent_moe:
            expert_output = self.experts(hidden_states, topk_output)
            wait_and_finalize_shared_experts()
            if shared_output is not None:
                expert_output = expert_output + shared_output
            return expert_output

        # Latent MoE: compress after routing, before experts
        if TYPE_CHECKING:
            assert (
                self.routed_expert_down_proj is not None
                and self.routed_expert_up_proj is not None
            )

        if routed_input is None:
            if hidden_states.shape[0] == 0:
                # Idle DP ranks must still enter the EP dispatch below so the
                # active replicas can exchange routed tokens.  Ascend's
                # quantized matmul does not accept an empty activation, so
                # materialize its shape-only result without launching GEMM.
                routed_input = hidden_states.new_empty((0, self.moe_hidden_size))
            else:
                routed_input, _ = self.routed_expert_down_proj(hidden_states)
        expert_output = run_experts(routed_input, topk_output)
        if expert_output.shape[0] == 0:
            # The EP combine returns one row per source token.  Keep the
            # source-side empty result while avoiding empty RMSNorm/up-proj
            # launches; the collective itself has already completed above.
            out = hidden_states.new_empty((0, hidden_states.shape[1]))
        else:
            latent = self._reduce_latent(expert_output)
            # up_proj is replicated, so the routed output is now fully reduced.
            out, _ = self.routed_expert_up_proj(latent)
        wait_and_finalize_shared_experts()
        if shared_output is not None:
            # tp1 shared experts (SP-MoE) are complete per-rank; TP-sharded
            # ones need the partial-sum reduction.
            if (
                self.tp_size > 1
                and not self._shared_experts_tp1
                and not self._shared_experts_tp_comm
            ):
                shared_output = self._reduce_shared(shared_output)
            out = _add3(out, shared_output, prefix_sum)
            return out
        out = out if prefix_sum is None else out + prefix_sum
        return out

    @cached_property
    def _moe_front_needs_dense_bf16(self) -> bool:
        """Whether routed_input must be repaired into a dense bf16 buffer.

        Only the SM100 trtllm-gen mxfp4 runner reads the front slice as it
        comes: its group quant (route_quant_fused / per_token_group_quant)
        takes both a strided row and an fp32 row. The SM90/SM120 cutlass mxfp4
        kernels return from apply() before that quant, and precision="bf16"
        skips it as well, so those keep the bf16 contract even though the
        runner backend is the same."""
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        method = self.experts.quant_method
        return not (
            isinstance(method, Mxfp4MoEMethod)
            and method.use_flashinfer
            and not method.use_marlin
            and method._fi_kernel == "trtllm_sm100"
            and method.flashinfer_mxfp4_moe_precision == "default"
            and method.hidden_size == self.moe_hidden_size
        )

    @cached_property
    def _route_quant_fuse_eligible(self) -> bool:
        """Whether to stage routed_input for the fused route+pack+quant launch
        (route_quant_handoff). Only the trtllm-gen SiTU runner with mxfp8
        activations consumes the staged quant, so only that runner stages."""
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        method = self.experts.quant_method
        return (
            isinstance(method, Mxfp4MoEMethod)
            and method.use_flashinfer
            and not method.use_marlin
            and method.flashinfer_mxfp4_moe_precision == "default"
            and self.experts.moe_runner_config.activation == "situ"
        )

    def _forward_routed(self, hidden_states, router_logits, routed_input, latent):
        if self._route_quant_fuse_eligible:
            route_quant_handoff.stage(routed_input)
        try:
            topk_output = self.topk(hidden_states, router_logits)
            with zero_copy_context.set_moe_output(latent):
                expert_output = self.experts(routed_input, topk_output)
        finally:
            route_quant_handoff.clear()
        if expert_output.data_ptr() != latent.data_ptr():
            latent.copy_(expert_output)

    def _forward_routed_deferred(self, hidden_states, router_logits, routed_input):
        """Routed experts with the in-op finalize skipped: returns the
        FlashInferTrtllmDeferredFinalizeOutput triple (permuted gemm2 output,
        expanded_idx_to_permuted_idx, expert_weights) for the finalize-fused
        all-reduce."""
        if self._route_quant_fuse_eligible:
            route_quant_handoff.stage(routed_input)
        try:
            topk_output = self.topk(hidden_states, router_logits)
            return self.experts.forward_deferred_finalize(routed_input, topk_output)
        finally:
            route_quant_handoff.clear()

    def _forward_shared(self, gate_up, shared_output):
        shared = self.shared_experts
        if TYPE_CHECKING:
            assert shared is not None and isinstance(
                shared.down_proj.weight, torch.Tensor
            )
        assert shared is not None
        _k3_bf16_gemm(
            shared.act_fn(gate_up),
            shared.down_proj.weight,
            out=shared_output,
        )

    def _get_fused_norm_params(self) -> tuple[torch.Tensor, float]:
        norm = self.routed_expert_norm
        assert self.fuse_ar_norm and norm is not None
        return norm.weight, norm.variance_epsilon

    def _forward_fused(
        self, hidden_states: torch.Tensor, *, prefix_sum: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fused-front pipeline: read hidden_states once through the merged
        [H, gate_up + E + latent] weight, then land both TP-partial sums in
        one flat symmetric [latent | shared] buffer with zero copies — the
        shared down GEMM writes its slice via out=, the MoE runner writes
        its top-k sum via the zero-copy context — and all-reduce the pair
        in a single collective (the symmetric mempool keeps the one-shot
        allreduce path; same trick as RowParallelLinear)."""
        if TYPE_CHECKING:  # NOTE: precondition for this case
            assert (
                self._front_w is not None
                and self._front_sizes is not None
                and self.moe_hidden_size is not None
                and self.shared_experts is not None
                and isinstance(self.shared_experts.down_proj.weight, torch.Tensor)
                and self.routed_expert_up_proj is not None
            )

        num_tokens, hidden_size = hidden_states.shape
        fused = _k3_bf16_gemm(
            hidden_states,
            self._front_w,
            out_dtype=torch.float32 if self._front_fp32 else None,
        )
        gate_up, router_logits, routed_input = torch.split(
            fused, self._front_sizes, dim=-1
        )
        if num_tokens > 1 and _is_hip and not _aiter_k3_opt:
            router_logits = router_logits.contiguous()
        if self._moe_front_needs_dense_bf16:
            # off an fp32 front the cast allocates the dense buffer, so the
            # contiguous() behind it is free; off a bf16 front it is the copy
            routed_input = routed_input.to(hidden_states.dtype).contiguous()
        latent_numel = num_tokens * self.moe_hidden_size
        if k3_ar_fusion.enabled():
            # the shared-expert AR is pull-only, so its input must be a
            # symm_buffer slice for every rank to resolve the same offset
            buf = k3_ar_fusion.symm_buffer(
                k3_ar_fusion.MOE_LATENT_SHARED,
                num_tokens,
                self.moe_hidden_size + hidden_size,
                hidden_states.dtype,
            ).view(-1)
        else:
            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)

        latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
        shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
        fused_norm = False
        if self.alt_stream is not None and k3_ar_fusion.enabled():
            defer_finalize = (
                self._defer_moe_finalize
                and self.fuse_ar_norm
                and k3_ar_fusion.finalize_push_fits(num_tokens)
            )
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if defer_finalize:
                deferred = self._forward_routed_deferred(
                    hidden_states, router_logits, routed_input
                )
            else:
                self._forward_routed(hidden_states, router_logits, routed_input, latent)
            with torch.cuda.stream(self.alt_stream):
                self._forward_shared(gate_up, shared_output)
                # low-SM pull so the side-stream AR leaves the SMs to the
                # routed GEMMs it overlaps (K3 dims are fixed; tuned here)
                k3_ar_fusion.all_reduce_low_sm(shared_output, num_blocks=4, unroll=8)
            current_stream.wait_stream(self.alt_stream)
            # The latent AR must stay serialized after the shared AR (both
            # reuse the v2 pull semaphores); the join above does it.
            if defer_finalize:
                # finalize folded into the push AR's staging pass; the norm
                # covers every latent row
                fused_norm = True
                k3_ar_fusion.finalize_all_reduce_push_norm(
                    latent,
                    deferred.gemm2_out,
                    deferred.expanded_idx_to_permuted_idx,
                    deferred.expert_weights,
                    *self._get_fused_norm_params(),
                )
            elif self.fuse_ar_norm:
                fused_norm = True
                k3_ar_fusion.all_reduce_norm(
                    latent.view(-1, self.moe_hidden_size),
                    *self._get_fused_norm_params(),
                    num_tokens=num_tokens,
                )
            else:
                k3_ar_fusion.all_reduce(latent)
            # the gemm_ag tail wants the normed latent straight out of the
            # fused-norm AR (its GEMV chains on it via PDL)
            if (
                fused_norm
                and self._gemm_ag_up_eligible
                and k3_ar_fusion.gemm_ag_up_fits(num_tokens)
            ):
                return k3_ar_fusion.gemm_ag_up_proj(
                    latent,
                    self.routed_expert_up_proj.weight,  # type: ignore
                    shared_output,
                    prefix_sum,
                )
        else:  # single collective over the flat [latent | shared] pair
            self._forward_shared(gate_up, shared_output)
            self._forward_routed(hidden_states, router_logits, routed_input, latent)
            if self.fuse_ar_norm and k3_ar_fusion.enabled():
                fused_norm = True
                k3_ar_fusion.all_reduce_norm(
                    buf.view(-1, k3_ar_fusion.NORM_DIM),
                    *self._get_fused_norm_params(),
                    num_tokens=num_tokens,
                )
            elif k3_ar_fusion.enabled():
                k3_ar_fusion.all_reduce(buf)
            else:
                buf = self._reduce_latent_and_shared(buf)

        latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
        shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
        if not fused_norm:
            latent = self._latent_norm(latent)
        out, _ = self.routed_expert_up_proj(latent)

        # prefetch_bc: b and c complete before the norm / up_proj chain
        # starts; only `a`'s producer can still be in flight at PDL entry.
        return _add3(out, shared_output, prefix_sum, prefetch_bc=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        prefix_sum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """A latent MoE, which writes the next stream itself, takes the pending
        prefix_sum: folded into the 3-way JIT tail add when covered, plain adds
        otherwise (bit-identical either way). A non-latent MoE leaves the
        residual add to its stage boundary and takes none."""
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if hidden_states.shape[0] > 0 and self._eligible_for_fused_front:
            out = self._forward_fused(hidden_states, prefix_sum=prefix_sum)
        else:
            out = self._forward_unfused(hidden_states, prefix_sum=prefix_sum)
        return out.view(num_tokens, hidden_size)


class KimiK3DeltaAttention(nn.Module):
    """KDA attention; optional full-rank gate."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        all_reduce_fusion: bool = False,
        bfa_alt_stream: Optional[torch.cuda.Stream] = None,
        sp_moe: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.all_reduce_fusion = all_reduce_fusion
        # Side stream for the [f_a|b] + f_b tiny GEMVs: they read only
        # hidden_states, so they overlap the wide [q,k,v,g] GEMM on the main
        # stream (graphed decode/verify only; SM bound as the MLA gate stream).
        self._bfa_alt_stream = bfa_alt_stream
        self._bfa_bs_limit = (
            (128 if get_platform().is_blackwell else 64)
            if bfa_alt_stream is not None
            else 0
        )
        self.tp_size = get_parallel().tp_size
        # KDA is an attention layer: head-sharded params follow the
        # attention-TP group, matching the mamba state cache sizing.
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.linear_attn_config["head_dim"]
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % self.attn_tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.attn_tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.use_full_rank_gate = config.linear_attn_config.get(
            "use_full_rank_gate", False
        )
        self._bfa_uses_block_fp8 = self.use_full_rank_gate and _uses_modelopt_fp8_pb_wo(
            quant_config, f"{prefix}.b_proj"
        )

        # The full-rank [q, k, v, g] merged projection is explicitly sharded
        # using attention-TP placement, so it also supports DP attention.
        # The low-rank fused path still uses full-TP-only projection helpers.
        # For the full-rank gate (K3) the checkpoint quantizes only the MoE
        # experts; attention linears resolve to UnquantizedLinearMethod, so a
        # non-None quant_config is fine for the merged projection.
        self.do_fuse_qkvbfg = quant_config is None and self.attn_tp_size == self.tp_size

        if self.use_full_rank_gate:
            # Fuse only the wide projections [q, k, v, g]: folding b (12/rank)
            # and f_a (128, replicated) in skews the output dim and degrades
            # GEMM kernel selection; they stay as separate tiny GEMVs. ROCm
            # reverses this below the token threshold
            # (_merge_kda_inproj_weights_hip).
            self.fused_qkvg_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [
                    projection_size,
                    projection_size,
                    projection_size,
                    projection_size,
                ],
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=f"{prefix}.fused_qkvg_proj",
            )
            self.split_sizes = [
                3 * projection_size // self.attn_tp_size,
                projection_size // self.attn_tp_size,
            ]
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=f"{prefix}.b_proj",
                # TP8 shards K3's 96 beta rows below the 128-row FP8 block.
                skip_block_quant_check=self._bfa_uses_block_fp8,
            )
            self.f_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )
            self.f_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=f"{prefix}.f_b_proj",
            )
            # Merged [f_a | b] weight, built after weight loading by
            # _merge_bfa_weights().
            self._bfa_w: Optional[torch.Tensor] = None
            self._bfa_f_b_w: Optional[torch.Tensor] = None
            if _is_hip:
                # ROCm only: _merge_kda_inproj_weights_hip() may merge the
                # whole [q,k,v,g | f_a | b] in-proj instead, making _bfa_w a
                # tail view of that buffer. _qkvgbfa_sizes is the split of the
                # buffer, and stays None when the fusion does not apply. These
                # attributes exist on ROCm only; every reader is _is_hip-gated.
                self._qkvgbfa_layer: Optional[SimpleNamespace] = None
                self._qkvgbfa_sizes: Optional[list[int]] = None
                self._qkvgbfa_bs_limit = (
                    envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS.get()
                )
        elif self.do_fuse_qkvbfg:
            self.qkvb_sizes = [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,
                self.fg_sizes,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkvbfg_a_proj",
            )
            self.split_sizes = [
                3 * projection_size // self.tp_size,
                self.num_heads // self.tp_size,
                2 * self.head_dim,
            ]
            _dtype = config.dtype
            if isinstance(_dtype, str):
                _dtype = getattr(torch, _dtype, torch.bfloat16)
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2, self.head_dim, projection_size, dtype=_dtype
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                v_head_size=self.head_v_dim,
                prefix=f"{prefix}.qkv_proj",
            )

            self.f_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )
            self.f_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=f"{prefix}.f_b_proj",
            )
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=f"{prefix}.b_proj",
            )

            if self.use_full_rank_gate:
                self.g_proj = ColumnParallelLinear(
                    self.hidden_size,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    parallel_group="attn_tp",
                    prefix=f"{prefix}.g_proj",
                )
            else:
                self.g_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.g_a_proj",
                )
                self.g_b_proj = ColumnParallelLinear(
                    self.head_dim,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    parallel_group="attn_tp",
                    prefix=f"{prefix}.g_b_proj",
                )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.attn_tp_size), dtype=torch.float32)
        )
        set_weight_attrs(
            self.dt_bias,
            {"weight_loader": sharded_weight_loader(0, parallel_group="attn_tp")},
        )

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            parallel_group="attn_tp",
            prefix=f"{prefix}.qkv_conv1d",
        )
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        # Checkpoint stores A_log as [head_dim]; the FLA kernel expects
        # local_num_heads. The loader handles both the old 4-D and the K3 1-D
        # formats by narrowing to the first num_heads then TP-sharding.
        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )

        def _a_log_weight_loader(
            param: torch.Tensor, loaded_weight: torch.Tensor
        ) -> None:
            tp_rank = self.attn_tp_rank
            shard_size = param.data.shape[2]  # local_num_heads
            start_idx = tp_rank * shard_size

            # Handle old 4-D checkpoint format: [1, 1, H, 1] -> [H]
            if loaded_weight.dim() == 4:
                loaded_weight = loaded_weight.view(loaded_weight.shape[2])
            # Now loaded_weight is 1-D (either [num_heads] or [head_dim]).
            # Narrow to the TP shard along the head dimension.
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
            # Reshape to match param shape [1, 1, local_num_heads, 1]
            param.data.copy_(loaded_weight.view(param.data.shape))

        set_weight_attrs(self.A_log, {"weight_loader": _a_log_weight_loader})

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            # The stage boundary completes the attention-TP sum.
            reduce_results=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
            # Symmetric memory of the attention-TP group, which sums the output;
            # off under all_reduce_fusion, whose caller owns the output buffer.
            use_dp_attention_reduce=not self.all_reduce_fusion,
            prefix=f"{prefix}.o_proj",
        )
        if self.all_reduce_fusion and not _o_proj_takes_output(self.o_proj):
            # the fused AR reduces o_proj's output in place out of a symmetric
            # buffer, which needs the GEMM to write into caller-owned storage
            self.all_reduce_fusion = False
            self.o_proj.use_dp_attention_reduce = True
        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        # KDA safe gate: checkpoint trained with gate_lower_bound=-5.0
        self.attn.lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        # Set by _prepare_fused_decode() once weights are loaded.
        self._kda_fused_decode_ready = False
        self._kda_hip_fused_decode_ready = False
        if sp_moe:
            _route_sp_o_proj_output(self.o_proj)

    def forward_qkvbfg(self, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        beta = self.b_proj(hidden_states)[0]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        if self.use_full_rank_gate:
            g_proj_states = self.g_proj(hidden_states)[0]
        else:
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        return qkv, beta, forget_gate, g_proj_states

    def _merge_bfa_weights(self) -> None:
        """Merge f_a_proj (head_dim outputs) + b_proj (heads/tp outputs).

        Both are skinny same-input GEMVs at decode: b lands in a cublas dot
        kernel pair, f_a in a splitK GEMM. One [H, head_dim + heads/tp (+pad)]
        GEMV replaces both. f_a leads so its output slice starts at offset 0,
        and the width is padded to a multiple of 8 so every fused-output row
        stays 16-byte aligned for vectorized consumers (tiny-GEMM on f_b).

        Called once after weight loading. Block-FP8 inputs are dequantized into
        the BF16 tiny-GEMM buffers here."""
        if not self.use_full_rank_gate:
            return
        if _is_npu:
            return
        if _is_hip and self._merge_kda_inproj_weights_hip():
            # Split-path f_b GEMM still uses this when the fused in-proj
            # is above the token threshold.
            self._bfa_f_b_w = self.f_b_proj.weight
            return
        mods = [self.f_a_proj, self.b_proj]
        if self._bfa_uses_block_fp8:
            weights = [_get_k3_dense_weight(mod) for mod in mods]
            sizes = [weight.shape[0] for weight in weights]
            pad = (-sum(sizes)) % 8
            if pad:
                weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
            self._bfa_w = torch.cat(weights, dim=0).contiguous()
            self._bfa_f_b_w = _get_k3_dense_weight(self.f_b_proj).contiguous()
        else:
            if any(getattr(mod, "weight", None) is None for mod in mods):
                return
            # ROCm Quark checkpoints: leave per-channel FP8 / MXFP4 weights on
            # the unfused b_proj/f_a_proj GEMVs; the merged buffer would drop
            # their scales.
            if _is_hip and not _is_unquantized_mergeable([mod.weight for mod in mods]):
                return
            self._bfa_w, sizes = _merge_weights_as_views(mods, pad_rows_to=8)
            self._bfa_f_b_w = self.f_b_proj.weight
        self._bfa_fa_size, self._bfa_b_size = sizes

    def _merge_kda_inproj_weights_hip(self) -> bool:
        """ROCm only: append the [f_a | b] tail to the wide [q,k,v,g] buffer so
        one GEMM covers the whole in-proj, and take _bfa_w as a tail view of
        that buffer. The merge is view-only, so the wide-only and whole-buffer
        weights both stay live and forward_qkvbfg_fused picks per batch size.

        Returns False when the fusion does not apply, leaving the caller to do
        the plain [f_a | b] merge."""
        if not self._may_fuse_kda_inproj():
            return False

        # [q,k,v,g | f_a | b | pad]; f_a/b keep the same relative order and the
        # same pad (both widths are 4 short of a multiple of 8), so the tail
        # view is byte-identical to the wide-only merge.
        merged, sizes = _merge_weights_as_views(
            [self.fused_qkvg_proj, self.f_a_proj, self.b_proj], pad_rows_to=8
        )
        self._bfa_fa_size, self._bfa_b_size = sizes[-2:]
        self._bfa_w = merged[sizes[0] :]
        # Stand-in "layer" so the fused GEMM goes through the same
        # quant_method.apply (and therefore the same backend choice) as the
        # wide projection, whose own .weight stays the 6144-row view for the
        # above-threshold split path. Not an nn.Module on purpose: this must
        # not add a duplicate entry to state_dict.
        self._qkvgbfa_layer = SimpleNamespace(weight=merged)
        self._qkvgbfa_sizes = [
            *self.split_sizes,  # q,k,v then g
            self._bfa_fa_size,
            self._bfa_b_size,
            merged.shape[0] - sum(sizes),  # alignment pad
        ]
        return True

    def _may_fuse_kda_inproj(self) -> bool:
        """Whether the [f_a|b] tail can share the wide projection's buffer.

        Needs the wide fused projection to exist and all three weights to be
        plain unquantized 2-D tensors of one dtype and width -- the checkpoint
        keeps attention in bf16, but a quantized variant would carry scales
        that a raw row-cat would silently drop."""
        if not (_is_hip and envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get()):
            return False
        if not (self.do_fuse_qkvbfg and self.use_full_rank_gate):
            return False
        # Block-FP8 in-proj needs dequantized BF16 buffers; a raw row-cat
        # would drop the scales. Leave fusion to the split [f_a|b] path.
        if self._bfa_uses_block_fp8:
            return False
        ws = [m.weight for m in (self.fused_qkvg_proj, self.f_a_proj, self.b_proj)]
        if not all(type(w.data) is torch.Tensor and w.dim() == 2 for w in ws):
            return False
        # Whitelist the dtype rather than only require the three to agree: the
        # merged buffer carries only .weight, so quantized weights that happen to
        # match each other still lose their per-channel scales.
        if not _is_unquantized_mergeable(ws):
            return False
        return len({(w.dtype, w.shape[1]) for w in ws}) == 1

    def _prepare_fused_decode(self) -> None:
        """Static inputs for the fused KDA decode kernel
        (kernels/ops/attention/kda_fused_decode): per-segment transposed fp32 conv
        weights [4, seg], dense fp32 conv bias, fp32 output-norm weight. Stashed on the
        attention layer for the KDA backend; when the shapes do not match
        the compiled kernel the stash stays unset and decode keeps the
        unfused chain. Called once from load_weights (after all weights are
        loaded, before cuda graph capture)."""
        if _is_hip:
            from sglang.kernels.ops.attention import kda_fused_decode_aiter_hip

            layer = self.attn
            w = layer.conv_weights
            f_b_weight = self.f_b_proj.weight
            backend = os.environ.get("SGLANG_ROCM_K3_KDA_FUSED_BACKEND", "").lower()
            backend_available = (
                backend == "aiter"
                and kda_fused_decode_aiter_hip.available(f_b_weight.device)
            )
            if (
                backend_available
                and w is not None
                and tuple(w.shape) == (3 * 12 * 128, 4)
                and w.dtype == torch.float32
                and f_b_weight.shape == (12 * 128, 128)
                and f_b_weight.dtype == torch.bfloat16
                and layer.A_log is not None
                and layer.A_log.numel() == 12
                and layer.A_log.dtype == torch.float32
                and layer.dt_bias is not None
                and tuple(layer.dt_bias.shape) == (12 * 128,)
                and layer.dt_bias.dtype == torch.float32
                and layer.lower_bound is not None
            ):
                norm_weight = self.o_norm.weight.data.to(torch.bfloat16).contiguous()
                f_b_weight = f_b_weight.view(12, 128, 128).contiguous()
                a_log = layer.A_log.detach().reshape(-1).contiguous()
                layer._k3_hip_fused_decode_args = (
                    f_b_weight,
                    norm_weight,
                    float(self.o_norm.eps),
                    a_log,
                )
                kda_fused_decode_aiter_hip.warmup(
                    f_b_weight=f_b_weight,
                    conv_weight=w,
                    A_log=a_log,
                    dt_bias=layer.dt_bias,
                    lower_bound=float(layer.lower_bound),
                    norm_weight=norm_weight,
                    norm_eps=float(self.o_norm.eps),
                )
                layer._k3_hip_fused_decode_backend = backend
                self._kda_hip_fused_decode_ready = True
            return
        layer = self.attn
        w = layer.conv_weights
        if _is_npu:
            return
        seg = 12 * 128  # compiled for H = HV = 12 heads of 128 (TP8)
        if (
            w is None
            or w.ndim != 2
            or w.shape != (3 * seg, 4)
            or w.dtype != torch.float32
            or layer.A_log is None
            or layer.A_log.numel() != 12
            or layer.A_log.dtype != torch.float32
            or layer.dt_bias is None
            or tuple(layer.dt_bias.shape) != (seg,)
            or layer.dt_bias.dtype != torch.float32
        ):
            rank0_log(
                "K3 fused KDA decode disabled: unexpected conv/A_log/dt_bias "
                f"layout (conv {None if w is None else tuple(w.shape)}, "
                f"A_log {None if layer.A_log is None else tuple(layer.A_log.shape)}, "
                f"dt_bias {None if layer.dt_bias is None else tuple(layer.dt_bias.shape)})"
            )
            return
        # Conv weights/bias stay fp32 (checkpoint dtype; the kernel loads
        # them as fp32, matching the triton chain's precision exactly).
        wt = w.t().contiguous()  # [4, 3*seg]
        bias = layer.bias
        conv_bias = (
            bias.float().contiguous()
            if bias is not None
            else torch.zeros(3 * seg, dtype=torch.float32, device=w.device)
        )
        layer._k3_fused_decode_args = (
            wt[:, :seg].contiguous(),
            wt[:, seg : 2 * seg].contiguous(),
            wt[:, 2 * seg :].contiguous(),
            conv_bias,
            layer.A_log.detach().reshape(-1),  # view; kernel wants [12]
            self.o_norm.weight.data.float().contiguous(),
            float(self.o_norm.eps),
        )
        self._kda_fused_decode_ready = True

    def forward_qkvbfg_fused(
        self, hidden_states: torch.Tensor, defer_f_b: bool = False
    ):
        if self.use_full_rank_gate:
            if self._bfa_w is not None:
                w = self._bfa_w
                n_fa, n_b = self._bfa_fa_size, self._bfa_b_size
                from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm as gemm

                if (
                    _is_hip
                    and self._qkvgbfa_sizes is not None
                    and 0 < hidden_states.shape[0] <= self._qkvgbfa_bs_limit
                ):
                    # ROCm only. One GEMM for the whole in-proj: the [f_a|b]
                    # tail rides the wide projection's bandwidth (~30% of the
                    # in-proj at decode on gfx950, SGLANG_ROCM_K3_FUSE_KDA_INPROJ).
                    fused_states = self.fused_qkvg_proj.quant_method.apply(
                        self._qkvgbfa_layer, hidden_states, None
                    )
                    qkv, g_proj_states, f_a, beta, _pad = torch.split(
                        fused_states, self._qkvgbfa_sizes, dim=-1
                    )
                    # Fused KDA decode consumes f_a and applies f_b itself.
                    forget_gate = f_a if defer_f_b else gemm(f_a, self._bfa_f_b_w)
                    return qkv, beta, forget_gate, g_proj_states

                if (
                    self._bfa_alt_stream is not None
                    and get_is_capture_mode()
                    and 0 < hidden_states.shape[0] <= self._bfa_bs_limit
                ):
                    # Fork before both branches; capture the main projection
                    # first to avoid CUDA graph replay stream expansion.
                    alt = self._bfa_alt_stream
                    cur = torch.cuda.current_stream()
                    alt.wait_stream(cur)
                    fused_states, _ = self.fused_qkvg_proj(hidden_states)
                    with torch.cuda.stream(alt):
                        bfa = gemm(hidden_states, w)
                        forget_gate = (
                            bfa[..., :n_fa]
                            if defer_f_b
                            else gemm(bfa[..., :n_fa], self._bfa_f_b_w)
                        )
                        beta = bfa[..., n_fa : n_fa + n_b]
                    qkv, g_proj_states = torch.split(
                        fused_states, self.split_sizes, dim=-1
                    )
                    cur.wait_stream(alt)
                    return qkv, beta, forget_gate, g_proj_states

                fused_states, _ = self.fused_qkvg_proj(hidden_states)
                qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)
                bfa = gemm(hidden_states, w)
                forget_gate = (
                    bfa[..., :n_fa]
                    if defer_f_b
                    else gemm(bfa[..., :n_fa], self._bfa_f_b_w)
                )
                beta = bfa[..., n_fa : n_fa + n_b]
            else:
                fused_states, _ = self.fused_qkvg_proj(hidden_states)
                qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)
                beta = self.b_proj(hidden_states)[0]
                f_a = self.f_a_proj(hidden_states)[0]
                forget_gate = f_a if defer_f_b else self.f_b_proj(f_a)[0]
        else:
            fused_states = self.fused_qkvbfg_a_proj(hidden_states)
            qkv, beta, fg_a_states = torch.split(fused_states, self.split_sizes, dim=-1)
            forget_gate, g_proj_states = self.fused_fg_b_proj(
                fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)
            )
        return qkv, beta, forget_gate, g_proj_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        defer_f_b = (
            self._kda_hip_fused_decode_ready and forward_batch.forward_mode.is_decode()
        )
        if self.do_fuse_qkvbfg or self.use_full_rank_gate:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states, defer_f_b=defer_f_b
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states
            )

        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unflatten(-1, (-1, self.head_dim))
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        # Fused KDA handoff (attempt-and-verify): offer the output-norm gate
        # so covered decode and target-verify kernels can fold gated RMSNorm
        # into the recurrence kernel. If the backend leaves the stash
        # unconsumed (env off or shape not covered), apply o_norm here as
        # before.
        fused_onorm = (self._kda_fused_decode_ready or defer_f_b) and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        if fused_onorm:
            self.attn._k3_onorm_gate = g_proj_states
            self.attn._k3_onorm_consumed = False
        if defer_f_b:
            self.attn._k3_deferred_f_b = True

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        if fused_onorm:
            self.attn._k3_onorm_gate = None
            fused_onorm = self.attn._k3_onorm_consumed
        if defer_f_b:
            self.attn._k3_deferred_f_b = False
        if not fused_onorm:
            norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
            core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)
        if self.all_reduce_fusion:
            out = _k3_symm_o_proj_out(self.o_proj, core_attn_out)
            partial, _ = self.o_proj(core_attn_out, output_tensor=out)
            return partial
        return self.o_proj(core_attn_out)[0]


class KimiK3MLAAttention(DeepseekV2AttentionMLA):
    """MLA with output gate for K3. Gate is applied in TP-local space before o_proj."""

    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        all_reduce_fusion: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        gate_alt_stream: Optional[torch.cuda.Stream] = None,
        sp_moe: bool = False,
    ) -> None:
        # ModelSlim can quantize K3 latent projections while still storing
        # MLA kv_b_proj as one dense tensor; only GGUF expert packs split K/V.
        split_gguf_kv_b = _uses_split_gguf_kv_b(quant_config)
        self.all_reduce_fusion = all_reduce_fusion
        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        # The fused Ascend split+RMSNorm path is not numerically equivalent for
        # Kimi-K3. Other MLA models retain the existing fused fast path.
        self._disable_npu_fused_split_qk_norm = True
        super().__init__(
            layer_id=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            quant_config=quant_config,
            prefix=prefix,
            config=config,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            skip_rope=True,
            # The stage boundary completes the attention-TP sum.
            reduce_results=False,
            alt_stream=alt_stream,
        )
        if split_gguf_kv_b:
            del self.fused_qkv_a_proj_with_mqa
            del self.kv_b_proj
            self.q_a_proj = ReplicatedLinear(
                config.hidden_size,
                config.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                config.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.has_fused_proj = False

            from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter

            for role in ("k", "v"):
                qweight = GGUFUninitializedParameter(requires_grad=False)
                set_weight_attrs(
                    qweight,
                    {
                        "is_gguf_weight": True,
                        "weight_loader": self._split_kv_b_weight_loader,
                    },
                )
                self.register_parameter(f"{role}_b_qweight", qweight)
                qweight_type = nn.Parameter(
                    torch.empty(1, dtype=torch.uint8), requires_grad=False
                )
                set_weight_attrs(
                    qweight_type,
                    {
                        "is_gguf_weight_type": True,
                        "weight_type": 0,
                        "ignore_warning": True,
                        "weight_loader": self._split_kv_b_weight_loader,
                    },
                )
                self.register_parameter(f"{role}_b_qweight_type", qweight_type)
            self._kimi_split_gguf_kv_b = True
        # Installed before the output-gate wrap below so the gate multiply is
        # applied to x before the fused GEMM+AR sees it.
        if self.all_reduce_fusion and not _o_proj_takes_output(self.o_proj):
            # the fused AR reduces o_proj's output in place out of a symmetric
            # buffer, which needs the GEMM to write into caller-owned storage
            self.all_reduce_fusion = False
        if self.all_reduce_fusion:
            # Hand the GEMM a slice of the persistent symmetric buffer
            # (k3_ar_fusion.symm_buffer); the fused AR reduces it in place.
            # The captured name must differ from the gate block's
            # `_orig_o_proj_forward` — closures capture the __init__ local by
            # reference, and reusing it would rebind to this wrapper (recursion).
            _symm_inner_o_proj_forward = self.o_proj.forward
            _symm_o_proj = self.o_proj

            def _symm_o_proj_forward(x, *args, **kwargs):
                return _symm_inner_o_proj_forward(
                    x,
                    *args,
                    output_tensor=_k3_symm_o_proj_out(_symm_o_proj, x),
                    **kwargs,
                )

            self.o_proj.forward = _symm_o_proj_forward
        else:
            # Allocate o_proj's output in the attention-TP group's symmetric
            # memory, where the stage boundary sums it.
            self.o_proj.use_dp_attention_reduce = True
        if self.use_output_gate:
            projection_size = config.num_attention_heads * config.v_head_dim
            # Shard by attn-TP to match the attention output (DSV2 MLA shards
            # heads across the attention-TP group, not the global TP group).
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                projection_size,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=f"{prefix}.g_proj",
            )
            # Output gate multiplies the TP-local attention output right
            # before o_proj; o_proj is invoked deep inside
            # DeepseekV2AttentionMLA forward cores, so wrap its forward at
            # the instance level (weights, reduce_results, loading untouched).
            self._gate_hidden_states = None
            self._gate_pending_stream = None
            self._gate_alt_stream = gate_alt_stream
            # Above this token count the attention-core kernels fill the SMs
            # on their own and the overlap only adds sync overhead (same
            # bound as deepseek_v4).
            self._gate_bs_limit = (
                (128 if get_platform().is_blackwell else 64)
                if self._gate_alt_stream is not None
                else 0
            )
            _orig_o_proj_forward = self.o_proj.forward

            def _gated_o_proj_forward(x, *args, **kwargs):
                gate_input = self._gate_hidden_states
                self._gate_hidden_states = None
                if gate_input is not None and not isinstance(x, tuple):
                    gate = self._compute_output_gate(gate_input)
                    from sglang.kernels.ops.attention import mla_output_gate

                    if mla_output_gate.covered(x, gate):
                        # One kernel for x * sigmoid(gate); double rounding
                        # matches the unfused pair bit-for-bit.
                        x = mla_output_gate.kimi_k3_mla_output_gate(x, gate)
                    else:
                        x = x * torch.sigmoid(gate)
                elif self._gate_pending_stream is not None:
                    # Even a skipped gate must close its capture branch.
                    torch.cuda.current_stream().wait_stream(self._gate_pending_stream)
                    self._gate_pending_stream = None
                return _orig_o_proj_forward(x, *args, **kwargs)

            self.o_proj.forward = _gated_o_proj_forward
        if sp_moe:
            _route_sp_o_proj_output(self.o_proj)

    @staticmethod
    def _split_kv_b_weight_loader(param, loaded_weight) -> None:
        from torch.nn.parameter import UninitializedParameter

        if getattr(param, "is_gguf_weight_type", False):
            param.weight_type = int(loaded_weight.item())
            param.data.copy_(loaded_weight.reshape_as(param))
            return
        if isinstance(param, UninitializedParameter):
            param.materialize(tuple(loaded_weight.shape), dtype=loaded_weight.dtype)
        param.data.copy_(loaded_weight)

    def dispatch_attn_forward_method(self, forward_batch) -> AttnForwardMethod:
        method = super().dispatch_attn_forward_method(forward_batch)
        if getattr(self, "_kimi_split_gguf_kv_b", False):
            return AttnForwardMethod.MLA
        return method

    def _fork_output_gate(self, hidden_states: torch.Tensor) -> None:
        """Fork early, but record the gate after attention to limit replay streams."""
        self._gate_pending_stream = None
        if (
            self._gate_alt_stream is not None
            and get_is_capture_mode()
            # Keep the fork and join within one capture segment.
            and not is_in_breakable_cuda_graph()
            and (0 < hidden_states.shape[0] <= self._gate_bs_limit)
        ):
            alt = self._gate_alt_stream
            alt.wait_stream(torch.cuda.current_stream())
            self._gate_pending_stream = alt

    def _compute_output_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alt = self._gate_pending_stream
        self._gate_pending_stream = None
        if alt is None:
            return self.g_proj(hidden_states)[0]
        with torch.cuda.stream(alt):
            gate, _ = self.g_proj(hidden_states)
        torch.cuda.current_stream().wait_stream(alt)
        return gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ):
        if self.use_output_gate:
            self._gate_hidden_states = hidden_states
            self._fork_output_gate(hidden_states)
        return super().forward(
            positions, hidden_states, forward_batch, zero_allocator, **kwargs
        )


def _is_moe_layer(config: KimiLinearConfig, layer_idx: int) -> bool:
    """Whether a layer runs a MoE rather than the dense MLP: past the dense
    prefix and on the MoE cadence."""
    return (
        config.is_moe
        and config.num_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def _shards_moe_rows() -> bool:
    """Whether an MoE layer runs on its attention-TP token shard (SP-MoE): an
    EP a2a backend with attention TP."""
    backend = get_moe_a2a_backend()
    return (
        backend.is_megamoe()
        or backend.is_flashinfer_megamoe()
        or backend.is_deepep()
        or backend.is_mooncake()
        or backend.is_ascend_fuseep()
        or backend.is_mori()
    ) and get_parallel().attn_tp_group.world_size > 1


def _fuses_attn_all_reduce(config: KimiLinearConfig) -> bool:
    """Whether the attention-residual bank asks for o_proj's all-reduce fused
    with the pending residual add."""
    attn_tp_size = get_parallel().attn_tp_size
    return (
        attn_tp_size > 1
        and attn_tp_size == get_parallel().tp_size
        and config.attn_res_block_size is not None
        and k3_ar_fusion.enabled()
    )


def _carries_bank_slices(config: KimiLinearConfig) -> bool:
    """Whether consecutive SP-MoE layers keep the stream and the
    attention-residual bank on each rank's attention-TP shard of the rows
    (SGLANG_K3_SP_ATTN_RES): each layer reads its attention input there, then
    gathers what it read. A rank then holds only its own rows of the bank, so
    neither a later pipeline rank nor a draft model's capture of the target's
    hidden states can take them; it is decided at construction. Nor can a
    dense layer after an SP-MoE one, whose FFN reads every row of the bank:
    every layer from the first MoE layer on must be one."""
    if not (
        _shards_moe_rows()
        and config.attn_res_block_size is not None
        and k3_sp_collective.enabled()
        and envs.SGLANG_K3_SP_ATTN_RES.get()
        and get_parallel().pp_size == 1
    ):
        return False
    moe = [_is_moe_layer(config, idx) for idx in range(config.num_hidden_layers)]
    if True not in moe or not all(moe[moe.index(True) :]):
        return False
    spec = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    return not (spec.is_eagle3() or spec.is_dflash_family())


class KimiK3DecoderLayer(nn.Module):
    """Decoder layer carrying the K3 attention-residual stream."""

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        attn_bank: Optional[AttnBank] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self._dp_attention = is_dp_attention_enabled()
        self._is_moe_layer = _is_moe_layer(config, layer_idx)
        # SP-MoE: with an EP a2a backend and attention TP, a MoE layer runs on
        # this rank's attention-TP shard of the rows.
        self._sp_moe = self._is_moe_layer and _shards_moe_rows()

        self.all_reduce_fusion = not self._sp_moe and _fuses_attn_all_reduce(config)

        # Attention
        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiK3DeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                all_reduce_fusion=self.all_reduce_fusion,
                sp_moe=self._sp_moe,
                # Shared with the MLA gate stream: KDA and MLA layers never
                # run concurrently within one forward, so the stream is free.
                bfa_alt_stream=(alt_streams[2] if alt_streams is not None else None),
            )
        else:
            self.self_attn = KimiK3MLAAttention(
                config=config,
                layer_idx=layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                all_reduce_fusion=self.all_reduce_fusion,
                sp_moe=self._sp_moe,
                alt_stream=alt_streams[1] if alt_streams is not None else None,
                gate_alt_stream=alt_streams[2] if alt_streams is not None else None,
            )

        # the attention drops the fusion when its o_proj cannot write into
        # caller-owned storage; the FFN read's fused all-reduce must agree
        self.all_reduce_fusion = self.self_attn.all_reduce_fusion

        # MLP / MoE
        if self._is_moe_layer:
            self.mlp = KimiK3MoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_streams[0] if alt_streams is not None else None,
            )
        else:
            self.mlp = KimiK3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Attention Residual
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.prev_valid_blocks = _cdiv(layer_idx, self.attn_res_block_size)
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

        # A latent MoE on this rank's rows adds the residual in its tail add and
        # writes the next stream itself; one gathered over attention DP returns
        # its output to this rank's tokens first.
        self._ffn_writes_stream = (
            self.use_attn_residuals
            and self._is_moe_layer
            and self.mlp.use_latent_moe
            and not (self._dp_attention and not self.mlp._ep_a2a)
        )
        self._declare_stages(config, layer_idx, attn_bank)

    def _declare_stages(self, config, layer_idx, attn_bank):
        """Declare this layer's attention and FFN stages: the reads and updates
        around them, the rows the FFN takes, and the kernels K3 supplies for
        the bank path."""
        # Under attention DP the FFN input is read on this rank's rows
        # once the attention's sum is complete, then gathered.
        attn_ops = {}
        ffn_ops = dict(read=NormReadout(reads_before_dp_gather=True))
        carries = _carries_bank_slices(config)
        tuned_gather = (
            k3_sp_collective.all_gather if k3_sp_collective.enabled() else None
        )
        if self.use_attn_residuals:
            bank_ops = AttnBankState(
                attn_bank,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.mlp_res_proj,
                self.mlp_res_norm,
                writes_block=self.is_block_write_layer,
                ffn_input_fusions=self._ffn_input_fusions(),
                fuses_slice_collectives=self._sp_moe
                and k3_sp_collective.enabled()
                and envs.SGLANG_K3_SP_ATTN_RES.get(),
                reads_slices=carries,
            ).residual_ops()
            attn_ops = dict(
                read=bank_ops.attn_readout,
                update=bank_ops.attn_update,
                # The rows an SP-MoE layer left on each rank's shard are
                # gathered in K3's tuned all-gather when it takes the batch:
                # before this read, or after it while the bank stays on the
                # shard.
                attn_tp_gather=(
                    tuned_gather
                    if _shards_moe_rows() and _is_moe_layer(config, layer_idx - 1)
                    else None
                ),
            )
            ffn_ops = dict(
                read=bank_ops.ffn_readout,
                update=(
                    REPLACE_AT_EXIT if self._ffn_writes_stream else bank_ops.ffn_update
                ),
            )
        self.attn_boundary, self.ffn_boundary = append_stages(
            (declare_attn(**attn_ops), self.input_layernorm),
            (
                declare_ffn(
                    **ffn_ops,
                    sparse=self._is_moe_layer,
                    next_layer_sparse=_is_moe_layer(config, layer_idx + 1),
                    # The TP width the dense MLP is built with.
                    dense_tp_size=(
                        None
                        if self._is_moe_layer
                        else get_group_rank_size(self.mlp.down_proj.tp_group)[1]
                    ),
                    # A latent MoE completes its output sum together with
                    # the latent reduction its norm needs.
                    output_complete=self._is_moe_layer and self.mlp.use_latent_moe,
                ),
                self.post_attention_layernorm,
            ),
        )

    def _ffn_input_fusions(self):
        """K3's kernels that complete the attention output's sum with the
        pending residual add ahead of the bank's FFN read: the fused
        all-reduce on every row, or the tuned reduce-scatter onto this rank's
        attention-TP shard (SP-MoE)."""
        if self.all_reduce_fusion:
            return (ReadoutFusion(SumGroup.ATTN_TP, _k3_all_reduce_add),)
        if self._sp_moe and k3_sp_collective.enabled():
            return (
                ReadoutFusion(SumGroup.ATTN_TP, _k3_reduce_scatter_add, scatters=True),
            )
        return ()

    def _run_self_attn(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # DP attention: idle ranks (padded to the global shape) have no
        # attention metadata; pass hidden_states through shape-preserving
        # (matching the idle skip in other DP-attention models).
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        # MLA layers with q_lora_rank read their latent inputs from the
        # attention-TP context.
        from sglang.srt.layers.layer_boundary import (
            AttentionInputs,
            get_attn_tp_context,
        )

        qkv_latent_func = getattr(self.self_attn, "prepare_qkv_latent", None)
        if qkv_latent_func is not None:
            attn_inputs = AttentionInputs(hidden_states, forward_batch, qkv_latent_func)
            get_attn_tp_context().set_attn_inputs(attn_inputs)

        result = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        if qkv_latent_func is not None:
            get_attn_tp_context().clear_attn_inputs()

        return result

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        capture_gathered: Optional[AuxHiddenStateAccumulator] = None,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture_gathered=capture_gathered
        )
        hidden_states = self._run_self_attn(
            hidden_states, positions, forward_batch, zero_allocator
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        if self._ffn_writes_stream:
            hidden_states = self.mlp(
                hidden_states,
                prefix_sum=residual_batch.written_residual(forward_batch),
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class KimiK3LinearModel(nn.Module):
    """K3 language-model backbone."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.dspark_layers_to_capture: Optional[list[int]] = None
        self._dp_attention = is_dp_attention_enabled()
        self.carries_bank_slices = _carries_bank_slices(config)

        if self.pp_group.is_first_rank:
            embedding_quant_config = (
                quant_config
                if quant_config is not None and quant_config.get_name() == "expert_pack"
                else None
            )
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=embedding_quant_config,
                prefix=f"{prefix}.embed_tokens",
                # Under DP attention each rank embeds only its local tokens:
                # reduce within the attention-TP group, not the full TP group.
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Alt streams threaded down to the layers. Slots:
        #   [0] MoE dual-stream shared-expert tail
        #   [1] DeepseekV2AttentionMLA base internals (forwarded; unused by K3)
        #   [2] MLA output-gate GEMM, overlaps the attention core
        # Disable on HIP code path.
        self.alt_streams = None if _is_hip else [torch.cuda.Stream() for _ in range(3)]

        # The attention-residual bank of one forward, which every layer's
        # stage boundaries read.
        self.attn_bank = AttnBank() if config.attn_res_block_size is not None else None
        # The layer stack's last FFN leaves its output on the rows the final
        # read takes.
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.attn_res_block_size is not None:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
            self._final_read = (
                self.norm
                if self.attn_bank is None
                else AttnBankOutputRead(
                    self.attn_bank,
                    self.output_attn_res_proj,
                    self.output_attn_res_norm,
                    self.norm,
                    # The rows a last SP-MoE layer leaves on each rank's
                    # shard are gathered in K3's tuned all-gather when it
                    # takes the batch.
                    attn_tp_gather=(
                        k3_sp_collective.all_gather
                        if k3_sp_collective.enabled()
                        and _shards_moe_rows()
                        and _is_moe_layer(config, config.num_hidden_layers - 1)
                        else None
                    ),
                    reads_attn_tp_slices=self.carries_bank_slices,
                )
            )
        else:
            self.norm = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiK3DecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
                attn_bank=self.attn_bank,
            ),
            prefix=f"{prefix}.layers",
            final_read=self._final_read if self.pp_group.is_last_rank else None,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_parallel().pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual_batch.start(forward_batch)
        elif self.attn_bank is not None:
            # The bank's wire carries the stream head written: the sending rank
            # folds it in before the handoff.
            hidden_states = residual_batch.set_written(
                pp_proxy_tensors["hidden_states"], forward_batch
            )
        else:
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=device,
        )

        if self.attn_bank is not None:
            attn_res_block_num = _cdiv(self.end_layer, self.config.attn_res_block_size)
            # A later pipeline rank inherits the rows the earlier ones banked.
            inherited = (
                None
                if self.pp_group.is_first_rank
                else pp_proxy_tensors["attn_res_bank"]
            )
            self.attn_bank.open(hidden_states, attn_res_block_num, inherited)

        packs_aux = self.packs_aux_hidden_states
        aux_hidden_states: AuxHiddenStateAccumulator = (
            AuxHiddenStatePacker.for_batch(
                forward_batch, len(self.dspark_layers_to_capture)
            )
            if packs_aux
            else AuxHiddenStateList()
        )
        captures = self.dspark_layers_to_capture or ()
        # On the standard residual path the stream a layer hands on is what the
        # next layer's attention input reads, captured there on every row (the
        # final norm's, after the last layer). The bank's is the mixture the
        # next read would form, captured after the layer.
        captures_at_input = self.attn_bank is None
        if (
            self.dspark_layers_to_capture is not None
            and not self.pp_group.is_first_rank
        ):
            if "dspark_hidden_states" in pp_proxy_tensors.tensors:
                aux_hidden_states.append(pp_proxy_tensors["dspark_hidden_states"])
            if self.start_layer - 1 in captures and not captures_at_input:
                aux_hidden_states.append(
                    self._dspark_capture_stream(
                        self.start_layer - 1, hidden_states, forward_batch
                    )
                )
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states = self.layers[i](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                    capture_gathered=(
                        aux_hidden_states
                        if captures_at_input and i - 1 in captures
                        else None
                    ),
                )
            if (
                not captures_at_input
                and i in captures
                and (i + 1 < self.end_layer or self.pp_group.is_last_rank)
            ):
                aux_hidden_states.append(
                    self._dspark_capture_stream(i, hidden_states, forward_batch)
                )

        if not self.pp_group.is_last_rank:
            if self.attn_bank is not None:
                # The wire carries the stream head beside the bank.
                hidden_states = residual_batch.fold(hidden_states, forward_batch)
            proxy_tensors = residual_batch.to_pp(hidden_states, forward_batch)
            if self.attn_bank is not None:
                proxy_tensors.tensors["attn_res_bank"] = (
                    self.attn_bank.require().block_residual
                )
                self.attn_bank.close()
            if aux_hidden_states:
                proxy_tensors.tensors["dspark_hidden_states"] = torch.cat(
                    aux_hidden_states, dim=-1
                )
            return proxy_tensors

        hidden_states = residual_batch.final_norm(
            hidden_states,
            forward_batch,
            self._final_read,
            capture=(
                aux_hidden_states.capture
                if captures_at_input and self.end_layer - 1 in captures
                else None
            ),
            skip_empty=True,
        )
        if self.attn_bank is not None:
            self.attn_bank.close()

        if packs_aux:
            return hidden_states, aux_hidden_states.finalize()
        if self.dspark_layers_to_capture is not None:
            return hidden_states, aux_hidden_states
        return hidden_states

    @property
    def packs_aux_hidden_states(self) -> bool:
        # PP stages keep the list: inherited captures arrive pre-concatenated.
        return (
            self.dspark_layers_to_capture is not None and self.pp_group.world_size == 1
        )

    def _dspark_capture_stream(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Stream value after `layer_idx` on the bank path: the pre-norm
        mixture its next consumer would compute (next layer's attention side;
        output side for the last layer). The standard path captures at the
        next input instead."""
        hidden_states = residual_batch.snapshot(hidden_states, forward_batch)
        if layer_idx + 1 < self.end_layer:
            next_layer = self.layers[layer_idx + 1]
            score_proj = next_layer.self_attention_res_proj
            score_norm = next_layer.self_attention_res_norm
            nvb = next_layer.prev_valid_blocks
        else:
            # Last layer: the model's own output-side aggregation weights.
            score_proj = self.output_attn_res_proj
            score_norm = self.output_attn_res_norm
            nvb = _cdiv(self.end_layer, self.config.attn_res_block_size)
        return aggregate_stream(
            hidden_states,
            self.attn_bank.require().block_residual,
            nvb,
            score_proj,
            score_norm,
        )


class KimiK3LinearForCausalLM(nn.Module):
    """Text-only K3 causal LM."""

    # ModelSlim describes quantization with the original checkpoint module
    # names. Register the runtime fused QKVG module so it can resolve the
    # q_proj scheme while the weight loader packs q/k/v/g into its shards.
    packed_modules_mapping = {
        "fused_qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
    }

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        if quant_config is not None:
            if isinstance(quant_config, ModelSlimConfig):
                model_mapping = {
                    **quant_config.packed_modules_mapping.get("model", {}),
                    **self.packed_modules_mapping,
                }
                quant_config.update_packed_modules_mapping({"model": model_mapping})
            else:
                # The loader seeded this mapping from the outer model class;
                # replacing it would drop those entries.
                quant_config.update_packed_modules_mapping(
                    {
                        **(quant_config.packed_modules_mapping or {}),
                        **self.packed_modules_mapping,
                    }
                )
        self.model = KimiK3LinearModel(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_pp_proxy_dspark_hidden_size(self) -> int:
        layers = self.model.dspark_layers_to_capture or []
        return self.config.hidden_size * sum(
            layer < self.model.start_layer - 1 for layer in layers
        )

    def get_aux_hidden_states_width(self) -> int:
        if not self.model.packs_aux_hidden_states:
            return 0
        return len(self.model.dspark_layers_to_capture) * self.config.hidden_size

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        if self.model.carries_bank_slices:
            raise RuntimeError(
                "the model keeps each rank's shard of the rows across SP-MoE "
                "layers (SGLANG_K3_SP_ATTN_RES), which leaves no layer output "
                "to capture whole"
            )
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = list(layer_ids)

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        # DFLASH target_layer_ids name layer outputs, which is what the DSPARK
        # taps already capture here, so reuse them without the usual +1 shift.
        self.set_dspark_layers_to_capture(layer_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        embeds = input_embeds if input_embeds is not None else inputs_embeds
        hidden_states = self.model(
            input_ids, positions, forward_batch, embeds, pp_proxy_tensors
        )
        if self.pp_group.is_last_rank:
            aux_hidden_states = None
            if self.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        return hidden_states

    def prepare_context_parallel_metadata_for_dcp(
        self,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens_sum: int,
        kv_buffer_shape: torch.Size,
        kv_cache_dtype,
        kv_cache_device,
        create_chunked_prefix_cache_kv_indices_fn,
    ):
        return prepare_decode_context_parallel_metadata(
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            seq_lens_sum=seq_lens_sum,
            kv_buffer_shape=kv_buffer_shape,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_device=kv_cache_device,
            create_chunked_prefix_cache_kv_indices_fn=create_chunked_prefix_cache_kv_indices_fn,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        use_full_rank_gate = bool(
            (self.config.linear_attn_config or {}).get("use_full_rank_gate", False)
        )
        if use_full_rank_gate:
            # Fused layout (K3): [q, k, v, g] column-parallel; b / f_a / f_b
            # are standalone modules loaded by name.
            fused_qkvbfg_mapping = [
                (".fused_qkvg_proj", ".q_proj", 0),
                (".fused_qkvg_proj", ".k_proj", 1),
                (".fused_qkvg_proj", ".v_proj", 2),
                (".fused_qkvg_proj", ".g_proj", 3),
            ]
        else:
            # Fused layout (low-rank gate): [q, k, v, b] + [f_a, g_a]
            fused_qkvbfg_mapping = [
                (".fused_qkvbfg_a_proj", ".q_proj", 0),
                (".fused_qkvbfg_a_proj", ".k_proj", 1),
                (".fused_qkvbfg_a_proj", ".v_proj", 2),
                (".fused_qkvbfg_a_proj", ".b_proj", 3),
                (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
                (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
                (".fused_fg_b_proj", ".f_b_proj", 0),
                (".fused_fg_b_proj", ".g_b_proj", 1),
            ]

        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            *fused_qkvbfg_mapping,
            # Unfused QKV path
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # Conv1d fusion
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]

        if self.config.is_moe:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Keyed by the `experts.<id>.<proj>.` fragment (see _EXPERT_WEIGHT_NAME).
        expert_params_lookup = {entry[1]: entry for entry in expert_params_mapping}
        assert all(
            _EXPERT_WEIGHT_NAME.fullmatch(key) for key in expert_params_lookup
        ), "ckpt expert names diverged from _EXPERT_WEIGHT_NAME; no expert would load"

        num_hidden_layers = self.config.num_hidden_layers
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if name.endswith(".weight_scale") and loaded_weight.ndim == 4:
                loaded_weight = loaded_weight[:, 0, :, 0]

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.model.start_layer or layer_id >= self.model.end_layer
            ):
                continue

            # Skip weights of layers outside a truncated config (e.g.
            # num_hidden_layers override for fast testing); the checkpoint may
            # carry more layers than the instantiated model.
            if ".layers." in name:
                _lid = name.split(".layers.")[1].split(".")[0]
                if _lid.isdigit() and int(_lid) >= num_hidden_layers:
                    continue

            # compressed-tensors MXFP4 stores as weight_packed; Mxfp4MoEMethod uses weight
            # (NPU keeps weight_packed for NPUCompressedTensorsW4A8mxfp4MoE).
            if "weight_packed" in name and not _is_npu:
                name = name.replace("weight_packed", "weight")

            # MLA: fuse q_a_proj + kv_a_proj_with_mqa → fused_qkv_a_proj_with_mqa
            if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
                is_q_a = ".q_a_proj." in name
                fused_name = name.replace(".q_a_proj.", ".fused_qkv_a_proj_with_mqa.")
                fused_name = fused_name.replace(
                    ".kv_a_proj_with_mqa.", ".fused_qkv_a_proj_with_mqa."
                )
                fused_name = _maybe_map_fp8_pb_scale_name(fused_name, params_dict)
                if fused_name in params_dict:
                    param = params_dict[fused_name]
                    if fused_name.endswith(".weight_scale_inv"):
                        offset = 0 if is_q_a else _cdiv(self.config.q_lora_rank, 128)
                        param.data[offset : offset + loaded_weight.shape[0]].copy_(
                            loaded_weight
                        )
                    elif is_q_a:
                        param.data[: loaded_weight.shape[0]].copy_(loaded_weight)
                    else:
                        q_lora_rank = self.config.q_lora_rank or 0
                        param.data[q_lora_rank:].copy_(loaded_weight)
                    loaded_params.add(fused_name)
                    continue

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                # Fused projections only apply to KDA layers
                if param_name in {
                    ".fused_qkvbfg_a_proj",
                    ".fused_fg_b_proj",
                    ".fused_qkvg_proj",
                }:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                    layer = self.model.layers[layer_id].self_attn
                    # Full-rank K3 always instantiates fused_qkvg_proj, including
                    # ModelSlim-quantized models. The low-rank fused modules are
                    # still conditional on do_fuse_qkvbfg.
                    if param_name == ".fused_qkvg_proj":
                        if not getattr(layer, "use_full_rank_gate", False):
                            continue
                    elif not getattr(layer, "do_fuse_qkvbfg", False):
                        continue
                if weight_name in {".q_proj", ".k_proj", ".v_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = _maybe_map_fp8_pb_scale_name(name, params_dict)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                expert_match = _EXPERT_WEIGHT_NAME.search(name)
                expert_entry = (
                    expert_params_lookup.get(expert_match.group(0))
                    if expert_match
                    else None
                )
                if expert_entry is not None:
                    param_name, weight_name, expert_id, shard_id = expert_entry
                    name = name.replace(weight_name, param_name)
                    # Skip experts of layers outside a truncated config (e.g.
                    # num_hidden_layers override), mirroring the non-expert
                    # `name not in params_dict` guard below.
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            expert_id=expert_id,
                            shard_id=shard_id,
                        )
                else:
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    name = _maybe_map_fp8_pb_scale_name(name, params_dict)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    if name.endswith(".b_proj.weight_scale_inv"):
                        # All TP ranks share K3's single beta output-scale block.
                        param.data.copy_(loaded_weight)
                    else:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        self.post_load_weights()
        return loaded_params

    def post_load_weights(self):
        # Also invoked by loader post-load hooks (DummyModelLoader,
        # ShardedStateLoader, remote-instance flows -- none of which call
        # load_weights), so e.g. dummy-weight benchmarks get the fused buffers.
        # Post-load: absorb kv_b_proj into w_kc and w_vc for MLA layers
        for layer_id in self.config.full_attention_layer_ids:
            if layer_id >= len(self.model.layers):
                continue  # truncated config (e.g. num_hidden_layers override)
            layer = self.model.layers[layer_id]
            if isinstance(layer, PPMissingLayer):
                continue
            self_attn = layer.self_attn
            if getattr(self_attn, "_kimi_split_gguf_kv_b", False):
                if int(self_attn.k_b_qweight_type.weight_type) != 2:
                    raise ValueError("Kimi-K3 MLA K projection must remain GGUF Q4_0")
                if int(self_attn.v_b_qweight_type.weight_type) != 10:
                    raise ValueError("Kimi-K3 MLA V projection must remain GGUF Q2_K")
                self_attn.use_deep_gemm_bmm = False
                continue
            kv_b_weight = _get_k3_dense_weight(self_attn.kv_b_proj)
            scale_folded_into_weight = False
            if _is_hip and kv_b_weight.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                scale = getattr(self_attn.kv_b_proj, "weight_scale", None)
                if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                    from sglang.srt.models.kimi_k3_rocm_quant import (
                        _k3_channel_fp8_to_bf16,
                    )

                    # Fold the per-channel scale while dim 0 is still the
                    # channel axis it indexes, i.e. before the head split.
                    kv_b_weight = _k3_channel_fp8_to_bf16(
                        self_attn.kv_b_proj, kv_b_weight
                    )
                    scale_folded_into_weight = True
            w_kc, w_vc = kv_b_weight.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            kv_b_scale = getattr(self_attn.kv_b_proj, "weight_scale", None)
            if _is_hip and (
                scale_folded_into_weight
                or not (
                    isinstance(kv_b_scale, torch.Tensor) and kv_b_scale.numel() == 1
                )
            ):
                # aiter's absorb GEMM dereferences w_scale as one scalar. A scale
                # folded into the now-bf16 w_kc/w_vc, a vector the branch above
                # could not fold, or the None quark leaves on a dequantized
                # narrow partition must keep DeepseekV2AttentionMLA's 1.0
                # default. Skip the assignment rather than reset afterwards:
                # assigning a Parameter registers it, and nn.Module then refuses
                # a float in its place.
                pass
            elif hasattr(self_attn.kv_b_proj, "weight_scale"):
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale

        # Post-load: precompute the attn-res combined score weights BEFORE
        # cuda graph capture (a lazy first call inside get_cw would bake the
        # multiply into every replay). Warm both dtypes (bf16 fast kernel,
        # fp32 triton fallback).
        def _warm_cw(proj, norm):
            get_cw(proj, norm, dtype=torch.bfloat16)
            get_cw(proj, norm)

        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if layer.use_attn_residuals:
                _warm_cw(layer.self_attention_res_proj, layer.self_attention_res_norm)
                _warm_cw(layer.mlp_res_proj, layer.mlp_res_norm)
        if hasattr(self.model, "output_attn_res_proj"):
            _warm_cw(self.model.output_attn_res_proj, self.model.output_attn_res_norm)

        # Post-load: merge the horizontally-fused decode weights (views of the
        # merged buffers, ~0 extra memory); must run before cuda graph capture.
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, KimiK3MoE):
                layer.mlp._merge_front_weights()
                # Convert the correction bias to fp32 once so the per-call
                # .to(float32) in topk is a no-op, not one upcast kernel per
                # MoE layer per step.
                bias = layer.mlp.gate.e_score_correction_bias
                if bias.dtype != torch.float32:
                    bias.data = bias.data.to(torch.float32)
            if isinstance(layer.self_attn, KimiK3DeltaAttention):
                layer.self_attn._merge_bfa_weights()
                layer.self_attn._prepare_fused_decode()

        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer) or not isinstance(
                layer.self_attn, KimiK3DeltaAttention
            ):
                continue
            if _is_npu:
                continue
            from sglang.kernels.ops.attention.fla.kda import (
                precompile_k3_recompute_w_u_kernel,
            )

            o_proj_weight = getattr(layer.self_attn.o_proj, "weight", None)
            if o_proj_weight is None:
                o_proj_weight = layer.self_attn.o_proj.qweight
            if precompile_k3_recompute_w_u_kernel(
                num_heads=layer.self_attn.local_num_heads,
                dtype=getattr(layer.self_attn.o_proj, "params_dtype", None)
                or o_proj_weight.dtype,
                device=layer.self_attn.dt_bias.device,
            ):
                rank0_log("Precompiled the Kimi-K3 KDA prefill kernel.")
            break


class KimiK3ForConditionalGeneration(nn.Module):
    """K3 multimodal wrapper: MoonViT3d tower + KimiK3LinearForCausalLM."""

    supports_cuda_vmm_feature_transport = True

    # Fused runtime module -> checkpoint shard names, so quant configs can
    # match fused prefixes against per-shard exclude_modules
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
        "fused_qkvbfg_a_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
        "fused_fg_b_proj": ["f_b_proj", "g_b_proj"],
    }
    encoder_media_processor_config = EncoderMediaProcessorConfig(
        image_decode_mode="nvjpeg_fancy",
        preserve_media_metadata=True,
    )

    # Raw HF checkpoint prefixes, before hf_to_sglang_mapper is applied.
    encoder_only_safetensors_weight_prefixes = (
        "vision_tower.",
        "mm_projector.",
    )

    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
        },
        orig_to_new_substr={
            "block_sparse_moe": "mlp",
        },
    )

    def __init__(
        self,
        config: KimiK3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # The dedicated K3 tower runs replicated (per-rank full weights);
        # shard work across ranks image-wise via the DP runner.
        self.use_data_parallel = True

        self.vision_tower = KimiK3VisionTower(config.vision_config)
        self.mm_projector = KimiK3MultiModalProjector(config.vision_config)

        self.language_model = None
        if not config.encoder_only:
            quant_description = getattr(quant_config, "quant_description", {})
            uses_wrapper_quant_prefix = any(
                isinstance(name, str) and name.startswith("language_model.")
                for name in quant_description
            )
            language_prefix = (
                maybe_prefix(prefix, "language_model")
                if uses_wrapper_quant_prefix
                else prefix
            )
            self.language_model = KimiK3LinearForCausalLM(
                config.text_config,
                quant_config,
                prefix=language_prefix,
            )

    @property
    def model(self):
        return self.language_model

    def __setattr__(self, name, value):
        if name == "model":
            return
        super().__setattr__(name, value)

    def post_load_weights(self):
        # Delegate so DummyModelLoader's post-load hook reaches the LM tower.
        if self.language_model is not None:
            self.language_model.post_load_weights()

    def precompile_kernels_after_loading(self) -> None:
        if self.config.language_only:
            return
        if self.vision_tower.precompile_fused_rope():
            logger.info("Precompiled dynamic-token fused K3 vision RoPE kernel")
        if self.vision_tower.precompile_attention_backend():
            logger.info("Precompiled Kimi-K3 vision FA4 kernel")

    def get_input_embeddings(self):
        if self.language_model is None:
            raise AttributeError(
                "get_input_embeddings() is not available in encoder-only mode"
            )
        return self.language_model.model.embed_tokens

    @property
    def lm_head(self):
        if self.language_model is None:
            raise AttributeError("lm_head is not available in encoder-only mode")
        return self.language_model.lm_head

    def get_pp_proxy_dspark_hidden_size(self) -> int:
        if self.language_model is None:
            return 0
        return self.language_model.get_pp_proxy_dspark_hidden_size()

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.language_model is None:
            raise AttributeError(
                "DFLASH layer capture is not available in encoder-only mode"
            )
        self.language_model.set_dflash_layers_to_capture(layer_ids)

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.language_model is None:
            raise AttributeError(
                "DSPARK layer capture is not available in encoder-only mode"
            )
        self.language_model.set_dspark_layers_to_capture(layer_ids)

    def get_aux_hidden_states_width(self) -> int:
        if self.language_model is None:
            return 0
        return self.language_model.get_aux_hidden_states_width()

    def preprocess_mm_for_encoder(
        self,
        mm_data,
        modality,
        config,
        *,
        image_processor=None,
        use_gpu_preprocessing=False,
    ):
        """Prepare per-image raw inputs for owner-side EPD preprocessing."""
        if modality != Modality.IMAGE:
            raise ValueError("Kimi-K3 encoder mode supports image input only")
        if image_processor is None:
            raise ValueError("Kimi-K3 encoder preprocessing needs an image processor")

        from sglang.srt.multimodal.kimi_k3_image_processing import (
            prepare_kimi_k3_encoder_inputs,
        )

        self._encoder_image_processor = image_processor
        return prepare_kimi_k3_encoder_inputs(
            mm_data,
            image_processor,
            use_gpu_preprocessing=use_gpu_preprocessing,
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        device = self.vision_tower.device
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        image_grid_thws = []
        for item in items:
            grid_thw = item.model_specific_data.get("image_grid_thw")
            if grid_thw is None:
                grid_thw = item.model_specific_data["grid_thws"]
            if grid_thw.shape[0] != 1:
                # One item must carry exactly one logical image so the DP
                # owner assignment and the bounded CUDA-IPC lease accounting
                # stay per-item; aggregated encoder inputs are split upstream
                # (EPD encode server) before reaching this point.
                raise ValueError(
                    "Kimi-K3 expects one vision grid per MultimodalDataItem; "
                    "split aggregated encoder inputs before get_image_feature()"
                )
            image_grid_thws.append(grid_thw)
        grid_thws_host = torch.concat(image_grid_thws, dim=0).cpu()
        grid_thw_list = grid_thws_host.tolist()

        def materialize_item_features(image_indices: List[int]) -> torch.Tensor:
            """Materialize only the images assigned to this vision-DP rank."""
            from sglang.srt.multimodal.encoder_preprocessing import (
                LOCAL_PREPROCESSED_KEY,
            )

            # Match the configured TP consumer count captured when the
            # tokenizer creates MmItemMemoryPool. A live attention subgroup
            # size could leave acknowledgements missing and strand the lease.
            ipc_consumer_count = max(get_parallel().tp_size, 1)
            device_index = device.index
            if device.type == "cuda" and device_index is None:
                device_index = torch.cuda.current_device()

            selected_items = []
            for image_index in image_indices:
                item = items[image_index]
                if device.type == "cuda":
                    item.reconstruct(
                        device_index, ipc_consumer_count=ipc_consumer_count
                    )
                selected_items.append(item)

            locally_preprocessed = [
                item.model_specific_data.get(LOCAL_PREPROCESSED_KEY, False)
                for item in selected_items
            ]
            if any(locally_preprocessed):
                if not all(locally_preprocessed):
                    raise ValueError(
                        "Kimi-K3 cannot mix local preprocessed and deferred images"
                    )
                return materialize_multimodal_features(
                    [item.feature for item in selected_items],
                    device=device,
                    dtype=target_dtype,
                )

            deferred = [
                item.model_specific_data.get(DEFERRED_PREPROCESSING_KEY)
                for item in selected_items
            ]
            if any(config is not None for config in deferred):
                materialized = [None] * len(selected_items)
                deferred_by_backend = {}
                for index, (item, config) in enumerate(zip(selected_items, deferred)):
                    if config is None:
                        if not isinstance(item.feature, torch.Tensor):
                            raise TypeError(
                                "Kimi-K3 image feature must be a torch.Tensor, "
                                f"got {type(item.feature)}"
                            )
                        materialized[index] = item.feature
                    else:
                        deferred_by_backend.setdefault(config.backend, []).append(index)

                for backend, indices in deferred_by_backend.items():
                    group_items = [selected_items[index] for index in indices]
                    group_configs = [deferred[index] for index in indices]
                    # Map backend-group positions through the rank-local shard to global grid rows.
                    global_indices = [image_indices[index] for index in indices]
                    first_config = group_configs[0]
                    if backend == "gpu":
                        from sglang.srt.multimodal.processors.kimi_k25 import (
                            _gpu_preprocess_images,
                        )

                        image_scale, image_bias = normalization_tensors(
                            first_config.image_mean,
                            first_config.image_std,
                            device,
                        )
                        pixel_values, produced_grids = _gpu_preprocess_images(
                            [item.feature for item in group_items],
                            [config.resize_config for config in group_configs],
                            image_scale,
                            image_bias,
                            self.vision_tower.patch_size,
                            to_chw=lambda image: to_chw_uint8(image, device=device),
                            post_resize=lambda x: fill_transparent_bg(
                                x, first_config.transparent_bg_config
                            ),
                        )
                        expected_grids = grid_thws_host[global_indices]
                        if not torch.equal(produced_grids.cpu(), expected_grids):
                            raise ValueError(
                                "Kimi-K3 deferred GPU preprocessing produced wrong grids"
                            )
                    elif backend == "cpu":
                        from sglang.srt.multimodal.kimi_k3_image_processing import (
                            materialize_kimi_k3_cpu_features,
                        )

                        pixel_values = materialize_kimi_k3_cpu_features(
                            group_items, self._encoder_image_processor
                        )
                    else:
                        raise ValueError(
                            f"Unsupported Kimi-K3 deferred preprocessing backend: {backend}"
                        )

                    patch_counts = [
                        int(grid_thws_host[index].prod().item())
                        for index in global_indices
                    ]
                    if sum(patch_counts) != pixel_values.shape[0]:
                        raise ValueError(
                            "Kimi-K3 deferred feature length does not match image grids"
                        )
                    for index, feature in zip(
                        indices, pixel_values.split(patch_counts), strict=True
                    ):
                        materialized[index] = feature

                return materialize_multimodal_features(
                    materialized,
                    device=device,
                    dtype=target_dtype,
                )

            features = []
            for item in selected_items:
                if not isinstance(item.feature, torch.Tensor):
                    raise TypeError(
                        "Kimi-K3 image feature must be a torch.Tensor, "
                        f"got {type(item.feature)}"
                    )
                features.append(item.feature)
            return materialize_multimodal_features(
                features, device=device, dtype=target_dtype
            )

        if self.use_data_parallel:
            from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model

            image_embeds = run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                None,
                grid_thw_list,
                rope_type="rope_2d",
                # K3's tower pools the temporal dimension away: a t>1 grid
                # still yields h*w/merge_area output embeddings, so the DP
                # gather length must ignore t.
                pool_temporal_dimension=True,
                pass_grid_thw_list=True,
                load_local_pixel_values=materialize_item_features,
                pixel_values_device=device,
                pixel_values_dtype=target_dtype,
            )
            return self.mm_projector(image_embeds)

        pixel_values = materialize_item_features(list(range(len(items))))
        image_embeds = self.vision_tower(pixel_values, grid_thws_host.to(device))
        return self.mm_projector(image_embeds)

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @property
    def start_layer(self) -> int:
        if self.language_model is None:
            return 0
        return self.language_model.model.start_layer

    @property
    def end_layer(self) -> int:
        if self.language_model is None:
            return self.config.text_config.num_hidden_layers
        return self.language_model.model.end_layer

    def prepare_context_parallel_metadata_for_dcp(
        self,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens_sum: int,
        kv_buffer_shape: torch.Size,
        kv_cache_dtype,
        kv_cache_device,
        create_chunked_prefix_cache_kv_indices_fn,
    ):
        return self.language_model.prepare_context_parallel_metadata_for_dcp(
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            seq_lens_sum=seq_lens_sum,
            kv_buffer_shape=kv_buffer_shape,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_device=kv_cache_device,
            create_chunked_prefix_cache_kv_indices_fn=create_chunked_prefix_cache_kv_indices_fn,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        mapper = getattr(self, "hf_to_sglang_mapper", None)
        if mapper is not None:
            weights = mapper.apply(weights)

        vision_params = (
            None
            if self.config.language_only
            else dict(self.named_parameters(remove_duplicate=False))
        )

        def stream_language_weights():
            for name, loaded_weight in weights:
                if "vision_tower" in name or "mm_projector" in name:
                    if vision_params is None:
                        continue
                    if name not in vision_params:
                        logger.warning("Unmapped vision weight: %s", name)
                        continue
                    param = vision_params[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    continue
                yield name.replace("language_model.", ""), loaded_weight

        if self.language_model is not None:
            self.language_model.load_weights(stream_language_weights())
        else:
            # The vision weights are loaded as a side effect of advancing this
            # streaming iterator.  Encoder-only mode must therefore drain it
            # even though it discards every language-model tensor.
            for _ in stream_language_weights():
                pass


EntryClass = [KimiK3ForConditionalGeneration, KimiK3LinearForCausalLM]
