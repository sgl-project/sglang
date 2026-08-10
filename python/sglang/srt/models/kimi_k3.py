# Kimi-K3 multimodal model: KimiLinear text backbone + MoonViT3d vision tower.
# Based on kimi_linear.py with K3-specific features:
#   - Attention Residual (attn_res_block_size)
#   - Latent MoE (routed_expert_hidden_size)
#   - SiTU activation
#   - MLA output gate (mla_use_output_gate)
#   - Full-rank KDA gate (use_full_rank_gate)

import logging
from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import (
    k3_ar_fusion,
    k3_gemm_ar,
    k3_sp_collective,
    zero_copy_context,
)
from sglang.srt.layers.activation import SiluAndMul, SituAndMul
from sglang.srt.layers.attn_residual import AttnResidual, aggregate_stream, get_cw
from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
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
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, MoEGate
from sglang.srt.models.kimi_k3_vl import (
    KimiK3MultiModalProjector,
    KimiK3VisionTower,
)
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.multimodal.kimi_k3_image_processing import (
    DEFERRED_PREPROCESSING_KEY,
    fill_transparent_bg,
    normalization_tensors,
    to_chw_uint8,
)
from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.srt.runtime_context import (
    configured_tp_size,
    get_exec,
    get_parallel,
    get_server_args,
)
from sglang.srt.utils import is_blackwell_supported, is_hip, make_layers
from sglang.srt.utils.common import (
    BumpAllocator,
    add_prefix,
    get_bool_env_var,
    rank0_log,
    require_mlp_sync,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)
_is_hip = is_hip()
_aiter_k3_opt = get_bool_env_var("SGLANG_AITER_K3_OPT")


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# MegaMoE SiTU sentinel: DeepGEMM 0.1.5.post1+ selects the K3 SiTU
# activation when activation_clamp == 0.03125 (2^-5: exactly representable and
# unused by any legitimate swiglu clamp; the host asserts clamp >= 0 so a
# negative sentinel is impossible). beta=4.0 / linear_beta=25.0 are baked into
# the DeepGEMM kernel.
_K3_MEGA_SITU_SENTINEL_CLAMP = 0.03125


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


# Fully fused KDA decode step (conv1d + delta rule + gated RMSNorm in one
# kernel, kernels/ops/attention/kda_fused_decode). The model hands the output-norm gate
# to the KDA backend via an attempt-and-verify stash on the attention layer;
# unconsumed stashes fall back to the unfused chain + o_norm here.


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


# DP attention helpers.
#
# K3 cannot use LayerCommunicator: the attn-res aggregation kernels replace
# input_layernorm / post_attention_layernorm, which the communicator expects
# to own. Instead the MLP/MoE modules gather/scatter around their own body:
# attention and the attn-res buffers stay in local (per-DP-rank) token space,
# the MLP/MoE runs on the DP-gathered global batch with plain full-TP
# semantics (its internal all-reduces are unchanged and required — the latent
# reduce must happen in latent space before the norm), and the delayed
# prefix_sum add stays local, applied after the scatter back.


def _dp_local_buffer_group():
    """Symmetric-memory group for the local DP buffer (mirrors
    CommunicateSummableTensorPairFn._scatter_hidden_states)."""
    parallel = get_parallel()
    if parallel.tp_size == parallel.attn_dp_size:
        return get_tp_group()
    return parallel.attn_tp_group


def _sp_all_gather_rows(hidden_states: torch.Tensor) -> torch.Tensor:
    """Reassemble contiguous token shards, using the tuned K3 AG when covered."""
    group = get_parallel().attn_tp_group
    hidden_states = hidden_states.contiguous()
    full = k3_sp_collective.all_gather(hidden_states)
    if full is None:
        full = torch.empty(
            (hidden_states.shape[0] * group.world_size, hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        group.all_gather_into_tensor(full, hidden_states)
    return full


def _sp_local_rows(hidden_states: torch.Tensor) -> slice:
    """Full-batch row interval owned by this rank's contiguous token shard."""
    group = get_parallel().attn_tp_group
    lo = group.rank_in_group * hidden_states.shape[0]
    return slice(lo, lo + hidden_states.shape[0])


class KimiK3MLP(nn.Module):
    """K3 MLP; SiLU or SiTU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        _tp_kwargs = (
            dict(tp_rank=tp_rank, tp_size=tp_size) if tp_size is not None else {}
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            **_tp_kwargs,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
            **_tp_kwargs,
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
        self._dp_attention = is_dp_attention_enabled()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        prefix_sum: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        # DP attention only when driven from the decoder layer (forward_batch
        # given); the shared-experts instance inside KimiK3MoE passes None and
        # runs on the already-gathered buffer.
        use_dp = self._dp_attention and forward_batch is not None
        if use_dp:
            local_hidden_states = hidden_states
            hidden_states = get_global_dp_buffer(get_tp_group())
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        if use_dp:
            global_out = hidden_states
            hidden_states = get_local_dp_buffer(_dp_local_buffer_group())
            dp_scatter(hidden_states, global_out, forward_batch)
        # TODO(dark): maybe fuse residual with all reduce of down projection
        if prefix_sum is not None:
            hidden_states = hidden_states + prefix_sum
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

    return add3.add3(a, b, c, prefetch_bc=prefetch_bc)


# One-shot log guard: proves the merged front is live (see _ep_front).
_EP_FRONT_LOGGED = False


def _o_proj_takes_output(o_proj: RowParallelLinear) -> bool:
    """Whether o_proj can write into caller-owned storage. ``apply_into`` is an
    optional quant-method capability; only the unquantized method has it."""
    return getattr(o_proj.quant_method, "apply_into", None) is not None


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
        self._dp_attention = is_dp_attention_enabled()

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

        # For MXFP4 compressed-tensors, replace quant_config with Mxfp4Config
        # so FusedMoE's weight_loader uses the MXFP4 fast path
        moe_quant_config = quant_config
        if quant_config is not None and getattr(quant_config, "quant_format", None):
            if "mxfp4" in quant_config.quant_format:
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
            renormalize=moe_renormalize,
            use_grouped_topk=True,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            # flashinfer_mxfp4 + situ consumes precomputed routing
            # (PackedPrecomputed): keep the radix router in the TopK layer
            # and hand its ids/weights to the MoE op. Other quantized paths
            # keep the runner-resolved format (marlin -> standard anyway,
            # bypassed only for the public logits-routing path).
            output_format=(
                TopKOutputFormat.STANDARD
                if quant_config is None
                or (
                    config.hidden_act == "situ"
                    and get_moe_runner_backend().is_flashinfer_mxfp4()
                )
                # mega pre-dispatch consumes raw topk_ids/topk_weights
                or get_moe_a2a_backend().is_megamoe()
                else None
            ),
        )

        # MegaMoE (deep_gemm fused a2a+GEMM over the EP symm buffer): a drop-in
        # replacement for the routed experts call below. K3 routes ALL batches
        # through it when enabled — the megamoe backend's non-mega fallback is
        # a StandardDispatcher without a2a, which is wrong for scattered
        # tokens — so SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK must
        # cover the per-rank prefill chunk. SiTU is selected inside the
        # DeepGEMM mega kernel via a sentinel activation_clamp with the K3
        # constants baked in.
        self._use_mega_moe = get_moe_a2a_backend().is_megamoe()
        self._mega_intermediate_size = moe_intermediate_size
        self._mega_top_k = config.num_experts_per_token
        if self._use_mega_moe:
            assert self.use_latent_moe and config.hidden_act == "situ"
            assert (
                config.activation_situ_beta,
                config.activation_situ_linear_beta,
            ) == (4.0, 25.0), (
                "mega SiTU kernel patch bakes beta=4.0/linear_beta=25.0; "
                "got a checkpoint with different constants"
            )

        # EP a2a backends (megamoe / DeepEP) move each row to its experts
        # directly, so the MoE region can consume whatever rows this rank
        # holds — an SP-MoE token shard (attn_tp > 1) or the DP-local batch
        # (DP attention) — with every global token dispatched exactly once.
        # No DP gather and no TP reduce is needed anywhere in the region.
        _a2a_backend = get_moe_a2a_backend()
        self._ep_a2a = _a2a_backend.is_megamoe() or _a2a_backend.is_deepep()

        # Defer the trtllm-gen finalize (top-k weighted unpermute) out of the
        # MoE op and fuse it into the push all-reduce's staging pass
        # (k3_ar_fusion.finalize_all_reduce_push_norm): the rank-local latent
        # never materializes. Only the situ packed-routing trtllm-gen path
        # serves the deferral; sizes beyond the push window fall back to the
        # in-op finalize at runtime (finalize_push_fits).
        self._defer_moe_finalize = (
            get_moe_runner_backend().is_flashinfer_mxfp4()
            and config.hidden_act == "situ"
        )

        # Shared experts (operate in original hidden_size space).
        # Replicate the shared-expert weights (tp1, DSv2 convention) under EP
        # a2a: the block runs on partial batches (shard / DP-local rows), and
        # a TP-sharded partial sum could never be reduced across ranks that
        # hold different tokens.
        self._shared_experts_tp1 = self._ep_a2a
        if self.num_shared_experts is not None and self.num_shared_experts > 0:
            shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiK3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
                **(dict(tp_rank=0, tp_size=1) if self._shared_experts_tp1 else {}),
            )
        else:
            self.shared_experts = None

        # SBO (single batch overlap): the shared experts read a fixed slab of
        # weights the routed path never touches (bf16 — the checkpoint leaves
        # shared_experts unquantized — and tp1-replicated under EP a2a, so
        # ~264 MB per layer per rank), while the routed path is a2a-latency
        # bound in decode with HBM mostly idle. Issue the shared experts on the
        # side stream so the two run concurrently instead of back to back; the
        # join happens right before the tail add. Measured on 2x4 GB300
        # (TP8/EP8 MegaMoE + SP-MoE): +4~5% output tok/s and −5% ITL over
        # bs 1–32, GSM8K unchanged — so it is on whenever the shape allows,
        # no flag.
        # EP a2a only: with plain-TP experts the fused front already lands both
        # partial sums in one collective (_forward_fused), a strictly better
        # overlap than two streams.
        self._sbo_shared_overlap = (
            self._ep_a2a
            and self.shared_experts is not None
            and self.alt_stream is not None
        )

        if self.use_latent_moe:
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=None,
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
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_norm = None
            self.routed_expert_up_proj = None

        # Static eligibility for fusing the fused-front latent all-reduce with
        # the RMSNorm epilogue (SGLANG_K3_AR_FUSION). The kernel views the flat
        # [latent | shared] buffer as [3N, NORM_DIM] rows and norms the first N,
        # so it requires latent width == NORM_DIM and shared width == 2*NORM_DIM
        # (K3: 3584 / 7168). Decided once here so the hot path only reads a bool
        # and never re-validates dims per forward.
        self.fuse_ar_norm = (
            self.routed_expert_norm is not None
            and self.moe_hidden_size == k3_ar_fusion.NORM_DIM
            and hidden_size == 2 * k3_ar_fusion.NORM_DIM
        )
        # Static eligibility for the column-parallel up_proj tail (gemm_ag):
        # per-rank 1/8-column GEMV -> multicast all-gather staged in the v2
        # push workspace -> spin-add3 with shared_output (+ prefix_sum),
        # replacing the replicated [3584, 7168] GEMM + _add3 (~1.5-2x at
        # decode sizes, 1/8 of the weight bytes read per rank). Kernel dims
        # are fixed to fuse_ar_norm's (3584 -> 7168) over TP8; per-batch
        # capacity checks live in k3_ar_fusion.gemm_ag_up_fits.
        self._gemm_ag_up_eligible = (
            self.fuse_ar_norm
            and self.tp_size == 8
            and self.routed_expert_up_proj is not None
            and isinstance(self.routed_expert_up_proj.weight, torch.Tensor)
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
        if self.shared_experts is not None and get_moe_a2a_backend().is_none():
            mods = [
                self.shared_experts.gate_up_proj,
                self.gate,
                self.routed_expert_down_proj,
            ]
        elif envs.SGLANG_K3_FUSED_FRONT.get():
            # EP a2a: the shared experts are tp1-replicated and run on the side
            # stream, so they stay out of the merge -- but the router gate and the
            # latent down-proj still read the same hidden_states, and merging just
            # those two is what lets one GEMM read the activations once. The gate
            # GEMM alone is only 896 rows, which is too few to use the machine
            # well; folded into the 3584-row down-proj it comes almost free.
            mods = [self.gate, self.routed_expert_down_proj]
        else:
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

        from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.mega_moe import _get_mega_moe_symm_buffer

        # In SP-MoE mode (KimiK3DecoderLayer reduce-scatters the o_proj
        # output) the incoming rows are already this rank's token shard, so
        # the fused a2a below dispatches each token exactly once. On the
        # non-scattered fallback path the rows are the full batch (redundant
        # across ranks but correct).
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
            get_moe_ep_group().device_group,
            num_experts=self.experts.num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=self._mega_top_k,
            hidden=self.moe_hidden_size,
            intermediate_hidden=self._mega_intermediate_size,
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
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self.experts.mega_l1_weights,
            self.experts.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            # Sentinel: selects the K3 SiTU branch in the DeepGEMM mega kernel
            # (beta=4.0 / linear_beta=25.0 baked in).
            activation_clamp=_K3_MEGA_SITU_SENTINEL_CLAMP,
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
        # Do NOT test cfg.scoring_func: it defaults to "softmax" and TopK
        # documents it as unused. What actually selects sigmoid is the
        # grouped-topk-with-correction-bias route (DSv3 noaux_tc), which calls
        # biased_grouped_topk and hardwires scoring_func="sigmoid".
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

    def _reduce_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Unfused-front latent tail: TP-partial routed sums must be reduced
        in latent space BEFORE the RMSNorm (sum(norm(x_i)) != norm(sum(x_i)))."""
        if not self._routed_needs_reduce:
            return self._latent_norm(latent)
        return self._latent_norm(tensor_model_parallel_all_reduce(latent))

    def _forward_unfused(
        self, hidden_states: torch.Tensor, *, prefix_sum: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Front section with three separate GEMMs, each reading
        hidden_states: shared-expert MLP, router gate, latent down-proj."""
        # Shared experts on original hidden_states. Under SBO they go to the
        # side stream and are joined at the tail (see _sbo_shared_overlap).
        #
        # Issued *after* the front, deliberately: alt_stream.wait_stream() makes
        # the side stream wait for whatever the main stream has enqueued so far,
        # so issuing here means the shared experts overlap the routed a2a rather
        # than the front GEMMs. The shared branch is the shorter of the two and
        # does not need a head start; running it against the front only takes
        # bandwidth away from the critical path.
        shared_output = None
        shared_event = None

        def issue_shared():
            nonlocal shared_output, shared_event
            if self.shared_experts is None or hidden_states.shape[0] == 0:
                return
            if self._sbo_shared_overlap:
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self.shared_experts(hidden_states)
                    shared_event = self.alt_stream.record_event()
            else:
                shared_output = self.shared_experts(hidden_states)

        # Front: gate + TopK (+ latent down-proj when the merged front covers it).
        # The gate and the latent down-proj read the same hidden_states, so the
        # merged-weight strategies compute both in one GEMM; see
        # kernels/ops/moe/moe_front.py for the strategy table.
        routed_input = self._ep_front(hidden_states)
        if routed_input is None:
            routed_input = self._ep_front_overlap(hidden_states)
        topk_output = None
        if routed_input is not None:
            topk_output, routed_input = routed_input
        else:
            # MoEGate produces fp32 router logits on CUDA (via linear_bf16_fp32
            # or dsv3_router_gemm); non-CUDA falls back to F.linear (bf16). The
            # fp32 logits reach the radix router from moe_fused_gate.
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)

        issue_shared()

        if not self.use_latent_moe:
            expert_output = self.experts(hidden_states, topk_output)
            if shared_event is not None:
                torch.cuda.current_stream().wait_event(shared_event)
            if shared_output is not None:
                expert_output = expert_output + shared_output
            if self.tp_size > 1:
                expert_output = tensor_model_parallel_all_reduce(expert_output)
            if prefix_sum is not None:
                expert_output = expert_output + prefix_sum
            return expert_output

        # Latent MoE: compress after routing, before experts
        if TYPE_CHECKING:
            assert (
                self.routed_expert_down_proj is not None
                and self.routed_expert_up_proj is not None
            )

        if routed_input is None:
            routed_input, _ = self.routed_expert_down_proj(hidden_states)
        expert_output = (
            self._forward_mega_experts(routed_input, topk_output)
            if self._use_mega_moe
            else self.experts(routed_input, topk_output)
        )
        latent = self._reduce_latent(expert_output)
        # up_proj is replicated, so the routed output is now fully reduced.
        out, _ = self.routed_expert_up_proj(latent)
        if shared_event is not None:
            # SBO join: as late as possible, so the side-stream shared experts
            # get the whole routed a2a + latent tail to hide under.
            torch.cuda.current_stream().wait_event(shared_event)
        if shared_output is not None:
            # tp1 shared experts (SP-MoE) are complete per-rank; TP-sharded
            # ones need the partial-sum reduction.
            if self.tp_size > 1 and not self._shared_experts_tp1:
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            return _add3(out, shared_output, prefix_sum)
        return out if prefix_sum is None else out + prefix_sum

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
                get_tp_group(), disabled=not is_allocation_symmetric()
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
            # NOTE: the latent AR must stay serialized after the shared AR
            # (both reuse the v2 pull semaphores; concurrent calls would
            # corrupt each other's barrier windows) — the join above does it.
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
                buf = tensor_model_parallel_all_reduce(buf)

        latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
        shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
        if not fused_norm:
            latent = self._latent_norm(latent)
        out, _ = self.routed_expert_up_proj(latent)

        # prefetch_bc: b (shared_output) was produced by the all-reduce and
        # c (prefix_sum) even earlier; the AR is a plain launch (full
        # barrier), so both are complete once the norm / up_proj GEMM chain
        # starts — only `a`'s producer can still be in flight at PDL entry.
        return _add3(out, shared_output, prefix_sum, prefetch_bc=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        prefix_sum: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        """A pending prefix_sum is always consumed here: folded into the
        3-way JIT tail add when covered, plain adds otherwise (bit-identical
        either way).

        Under DP attention with TP-sharded experts (a2a none, forward_batch
        given) the experts run on the DP-gathered global batch — the internal
        reduces stay over the full TP group, which is exactly right in
        gathered space — while prefix_sum stays in local token space, added
        after the scatter back. EP a2a backends skip the gather entirely:
        dispatching the DP-local rows (or the SP-MoE shard of them the
        decoder layer already produced) covers every global token exactly
        once, and prefix_sum is consumed in the tail add like the non-DP
        path — gathering first would just replicate the whole batch onto
        every rank (tp-fold redundant compute + a2a traffic)."""
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        use_dp = self._dp_attention and forward_batch is not None and not self._ep_a2a
        if use_dp:
            local_hidden_states = hidden_states
            hidden_states = get_global_dp_buffer(get_tp_group())
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
            dp_prefix_sum, prefix_sum = prefix_sum, None
        if hidden_states.shape[0] > 0 and self._eligible_for_fused_front:
            out = self._forward_fused(hidden_states, prefix_sum=prefix_sum)
        else:
            out = self._forward_unfused(hidden_states, prefix_sum=prefix_sum)
        if use_dp:
            global_out = out
            out = get_local_dp_buffer(_dp_local_buffer_group())
            dp_scatter(out, global_out, forward_batch)
            if dp_prefix_sum is not None:
                out = out + dp_prefix_sum
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
        **kwargs,
    ) -> None:
        super().__init__()
        self.all_reduce_fusion = all_reduce_fusion
        # Side stream for the [f_a|b] + f_b tiny GEMVs: they read only
        # hidden_states, so they can run concurrently with the wide fused
        # [q,k,v,g] GEMM on the main stream (graphed decode/verify only).
        # Same SM bound rationale as the MLA gate stream.
        self._bfa_alt_stream = bfa_alt_stream
        self._bfa_bs_limit = (
            (128 if is_blackwell_supported() else 64)
            if bfa_alt_stream is not None
            else 0
        )
        self.tp_size = get_parallel().tp_size
        # KDA is an attention layer: all head-sharded params must follow the
        # attention-TP group (= tp under plain TP, = 1 under DP attention),
        # matching the mamba state cache sizing (KimiLinearCacheParams uses
        # get_attention_tp_size). Mirrors GLM5-next's head_shard_size pattern.
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.v_head_dim
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % self.attn_tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.attn_tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.use_full_rank_gate = config.linear_attn_config.get(
            "use_full_rank_gate", False
        )

        # The fused path hardcodes tp_size sharding, so require attn_tp == tp.
        # For the full-rank gate (K3) the checkpoint quantizes only the MoE
        # experts; attention linears resolve to UnquantizedLinearMethod, so a
        # non-None quant_config is fine for the merged projection.
        self.do_fuse_qkvbfg = self.attn_tp_size == self.tp_size and (
            quant_config is None or self.use_full_rank_gate
        )

        if self.do_fuse_qkvbfg and self.use_full_rank_gate:
            # Fuse only the alignment-friendly wide projections [q, k, v, g]
            # (6144/rank at TP8). Folding b (12/rank) and f_a (128, replicated)
            # in as well skews the output dim to 6284 and measurably degrades
            # the GEMM kernel selection; they stay as separate tiny GEMVs.
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
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=f"{prefix}.fused_qkvg_proj",
            )
            self.split_sizes = [
                3 * projection_size // self.tp_size,
                projection_size // self.tp_size,
            ]
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=f"{prefix}.b_proj",
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
                tp_rank=self.attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=f"{prefix}.f_b_proj",
            )
            # Merged [f_a | b] weight, built after weight loading by
            # _merge_bfa_weights().
            self._bfa_w: Optional[torch.Tensor] = None
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
            attn_tp_rank = self.attn_tp_rank
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=self.attn_tp_size,
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
                tp_rank=attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=f"{prefix}.f_b_proj",
            )
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=self.attn_tp_size,
                prefix=f"{prefix}.b_proj",
            )

            if self.use_full_rank_gate:
                self.g_proj = ColumnParallelLinear(
                    self.hidden_size,
                    projection_size,
                    bias=False,
                    quant_config=quant_config,
                    tp_rank=attn_tp_rank,
                    tp_size=self.attn_tp_size,
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
                    tp_rank=attn_tp_rank,
                    tp_size=self.attn_tp_size,
                    prefix=f"{prefix}.g_b_proj",
                )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.attn_tp_size), dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=f"{prefix}.qkv_conv1d",
        )
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        # K3 checkpoint stores A_log as [head_dim] (128), but the FLA kernel
        # expects exactly local_num_heads elements.  We define the param as
        # [1, 1, local_num_heads, 1] (matching the kimi_linear.py convention)
        # and attach a custom weight_loader that handles both the old 4-D
        # format and the K3 1-D [head_dim] format by narrowing to the first
        # num_heads elements then TP-sharding.
        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )

        def _a_log_weight_loader(
            param: torch.Tensor, loaded_weight: torch.Tensor
        ) -> None:
            tp_rank = get_parallel().attn_tp_rank
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
            # SGLANG_K3_AR_FUSION: keep the o_proj output TP-partial and
            # complete the reduce at the decoder layer via the fused MNNVL
            # all-reduce (which can fold the attn-res prefix add in). Only
            # valid when the attn TP group is the full TP group (the fused
            # comm lives there).
            reduce_results=not self.all_reduce_fusion,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            # Reduce within the attn-TP group: the default reduce path uses
            # the full-TP collective, which at attn_tp>1 is both the wrong
            # group (sums across DP groups) and asymmetric vs idle DP ranks
            # (deadlocks the per-layer DP gather). Off under all_reduce_fusion:
            # the fused AR does the reduce itself (reduce_results=False) and the
            # forward hands o_proj a slice of a persistent symmetric region —
            # leaving this True would wrap the GEMM in
            # use_symmetric_memory(attn_tp), which allocates its own output and
            # so defeats the caller-owned buffer. At the fusion config
            # attn_tp==tp so the fused full-TP reduce is the same group anyway.
            use_dp_attention_reduce=not self.all_reduce_fusion,
            prefix=f"{prefix}.o_proj",
        )
        if self.all_reduce_fusion and not _o_proj_takes_output(self.o_proj):
            # the fused AR reduces o_proj's output in place out of a symmetric
            # buffer, which needs the GEMM to write into caller-owned storage
            self.all_reduce_fusion = False
            self.o_proj.reduce_results = True
            self.o_proj.use_dp_attention_reduce = True
        k3_gemm_ar.maybe_wrap_o_proj(self.o_proj)
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

        Called once from load_weights (after all weights are loaded, before
        cuda graph capture)."""
        if not self.use_full_rank_gate:
            return
        self._bfa_w, sizes = _merge_weights_as_views(
            [self.f_a_proj, self.b_proj], pad_rows_to=8
        )
        self._bfa_fa_size, self._bfa_b_size = sizes

    def _prepare_fused_decode(self) -> None:
        """Static inputs for the fused KDA decode kernel
        (kernels/ops/attention/kda_fused_decode): per-segment transposed fp32 conv
        weights [4, seg], dense fp32 conv bias, fp32 output-norm weight. Stashed on the
        attention layer for the KDA backend; when the shapes do not match
        the compiled kernel the stash stays unset and decode keeps the
        unfused chain. Called once from load_weights (after all weights are
        loaded, before cuda graph capture)."""
        if _is_hip:
            # The fused KDA decode kernel is NVIDIA-only
            return
        layer = self.attn
        w = layer.conv_weights
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

    def forward_qkvbfg_fused(self, hidden_states: torch.Tensor):
        if self.use_full_rank_gate:
            if self._bfa_w is not None:
                w = self._bfa_w
                n_fa, n_b = self._bfa_fa_size, self._bfa_b_size
                from sglang.kernels.ops.kimi_k3 import kimi_k3_tiny_gemm as gemm

                if (
                    self._bfa_alt_stream is not None
                    and get_is_capture_mode()
                    and 0 < hidden_states.shape[0] <= self._bfa_bs_limit
                ):
                    # Issue the tiny [f_a|b] + f_b GEMVs on the side stream,
                    # then the wide [q,k,v,g] GEMM on the main stream; both
                    # read only hidden_states. Join before the split's
                    # consumers touch beta/forget_gate.
                    alt = self._bfa_alt_stream
                    cur = torch.cuda.current_stream()
                    alt.wait_stream(cur)
                    with torch.cuda.stream(alt):
                        bfa = gemm(hidden_states, w)
                        forget_gate = gemm(bfa[..., :n_fa], self.f_b_proj.weight)
                        beta = bfa[..., n_fa : n_fa + n_b]
                    fused_states, _ = self.fused_qkvg_proj(hidden_states)
                    qkv, g_proj_states = torch.split(
                        fused_states, self.split_sizes, dim=-1
                    )
                    cur.wait_stream(alt)
                    return qkv, beta, forget_gate, g_proj_states

                fused_states, _ = self.fused_qkvg_proj(hidden_states)
                qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)
                bfa = gemm(hidden_states, w)
                forget_gate = gemm(bfa[..., :n_fa], self.f_b_proj.weight)
                beta = bfa[..., n_fa : n_fa + n_b]
            else:
                fused_states, _ = self.fused_qkvg_proj(hidden_states)
                qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)
                beta = self.b_proj(hidden_states)[0]
                forget_gate = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
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
        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states
            )

        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unflatten(-1, (-1, self.head_dim))
            if not forward_batch.forward_mode.is_target_verify():
                # Only chunk_kda (extend) wants pre-activated beta; the verify
                # kernel sigmoids it in-kernel like decode.
                beta = beta.float().sigmoid()
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        # Fused KDA handoff (attempt-and-verify): offer the output-norm gate
        # so covered decode and target-verify kernels can fold gated RMSNorm
        # into the recurrence kernel. If the backend leaves the stash
        # unconsumed (env off or shape not covered), apply o_norm here as
        # before.
        fused_onorm = self._kda_fused_decode_ready and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        if fused_onorm:
            self.attn._k3_onorm_gate = g_proj_states
            self.attn._k3_onorm_consumed = False

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        if fused_onorm:
            self.attn._k3_onorm_gate = None
            fused_onorm = self.attn._k3_onorm_consumed
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
    ) -> None:
        self.all_reduce_fusion = all_reduce_fusion
        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
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
            reduce_results=not self.all_reduce_fusion,
            alt_stream=alt_stream,
        )
        # Installed before the output-gate wrap below so the gate multiply is
        # applied to x before the fused GEMM+AR sees it.
        if self.all_reduce_fusion and not _o_proj_takes_output(self.o_proj):
            # the fused AR reduces o_proj's output in place out of a symmetric
            # buffer, which needs the GEMM to write into caller-owned storage
            self.all_reduce_fusion = False
            self.o_proj.reduce_results = True
            self.o_proj.use_dp_attention_reduce = True
        k3_gemm_ar.maybe_wrap_o_proj(self.o_proj)
        if self.all_reduce_fusion:
            # reduce_results=False was passed through super().__init__ above;
            # the fused all-reduce does the reduce itself and reduces the o_proj
            # output in place, so hand the GEMM a slice of the persistent
            # symmetric buffer (k3_ar_fusion.symm_buffer) and do NOT set
            # use_dp_attention_reduce — its inner attn_tp symm_ctx allocates its
            # own output and would defeat the caller-owned buffer. At the fusion
            # config (attn_tp==tp) the fused full-TP reduce is the same group as
            # the attn_tp reduce.
            # The wrap is installed before the output-gate wrap below so the
            # gate multiply stays outside it and only the o_proj GEMM writes the
            # region slice. NOTE: the captured name must differ from the
            # gate block's `_orig_o_proj_forward` — closures capture the
            # __init__ local by reference, and reusing the name would rebind it
            # to this wrapper (infinite recursion + nested pool enter).
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
            # K3 has no LayerCommunicator, so o_proj (reduce_results=True by
            # default here, unlike deepseek's communicator flow) must reduce
            # within the attn-TP group itself — the default full-TP collective
            # is the wrong group at attn_tp>1 and deadlocks against idle DP
            # ranks.
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
                tp_rank=get_parallel().attn_tp_rank,
                tp_size=get_parallel().attn_tp_size,
                prefix=f"{prefix}.g_proj",
            )
            # Output gate must multiply the TP-local attention output right
            # before o_proj (vLLM: attn_out * sigmoid(g_proj(hidden_states))).
            # o_proj is invoked deep inside DeepseekV2AttentionMLA forward
            # cores, so wrap its forward at the instance level; the module
            # itself (weights, reduce_results, loading path) is untouched.
            self._gate_hidden_states = None
            # (gate, producer stream) issued on the alt stream by forward();
            # None when the lazy path computes the gate here instead.
            self._gate_precomputed = None
            self._gate_alt_stream = gate_alt_stream
            # Above this token count the attention-core kernels fill the SMs
            # on their own and the overlap only adds sync overhead (same
            # bound as deepseek_v4).
            self._gate_bs_limit = (
                (128 if is_blackwell_supported() else 64)
                if self._gate_alt_stream is not None
                else 0
            )
            _orig_o_proj_forward = self.o_proj.forward

            def _gated_o_proj_forward(x, *args, **kwargs):
                gate_input = self._gate_hidden_states
                self._gate_hidden_states = None
                precomputed = self._gate_precomputed
                self._gate_precomputed = None
                if precomputed is not None:
                    # Use wait_stream rather than an explicit event so the
                    # breakable-CUDA-graph runner can track the side-stream
                    # join across graph-segment boundaries.
                    torch.cuda.current_stream().wait_stream(precomputed[1])
                if gate_input is not None and not isinstance(x, tuple):
                    gate = (
                        precomputed[0]
                        if precomputed is not None
                        else self.g_proj(gate_input)[0]
                    )
                    from sglang.kernels.ops.kimi_k3 import mla_output_gate

                    if mla_output_gate.covered(x, gate):
                        # One kernel for x * sigmoid(gate); double rounding
                        # matches the unfused pair bit-for-bit.
                        x = mla_output_gate.kimi_k3_mla_output_gate(x, gate)
                    else:
                        x = x * torch.sigmoid(gate)
                return _orig_o_proj_forward(x, *args, **kwargs)

            self.o_proj.forward = _gated_o_proj_forward

    def _precompute_output_gate(self, hidden_states: torch.Tensor) -> None:
        """Issue the output-gate GEMM on the alt stream so it overlaps the
        attention core; the lazy path in the o_proj wrap otherwise computes
        it on the critical path right before the gate multiply. The gate
        tensor stays referenced via _gate_precomputed until the wrap joins,
        so its memory cannot be reused while the alt stream still writes."""
        self._gate_precomputed = None
        if (
            self._gate_alt_stream is not None
            and get_is_capture_mode()
            # The attention-core break ends the segment between the alt-stream
            # event record and the o_proj-side wait, so under breakable capture
            # the wait would cross graph segments; use the lazy path instead.
            and not is_in_breakable_cuda_graph()
            and (0 < hidden_states.shape[0] <= self._gate_bs_limit)
        ):
            alt = self._gate_alt_stream
            alt.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(alt):
                gate, _ = self.g_proj(hidden_states)
            self._gate_precomputed = (gate, alt)

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
            self._precompute_output_gate(hidden_states)
        return super().forward(
            positions, hidden_states, forward_batch, zero_allocator, **kwargs
        )


class KimiK3DecoderLayer(nn.Module):
    """Decoder layer carrying the K3 attention-residual stream."""

    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.is_moe = config.is_moe
        self.layer_idx = layer_idx
        self._dp_attention = is_dp_attention_enabled()
        # mlp-sync (DP attention OR MoE a2a/EP) pads extend batches to
        # attn_tp multiples; attention must then run on the real rows only.
        self._trim_padded_attn = require_mlp_sync(get_server_args())
        # A layer runs MoE (vs a plain dense MLP) iff it is past the dense
        # prefix and on the MoE cadence — same predicate the mlp construction
        # below uses.
        self._is_moe_layer = (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        # SP-MoE (EP a2a backend — megamoe or DeepEP): o_proj defers its
        # attention-TP reduction; this layer completes it as a reduce-scatter
        # so the whole MoE region (agg2, norms, gate, latent projs, tp1
        # shared experts, EP a2a dispatch) runs on 1/attn_tp of the rows,
        # then all-gathers rows back after the MoE tail add. RS+AG moves the
        # same bytes the o_proj all-reduce did, the shared-expert all-reduce
        # disappears via tp1 weights, and each rank dispatches only its shard
        # through the a2a (kills the attn_tp-fold dispatch redundancy) —
        # strictly less communication + MoE-front compute /attn_tp. Works the
        # same under DP attention: the attn_tp group is then the
        # within-replica subgroup, rows are the DP-local batch, and
        # KimiK3MoE skips the DP gather under EP a2a so the shard flows
        # straight into the a2a. With attn_tp == 1 (full DP attention) there
        # is no attention reduce to convert — the MoE-side gather skip alone
        # removes the replication. Dense layers are excluded: their
        # column-parallel MLP has no per-token decomposition that survives a
        # token shard.
        _a2a_backend = get_moe_a2a_backend()
        self._sp_moe = (
            (_a2a_backend.is_megamoe() or _a2a_backend.is_deepep())
            and self._is_moe_layer
            and get_parallel().attn_tp_group.world_size > 1
        )

        # The fused all-reduce only serves the attn-res path (attn_res is
        # config-static), so the standard path stays byte-for-byte untouched
        # and always sees a reduced attention output.
        # Mutually exclusive with SP-MoE: both complete o_proj's deferred
        # reduction, but SP-MoE reduce-scatters to a shard whereas the fusion
        # produces the full batch in a symm buffer — an SP-MoE layer builds
        # o_proj in the plain deferred-reduce config (reduce_results forced off
        # below) and reduce-scatters instead.
        attn_tp_size = get_parallel().attn_tp_size
        self.all_reduce_fusion = (
            not self._sp_moe
            and attn_tp_size > 1
            and attn_tp_size == get_parallel().tp_size
            and config.attn_res_block_size is not None
            and k3_ar_fusion.enabled()
        )

        # Attention
        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiK3DeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                all_reduce_fusion=self.all_reduce_fusion,
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
                alt_stream=alt_streams[1] if alt_streams is not None else None,
                gate_alt_stream=alt_streams[2] if alt_streams is not None else None,
            )

        # the attention drops the fusion when its o_proj cannot write into
        # caller-owned storage; the layer's own AR call-site must agree
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

        if self._sp_moe:
            # o_proj emits TP-partial sums; _finish_attn_reduce completes the
            # reduction (RS on the clean attn-res path, AR on fallbacks).
            o_proj = getattr(self.self_attn, "o_proj", None)
            assert o_proj is not None, "SP-MoE requires attention exposing o_proj"
            o_proj.reduce_results = False
            if k3_sp_collective.enabled():
                # The table selects NVLS pull RS for larger token buckets.
                # Only those o_proj outputs come from the persistent symmetric
                # buffer; small push RS keeps the regular graph allocator.
                _sp_inner_o_proj_forward = o_proj.forward

                def _sp_o_proj_forward(x, *args, **kwargs):
                    output_rows = k3_sp_collective.get_o_proj_output_rows(x.shape[0])
                    if k3_sp_collective.requires_symmetric_rs(output_rows, x.device):
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

    def _finish_attn_reduce(
        self,
        attn_out: torch.Tensor,
        allow_scatter: bool,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int, bool]:
        """Complete o_proj's deferred TP reduction under SP-MoE.

        Returns (reduced tensor, shard row offset, residual_fused); offset is
        -1 when the result covers the full batch (non-SP mode, or fallback
        all-reduce for row counts not divisible by attn_tp)."""
        if not self._sp_moe:
            return attn_out, -1, False
        group = get_parallel().attn_tp_group
        num_tokens = attn_out.shape[0]
        if allow_scatter and num_tokens > 0 and num_tokens % group.world_size == 0:
            shard = num_tokens // group.world_size
            custom_out = k3_sp_collective.reduce_scatter_res(attn_out, residual)
            if custom_out is not None:
                return (
                    custom_out,
                    group.rank_in_group * shard,
                    residual is not None,
                )
            out = torch.empty(
                (shard, attn_out.shape[1]),
                dtype=attn_out.dtype,
                device=attn_out.device,
            )
            group.reduce_scatter_tensor(out, attn_out)
            return out, group.rank_in_group * shard, False
        return group.all_reduce(attn_out), -1, False

    def _run_self_attn(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # DP attention: idle ranks (padded to the global shape) have no
        # attention metadata; pass hidden_states through shape-preserving
        # (same as the LayerCommunicator models' is_idle skip).
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        # mlp-sync (DP attention OR MoE a2a/EP — require_mlp_sync) pads
        # extend batches to a multiple of attn_tp_size
        # (prepare_mlp_sync_batch ceil_align), but the attention metadata
        # (qo_indptr / query_start_loc) covers only the real tokens — the
        # flashinfer ragged prefill rejects the row mismatch, and silent
        # paths write the padded rows' garbage KV through the zero-padded
        # out_cache_loc entries (clobbering pool slot 0 → cross-request
        # corruption). Run attention on the real rows and zero-pad the
        # output back; padded rows are discarded downstream.
        num_padded = hidden_states.shape[0]
        num_real = num_padded
        if self._trim_padded_attn and forward_batch.forward_mode.is_extend():
            extend_lens = forward_batch.extend_seq_lens_cpu
            if extend_lens is not None:
                num_real = min(int(sum(extend_lens)), num_padded)
        if num_real != num_padded:
            with k3_sp_collective.o_proj_output_rows(num_padded):
                attn_out = self._run_self_attn_inner(
                    hidden_states[:num_real],
                    positions[:num_real],
                    forward_batch,
                    zero_allocator,
                )
            padded_o_proj = k3_sp_collective.finish_padded_o_proj_output(
                attn_out, num_padded
            )
            if padded_o_proj is not None:
                return padded_o_proj
            out = hidden_states.new_zeros(num_padded, attn_out.shape[-1])
            out[:num_real] = attn_out
            return out
        return self._run_self_attn_inner(
            hidden_states, positions, forward_batch, zero_allocator
        )

    def _run_self_attn_inner(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        # For MLA layers with q_lora_rank, set up communicator attn_inputs
        # before the forward call (normally done by LayerCommunicator).
        from sglang.srt.layers.communicator import (
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
        residual: Optional[torch.Tensor],
        attn_res: Optional[AttnResidual],
        zero_allocator: BumpAllocator,
        input_sharded: bool = False,
        keep_sharded: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        if attn_res is not None:
            return self._forward_attn_residual(
                positions,
                hidden_states,
                residual,
                attn_res,
                forward_batch,
                zero_allocator,
                input_sharded,
                keep_sharded,
            )

        assert not input_sharded
        # Standard residual path
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self._run_self_attn(
            hidden_states, positions, forward_batch, zero_allocator
        )
        # standard path returns a full-size residual to the next layer, so
        # complete the deferred o_proj reduction as a plain all-reduce
        hidden_states, _, _ = self._finish_attn_reduce(
            hidden_states, allow_scatter=False
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        return hidden_states, residual, False

    def _forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        attn_res: AttnResidual,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        input_sharded: bool,
        keep_sharded: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        # Between attn-res layers hidden_states carries the previous layer's
        # un-added MLP delta and prefix_sum the prefix it extends (None at
        # stream start / PP entry, where hidden_states already is the head).

        # ---- Aggregation 1: attention side. Write layers snapshot the
        # pre-attention prefix into the bank in the same call (fused into
        # the fast kernel; standalone copy on other paths). ----
        if input_sharded:
            assert self._sp_moe
            input_rows = _sp_local_rows(hidden_states)
            fused_ag = attn_res.forward_sp_all_gather(
                hidden_states,
                prefix_sum,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                rows=input_rows,
                write=self.is_block_write_layer,
            )
            if fused_ag is not None:
                hidden_states, prefix_sum = fused_ag
            else:
                hidden_states, prefix_sum = attn_res.forward(
                    hidden_states,
                    prefix_sum,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    rows=input_rows,
                    write=self.is_block_write_layer,
                )
                # Aggregate/norm and snapshot only this rank's rows, then
                # gather the normalized tensor consumed by attention.
                hidden_states = _sp_all_gather_rows(hidden_states)
        else:
            hidden_states, prefix_sum = attn_res.forward(
                hidden_states,
                prefix_sum,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                write=self.is_block_write_layer,
            )
        if self.is_block_write_layer:
            prefix_sum = None

        # ---- Attention ----
        hidden_states = self._run_self_attn(
            hidden_states, positions, forward_batch, zero_allocator
        )

        # ---- Complete o_proj's deferred reduction ----
        # SP-MoE takes precedence (reduce-scatter to this rank's token shard);
        # otherwise the fused all-reduce when enabled; otherwise o_proj already
        # reduced itself (use_dp_attention_reduce, on when neither is active).
        rows = None
        shard_lo = -1
        agg2_fused = False
        if self._sp_moe:
            group = get_parallel().attn_tp_group
            if (
                hidden_states.shape[0] > 0
                and hidden_states.shape[0] % group.world_size == 0
            ):
                shard = hidden_states.shape[0] // group.world_size
                fused_rows = slice(
                    group.rank_in_group * shard,
                    (group.rank_in_group + 1) * shard,
                )
                fused_rs = attn_res.forward_sp_reduce_scatter(
                    hidden_states,
                    prefix_sum,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                    rows=fused_rows,
                )
            else:
                fused_rs = None
            if fused_rs is not None:
                hidden_states, prefix_sum = fused_rs
                rows = fused_rows
                shard_lo = fused_rows.start
                agg2_fused = True
            else:
                hidden_states, shard_lo, residual_fused = self._finish_attn_reduce(
                    hidden_states, allow_scatter=True, residual=prefix_sum
                )
                if shard_lo >= 0:
                    rows = slice(shard_lo, shard_lo + hidden_states.shape[0])
                    if residual_fused:
                        prefix_sum = None
                    elif prefix_sum is not None:
                        # Shard carry already holds the destination-local prefix;
                        # the first sharded layer still holds a full-batch prefix.
                        if prefix_sum.shape[0] != hidden_states.shape[0]:
                            prefix_sum = prefix_sum[rows]
        elif self.all_reduce_fusion:
            # Complete the o_proj reduce here, folding the pending prefix add
            # into the fused all-reduce; attn_res then takes the pre-added
            # tensor through its prefix_sum=None branch (same semantics:
            # (normed, new_prefix) with new_prefix = prefix + attn_out).
            hidden_states = k3_ar_fusion.all_reduce(hidden_states, prefix_sum)
            prefix_sum = None

        # ---- Aggregation 2: MLP side (on the shard under SP-MoE) ----
        if not agg2_fused:
            hidden_states, prefix_sum = attn_res.forward(
                hidden_states,
                prefix_sum,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
                rows=rows,
            )

        # ---- MLP (consumes +prefix_sum: MoE folds it into the 3-way tail
        # add, dense adds it after down_proj) ----
        out = self.mlp(
            hidden_states, prefix_sum=prefix_sum, forward_batch=forward_batch
        )
        if shard_lo >= 0:
            if keep_sharded:
                return out, None, True
            out = _sp_all_gather_rows(out)
        return out, None, False


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
        self.pp_group = get_pp_group()
        self.dspark_layers_to_capture: Optional[list[int]] = None
        self._dp_attention = is_dp_attention_enabled()
        self._trim_padded_attn = require_mlp_sync(get_server_args())

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
                # Under DP attention each rank embeds only its local tokens:
                # reduce within the attention-TP group, not the full TP group.
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Multi-stream pool (deepseek_v4 pattern): every alt stream is
        # constructed here and threaded down to the layers. Slots:
        #   [0] MoE dual-stream shared-expert tail
        #   [1] DeepseekV2AttentionMLA base internals (forwarded; unused by K3)
        #   [2] MLA output-gate GEMM, overlaps the attention core
        # (The attn-res bank write no longer needs a stream: it is fused
        # into the agg1 fast kernel, see AttnResidual.forward(write=True).)
        # Disable on HIP code path.
        self.alt_streams = None if _is_hip else [torch.cuda.Stream() for _ in range(3)]

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiK3DecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

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
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            if TYPE_CHECKING:
                assert isinstance(hidden_states, torch.Tensor)
                assert isinstance(residual, torch.Tensor | None)

        # mlp-sync (DP attention OR MoE a2a/EP) pads extend batches to a
        # multiple of attn_tp_size; attention layers run on the real rows
        # only (_run_self_attn trims), so the KV write locations must match
        # the trimmed length. positions and hidden_states keep the padded
        # length for the DP gather/scatter and the MoE.
        if (
            self._trim_padded_attn
            and forward_batch.forward_mode.is_extend()
            and forward_batch.out_cache_loc is not None
            and forward_batch.extend_seq_lens_cpu is not None
        ):
            num_real = int(sum(forward_batch.extend_seq_lens_cpu))
            if forward_batch.out_cache_loc.shape[0] > num_real:
                forward_batch.out_cache_loc = forward_batch.out_cache_loc[:num_real]

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=device,
        )

        attn_res = None
        if self.config.attn_res_block_size is not None:
            attn_res_block_num = _cdiv(self.end_layer, self.config.attn_res_block_size)
            attn_res = AttnResidual(
                hidden_states,
                attn_res_block_num,
                block_residual=residual,
            )
            residual = None

        # Carry the raw residual stream as a token shard across consecutive
        # SP-MoE layers. PP transfer and dspark capture require full tensors,
        # so those uncommon paths keep the established gather-per-layer flow.
        sp_attn_res = (
            attn_res is not None
            and envs.SGLANG_K3_SP_ATTN_RES.get()
            and self.pp_group.world_size == 1
            and self.dspark_layers_to_capture is None
            and k3_sp_collective.enabled()
        )
        sp_sharded = False
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if sp_sharded and not self.layers[i]._sp_moe:
                hidden_states = _sp_all_gather_rows(hidden_states)
                sp_sharded = False
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual, sp_sharded = self.layers[i](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    attn_res=attn_res,
                    zero_allocator=zero_allocator,
                    input_sharded=sp_sharded,
                    keep_sharded=sp_attn_res,
                )
            if (
                self.dspark_layers_to_capture is not None
                and i in self.dspark_layers_to_capture
            ):
                aux_hidden_states.append(
                    self._dspark_capture_stream(i, hidden_states, residual, attn_res)
                )

        if not self.pp_group.is_last_rank:
            assert not sp_sharded
            if attn_res is not None:
                if residual is not None:
                    # Materialize the delayed MLP add: the wire carries the
                    # full stream head (bit-identical to the fused fold).
                    hidden_states = residual + hidden_states
                residual = attn_res.block_residual  # raw bank across ranks
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if attn_res is not None:
                # ---- Final aggregation (output side, folds delayed add) ----
                if sp_sharded:
                    output_rows = _sp_local_rows(hidden_states)
                    fused_output = attn_res.forward_sp_all_gather(
                        hidden_states,
                        residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                        self.norm,
                        rows=output_rows,
                    )
                    if fused_output is not None:
                        hidden_states, _ = fused_output
                    else:
                        hidden_states, _ = attn_res.forward(
                            hidden_states,
                            residual,
                            self.output_attn_res_proj,
                            self.output_attn_res_norm,
                            self.norm,
                            rows=output_rows,
                        )
                        hidden_states = _sp_all_gather_rows(hidden_states)
                else:
                    hidden_states, _ = attn_res.forward(
                        hidden_states,
                        residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                        self.norm,
                    )
            else:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if self.dspark_layers_to_capture is not None:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _dspark_capture_stream(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        attn_res: Optional[AttnResidual],
    ) -> torch.Tensor:
        """Stream value after `layer_idx`: the pre-norm mixture its next
        consumer would compute (next layer's attention side; output side
        for the last layer)."""
        if attn_res is None:
            return hidden_states if residual is None else hidden_states + residual
        if residual is not None:
            # Materialize a delayed MLP add (mirrors the PP-wire fold).
            hidden_states = residual + hidden_states
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
            hidden_states, attn_res.block_residual, nvb, score_proj, score_norm
        )


class KimiK3LinearForCausalLM(nn.Module):
    """Text-only K3 causal LM."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = KimiK3LinearModel(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pp_group = get_pp_group()
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

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.pp_group.world_size > 1:
            # Capture layers living on non-last PP ranks would be silently
            # skipped (the flag is only set on the last rank).
            raise NotImplementedError("DSPARK aux hidden capture requires PP=1.")
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = list(layer_ids)

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

        num_hidden_layers = self.config.num_hidden_layers
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}

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
            if "weight_packed" in name:
                name = name.replace("weight_packed", "weight")

            # MLA: fuse q_a_proj + kv_a_proj_with_mqa → fused_qkv_a_proj_with_mqa
            if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
                fused_name = name.replace(".q_a_proj.", ".fused_qkv_a_proj_with_mqa.")
                fused_name = fused_name.replace(
                    ".kv_a_proj_with_mqa.", ".fused_qkv_a_proj_with_mqa."
                )
                if fused_name in params_dict:
                    param = params_dict[fused_name]
                    if ".q_a_proj." in name:
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
                    if not getattr(layer, "do_fuse_qkvbfg", False):
                        continue
                if weight_name in {".q_proj", ".k_proj", ".v_proj"}:
                    layer_id = int(name.split(".")[2])
                    if not self.config.is_kda_layer(layer_id):
                        continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip experts of layers outside a truncated config (e.g.
                    # num_hidden_layers override), mirroring the non-expert
                    # `name not in params_dict` guard below.
                    if name not in params_dict:
                        break
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
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
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        self.post_load_weights()

    def post_load_weights(self):
        # Also invoked by loader post-load hooks (DummyModelLoader,
        # ShardedStateLoader, remote-instance flows -- none of which call
        # load_weights), so e.g. dummy-weight benchmarks get w_kc/w_vc and
        # the fused buffers too. Same pattern as deepseek_v4.
        # Post-load: absorb kv_b_proj into w_kc and w_vc for MLA layers
        for layer_id in self.config.full_attention_layer_ids:
            if layer_id >= len(self.model.layers):
                continue  # truncated config (e.g. num_hidden_layers override)
            layer = self.model.layers[layer_id]
            if isinstance(layer, PPMissingLayer):
                continue
            self_attn = layer.self_attn
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            if hasattr(self_attn.kv_b_proj, "weight_scale"):
                self_attn.w_scale = self_attn.kv_b_proj.weight_scale

        # Post-load: precompute the attn-res combined score weights BEFORE
        # cuda graph capture (a lazy first call inside get_cw would bake the
        # multiply into every captured graph replay otherwise). Warm both
        # dtypes: the fast kernel consumes bf16, the triton fallback fp32.
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

        # Post-load: merge the horizontally-fused decode weights. Module
        # weights are re-pointed to views of the merged buffers (net extra
        # memory ~0), so this must run after all weights are loaded and
        # before cuda graph capture.
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, KimiK3MoE):
                layer.mlp._merge_front_weights()
                # The router consumes the correction bias in fp32; convert the
                # bf16 checkpoint values once (exact) so the per-call
                # .to(float32) in topk becomes a no-op instead of one upcast
                # kernel per MoE layer per step.
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
            from sglang.kernels.ops.attention.fla.kda import (
                precompile_k3_recompute_w_u_kernel,
            )

            if precompile_k3_recompute_w_u_kernel(
                num_heads=layer.self_attn.local_num_heads,
                dtype=layer.self_attn.o_proj.weight.dtype,
                device=layer.self_attn.dt_bias.device,
            ):
                rank0_log("Precompiled the Kimi-K3 KDA prefill kernel.")
            break


class KimiK3ForConditionalGeneration(nn.Module):
    """K3 multimodal wrapper: MoonViT3d tower + KimiK3LinearForCausalLM."""

    supports_cuda_vmm_feature_transport = True

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
            self.language_model = KimiK3LinearForCausalLM(
                config.text_config,
                quant_config,
                prefix="",
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

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.language_model is None:
            raise AttributeError(
                "DSPARK layer capture is not available in encoder-only mode"
            )
        self.language_model.set_dspark_layers_to_capture(layer_ids)

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
            # Same source as MmItemMemoryPool.try_to_recycle(), which waits on
            # configured_tp_size(): the live world size agrees once dist is up,
            # but a refcount that disagrees with the waiter would strand items.
            ipc_consumer_count = max(configured_tp_size(), 1)
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

            deferred = [
                item.model_specific_data.get(DEFERRED_PREPROCESSING_KEY)
                for item in selected_items
            ]
            if any(config is not None for config in deferred):
                if not all(config is not None for config in deferred):
                    raise ValueError(
                        "Kimi-K3 cannot mix deferred and preprocessed image features"
                    )
                from sglang.srt.multimodal.processors.kimi_k25 import (
                    _gpu_preprocess_images,
                )

                first_config = deferred[0]
                image_scale, image_bias = normalization_tensors(
                    first_config["image_mean"], first_config["image_std"], device
                )
                pixel_values, _ = _gpu_preprocess_images(
                    [item.feature for item in selected_items],
                    [config["resize_config"] for config in deferred],
                    image_scale,
                    image_bias,
                    self.vision_tower.patch_size,
                    to_chw=lambda image: to_chw_uint8(image, device=device),
                    post_resize=lambda x: fill_transparent_bg(
                        x, first_config["transparent_bg_config"]
                    ),
                )
                return pixel_values.to(dtype=target_dtype)

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

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
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


EntryClass = [KimiK3ForConditionalGeneration]
