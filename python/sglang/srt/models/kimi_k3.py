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
import triton
import triton.language as tl
from torch import nn

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import k3_ar_fusion, zero_copy_context
from sglang.srt.layers.activation import SiluAndMul, SituAndMul
from sglang.srt.layers.attn_residual import (
    AttnResidual,
    BaseAttnResidual,
    aggregate_stream,
)
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
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils import PPMissingLayer
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
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import make_layers
from sglang.srt.utils.common import (
    BumpAllocator,
    add_prefix,
    rank0_log,
    require_mlp_sync,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# Latent MoE TP reduction strategy for the UNFUSED-front path only (the
# fused front always lands both partial sums in one symmetric
# [latent | shared] buffer and all-reduces once; see _forward_fused):
#   "baseline" - two separate all-reduces (routed latent, then shared)
#   "fi_fused" - flashinfer fused allreduce+rmsnorm for the latent reduce
#   "concat"   - accepted for compat, behaves as "baseline" here (the
#                single-collective tail now rides the fused front)
# A/B history, pre-zero-copy concat path, 8xB300 bs=1 decode (2026-07-03):
# baseline 35.2 tok/s beat concat 33.2 (21.5KB message falls off the
# one-shot allreduce path into two-shot) and fi_fused 33.0. On MULTI-NODE
# TP the trade flipped: one 21.5KB NCCL collective beat a 7KB + 14KB pair
# (2x4 GB300 MNNVL bs=1: 22.05 -> 21.36 ms ITL). Re-benchmark now that the
# fused-front buffer is written with zero copies.
_K3_MOE_REDUCE_MODE = envs.SGLANG_K3_MOE_REDUCE_MODE.get()


def _resolve_moe_reduce_mode() -> str:
    if _K3_MOE_REDUCE_MODE is not None:
        return _K3_MOE_REDUCE_MODE
    try:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(get_tp_group().cpu_group, source_rank=0)):
            return "concat"
    except Exception:
        pass
    return "baseline"


# Horizontal fusion of same-input GEMMs (decode is launch/BW bound):
#   moe_front: shared gate_up + router gate + latent down_proj -> one GEMM
#   kda_bfa:   KDA b_proj + f_a_proj -> one GEMV
_K3_FUSE_MOE_FRONT = envs.SGLANG_K3_FUSE_MOE_FRONT.get()

# MegaMoE SiTU sentinel: the patched deep_gemm mega kernel selects the K3 SiTU
# activation when activation_clamp == 0.03125 (2^-5: exactly representable and
# unused by any legitimate swiglu clamp; the host asserts clamp >= 0 so a
# negative sentinel is impossible). beta=4.0 / linear_beta=25.0 are baked into
# the kernel patch — see p0-wideep/scripts/v2/apply_deepgemm_situ_patch.py.
_K3_MEGA_SITU_SENTINEL_CLAMP = 0.03125


def _k3_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """F.linear / torch.mm with the same TGV dispatch module-level GEMMs get
    through UnquantizedLinearMethod. The fused MoE front and the deferred
    shared down GEMM call torch directly on raw merged weights, so the
    --bf16-gemm-backend cutedsl selection would silently skip them.

    TGV has no out= form; the copy into `out` (a contiguous view of the
    concat-allreduce buffer) is [T, H] at decode sizes and only taken when
    the heuristic already judged the TGV win larger."""
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend

        if get_bf16_gemm_backend().is_cutedsl():
            from sglang.jit_kernel.cutedsl_bf16_gemm import (
                cutedsl_bf16_gemm,
                use_cutedsl_bf16_gemm,
            )

            if use_cutedsl_bf16_gemm(x.shape[0], weight.shape[0], weight.shape[1]):
                y = cutedsl_bf16_gemm(x, weight)
                if out is None:
                    return y
                out.copy_(y)
                return out
    if out is None:
        return torch.nn.functional.linear(x, weight)
    return torch.mm(x, weight.t(), out=out)


_K3_FUSE_KDA_BFA = envs.SGLANG_K3_FUSE_KDA_BFA.get()
# Use the dedicated CUDA decode-GEMV kernel for the skinny KDA projections
# (b+f_a merged, f_b) instead of cublas gemvx/dot dispatch.
_K3_DECODE_GEMV = envs.SGLANG_K3_DECODE_GEMV.get()
# MLA output gate x * sigmoid(g) in one kernel instead of two elementwise.
_K3_FUSE_O_GATE = envs.SGLANG_K3_FUSE_O_GATE.get()
# Fully fused KDA decode step (conv1d + delta rule + gated RMSNorm in one
# kernel, jit_kernel/kda_fused_decode). The model hands the output-norm gate
# to the KDA backend via an attempt-and-verify stash on the attention layer;
# unconsumed stashes fall back to the unfused chain + o_norm here.
_K3_KDA_FUSED_DECODE = envs.SGLANG_KDA_FUSED_DECODE.get()


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


# "fused" = new single-kernel aggregation (attn_residual.py)
# "torch" = pytorch reference (attn_residual.py)
# "legacy" = existing multi-kernel path (below)
_K3_ATTN_RES_MODE = envs.SGLANG_K3_ATTN_RES_MODE.get()

_ATTN_RES_MAX_B = 16
_ATTN_RES_BLOCK_H = 1024


@triton.jit
def _attn_res_scores_kernel(
    prefix_ptr,  # [T, H]
    block_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H] fp32 precombined rmsnorm_weight * proj_weight (set 1)
    scores_ptr,  # [T, MAX_B] fp32 out (set 1)
    cw2_ptr,  # [H] fp32 precombined weights (set 2, optional)
    scores2_ptr,  # [T, MAX_B] fp32 out (set 2, optional)
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_SECOND: tl.constexpr,
):
    """Per-(token, row) score(s): s_j = dot(RMSNorm(v_j), pw). Row NVB is
    prefix_sum. With HAS_SECOND, also computes the score under a second
    (norm, proj) weight set in the same pass — the rows are frozen between
    block writes, so the MLP-side aggregation of the same layer can reuse
    them without re-reading v (halves pass-1 memory traffic per layer)."""
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    if j > NVB:
        return
    sumsq = 0.0
    dotv = 0.0
    dotv2 = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if j < NVB:
            v_raw = tl.load(block_ptr + pid_t * stride_bm + j * stride_bb + offs_h)
        else:
            v_raw = tl.load(prefix_ptr + pid_t * stride_pm + offs_h)
        v = v_raw.to(tl.float32)
        cw = tl.load(cw_ptr + offs_h)
        sumsq += tl.sum(v * v)
        dotv += tl.sum(v * cw)
        if HAS_SECOND:
            cw2 = tl.load(cw2_ptr + offs_h)
            dotv2 += tl.sum(v * cw2)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + j, dotv * rrms)
    if HAS_SECOND:
        tl.store(scores2_ptr + pid_t * stride_sm + j, dotv2 * rrms)


@triton.jit
def _attn_res_combine_norm_kernel(
    prefix_ptr,
    block_ptr,
    scores_ptr,  # [T, MAX_B] fp32
    out_nw_ptr,  # [H] output RMSNorm weight
    out_ptr,  # [T, H] normed aggregate
    NVB,
    out_eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_B: tl.constexpr,
):
    """out[t] = RMSNorm(sum_j softmax(scores)_j * v_j) * out_nw. One CTA per
    token, full-H loop (the fused norm needs the full-row sumsq), fusing the
    input_layernorm / post_attention_layernorm that always follows the
    aggregation."""
    pid_t = tl.program_id(0)

    offs_b = tl.arange(0, MAX_B)
    mask_b = offs_b <= NVB
    scores = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b,
        mask=mask_b,
        other=float("-inf"),
    )
    m = tl.max(scores, axis=0)
    e = tl.where(mask_b, tl.exp(scores - m), 0.0)
    p = e / tl.sum(e, axis=0)

    # Pass 1 over H: aggregate + accumulate sumsq for the fused norm.
    sumsq = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        acc = tl.zeros([BLOCK_H], tl.float32)
        for j in range(0, NVB + 1):
            if j < NVB:
                v_raw = tl.load(block_ptr + pid_t * stride_bm + j * stride_bb + offs_h)
            else:
                v_raw = tl.load(prefix_ptr + pid_t * stride_pm + offs_h)
            p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
            acc += p_j * v_raw.to(tl.float32)
        sumsq += tl.sum(acc * acc)
        # Stash the raw aggregate chunk; renormalized in pass 2.
        tl.store(out_ptr + pid_t * stride_om + offs_h, acc.to(out_ptr.dtype.element_ty))
    rrms = 1.0 / tl.sqrt(sumsq / H + out_eps)
    # Pass 2: scale in place with the norm weight.
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        a = tl.load(out_ptr + pid_t * stride_om + offs_h).to(tl.float32)
        out_nw = tl.load(out_nw_ptr + offs_h).to(tl.float32)
        tl.store(
            out_ptr + pid_t * stride_om + offs_h,
            (a * rrms * out_nw).to(out_ptr.dtype.element_ty),
        )


def _attn_res_cw(proj: ReplicatedLinear, norm: RMSNorm) -> torch.Tensor:
    """Cached fp32 product of the (frozen) rmsnorm weight and score-proj
    weight: dot(RMSNorm(v), pw) == dot(v, nw*pw) / rms(v). fp32 keeps the
    in-kernel math identical to loading both factors separately."""
    cw = getattr(proj, "_attn_res_cw", None)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.view(-1).float()).contiguous()
        proj._attn_res_cw = cw
    return cw


def _attn_res_scores(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
    proj2: Optional[ReplicatedLinear] = None,
    norm2: Optional[RMSNorm] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    T, H = prefix_sum.shape
    scores = torch.empty(
        (T, _ATTN_RES_MAX_B), dtype=torch.float32, device=prefix_sum.device
    )
    has_second = proj2 is not None
    scores2 = (
        torch.empty_like(scores)
        if has_second
        else scores  # dummy, unused when HAS_SECOND=False
    )
    cw = _attn_res_cw(proj, norm)
    _attn_res_scores_kernel[(T, num_valid_blocks + 1)](
        prefix_sum,
        block_residual,
        cw,
        scores,
        _attn_res_cw(proj2, norm2) if has_second else cw,
        scores2,
        num_valid_blocks,
        norm.variance_epsilon,
        prefix_sum.stride(0),
        block_residual.stride(0),
        block_residual.stride(1),
        scores.stride(0),
        H=H,
        BLOCK_H=_ATTN_RES_BLOCK_H,
        HAS_SECOND=has_second,
        num_warps=8,
    )
    return scores, (scores2 if has_second else None)


def _attn_res_prefix_update(
    old: Optional[torch.Tensor],
    delta: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    scores: torch.Tensor,
    idx: int,
) -> torch.Tensor:
    """Residual update + score of the updated prefix row into scores[:, idx].

    The add stays a plain (parallel) elementwise op; only the score is a
    kernel. A fully fused single-CTA-per-token version was measured SLOWER at
    bs=1 (serial 28KB chain on one SM vs parallel add + tiny score kernel).
    """
    new = delta if old is None else old + delta
    T, H = new.shape
    # Reuse the scores kernel for a single row: NVB=0 makes row 0 take the
    # prefix path; aim the output at column `idx` via a pointer offset.
    scores_at_idx = scores[:, idx:]
    cw = _attn_res_cw(proj, norm)
    _attn_res_scores_kernel[(T, 1)](
        new,
        new,  # block_ptr unused when NVB == 0
        cw,
        scores_at_idx,
        cw,  # dummy second set (HAS_SECOND=False)
        scores_at_idx,
        0,
        norm.variance_epsilon,
        new.stride(0),
        0,
        0,
        scores.stride(0),
        H=H,
        BLOCK_H=_ATTN_RES_BLOCK_H,
        HAS_SECOND=False,
        num_warps=8,
    )
    return new


@triton.jit
def _attn_res_combine_kernel(
    prefix_ptr,
    block_ptr,
    scores_ptr,  # [T, MAX_B] fp32
    out_ptr,  # [T, H]
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_B: tl.constexpr,
):
    """out[t, chunk] = sum_j softmax(scores)_j * v_j[chunk]. (T, H-chunks)
    grid: the small-T/decode variant — a fused output norm is impossible here
    (needs full-row sumsq), but the chunked grid keeps SMs busy at bs=1."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    offs_b = tl.arange(0, MAX_B)
    mask_b = offs_b <= NVB
    scores = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b,
        mask=mask_b,
        other=float("-inf"),
    )
    m = tl.max(scores, axis=0)
    e = tl.where(mask_b, tl.exp(scores - m), 0.0)
    p = e / tl.sum(e, axis=0)

    acc = tl.zeros([BLOCK_H], tl.float32)
    for j in range(0, NVB + 1):
        if j < NVB:
            v_raw = tl.load(block_ptr + pid_t * stride_bm + j * stride_bb + offs_h)
        else:
            v_raw = tl.load(prefix_ptr + pid_t * stride_pm + offs_h)
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * v_raw.to(tl.float32)
    tl.store(out_ptr + pid_t * stride_om + offs_h, acc.to(out_ptr.dtype.element_ty))


# Below this token count, the norm-fused (T,)-grid combine cannot fill the
# SMs (one CTA per token); use the chunked combine + separate norm instead.
# Env-overridable for A/B testing.
# A/B on 8xB300 (2026-07-03): the norm-fused (T,)-grid combine + shared
# dual-scores path LOSES at every tested size — bs=1 (single-CTA rows) AND
# bs=64 decode (650 vs 673 tok/s: 64 CTAs underfill 148 SMs), while prefill
# (T>=1024) shows no measurable difference (attn-res is a tiny fraction of
# GEMM-dominated prefill). Disabled by default; kernels kept for retuning.
_ATTN_RES_FUSED_NORM_MIN_T = envs.SGLANG_K3_ATTN_RES_FUSED_MIN_T.get()


def _attn_res_combine_norm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    scores: torch.Tensor,
    num_valid_blocks: int,
    out_norm: RMSNorm,
) -> torch.Tensor:
    T, H = prefix_sum.shape
    out = torch.empty_like(prefix_sum)
    if T < _ATTN_RES_FUSED_NORM_MIN_T:
        _attn_res_combine_kernel[(T, H // _ATTN_RES_BLOCK_H)](
            prefix_sum,
            block_residual,
            scores,
            out,
            num_valid_blocks,
            prefix_sum.stride(0),
            block_residual.stride(0),
            block_residual.stride(1),
            scores.stride(0),
            out.stride(0),
            BLOCK_H=_ATTN_RES_BLOCK_H,
            MAX_B=_ATTN_RES_MAX_B,
            num_warps=4,
        )
        return out_norm(out)
    _attn_res_combine_norm_kernel[(T,)](
        prefix_sum,
        block_residual,
        scores,
        out_norm.weight,
        out,
        num_valid_blocks,
        out_norm.variance_epsilon,
        prefix_sum.stride(0),
        block_residual.stride(0),
        block_residual.stride(1),
        scores.stride(0),
        out.stride(0),
        H=H,
        BLOCK_H=_ATTN_RES_BLOCK_H,
        MAX_B=_ATTN_RES_MAX_B,
        num_warps=8,
    )
    return out


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
    out_norm: RMSNorm,
) -> torch.Tensor:
    """Aggregation + fused following RMSNorm (every aggregation in K3 is
    immediately followed by one). num_valid_blocks must be > 0."""
    scores, _ = _attn_res_scores(
        prefix_sum, block_residual, proj, norm, num_valid_blocks
    )
    return _attn_res_combine_norm(
        prefix_sum, block_residual, scores, num_valid_blocks, out_norm
    )


def _apply_attn_res_torch(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
) -> torch.Tensor:
    """Eager reference implementation (kept for testing/fallback)."""
    if num_valid_blocks <= 0:
        return prefix_sum

    v = torch.cat(
        (block_residual[:, :num_valid_blocks, :], prefix_sum.unsqueeze(1)), dim=1
    )
    k = norm(v)
    probs = (k @ proj.weight.squeeze(0)).softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v).squeeze(1)
    return hidden_states


# ---------------------------------------------------------------------------
# DP attention helpers
#
# K3 cannot use LayerCommunicator: the attn-res aggregation kernels replace
# input_layernorm / post_attention_layernorm, which the communicator expects
# to own. Instead the MLP/MoE modules gather/scatter around their own body:
# attention and the attn-res buffers stay in local (per-DP-rank) token space,
# the MLP/MoE runs on the DP-gathered global batch with plain full-TP
# semantics (its internal all-reduces are unchanged and required — the latent
# reduce must happen in latent space before the norm), and the delayed
# prefix_sum add stays local, applied after the scatter back.
# ---------------------------------------------------------------------------


def _dp_local_buffer_group():
    """Symmetric-memory group for the local DP buffer (mirrors
    CommunicateSummableTensorPairFn._scatter_hidden_states)."""
    parallel = get_parallel()
    if parallel.tp_size == parallel.attn_dp_size:
        return get_tp_group()
    return parallel.attn_tp_group


# ---------------------------------------------------------------------------
# KimiK3MLP — supports both SiLU and SiTU
# ---------------------------------------------------------------------------


class KimiK3MLP(nn.Module):
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


# ---------------------------------------------------------------------------
# KimiK3MoE — with Latent MoE support
# ---------------------------------------------------------------------------


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
    from sglang.jit_kernel import add3

    return add3.add3(a, b, c, prefetch_bc=prefetch_bc)


class KimiK3MoE(nn.Module):
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
        num_experts = config.num_experts
        moe_renormalize = config.moe_renormalize
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        self._dp_attention = is_dp_attention_enabled()

        # Latent MoE
        self.use_latent_moe = config.routed_expert_hidden_size is not None
        self.moe_reduce_mode = _resolve_moe_reduce_mode()
        # Merged front weight ([H, gate_up + E + latent]), built after weight
        # loading by _merge_front_weights().
        self._front_w: Optional[torch.Tensor] = None
        self._front_sizes: Optional[List[int]] = None
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
        # cover the per-rank prefill chunk. SiTU is selected inside the patched
        # mega kernel via a sentinel activation_clamp with the K3 constants
        # baked in (see p0-wideep scripts/v2/apply_deepgemm_situ_patch.py).
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

        # The flashinfer_mxfp4 (trtllm-gen) runner quantizes routed_input with
        # the strided-input JIT group quant (_use_jit_mxfp8_quant in mxfp4.py),
        # so the fused-front split view can be consumed as is; other runners
        # (e.g. marlin) require a dense buffer.
        self._moe_front_needs_contiguous = (
            not get_moe_runner_backend().is_flashinfer_mxfp4()
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

        # Latent MoE projections
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
        if not (
            _K3_FUSE_MOE_FRONT
            and self.use_latent_moe
            and self.shared_experts is not None
        ):
            return
        mods = [
            self.shared_experts.gate_up_proj,
            self.gate,
            self.routed_expert_down_proj,
        ]
        dtypes = {m.weight.dtype for m in mods}
        if len(dtypes) != 1 or dtypes.pop() not in (torch.bfloat16, torch.float16):
            return
        self._front_w, self._front_sizes = _merge_weights_as_views(mods)
        # NOTE: invalidate the cached property
        self.__dict__.pop("_eligible_for_fused_front", None)

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
            _K3_FUSE_MOE_FRONT
            and self.use_latent_moe
            and self.shared_experts is not None
            and self._front_w is not None
            and get_moe_a2a_backend().is_none()
            and self.shared_experts.down_proj.weight.dtype
            in (torch.bfloat16, torch.float16)
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

        from sglang.jit_kernel.dsv4 import mega_moe_pre_dispatch
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
            # sentinel: selects the K3 SiTU branch in the patched mega kernel
            # (beta=4.0 / linear_beta=25.0 baked in); see
            # p0-wideep/scripts/v2/apply_deepgemm_situ_patch.py
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

    def _fi_fused_reduce_norm(self, latent: torch.Tensor) -> torch.Tensor:
        """Fuse the latent all-reduce with the RMSNorm epilogue; fall back to
        plain all-reduce + norm when flashinfer declines the shape."""
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        norm = self.routed_expert_norm
        assert norm is not None  # regime "fi_fused" requires the norm
        zero_res = torch.zeros_like(latent)
        norm_out, _ = flashinfer_allreduce_residual_rmsnorm(
            latent,
            zero_res,
            norm.weight,
            eps=norm.variance_epsilon,
        )
        if norm_out is not None:
            return norm_out
        return self._latent_norm(tensor_model_parallel_all_reduce(latent))

    def _reduce_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Unfused-front latent tail: TP-partial routed sums must be reduced
        in latent space BEFORE the RMSNorm (sum(norm(x_i)) != norm(sum(x_i)))."""
        if not self._routed_needs_reduce:
            return self._latent_norm(latent)
        if self.moe_reduce_mode == "fi_fused" and self.routed_expert_norm is not None:
            return self._fi_fused_reduce_norm(latent)
        return self._latent_norm(tensor_model_parallel_all_reduce(latent))

    def _forward_unfused(
        self, hidden_states: torch.Tensor, *, prefix_sum: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Front section with three separate GEMMs, each reading
        hidden_states: shared-expert MLP, router gate, latent down-proj."""
        # Shared experts on original hidden_states
        shared_output = None
        if self.shared_experts is not None and hidden_states.shape[0] > 0:
            shared_output = self.shared_experts(hidden_states)

        # Gate + TopK (on original hidden_states for correct token count)
        # MoEGate produces fp32 router logits on CUDA (via linear_bf16_fp32
        # or dsv3_router_gemm); non-CUDA falls back to F.linear (bf16).
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)

        if not self.use_latent_moe:
            expert_output = self.experts(hidden_states, topk_output)
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

        routed_input, _ = self.routed_expert_down_proj(hidden_states)
        expert_output = (
            self._forward_mega_experts(routed_input, topk_output)
            if self._use_mega_moe
            else self.experts(routed_input, topk_output)
        )
        latent = self._reduce_latent(expert_output)
        # up_proj is replicated, so the routed output is now fully reduced.
        out, _ = self.routed_expert_up_proj(latent)
        if shared_output is not None:
            # tp1 shared experts (SP-MoE) are complete per-rank; TP-sharded
            # ones need the partial-sum reduction.
            if self.tp_size > 1 and not self._shared_experts_tp1:
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            return _add3(out, shared_output, prefix_sum)
        return out if prefix_sum is None else out + prefix_sum

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
        fused = _k3_bf16_gemm(hidden_states, self._front_w)
        gate_up, router_logits, routed_input = torch.split(
            fused, self._front_sizes, dim=-1
        )
        topk_output = self.topk(hidden_states, router_logits)
        latent_numel = num_tokens * self.moe_hidden_size
        if k3_ar_fusion.enabled():
            # fused MNNVL AR: the buffer lives in the torch symm pool so the
            # NVLS 2shot can reduce it in place (small sizes take the push)
            with k3_ar_fusion.symm_alloc():
                buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)
        else:
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                buf = hidden_states.new_empty(latent_numel + num_tokens * hidden_size)

        latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
        shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
        _k3_bf16_gemm(
            self.shared_experts.act_fn(gate_up),
            self.shared_experts.down_proj.weight,
            out=shared_output,
        )

        # NOTE: Marlin need contiguous input; bs = 1 is already contiguous.
        # flashinfer_mxfp4 keeps the strided view (its quant reads strides).
        if num_tokens > 1 and self._moe_front_needs_contiguous:
            routed_input = routed_input.contiguous()
        with zero_copy_context.set_moe_output(latent):
            expert_output = self.experts(routed_input, topk_output)
        if expert_output.data_ptr() != latent.data_ptr():
            latent.copy_(expert_output)

        # Re-derive both views from the returned tensor: all_reduce is NOT
        # guaranteed in-place (custom-AR / pymscclpp paths return a new
        # tensor; only the symmetric-mempool pynccl path and the K3 fused
        # MNNVL path reduce in place).
        fused_norm = False
        if k3_ar_fusion.enabled():
            if self.fuse_ar_norm:
                fused_norm = True
                norm = self.routed_expert_norm
                assert norm is not None
                k3_ar_fusion.all_reduce_norm(buf, norm.weight, norm.variance_epsilon)
            else:
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
        use_dp = (
            self._dp_attention and forward_batch is not None and not self._ep_a2a
        )
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


# ---------------------------------------------------------------------------
# KimiK3DeltaAttention — KDA with full-rank gate option
# ---------------------------------------------------------------------------


class KimiK3DeltaAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: KimiLinearConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        all_reduce_fusion: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.all_reduce_fusion = all_reduce_fusion
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

        # Decide fusion strategy
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
            # forward wraps o_proj in k3 symm_alloc — leaving this True would
            # nest use_symmetric_memory(attn_tp) inside symm_alloc and misroute
            # the GEMM output away from the k3 pool (rendezvous miss). At the
            # fusion config attn_tp==tp so the fused full-TP reduce is the same
            # group anyway.
            use_dp_attention_reduce=not self.all_reduce_fusion,
            prefix=f"{prefix}.o_proj",
        )
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
        stays 16-byte aligned for vectorized consumers (decode_gemv on f_b).

        Called once from load_weights (after all weights are loaded, before
        cuda graph capture)."""
        if not (_K3_FUSE_KDA_BFA and self.use_full_rank_gate):
            return
        self._bfa_w, sizes = _merge_weights_as_views(
            [self.f_a_proj, self.b_proj], pad_rows_to=8
        )
        self._bfa_fa_size, self._bfa_b_size = sizes

    def _prepare_fused_decode(self) -> None:
        """Static inputs for the fused KDA decode kernel
        (jit_kernel/kda_fused_decode): per-segment transposed fp32 conv
        weights [4, seg], dense fp32 conv bias, fp32 output-norm weight. Stashed on the
        attention layer for the KDA backend; when the shapes do not match
        the compiled kernel the stash stays unset and decode keeps the
        unfused chain. Called once from load_weights (after all weights are
        loaded, before cuda graph capture)."""
        if not _K3_KDA_FUSED_DECODE:
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
            fused_states, _ = self.fused_qkvg_proj(hidden_states)
            qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)
            if self._bfa_w is not None:
                w = self._bfa_w
                n_fa, n_b = self._bfa_fa_size, self._bfa_b_size
                if _K3_DECODE_GEMV:
                    from sglang.jit_kernel.kimi_k3 import kimi_k3_tiny_gemm as gemm

                    bfa = gemm(hidden_states, w)
                    forget_gate = gemm(bfa[..., :n_fa], self.f_b_proj.weight)
                else:
                    bfa = torch.nn.functional.linear(hidden_states, w)
                    forget_gate = self.f_b_proj(bfa[..., :n_fa])[0]
                beta = bfa[..., n_fa : n_fa + n_b]
            else:
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

        # Fused-decode handoff (attempt-and-verify): offer the output-norm
        # gate to the KDA backend so decode can run conv1d + delta rule +
        # gated RMSNorm as one kernel. If the backend leaves the stash
        # unconsumed (env off, shape not covered, non-decode), apply o_norm
        # here as before.
        fused_onorm = (
            self._kda_fused_decode_ready and forward_batch.forward_mode.is_decode()
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
            with k3_ar_fusion.symm_alloc():
                partial, _ = self.o_proj(core_attn_out)
            return partial
        return self.o_proj(core_attn_out)[0]


# ---------------------------------------------------------------------------
# KimiK3MLAAttention — MLA with optional output gate
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(DeepseekV2AttentionMLA):
    """MLA with output gate for K3. Gate is applied in TP-local space before o_proj."""

    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        all_reduce_fusion: bool = False,
        prefix: str = "",
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
        )
        if self.all_reduce_fusion:
            # reduce_results=False was passed through super().__init__ above;
            # the fused all-reduce does the reduce itself and needs the o_proj
            # output in the k3 symm pool, so install the symm-pool wrap and do
            # NOT set use_dp_attention_reduce (its inner attn_tp symm_ctx would
            # nest inside symm_alloc and misroute the GEMM output away from the
            # k3 pool → rendezvous miss). At the fusion config (attn_tp==tp) the
            # fused full-TP reduce is the same group as the attn_tp reduce.
            # The wrap is installed before the output-gate wrap below so the
            # gate multiply stays outside the pool and only the o_proj GEMM
            # output lands in it. NOTE: the captured name must differ from the
            # gate block's `_orig_o_proj_forward` — closures capture the
            # __init__ local by reference, and reusing the name would rebind it
            # to this wrapper (infinite recursion + nested pool enter).
            _symm_inner_o_proj_forward = self.o_proj.forward

            def _symm_o_proj_forward(x, *args, **kwargs):
                with k3_ar_fusion.symm_alloc():
                    return _symm_inner_o_proj_forward(x, *args, **kwargs)

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
            _orig_o_proj_forward = self.o_proj.forward

            def _gated_o_proj_forward(x, *args, **kwargs):
                gate_input = self._gate_hidden_states
                self._gate_hidden_states = None
                if gate_input is not None and not isinstance(x, tuple):
                    gate, _ = self.g_proj(gate_input)
                    from sglang.jit_kernel.kimi_k3 import mla_output_gate

                    if _K3_FUSE_O_GATE and mla_output_gate.covered(x, gate):
                        # One kernel for x * sigmoid(gate); double rounding
                        # matches the unfused pair bit-for-bit.
                        x = mla_output_gate.kimi_k3_mla_output_gate(x, gate)
                    else:
                        x = x * torch.sigmoid(gate)
                return _orig_o_proj_forward(x, *args, **kwargs)

            self.o_proj.forward = _gated_o_proj_forward

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
        return super().forward(
            positions, hidden_states, forward_batch, zero_allocator, **kwargs
        )


# ---------------------------------------------------------------------------
# KimiK3DecoderLayer — with Attention Residual
# ---------------------------------------------------------------------------


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
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

        # The fused all-reduce only serves the attn-res clean path (the
        # production path: attn_res is config-static and the clean/legacy
        # choice is env-static), so the legacy and standard paths stay
        # byte-for-byte untouched and always see a reduced attention output.
        # Mutually exclusive with SP-MoE: both complete o_proj's deferred
        # reduction, but SP-MoE reduce-scatters to a shard whereas the fusion
        # produces the full batch in a symm buffer — an SP-MoE layer builds
        # o_proj in the plain deferred-reduce config (reduce_results forced off
        # below) and reduce-scatters instead.
        attn_tp_size = get_parallel().attn_tp_size
        self.all_reduce_fusion = (
            not self._sp_moe
            and attn_tp_size > 1
            and attn_tp_size == get_tensor_model_parallel_world_size()
            and config.attn_res_block_size is not None
            and _K3_ATTN_RES_MODE != "legacy"
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
            )
        else:
            self.self_attn = KimiK3MLAAttention(
                config=config,
                layer_idx=layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                all_reduce_fusion=self.all_reduce_fusion,
            )

        # MLP / MoE
        if self._is_moe_layer:
            self.mlp = KimiK3MoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
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
            self.block_write_idx = layer_idx // self.attn_res_block_size
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

        # dispatch to impl
        self._forward_attn_residual = (
            self._forward_attn_residual_legacy
            if _K3_ATTN_RES_MODE == "legacy"
            else self._forward_attn_residual_clean
        )

        if self._sp_moe:
            # o_proj emits TP-partial sums; _finish_attn_reduce completes the
            # reduction (RS on the clean attn-res path, AR on fallbacks).
            o_proj = getattr(self.self_attn, "o_proj", None)
            assert o_proj is not None, "SP-MoE requires attention exposing o_proj"
            o_proj.reduce_results = False

    def _finish_attn_reduce(
        self, attn_out: torch.Tensor, allow_scatter: bool
    ) -> tuple[torch.Tensor, int]:
        """Complete o_proj's deferred TP reduction under SP-MoE.

        Returns (reduced tensor, shard row offset); offset is -1 when the
        result covers the full batch (non-SP mode, or fallback all-reduce for
        row counts not divisible by attn_tp)."""
        if not self._sp_moe:
            return attn_out, -1
        group = get_parallel().attn_tp_group
        num_tokens = attn_out.shape[0]
        if allow_scatter and num_tokens > 0 and num_tokens % group.world_size == 0:
            shard = num_tokens // group.world_size
            out = torch.empty(
                (shard, attn_out.shape[1]),
                dtype=attn_out.dtype,
                device=attn_out.device,
            )
            group.reduce_scatter_tensor(out, attn_out)
            return out, group.rank_in_group * shard
        return group.all_reduce(attn_out), -1

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
            attn_out = self._run_self_attn_inner(
                hidden_states[:num_real],
                positions[:num_real],
                forward_batch,
                zero_allocator,
            )
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
        attn_res: Optional[BaseAttnResidual],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attn_res is not None:
            return self._forward_attn_residual(
                positions,
                hidden_states,
                residual,
                attn_res,
                forward_batch,
                zero_allocator,
            )

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
        hidden_states, _ = self._finish_attn_reduce(hidden_states, allow_scatter=False)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        return hidden_states, residual

    def _forward_attn_residual_clean(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        attn_res: BaseAttnResidual,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Between attn-res layers hidden_states carries the previous layer's
        # un-added MLP delta and prefix_sum the prefix it extends (None at
        # stream start / PP entry, where hidden_states already is the head).

        # ---- Aggregation 1: attention side ----
        hidden_states, prefix_sum = attn_res.forward(
            hidden_states,
            prefix_sum,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
            self.input_layernorm,
        )

        # ---- Write snapshot (before attention, using pre-update prefix) ----
        if self.is_block_write_layer:
            attn_res.write(prefix_sum)
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
        if self._sp_moe:
            hidden_states, shard_lo = self._finish_attn_reduce(
                hidden_states, allow_scatter=True
            )
            if shard_lo >= 0:
                rows = slice(shard_lo, shard_lo + hidden_states.shape[0])
                if prefix_sum is not None:
                    prefix_sum = prefix_sum[rows]
        elif self.all_reduce_fusion:
            # Complete the o_proj reduce here, folding the pending prefix add
            # into the fused all-reduce; attn_res then takes the pre-added
            # tensor through its prefix_sum=None branch (same semantics:
            # (normed, new_prefix) with new_prefix = prefix + attn_out).
            hidden_states = k3_ar_fusion.all_reduce(hidden_states, prefix_sum)
            prefix_sum = None

        # ---- Aggregation 2: MLP side (on the shard under SP-MoE) ----
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
        out = self.mlp(hidden_states, prefix_sum=prefix_sum, forward_batch=forward_batch)
        if shard_lo >= 0:
            # reassemble the batch: contiguous shards concatenate in rank order
            group = get_parallel().attn_tp_group
            full = torch.empty(
                (out.shape[0] * group.world_size, out.shape[1]),
                dtype=out.dtype,
                device=out.device,
            )
            group.all_gather_into_tensor(full, out.contiguous())
            out = full
        return out, None

    def _forward_attn_residual_legacy(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        attn_res: BaseAttnResidual,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert prefix_sum is None, "legacy attn-res path does not support extra input"
        block_residual = attn_res.block_residual
        prefix_sum = hidden_states
        nvb = self.prev_valid_blocks
        mlp_valid_blocks = nvb + (1 if self.is_block_write_layer else 0)
        # The dual-score pass (attention-side + MLP-side frozen-row scores in
        # one read) and the norm-fused combine are BANDWIDTH optimizations:
        # at small T (decode) all row-CTAs run in parallel anyway, so sharing
        # saves no wall clock while the extra cw2 loads and the single-CTA
        # prefix-score kernel sit on the critical path. Dispatch by T.
        use_shared_scores = (
            prefix_sum.shape[0] >= _ATTN_RES_FUSED_NORM_MIN_T and nvb > 0
        )

        if use_shared_scores:
            # scores2 row layout matches the MLP-side aggregation:
            #   - non-write layers: rows 0..nvb-1 frozen blocks; index nvb is
            #     overwritten below with the updated-prefix score.
            #   - write layers: the new block IS prefix_sum, whose mlp-score
            #     landed at index nvb == block_write_idx; the updated
            #     prefix's score goes to index nvb+1.
            scores, scores2 = _attn_res_scores(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                nvb,
                proj2=self.mlp_res_proj,
                norm2=self.mlp_res_norm,
            )
        elif nvb > 0:
            scores, _ = _attn_res_scores(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                nvb,
            )

        if nvb > 0:
            hidden_states = _attn_res_combine_norm(
                prefix_sum, block_residual, scores, nvb, self.input_layernorm
            )
        else:
            # Layer 0: aggregation is a passthrough; just norm.
            hidden_states = self.input_layernorm(prefix_sum)

        if self.is_block_write_layer:
            block_residual[:, self.block_write_idx, :].copy_(prefix_sum)

        attn_out = self._run_self_attn(
            hidden_states, positions, forward_batch, zero_allocator
        )
        # legacy path updates full-batch prefix/snapshots post-attention:
        # complete the deferred o_proj reduction as a plain all-reduce
        attn_out, _ = self._finish_attn_reduce(attn_out, allow_scatter=False)

        if use_shared_scores:
            # Residual update + only the updated-prefix score (frozen-row
            # scores were computed above).
            prefix_sum = _attn_res_prefix_update(
                None if self.is_block_write_layer else prefix_sum,
                attn_out,
                self.mlp_res_proj,
                self.mlp_res_norm,
                scores2,
                mlp_valid_blocks,
            )
        else:
            if self.is_block_write_layer:
                prefix_sum = attn_out
            else:
                prefix_sum = prefix_sum + attn_out
            scores2, _ = _attn_res_scores(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
            )

        hidden_states = _attn_res_combine_norm(
            prefix_sum,
            block_residual,
            scores2,
            mlp_valid_blocks,
            self.post_attention_layernorm,
        )
        return (
            self.mlp(hidden_states, prefix_sum=prefix_sum, forward_batch=forward_batch),
            None,
        )


# ---------------------------------------------------------------------------
# KimiK3LinearModel — language model backbone
# ---------------------------------------------------------------------------


class KimiK3LinearModel(nn.Module):
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

        self.alt_stream = torch.cuda.Stream()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: KimiK3DecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
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
            # NOTE: assert to bypass the annoying typing errors
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
                self.config.attn_res_block_size,
                block_residual=residual,
            )
            residual = None

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = self.layers[i](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    attn_res=attn_res,
                    zero_allocator=zero_allocator,
                )
            if (
                self.dspark_layers_to_capture is not None
                and i in self.dspark_layers_to_capture
            ):
                aux_hidden_states.append(
                    self._dspark_capture_stream(i, hidden_states, residual, attn_res)
                )

        if not self.pp_group.is_last_rank:
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
                if _K3_ATTN_RES_MODE == "legacy":
                    hidden_states = _apply_attn_res_fused(
                        hidden_states,
                        attn_res.block_residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                        attn_res_block_num,
                        out_norm=self.norm,
                    )
                else:
                    # ---- Final aggregation (output side, folds delayed add) ----
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
        attn_res: Optional[BaseAttnResidual],
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


# ---------------------------------------------------------------------------
# KimiK3LinearForCausalLM — text-only causal LM
# ---------------------------------------------------------------------------


class KimiK3LinearForCausalLM(nn.Module):
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
                use_attn_tp_group=get_server_args().enable_dp_lm_head,
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
        # cuda graph capture (the lazy path inside _attn_res_cw would bake
        # the multiply into every captured graph replay otherwise).
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if getattr(layer, "use_attn_residuals", False):
                _attn_res_cw(
                    layer.self_attention_res_proj, layer.self_attention_res_norm
                )
                _attn_res_cw(layer.mlp_res_proj, layer.mlp_res_norm)
        if hasattr(self.model, "output_attn_res_proj"):
            _attn_res_cw(
                self.model.output_attn_res_proj, self.model.output_attn_res_norm
            )

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


# ---------------------------------------------------------------------------
# KimiK3ForConditionalGeneration — multimodal wrapper
# ---------------------------------------------------------------------------


class KimiK3ForConditionalGeneration(nn.Module):
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
        self.language_model.post_load_weights()

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    @property
    def lm_head(self):
        return self.language_model.lm_head

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        self.language_model.set_dspark_layers_to_capture(layer_ids)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        device = self.vision_tower.device
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=device, dtype=target_dtype
        )
        image_grid_thws = []
        for item in items:
            grid_thw = item.model_specific_data.get("image_grid_thw")
            if grid_thw is None:
                grid_thw = item.model_specific_data["grid_thws"]
            image_grid_thws.append(grid_thw)
        grid_thws = torch.concat(image_grid_thws, dim=0).to(device)

        if self.use_data_parallel:
            from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model

            image_embeds = run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thws.tolist(),
                rope_type="rope_2d",
            )
            return self.mm_projector(image_embeds)

        image_embeds = self.vision_tower(pixel_values, grid_thws)
        return self.mm_projector(image_embeds)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @property
    def start_layer(self) -> int:
        return self.language_model.model.start_layer

    @property
    def end_layer(self) -> int:
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

        vision_params = dict(self.named_parameters(remove_duplicate=False))

        def stream_language_weights():
            for name, loaded_weight in weights:
                if "vision_tower" in name or "mm_projector" in name:
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

        self.language_model.load_weights(stream_language_weights())
        if self.vision_tower.precompile_fused_rope():
            logger.info("Precompiled dynamic-token fused K3 vision RoPE kernel")
        if self.vision_tower.precompile_attention_backend():
            logger.info("Precompiled Kimi-K3 vision FA4 kernel")

    @property
    def stacked_params_mapping(self):
        return getattr(self.language_model, "stacked_params_mapping", [])

    @property
    def expert_params_mapping(self):
        return getattr(self.language_model, "expert_params_mapping", [])


EntryClass = [KimiK3ForConditionalGeneration]
