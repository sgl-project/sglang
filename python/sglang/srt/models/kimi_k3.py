# Kimi-K3 multimodal model: KimiLinear text backbone + MoonViT3d vision tower.
# Based on kimi_linear.py with K3-specific features:
#   - Attention Residual (attn_res_block_size)
#   - Latent MoE (routed_expert_hidden_size)
#   - SiTU activation
#   - MLA output gate (mla_use_output_gate)
#   - Full-rank KDA gate (use_full_rank_gate)

import os as _os
import sys as _sys
from collections.abc import Iterable
from typing import List, Optional, Tuple

# NPU detection (before any sglang / triton import).
_is_npu = False
try:
    import torch as _detect_torch

    _is_npu = (
        hasattr(_detect_torch, "npu")
        and hasattr(_detect_torch.npu, "is_available")
        and _detect_torch.npu.is_available()
    )
except Exception:
    pass

# NPU env defaults (must be set before sglang.environ reads).
if _is_npu:
    _NPU_ENV_DEFAULTS = {
        "SGLANG_K3_ATTN_RES_MODE": "torch",
        "SGLANG_K3_FUSE_MOE_FRONT": "0",
        "SGLANG_K3_FUSE_KDA_BFA": "0",
        "SGLANG_K3_DECODE_GEMV": "0",
        "SGLANG_K3_TAIL_FUSE": "0",
        "SGLANG_K3_FUSE_O_GATE": "0",
        "SGLANG_K3_ATTN_RES_FUSED_MIN_T": "999999",
        "SGLANG_K3_MOE_REDUCE_MODE": "baseline",
    }
    for _k, _v in _NPU_ENV_DEFAULTS.items():
        _os.environ.setdefault(_k, _v)

# Stub CUDA-only modules before they are imported.
if _is_npu:
    _STUB_MODULES = {
        "triton": None,
        "triton.language": None,
        "sglang.srt.distributed.device_communicators.pynccl_allocator": None,
        "sglang.srt.layers.flashinfer_comm_fusion": None,
        "triton_kernels": None,
        "triton_kernels.numerics_details": None,
        "triton_kernels.numerics_details.mxfp": None,
    }
    for _mod_name in _STUB_MODULES:
        if _mod_name not in _sys.modules:
            try:
                __import__(_mod_name)
            except (ImportError, ModuleNotFoundError):
                _stub = type(_sys)(_mod_name.split(".")[-1])
                _stub.__spec__ = None  # avoid find_spec ValueError
                if _mod_name == "triton":
                    # identity decorator: @triton.jit kernels are defined but not called on NPU.
                    _stub.jit = lambda fn: fn
                _sys.modules[_mod_name] = _stub

import torch

# Stub torch.cuda.Stream for NPU.
if _is_npu:

    class _NoopStream:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    torch.cuda.Stream = _NoopStream  # type: ignore[assignment]
    torch.cuda.current_stream = lambda *a, **kw: _NoopStream()
    torch.cuda.default_stream = lambda *a, **kw: _NoopStream()

import triton
import triton.language as tl
from torch import nn

from sglang.srt.configs.kimi_k3 import KimiK3Config
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.distributed import (
    divide,
    get_pp_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
if not _is_npu:
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
else:
    use_symmetric_memory = None
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.activation import SiluAndMul, SituAndMul
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.dp_attention import is_allocation_symmetric
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
from sglang.srt.layers.moe import single_token_handoff
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
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
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.kimi_k25 import (
    K2VLMultiModalProjector,
    MoonViT3dPretrainedModel,
    mm_projection_auto,
)
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, make_layers
from sglang.srt.utils.common import BumpAllocator, add_prefix, set_weight_attrs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# Latent MoE TP reduction strategy:
#   "baseline" - two separate all-reduces (routed latent, then shared)
#   "concat"   - concat routed latent + shared partials, single all-reduce
#   "fi_fused" - flashinfer fused allreduce+rmsnorm for the latent reduce
# A/B on 8xB300 bs=1 decode (2026-07-03): baseline 35.2 tok/s beats
# concat 33.2 (21.5KB message falls off the one-shot allreduce path into
# two-shot) and fi_fused 33.0 (lamport-buffer kernel + zero-residual
# overhead loses to custom one-shot + tiny rmsnorm at 7KB messages).
# On MULTI-NODE TP the trade flips: single-node custom one-shot is
# unavailable, both reduces go through NCCL, and one 21.5KB collective
# beats a 7KB + 14KB pair (2x4 GB300 MNNVL bs=1: 22.05 -> 21.36 ms ITL).
# Default resolves by topology in KimiK3MoE.__init__; env overrides.
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
_K3_FUSE_KDA_BFA = envs.SGLANG_K3_FUSE_KDA_BFA.get()
# Use the dedicated CUDA decode-GEMV kernel for the skinny KDA projections
# (b+f_a merged, f_b) instead of cublas gemvx/dot dispatch.
_K3_DECODE_GEMV = envs.SGLANG_K3_DECODE_GEMV.get()
# Cross-op tail fusions (decode): residual add fused into the attn_res score
# kernel, and the MoE tail 3-way add (up + shared + residual) in one kernel.
_K3_TAIL_FUSE = envs.SGLANG_K3_TAIL_FUSE.get()
# MLA output gate x * sigmoid(g) in one kernel instead of two elementwise.
_K3_FUSE_O_GATE = envs.SGLANG_K3_FUSE_O_GATE.get()


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
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
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

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        # NPU: ensure contiguous strides for downstream ops.
        if not gate_up.is_contiguous():
            gate_up = gate_up.contiguous()
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x if residual is None else x + residual

    def forward_from_gate_up(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Same as forward() but with the gate_up GEMM already computed
        (e.g. as a slice of a horizontally-fused projection)."""
        if not gate_up.is_contiguous():
            gate_up = gate_up.contiguous()
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# KimiK3MoE — with Latent MoE support
# ---------------------------------------------------------------------------


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

        # Gate
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))

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
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        # Shared experts (operate in original hidden_size space)
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

    def _tail_fuse_applicable(
        self, num_tokens: int, residual: Optional[torch.Tensor]
    ) -> bool:
        """Tail-fusion regime (single-token decode on the concat-reduce path):
        defer the shared down GEMM so it writes its partial sum directly into
        the concat-AR buffer (no torch.cat), and fold up_out + shared +
        residual into one moe_tail_add kernel at the end."""
        return (
            _K3_TAIL_FUSE
            and self.use_latent_moe
            and self.moe_reduce_mode == "concat"
            and num_tokens == 1
            and residual is not None
            and self.tp_size > 1
            and get_moe_a2a_backend().is_none()
        )

    def _forward_front(self, hidden_states: torch.Tensor, tail_fuse: bool):
        """Front section: shared-expert activation, routing, and the latent
        down-projection.

        The fused regime reads hidden_states once through the merged
        [H, gate_up + E + latent] weight; otherwise three separate GEMMs.
        Returns (routed_input, topk_output, shared_output, shared_act);
        shared_act is non-None when the shared down GEMM is deferred to the
        concat buffer (tail fusion), with shared_output aliasing it as the
        usual non-None marker.
        """
        use_fused_front = (
            self._front_w is not None
            and hidden_states.shape[0] > 0
            and self._front_w.dtype == hidden_states.dtype
        )
        if not use_fused_front:
            # Shared experts on original hidden_states
            shared_output = None
            if self.shared_experts is not None and hidden_states.shape[0] > 0:
                shared_output = self.shared_experts(hidden_states)

            # Gate + TopK (on original hidden_states for correct token count)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)

            # Latent MoE: compress after routing, before experts
            if self.use_latent_moe:
                routed_input, _ = self.routed_expert_down_proj(hidden_states)
            else:
                routed_input = hidden_states
            return routed_input, topk_output, shared_output, None

        fused = torch.nn.functional.linear(hidden_states, self._front_w)
        gate_up, router_logits, routed_input = torch.split(
            fused, self._front_sizes, dim=-1
        )
        if hidden_states.shape[0] > 1:
            # Downstream kernels want contiguous inputs; free for T==1.
            gate_up = gate_up.contiguous()
            router_logits = router_logits.contiguous()
            routed_input = routed_input.contiguous()
        shared_act = None
        if tail_fuse and self.shared_experts.down_proj.weight.dtype == torch.bfloat16:
            shared_act = self.shared_experts.act_fn(gate_up)
            shared_output = shared_act  # non-None marker; down GEMM deferred
        else:
            shared_output = self.shared_experts.forward_from_gate_up(gate_up)
        topk_output = self.topk(hidden_states, router_logits)
        return routed_input, topk_output, shared_output, shared_act

    def _concat_reduce(
        self,
        final_hidden_states: torch.Tensor,
        shared_output: torch.Tensor,
        shared_act: Optional[torch.Tensor],
        hidden_size: int,
        buf: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One NCCL call instead of two: all-reduce the routed latent (3584)
        and shared (7168) partial sums as a single [latent | shared] buffer.

        The buffer is allocated in the NCCL symmetric mempool so the
        all-reduce takes the symmetric-kernel one-shot path instead of
        falling back to ring (same trick as RowParallelLinear). When the
        shared down GEMM was deferred (shared_act), it writes its partial sum
        straight into the buffer slice and the routed latent is one small
        copy — replacing down-GEMM-then-cat (one fewer 14KB round trip).
        """
        if buf is None:
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                if shared_act is not None:
                    buf = final_hidden_states.new_empty(
                        final_hidden_states.shape[0],
                        self.moe_hidden_size + hidden_size,
                    )
                else:
                    buf = torch.cat((final_hidden_states, shared_output), dim=-1)
        if shared_act is not None:
            torch.mm(
                shared_act,
                self.shared_experts.down_proj.weight.t(),
                out=buf[..., self.moe_hidden_size :],
            )
            if final_hidden_states.data_ptr() != buf.data_ptr():
                buf[..., : self.moe_hidden_size].copy_(final_hidden_states)
        buf = tensor_model_parallel_all_reduce(buf)
        return (
            buf[..., : self.moe_hidden_size].contiguous(),
            buf[..., self.moe_hidden_size :],
        )

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor = None
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        tail_fuse = self._tail_fuse_applicable(num_tokens, residual)
        routed_input, topk_output, shared_output, shared_act = self._forward_front(
            hidden_states, tail_fuse
        )

        # Deferred-tail regime: preallocate the concat-AR buffer and hand its
        # routed slice to the MoE runner as the final-sum destination, so the
        # top-k sum writes it directly and _concat_reduce drops its memcpy
        # (attempt-and-verify: an unconsumed handoff falls back to the copy).
        buf = None
        if shared_act is not None:
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                buf = routed_input.new_empty(1, self.moe_hidden_size + hidden_size)
            single_token_handoff.stash_output_destination(
                routed_input, buf[:, : self.moe_hidden_size]
            )

        # Experts
        final_hidden_states = self.experts(routed_input, topk_output)
        if buf is not None:
            single_token_handoff.clear_output_destination()

        # With an a2a backend (deepep etc.), the combine step already returns
        # the COMPLETE routed sum; all-reducing again would multiply by tp_size.
        # Only plain-TP partial sums (a2a=none) need the reduction here.
        routed_needs_reduce = self.tp_size > 1 and get_moe_a2a_backend().is_none()

        if self.use_latent_moe:
            # TP-partial routed outputs must be summed in latent space BEFORE
            # non-linear transforms (RMSNorm): sum(RMSNorm(x_i)) != RMSNorm(sum(x_i)).
            did_fused_norm = False
            did_shared_reduce = False
            if (
                routed_needs_reduce
                and shared_output is not None
                and self.moe_reduce_mode == "concat"
            ):
                final_hidden_states, shared_output = self._concat_reduce(
                    final_hidden_states, shared_output, shared_act, hidden_size, buf
                )
                did_shared_reduce = True
            elif routed_needs_reduce:
                if (
                    self.moe_reduce_mode == "fi_fused"
                    and self.routed_expert_norm is not None
                ):
                    # Fuse the latent all-reduce with the RMSNorm epilogue.
                    from sglang.srt.layers.flashinfer_comm_fusion import (
                        flashinfer_allreduce_residual_rmsnorm,
                    )

                    zero_res = torch.zeros_like(final_hidden_states)
                    norm_out, _ = flashinfer_allreduce_residual_rmsnorm(
                        final_hidden_states,
                        zero_res,
                        self.routed_expert_norm.weight,
                        eps=self.routed_expert_norm.variance_epsilon,
                    )
                    if norm_out is not None:
                        final_hidden_states = norm_out
                        did_fused_norm = True
                if not did_fused_norm:
                    final_hidden_states = tensor_model_parallel_all_reduce(
                        final_hidden_states
                    )
            if self.routed_expert_norm is not None and not did_fused_norm:
                final_hidden_states = self.routed_expert_norm(final_hidden_states)
            # up_proj is replicated, so the routed output is now fully reduced.
            final_hidden_states, _ = self.routed_expert_up_proj(final_hidden_states)
            if shared_output is not None:
                if self.tp_size > 1 and not did_shared_reduce:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)
                from sglang.jit_kernel.kimi_k3 import moe_tail_add

                if (
                    _K3_TAIL_FUSE
                    and residual is not None
                    and moe_tail_add.covered(
                        final_hidden_states,
                        shared_output,
                        residual.view(-1, hidden_size),
                    )
                ):
                    # out = bf16(bf16(up + shared) + residual): one kernel, double
                    # rounding matches the unfused add pair bit-for-bit. Applies
                    # at any T (prefill included); shared_output may be a
                    # row-strided slice of the concat-allreduce buffer.
                    final_hidden_states = moe_tail_add.kimi_k3_moe_tail_add(
                        final_hidden_states,
                        shared_output,
                        residual.view(-1, hidden_size),
                    )
                    residual = None  # consumed
                else:
                    final_hidden_states = final_hidden_states + shared_output
        else:
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
            if self.tp_size > 1:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
        if residual is not None:
            final_hidden_states = final_hidden_states + residual.view(-1, hidden_size)
        return final_hidden_states.view(num_tokens, hidden_size)


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
        **kwargs,
    ) -> None:
        super().__init__()
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
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
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

    def forward_qkvbfg_fused(self, hidden_states: torch.Tensor):
        if self.use_full_rank_gate:
            fused_states, _ = self.fused_qkvg_proj(hidden_states)
            qkv, g_proj_states = torch.split(fused_states, self.split_sizes, dim=-1)
            if self._bfa_w is not None:
                w = self._bfa_w
                n_fa, n_b = self._bfa_fa_size, self._bfa_b_size
                # decode_gemv re-reads the weight once per token (CTA-per-output),
                # so its traffic scales linearly with T: measured 2us/launch at
                # T=1 but 41us at T=64 (5ms/step regression). Break-even vs the
                # cublas GEMV pair is around T~8.
                if _K3_DECODE_GEMV and hidden_states.shape[0] <= 8:
                    from sglang.jit_kernel.kimi_k3.decode_gemv import decode_gemv

                    bfa = decode_gemv(hidden_states, w)
                    forget_gate = decode_gemv(bfa[..., :n_fa], self.f_b_proj.weight)
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
            beta = beta.float().sigmoid()
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)

        return self.o_proj(core_attn_out)[0]


# ---------------------------------------------------------------------------
# KimiK3MLAAttention — MLA with optional output gate
# ---------------------------------------------------------------------------


class KimiK3MLAAttention(DeepseekV2AttentionMLA):
    """MLA with output gate for K3. Gate is applied in TP-local space before o_proj."""

    def __init__(self, config, layer_idx, quant_config=None, prefix=""):
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
        )
        # fused_split_qk_norm expects a .bias attribute on the norms.
        for _attr, _norm in [("q_a_layernorm", getattr(self, "q_a_layernorm", None)),
                              ("kv_a_layernorm", getattr(self, "kv_a_layernorm", None))]:
            if _norm is not None and not hasattr(_norm, "bias"):
                _norm.bias = None

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

        # Attention
        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiK3DeltaAttention(
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiK3MLAAttention(
                config=config,
                layer_idx=layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        # MLP / MoE
        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
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

    def _run_self_attn(
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
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_attn_residuals:
            assert residual is not None
            if _K3_ATTN_RES_MODE == "legacy":
                return self.forward_attn_residual(
                    positions, hidden_states, residual, forward_batch, zero_allocator
                )
            return self._forward_attn_residual_clean(
                positions, hidden_states, residual, forward_batch, zero_allocator
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

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_attn_residual_clean(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.layers.attn_residual import (
            attn_res_aggregate,
            attn_res_aggregate_fused_add,
        )

        prefix_sum = hidden_states
        nvb = self.prev_valid_blocks

        # ---- Aggregation 1: attention side ----
        if nvb > 0:
            hidden_states = attn_res_aggregate(
                prefix_sum,
                block_residual,
                nvb,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = self.input_layernorm(prefix_sum)

        # ---- Write snapshot (before attention, using pre-update prefix) ----
        if self.is_block_write_layer:
            block_residual[:, self.block_write_idx, :].copy_(prefix_sum)

        # ---- Attention ----
        attn_out = self._run_self_attn(
            hidden_states, positions, forward_batch, zero_allocator
        )
        # NPU: ensure contiguous strides for downstream reduce/collective ops.
        if attn_out is not None and not attn_out.is_contiguous():
            attn_out = attn_out.contiguous()

        # ---- Residual update + Aggregation 2 (MLP side) ----
        mlp_nvb = nvb + (1 if self.is_block_write_layer else 0)
        if self.is_block_write_layer:
            prefix_sum = attn_out
            hidden_states = attn_res_aggregate(
                prefix_sum,
                block_residual,
                mlp_nvb,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        elif _K3_TAIL_FUSE:
            # Residual add fused into the score kernel (bit-identical rounding);
            # the summed prefix is materialized by the kernel for combine and
            # for the accumulate below.
            hidden_states, prefix_sum = attn_res_aggregate_fused_add(
                prefix_sum,
                attn_out,
                block_residual,
                mlp_nvb,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum = prefix_sum + attn_out
            hidden_states = attn_res_aggregate(
                prefix_sum,
                block_residual,
                mlp_nvb,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )

        # ---- MLP + accumulate ----
        prefix_sum = self.mlp(hidden_states, residual=prefix_sum)
        return prefix_sum, block_residual

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        hidden_states = self.mlp(hidden_states)
        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual


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

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
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

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=device,
        )

        use_attn_res = self.config.attn_res_block_size is not None

        if use_attn_res:
            attn_res_block_num = _cdiv(self.end_layer, self.config.attn_res_block_size)
            block_residual = hidden_states.new_empty(
                hidden_states.size(0), attn_res_block_num, hidden_states.size(1)
            )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            residual = block_residual

        for i in range(self.start_layer, self.end_layer):
            ctx = get_global_expert_distribution_recorder().with_current_layer(i)
            with ctx:
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if use_attn_res:
                if _K3_ATTN_RES_MODE == "legacy":
                    hidden_states = _apply_attn_res_fused(
                        hidden_states,
                        residual,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                        attn_res_block_num,
                        out_norm=self.norm,
                    )
                else:
                    from sglang.srt.layers.attn_residual import attn_res_aggregate

                    hidden_states = attn_res_aggregate(
                        hidden_states,
                        residual,
                        attn_res_block_num,
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                        self.norm,
                    )
            else:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


# ---------------------------------------------------------------------------
# KimiK3LinearForCausalLM — text-only causal LM
# ---------------------------------------------------------------------------


def _coerce_quant_weight_dtype(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    """Match a checkpoint weight's dtype to its target parameter's dtype.

    INT4 MoE packed weights are stored as uint8 in the checkpoint but allocated
    as int32 (8 int4 per int32); view the buffer as int32 to preserve the packed
    bits. Floating-point scales are cast to the param dtype. No-op otherwise.
    """
    if loaded_weight.dtype == param.dtype:
        return loaded_weight
    if param.dtype == torch.int32 and loaded_weight.dtype == torch.uint8:
        return loaded_weight.view(torch.int32)
    if loaded_weight.dtype.is_floating_point and param.dtype.is_floating_point:
        return loaded_weight.to(param.dtype)
    return loaded_weight


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
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)

    def get_input_embeddings(self):
        return self.model.embed_tokens

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
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

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

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}

            # compressed-tensors MXFP4 stores as weight_packed; Mxfp4MoEMethod uses weight
            if "weight_packed" in name:
                name = name.replace("weight_packed", "weight")

            # Reduced-layer support: skip weights for layer indices >= num_hidden_layers.
            if ".layers." in name:
                _layer_id = name.split(".layers.", 1)[1].split(".", 1)[0]
                if _layer_id.isdigit() and int(_layer_id) >= self.config.num_hidden_layers:
                    continue

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
                loaded_weight = _coerce_quant_weight_dtype(param, loaded_weight)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for idx, (param_name, weight_name, expert_id, shard_id) in enumerate(
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Layer reduced: checkpoint has weights for absent layers; skip.
                        break
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    loaded_weight = _coerce_quant_weight_dtype(param, loaded_weight)
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
                    loaded_weight = _coerce_quant_weight_dtype(param, loaded_weight)
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        # Post-load: absorb kv_b_proj into w_kc and w_vc for MLA layers
        for layer_id in self.config.full_attention_layer_ids:
            layer = self.model.layers[layer_id]
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


# ---------------------------------------------------------------------------
# KimiK3ForConditionalGeneration — multimodal wrapper
# ---------------------------------------------------------------------------


class KimiK3ForConditionalGeneration(nn.Module):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
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

        # Ensure vision_config has aliases needed by MoonViT3dPretrainedModel.
        # HF's trust_remote_code config uses vt_* prefixed names; the K25 vision
        # code expects unprefixed aliases plus video_attn_type.
        vc = config.vision_config
        _vc_aliases = {
            "hidden_size": ("vt_hidden_size", 1024),
            "num_attention_heads": ("vt_num_attention_heads", 12),
            "num_hidden_layers": ("vt_num_hidden_layers", 27),
            "intermediate_size": ("vt_intermediate_size", 4096),
            "video_attn_type": (None, "spatial_temporal"),
        }
        for attr, (src, default) in _vc_aliases.items():
            if not hasattr(vc, attr):
                setattr(vc, attr, getattr(vc, src, default) if src else default)

        # K3 vision has 12 heads which may not divide TP size; force DP for encoder
        self.use_data_parallel = True

        self.vision_tower = MoonViT3dPretrainedModel(
            config.vision_config,
            use_data_parallel=self.use_data_parallel,
            quant_config=None,
            prefix="vision_tower",
        )
        self.mm_projector = K2VLMultiModalProjector(config.vision_config)

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

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

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
            image_features = self.mm_projector(image_embeds)
            return image_features

        image_embeds = self.vision_tower(pixel_values, grid_thws)
        proj_out = mm_projection_auto(self.mm_projector, image_embeds)
        return torch.cat(proj_out, dim=0)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @property
    def start_layer(self) -> int:
        return self.language_model.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.language_model.model.end_layer

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
                    vname = (
                        name.replace(r"wqkv.", r"attn.qkv_proj.")
                        .replace(r"wo.", r"attn.proj.")
                        .replace("mm_projector.proj.0", "mm_projector.linear_1")
                        .replace("mm_projector.proj.2", "mm_projector.linear_2")
                    )
                    if vname not in vision_params:
                        continue
                    param = vision_params[vname]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    continue
                yield name.replace("language_model.", ""), loaded_weight

        self.language_model.load_weights(stream_language_weights())

    @property
    def stacked_params_mapping(self):
        return getattr(self.language_model, "stacked_params_mapping", [])

    @property
    def expert_params_mapping(self):
        return getattr(self.language_model, "expert_params_mapping", [])


# Register in the transformers namespace for resolve_transformers_arch.
import transformers as _transformers
_transformers.KimiK3ForConditionalGeneration = KimiK3ForConditionalGeneration

EntryClass = [KimiK3ForConditionalGeneration]
