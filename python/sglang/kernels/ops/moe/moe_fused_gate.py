from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.moe import moe_route_radix

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
    "softmax": 2,
}


@cache_once
def _jit_moe_fused_gate_module() -> Module:
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")],
    )


@cache_once
def can_use_moe_fused_gate() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_moe_fused_gate_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT MoE fused gate kernel: {e}")
        return False


def moe_fused_gate_jit(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert scoring_func_int is not None, (
        f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    )

    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"

    num_rows, _ = input.shape
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    module = _jit_moe_fused_gate_module()
    module.moe_fused_gate(
        input,
        bias,
        output,
        indices,
        topk,
        scoring_func_int,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    return output, indices


@triton.jit
def _router_triton_kernel(
    scores_ptr,  # [M, N (+ SHARED_SINK)] fp32, GEMM output (raw logits)
    bias_ptr,  # [N]    fp32/fp16/bf16 (upcast to fp32 on load)
    out_weights_ptr,  # [M, K] fp32 (LOGSIGMOID_SINK: [M, K_ROUTED])
    out_indices_ptr,  # [M, K] int32 (LOGSIGMOID_SINK: [M, K_ROUTED])
    out_shared_ptr,  # [M, SHARED_SINK] fp32 (LOGSIGMOID_SINK only)
    out_packed_ptr,  # [M, K_ROUTED] int32 (LOGSIGMOID_SINK + RETURN_PACKED only)
    global_scale_ptr,  # [1] fp32 (HAS_GLOBAL_SCALE only)
    M,
    routed_scaling_factor,
    moe_softcapping,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts / sink columns)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts - SHARED_SINK
    BLOCK_M: tl.constexpr,  # rows processed per program (row tiling)
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    N_GROUP: tl.constexpr,  # expert groups (1 = ungrouped)
    TOPK_GROUP: tl.constexpr,  # groups kept per token (grouped routing)
    EXPERTS_PER_GROUP: tl.constexpr,  # N // N_GROUP
    BLOCK_G: tl.constexpr,  # >= N_GROUP, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    HAS_BIAS: tl.constexpr,
    USE_PDL: tl.constexpr,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    # Epilogue parameterization. EPILOGUE 0 (SUM_NORM) is the historical
    # behaviour and the only one any existing call site reaches. EPILOGUE 1
    # (LOGSIGMOID_SINK) is Inkling's gate: the last SHARED_SINK columns of the
    # row are shared-expert sink logits that take no bias and never enter the
    # top-k, but do join the normalizer, and the quantity normalized is the
    # winner's RAW logit rather than its activated score.
    EPILOGUE: tl.constexpr = 0,
    SHARED_SINK: tl.constexpr = 0,
    HAS_GLOBAL_SCALE: tl.constexpr = False,
    RETURN_PACKED: tl.constexpr = False,
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Prefetch a real bias before the PDL wait. Plain softmax routing has no
    # bias, so keep the zero value in registers rather than materializing and
    # clearing a device tensor for every routing call.
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros([BLOCK_N], dtype=tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(
        tl.float32
    )  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = tl.sigmoid(scores)
        biased = activated + bias[None, :]
    elif SCORING_FUNC == 1:
        # sqrt(softplus(x)) = sqrt(log1p(exp(x))); guard against overflow when x is large.
        sp = tl.where(scores > 20.0, scores, tl.log(1.0 + tl.exp(scores)))
        activated = tl.sqrt(sp)
        biased = activated + bias[None, :]
    else:
        # softmax over the row: weight is the softmax probability (bias kept), with
        # optional tanh softcapping. Ranking by the (softcapped, biased) logit is
        # monotonic with the softmax prob, so the topk loop below ranks on `biased`.
        logit = scores
        if HAS_SOFTCAP:
            # tanh(z) = 2*sigmoid(2z) - 1 (avoids relying on tl.math.tanh availability).
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + bias[None, :]
        biased = tl.where(mask_n[None, :], biased, -float("inf"))
        row_max = tl.max(biased, axis=1)[:, None]  # [BLOCK_M, 1]
        exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
        row_sum = tl.sum(exp_row, axis=1)[:, None]  # [BLOCK_M, 1]
        activated = exp_row / row_sum

    biased = tl.where(mask_n[None, :], biased, -float("inf"))  # [BLOCK_M, BLOCK_N]

    # Map NaN -> a finite floor
    biased = tl.where(biased == biased, biased, -1e30)  # [BLOCK_M, BLOCK_N]

    # Grouped routing (DeepSeek-V3 noaux_tc): per-group score = sum of the top-2
    # biased values; keep TOPK_GROUP groups (lowest group id wins ties); mask the
    # experts of dropped groups to -inf before the top-k below. Weight is still the
    # bias-free `activated`. Constexpr N_GROUP <= 1 skips this entirely (ungrouped).
    if N_GROUP > 1:
        offs_g = tl.arange(0, BLOCK_G)  # [BLOCK_G]
        group_of_n = offs_n // EXPERTS_PER_GROUP  # [BLOCK_N]
        group_score = tl.full([BLOCK_M, BLOCK_G], -float("inf"), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (group_of_n[None, :] == g) & mask_n[None, :]
            vals = tl.where(in_g, biased, -float("inf"))
            top1 = tl.max(vals, axis=1)[:, None]  # [BLOCK_M, 1]
            vals2 = tl.where(vals >= top1, -float("inf"), vals)
            top2 = tl.max(vals2, axis=1)[:, None]  # [BLOCK_M, 1]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

        gcur = group_score
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]  # [BLOCK_M, 1]
            glane = tl.where(gcur == gmax, offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]  # [BLOCK_M, 1] lowest-id on ties
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            gcur = tl.where(offs_g[None, :] == win_g, -float("inf"), gcur)
        biased = tl.where(keep > 0.0, biased, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    # The quantity carried out of the top-k loop is not the quantity it ranks on.
    # SUM_NORM normalizes the activated score, so it carries `activated`.
    # LOGSIGMOID_SINK normalizes the RAW logit, so it carries `scores` -- ranking
    # still happens on `biased`. Two quantities per column is what makes a plain
    # "top-k and keep the winning score" kernel unusable for Inkling.
    if EPILOGUE == 1:
        carried = scores
    else:
        carried = activated

    cur = biased  # [BLOCK_M, BLOCK_N]
    for k in tl.static_range(K_ROUTED):
        max_val = tl.max(cur, axis=1)[:, None]  # [BLOCK_M, 1]
        is_max = cur == max_val
        lane_id = tl.where(is_max, offs_n[None, :], N + 1)  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)  # [BLOCK_M, 1]
        win_carried = tl.sum(
            tl.where(offs_n[None, :] == win_lane, carried, 0.0), axis=1
        )[:, None]  # [BLOCK_M, 1]
        slot = offs_k[None, :] == k  # [1, BLOCK_K]
        selected_vals = tl.where(slot, win_carried, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        cur = tl.where(offs_n[None, :] == win_lane, -float("inf"), cur)

    if EPILOGUE == 1:
        # Sink columns N .. N + SHARED_SINK of the same row occupy slots
        # K_ROUTED .. K. They take no bias (bias is [N], not [N + SHARED_SINK])
        # and never enter the top-k, but they do join the normalizer.
        offs_s = offs_k - K_ROUTED  # [BLOCK_K]
        mask_s = (offs_s >= 0) & (offs_s < SHARED_SINK)
        sink_raw = tl.load(
            scores_ptr
            + offs_m[:, None] * stride_sm
            + (N + offs_s[None, :]) * stride_sn,
            mask=mask_m[:, None] & mask_s[None, :],
            other=0.0,
        ).to(tl.float32)
        active = tl.where(mask_s[None, :], sink_raw, selected_vals)

        # exp(logsigmoid(x) - logsumexp(logsigmoid(x))) over the K active slots.
        # Algebraically sigmoid(x) / sum(sigmoid(x)), but fp32 sigmoid underflows
        # to 0 below x ~ -104, so the explicit form divides 0/0 and returns NaN
        # once every active logit is that negative.
        # log(sigmoid(x)) = min(x, 0) - log1p(exp(-|x|)), exact for large |x|.
        lp = tl.minimum(active, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(active)))
        lp = tl.where(mask_k_total[None, :], lp, float("-inf"))
        e = tl.where(
            mask_k_total[None, :], tl.exp(lp - tl.max(lp, axis=1)[:, None]), 0.0
        )
        selected_vals = e / tl.sum(e, axis=1, keep_dims=True)
        # route_scale, then the model's scalar global_scale. RENORMALIZE /
        # APPLY_SCALE do not apply: this epilogue always renormalizes and always
        # scales (the sum over routed + sink weights is route_scale*global_scale).
        selected_vals = selected_vals * routed_scaling_factor
        if HAS_GLOBAL_SCALE:
            selected_vals = selected_vals * tl.load(global_scale_ptr)

        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

        # Routed weights/ids are [M, K_ROUTED]; the sink gammas get their own
        # [M, SHARED_SINK] output rather than trailing slots of the routed one.
        store_mask = mask_m[:, None] & mask_k_routed[None, :]
        if RETURN_PACKED:
            weight_bits = (
                selected_vals.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
            )
            tl.store(
                out_packed_ptr + offs_m[:, None] * K_ROUTED + offs_k[None, :],
                (selected_idx << 16) | weight_bits,
                mask=store_mask,
            )
        else:
            tl.store(
                out_weights_ptr
                + offs_m[:, None] * stride_wm
                + offs_k[None, :] * stride_wk,
                selected_vals,
                mask=store_mask,
            )
            tl.store(
                out_indices_ptr
                + offs_m[:, None] * stride_im
                + offs_k[None, :] * stride_ik,
                selected_idx,
                mask=store_mask,
            )
        tl.store(
            out_shared_ptr + offs_m[:, None] * SHARED_SINK + offs_s[None, :],
            selected_vals,
            mask=mask_m[:, None] & mask_s[None, :],
        )
    else:
        routed_sum = tl.sum(
            tl.where(mask_k_routed[None, :], selected_vals, 0.0), axis=1
        )[:, None]  # [BLOCK_M, 1]

        # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
        # id = num_experts + (slot - K_ROUTED).
        if K_ROUTED < K:
            is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
            shared_weight = routed_sum / routed_scaling_factor  # [BLOCK_M, 1]
            shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]  # [1, BLOCK_K]
            selected_vals = tl.where(is_shared, shared_weight, selected_vals)
            selected_idx = tl.where(is_shared, shared_idx, selected_idx)

        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

        if RENORMALIZE:
            norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)  # [BLOCK_M, 1]
            selected_vals = selected_vals / norm
        if APPLY_SCALE:
            selected_vals = selected_vals * routed_scaling_factor

        out_w_ptr = (
            out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
        )
        out_i_ptr = (
            out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        )
        store_mask = mask_m[:, None] & mask_k_total[None, :]
        tl.store(out_w_ptr, selected_vals, mask=store_mask)
        tl.store(out_i_ptr, selected_idx, mask=store_mask)


@debug_kernel_api
def moe_fused_gate(
    scores: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
    shared_sink: int = 0,
    global_scale: torch.Tensor | None = None,
    return_packed: bool = False,
) -> (
    Tuple[torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]
):
    """Triton fused router: scoring + bias + topk + (optional) renorm/scale.

    Mirrors the semantics of :func:`moe_fused_gate_jit` (the CUDA JIT kernel).
    With ``num_expert_group > 1`` it performs DeepSeek-V3 grouped routing
    (per-group top-2-sum group scores, keep ``topk_group`` groups, then top-k
    within). The first argument is named ``scores`` (raw GEMM logits) to match
    the existing call sites.

    ``shared_sink > 0`` selects the Inkling gate epilogue: ``scores`` is
    ``[M, num_experts + shared_sink]``, the trailing ``shared_sink`` columns are
    shared-expert sink logits that take no bias and never enter the top-k but do
    join the normalizer, the normalized quantity is the winner's RAW logit rather
    than its activated score, and the weights are renormalized in log space (see
    :mod:`sglang.kernels.ops.moe.sigmoid_gate_topk_renorm`). It returns a 4-tuple
    ``(routed_weights, topk_indices, shared_weights, packed_topk)``.

    At the default ``shared_sink == 0`` every code path, launch geometry and
    output tensor is unchanged and the return value is the historical 2-tuple.
    """
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert scoring_func_int is not None, (
        f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    )
    assert scores.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "scores must be float32/float16/bfloat16"
    assert scores.ndim == 2, "scores must be 2D"
    assert shared_sink >= 0, f"{shared_sink=} must be non-negative"
    if bias is None:
        assert scoring_func.lower() == "softmax", (
            "bias is required for non-softmax routing"
        )
        assert shared_sink == 0, "shared_sink requires a bias"
    else:
        # The kernel loads the bias and upcasts it to fp32 in-register (see
        # _router_triton_kernel), so a non-fp32 bias (DeepSeek-V4 stores the
        # correction bias in bf16) needs no host-side cast/copy.
        assert bias.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ), "bias must be float32/float16/bfloat16"
        assert bias.ndim == 1, "bias must be 1D"
        assert scores.size(1) == bias.size(0) + shared_sink, (
            "scores must be [M, num_experts + shared_sink] and bias [num_experts]"
        )
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    if shared_sink > 0:
        # The sink columns are read directly off the score row, so only
        # column-stride-1 is required -- Inkling's gate logits are a [T, 258]
        # slice of a padded [T, 264] fp32 GEMM output, i.e. not contiguous but
        # column-contiguous, and need no copy.
        assert scores.stride(1) == 1, f"{scores.stride()=} needs column stride 1"
        assert num_fused_shared_experts == 0, "shared_sink and fused shared experts"
        assert num_expert_group <= 1, "shared_sink does not support grouped routing"
        assert scoring_func.lower() == "sigmoid", "shared_sink is sigmoid-gate only"
        assert moe_softcapping == 0.0, "shared_sink does not support softcapping"
    else:
        assert global_scale is None, "global_scale requires shared_sink > 0"
        assert not return_packed, "return_packed requires shared_sink > 0"
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    # K3 radix-select fast path: native-CUDA radix-select replaces the 16
    # dependent argmax rounds (single CTA per token; ids bit-identical to this
    # triton kernel incl. ties).
    # The radix kernel keeps keys register-resident and returns winners in
    # expert-id order (skipping the biased-descending sort; downstream MoE
    # kernels are order-insensitive). It is 3.1-3.5x faster than the Triton
    # kernel at [1..8192, 896] top-16 on B200.
    if (
        scoring_func.lower() == "sigmoid"
        and num_fused_shared_experts == 0
        and num_expert_group <= 1
        and moe_softcapping == 0.0
        and shared_sink == 0
    ):
        radix_args = (
            scores,
            bias,
            topk,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        if moe_route_radix.covered(scores, bias, topk):
            return moe_route_radix.route_radix(*radix_args, sorted=False)

    M = scores.shape[0]
    # bias is [N] while the score row is [N + shared_sink], and bias may be None
    # for plain softmax routing, so derive N from the row rather than the bias.
    N = scores.shape[1] - shared_sink
    if shared_sink > 0:
        # Sink columns take slots K_routed .. K of the normalizer, so they widen
        # K without widening the routed outputs.
        K = topk + shared_sink
        K_routed = topk
    else:
        K = topk
        K_routed = topk - num_fused_shared_experts
    if num_expert_group > 1:
        assert N % num_expert_group == 0, "num_experts must be divisible by group count"
        assert 1 <= topk_group <= num_expert_group, "invalid topk_group"
    experts_per_group = N // num_expert_group
    BLOCK_G = triton.next_power_of_2(num_expert_group)

    shared_weights = packed_topk = None
    if shared_sink > 0:
        shared_weights = torch.empty(
            (M, shared_sink), dtype=torch.float32, device=scores.device
        )
        # In packed mode the kernel writes only the packed tensor; the unused
        # pointer args still need a valid (never-stored) address.
        if return_packed:
            packed_topk = torch.empty(
                (M, K_routed), dtype=torch.int32, device=scores.device
            )
            weights = indices = None
            w_arg = i_arg = p_arg = packed_topk
        else:
            weights = torch.empty(
                (M, K_routed), dtype=torch.float32, device=scores.device
            )
            indices = torch.empty(
                (M, K_routed), dtype=torch.int32, device=scores.device
            )
            w_arg, i_arg, p_arg = weights, indices, indices
        s_arg = shared_weights
    else:
        weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
        indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)
        w_arg, i_arg, p_arg, s_arg = weights, indices, indices, indices

    BLOCK_N = triton.next_power_of_2(N)  # 256 -> 256, 384 -> 512
    BLOCK_K = triton.next_power_of_2(K)  # 6 -> 8, 8 -> 8
    # Single warp per program keeps the per-row top-k reductions on cheap warp
    # shuffles; pack a few rows per program only when N is small so tiny launches
    # stay occupancy-bound. Swept on H100/B200: this beats the AOT kernels across
    # shapes, whereas larger tiles / more warps regress (register pressure).
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    # For wide rows (e.g. Kimi K3: 896 experts, BLOCK_N 1024) the K sequential
    # argmax passes dominate and benefit from more warps despite the
    # cross-warp reduction cost.
    num_warps = 1 if BLOCK_N <= 512 else 4
    grid = (triton.cdiv(M, BLOCK_M),)
    use_pdl = is_arch_support_pdl()
    extra = {"launch_pdl": True} if use_pdl else {}
    _router_triton_kernel[grid](
        scores,
        bias if bias is not None else scores,
        w_arg,
        i_arg,
        s_arg,
        p_arg,
        global_scale if global_scale is not None else scores,
        M,
        float(routed_scaling_factor),
        float(moe_softcapping),
        N=N,
        K=K,
        K_ROUTED=K_routed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        N_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=experts_per_group,
        BLOCK_G=BLOCK_G,
        SCORING_FUNC=scoring_func_int,
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        HAS_BIAS=bias is not None,
        USE_PDL=use_pdl,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=w_arg.stride(0),
        stride_wk=w_arg.stride(1),
        stride_im=i_arg.stride(0),
        stride_ik=i_arg.stride(1),
        EPILOGUE=1 if shared_sink > 0 else 0,
        SHARED_SINK=shared_sink,
        HAS_GLOBAL_SCALE=global_scale is not None,
        RETURN_PACKED=bool(return_packed),
        num_warps=num_warps,
        **extra,
    )
    if shared_sink > 0:
        return weights, indices, shared_weights, packed_topk
    return weights, indices
