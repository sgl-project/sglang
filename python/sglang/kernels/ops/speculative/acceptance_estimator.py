"""Kernels for the online draft acceptance estimator.

``max_log_odds`` turns draft scores into the estimator's feature, the log-odds
of the most likely draft token, and optionally the logistic prediction from it.
``observe_acceptance`` folds verified drafts into the per-round Newton (IRLS)
statistics and, every ``refit_interval`` calls, refits the logistic
coefficients in place. Both read their state from device tensors only, so they
never synchronize with the host and can be captured in a CUDA graph.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Columns of the per-position statistics: the X^T W X arrowhead entries a, b, c,
# the X^T r entries g0, g1, and the observation count n.
NUM_ACCEPTANCE_STATS = 6
# A refit halves a slope step that would make the slope non-positive at most
# this many times before dropping it.
_MAX_SLOPE_HALVINGS = 8


@triton.jit
def _max_log_odds_kernel(
    scores_ptr,
    stride_req,
    stride_pos,
    stride_cand,
    temperatures_ptr,
    temperatures_stride,
    slope_ptr,
    intercepts_ptr,
    out_ptr,
    probs_ptr,
    num_positions,
    num_candidates,
    MAX_LOG_ODDS: tl.constexpr,
    WITH_PROBS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // num_positions
    pos = row % num_positions

    # Rows with a non-positive temperature are greedy; score them at T = 1.
    temperature = tl.load(temperatures_ptr + req * temperatures_stride).to(tl.float32)
    temperature = tl.where(temperature > 0.0, temperature, 1.0)

    row_ptr = scores_ptr + req.to(tl.int64) * stride_req + pos.to(tl.int64) * stride_pos

    # logit(q_max) = log(q_max / (1 - q_max)) = -log(sum_{z != argmax} e^(z - m)),
    # with m the max score. Track the running max and the sum over every other
    # candidate directly, so confident rows keep their precision instead of
    # cancelling in (sum_z e^(z - m)) - 1.
    run_max = float("-inf")
    run_rest = 0.0
    for start in range(0, num_candidates, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_candidates
        s = tl.load(row_ptr + offs * stride_cand, mask=mask, other=float("-inf")).to(
            tl.float32
        )
        s = s / temperature
        blk_max = tl.max(s, axis=0)
        # Exclude exactly one instance of the block max; ties stay in the rest.
        blk_arg = tl.min(tl.where(s == blk_max, offs, num_candidates), axis=0)
        e = tl.where((s > float("-inf")) & (offs != blk_arg), tl.exp(s - blk_max), 0.0)
        blk_rest = tl.sum(e, axis=0)
        # Merge the block into the running pair; the smaller max joins the rest.
        rest_new_max = blk_rest + (run_rest + 1.0) * tl.exp(run_max - blk_max)
        rest_old_max = run_rest + tl.where(
            blk_max > float("-inf"), (blk_rest + 1.0) * tl.exp(blk_max - run_max), 0.0
        )
        run_rest = tl.where(blk_max > run_max, rest_new_max, rest_old_max)
        run_max = tl.maximum(run_max, blk_max)

    feature = tl.where(run_rest > 0.0, -tl.log(run_rest), MAX_LOG_ODDS)
    feature = tl.minimum(tl.maximum(feature, -MAX_LOG_ODDS), MAX_LOG_ODDS)
    tl.store(out_ptr + row, feature)
    if WITH_PROBS:
        bias = tl.load(intercepts_ptr + pos)
        tl.store(probs_ptr + row, tl.sigmoid(tl.load(slope_ptr) * feature + bias))


def max_log_odds(
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    clamp: float,
    slope: Optional[torch.Tensor] = None,
    intercepts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Log-odds of the most likely candidate of each temperature-scaled row.

    scores: [bs, num_positions, num_candidates] logits; temperatures: one per
    request (any shape with bs elements). Returns the float32 log-odds
    [bs, num_positions], clamped to +-clamp, and, when the float32 logistic
    coefficients slope [1] and intercepts [num_positions] are given, the
    probabilities sigmoid(slope * x + intercepts[k]) from the same launch
    (otherwise None).

    One program handles one row and walks its candidates in blocks, so the
    kernel is sized for candidate lists; a full-vocabulary row runs serially.
    """
    bs, num_positions, num_candidates = scores.shape
    temperatures = temperatures.reshape(-1)
    assert temperatures.numel() == bs, (temperatures.shape, scores.shape)
    with_probs = slope is not None
    assert with_probs == (intercepts is not None)
    if with_probs:
        assert slope.numel() == 1 and intercepts.shape == (num_positions,)
        assert slope.dtype == intercepts.dtype == torch.float32
    out = torch.empty((bs, num_positions), dtype=torch.float32, device=scores.device)
    probs = torch.empty_like(out) if with_probs else None
    if out.numel() == 0:
        return out, probs
    block_n = min(triton.next_power_of_2(num_candidates), 4096)
    _max_log_odds_kernel[(bs * num_positions,)](
        scores,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        temperatures,
        temperatures.stride(0),
        slope if with_probs else out,
        intercepts if with_probs else out,
        out,
        probs if with_probs else out,
        num_positions,
        num_candidates,
        MAX_LOG_ODDS=clamp,
        WITH_PROBS=with_probs,
        BLOCK_N=block_n,
        num_warps=1 if block_n <= 256 else 4,
    )
    return out, probs


@triton.jit(do_not_specialize=["num_reqs", "refit_interval"])
def _observe_acceptance_kernel(
    x_ptr,
    x_stride_req,
    x_stride_pos,
    num_verified_ptr,
    num_accepted_ptr,
    slope_ptr,
    intercepts_ptr,
    stats_ptr,
    num_calls_ptr,
    num_reqs,
    refit_interval,
    NUM_POSITIONS: tl.constexpr,
    L2: tl.constexpr,
    DAMPING: tl.constexpr,
    MAX_SLOPE_HALVINGS: tl.constexpr,
    NUM_STATS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    k_mask = k < NUM_POSITIONS
    slope = tl.load(slope_ptr)
    bias = tl.load(intercepts_ptr + k, mask=k_mask, other=0.0)

    # This call's contribution to the normal equations. Observation i at
    # position k has design row [x_i, e_k], IRLS weight w_i = p_i (1 - p_i) and
    # residual y_i - p_i, so it adds w x^2 to a, w x to b_k, w to c_k, r x to
    # g0 and r to g1_k. a and g0 are kept per position and summed at refit.
    a_k = tl.zeros([BLOCK_K], dtype=tl.float32)
    b_k = tl.zeros([BLOCK_K], dtype=tl.float32)
    c_k = tl.zeros([BLOCK_K], dtype=tl.float32)
    g0_k = tl.zeros([BLOCK_K], dtype=tl.float32)
    g1_k = tl.zeros([BLOCK_K], dtype=tl.float32)
    n_k = tl.zeros([BLOCK_K], dtype=tl.float32)
    for start in range(0, num_reqs, BLOCK_R):
        r = start + tl.arange(0, BLOCK_R)
        r_mask = r < num_reqs
        verified = tl.load(num_verified_ptr + r, mask=r_mask, other=0)
        accepted = tl.load(num_accepted_ptr + r, mask=r_mask, other=0)
        # The accepted drafts and the first rejected one are observed; drafts
        # after the first rejection, or never verified, carry no label.
        observed = (
            r_mask[:, None]
            & k_mask[None, :]
            & (k[None, :] <= accepted[:, None])
            & (k[None, :] < verified[:, None])
        )
        x = tl.load(
            x_ptr + r.to(tl.int64)[:, None] * x_stride_req + k[None, :] * x_stride_pos,
            mask=observed,
            other=0.0,
        ).to(tl.float32)
        p = tl.sigmoid(slope * x + bias[None, :])
        label = tl.where(k[None, :] < accepted[:, None], 1.0, 0.0)
        w = tl.where(observed, p * (1.0 - p), 0.0)
        resid = tl.where(observed, label - p, 0.0)
        a_k += tl.sum(w * x * x, axis=0)
        b_k += tl.sum(w * x, axis=0)
        c_k += tl.sum(w, axis=0)
        g0_k += tl.sum(resid * x, axis=0)
        g1_k += tl.sum(resid, axis=0)
        n_k += tl.sum(observed.to(tl.float32), axis=0)

    s_ptr = stats_ptr + k * NUM_STATS
    a_k += tl.load(s_ptr + 0, mask=k_mask, other=0.0)
    b_k += tl.load(s_ptr + 1, mask=k_mask, other=0.0)
    c_k += tl.load(s_ptr + 2, mask=k_mask, other=0.0)
    g0_k += tl.load(s_ptr + 3, mask=k_mask, other=0.0)
    g1_k += tl.load(s_ptr + 4, mask=k_mask, other=0.0)
    n_k += tl.load(s_ptr + 5, mask=k_mask, other=0.0)

    num_calls = tl.load(num_calls_ptr) + 1
    # Every warp must read the state before any warp overwrites it below, or a
    # late warp could see the new call count and disagree on the refit.
    tl.debug_barrier()
    refit = num_calls >= refit_interval
    if refit:
        # One damped Newton step on the round's statistics: solve the arrowhead
        # system (X^T W X + L2 I) [d_slope, d_bias_0 .. d_bias_n-1] = X^T r,
        #   [ a+L2  b_0      ...  b_n-1     ] [ d_slope    ]   [ g0     ]
        #   [ b_0   c_0+L2                  ] [ d_bias_0   ] = [ g1_0   ]
        #   [ ...          ...              ] [ ...        ]   [ ...    ]
        #   [ b_n-1             c_n-1+L2    ] [ d_bias_n-1 ]   [ g1_n-1 ]
        # by eliminating the diagonal block (Schur complement on the slope).
        c_reg = tl.where(k_mask, c_k + L2, 1.0)
        a = tl.sum(a_k, axis=0) + L2
        g0 = tl.sum(g0_k, axis=0)
        step_slope = (g0 - tl.sum(b_k * g1_k / c_reg, axis=0)) / (
            a - tl.sum(b_k * b_k / c_reg, axis=0)
        )
        # The slope learns from every position, so it is damped by the round's
        # total observation count; each intercept by its own.
        total_n = tl.sum(n_k, axis=0)
        step_slope *= total_n / (total_n + DAMPING)
        # From an overconfident fit the predictions saturate, the curvature is
        # small and the step overshoots past zero. Backtrack: halve the step
        # until the slope stays positive. Drop non-finite steps and steps that
        # still cross zero after MAX_SLOPE_HALVINGS halvings.
        for _ in tl.static_range(MAX_SLOPE_HALVINGS):
            step_slope = tl.where(
                slope + step_slope > 0.0, step_slope, step_slope * 0.5
            )
        valid = (tl.abs(step_slope) < float("inf")) & (slope + step_slope > 0.0)
        step_slope = tl.where(valid, step_slope, 0.0)
        step_bias = (g1_k - b_k * step_slope) / c_reg
        step_bias *= n_k / (n_k + DAMPING)
        step_bias = tl.where(tl.abs(step_bias) < float("inf"), step_bias, 0.0)
        tl.store(slope_ptr, slope + step_slope)
        tl.store(intercepts_ptr + k, bias + step_bias, mask=k_mask)

    # A refit consumes the round: the next one starts from clean statistics.
    tl.store(s_ptr + 0, tl.where(refit, 0.0, a_k), mask=k_mask)
    tl.store(s_ptr + 1, tl.where(refit, 0.0, b_k), mask=k_mask)
    tl.store(s_ptr + 2, tl.where(refit, 0.0, c_k), mask=k_mask)
    tl.store(s_ptr + 3, tl.where(refit, 0.0, g0_k), mask=k_mask)
    tl.store(s_ptr + 4, tl.where(refit, 0.0, g1_k), mask=k_mask)
    tl.store(s_ptr + 5, tl.where(refit, 0.0, n_k), mask=k_mask)
    tl.store(num_calls_ptr, tl.where(refit, 0, num_calls))


def observe_acceptance(
    x: torch.Tensor,
    num_verified: torch.Tensor,
    num_accepted: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    stats: torch.Tensor,
    num_calls: torch.Tensor,
    *,
    refit_interval: int,
    l2: float,
    damping: float,
    block_reqs: int,
) -> None:
    """Accumulate one batch of verified drafts and refit when a round ends.

    x: float [bs, K] features the drafts were scored with; num_verified /
    num_accepted: int [bs] drafts the target verified / accepted per request.
    slope [1], intercepts [K], stats [K, NUM_ACCEPTANCE_STATS] and the int32
    call counter num_calls [1] are updated in place by a single program, so the
    result is deterministic.
    """
    num_reqs, num_positions = x.shape
    assert intercepts.shape == (num_positions,), (intercepts.shape, x.shape)
    assert (
        stats.shape == (num_positions, NUM_ACCEPTANCE_STATS) and stats.is_contiguous()
    )
    _observe_acceptance_kernel[(1,)](
        x,
        x.stride(0),
        x.stride(1),
        num_verified,
        num_accepted,
        slope,
        intercepts,
        stats,
        num_calls,
        num_reqs,
        refit_interval,
        NUM_POSITIONS=num_positions,
        L2=l2,
        DAMPING=damping,
        MAX_SLOPE_HALVINGS=_MAX_SLOPE_HALVINGS,
        NUM_STATS=NUM_ACCEPTANCE_STATS,
        BLOCK_R=block_reqs,
        BLOCK_K=triton.next_power_of_2(num_positions),
        num_warps=4,
    )
