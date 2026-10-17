from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

_MAX_LOG_ODDS = 40.0

# Cold-start coefficients from vllm-project/vllm#52228.
_INIT_SLOPE = 0.5
_INIT_INTERCEPT = -0.2

# Saturated rows carry ~0 Newton weight but +-1 residuals; unclipped steps diverge.
_MAX_STEP = 1.0
_MAX_SLOPE = 4.0
_MAX_INTERCEPT = 10.0


@triton.jit
def _predict_kernel(
    draft_probs_ptr,
    draft_probs_stride,
    features_ptr,
    predictions_ptr,
    slope_ptr,
    intercepts_ptr,
    num_steps,
    MAX_LOG_ODDS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    mask = k < num_steps
    q = tl.load(draft_probs_ptr + row * draft_probs_stride + k, mask=mask, other=0.5)
    q = tl.minimum(tl.maximum(q.to(tl.float32), 0.0), 1.0)
    feature = tl.log(q) - tl.log(1.0 - q)
    feature = tl.where(feature == feature, feature, 0.0)
    feature = tl.minimum(tl.maximum(feature, -MAX_LOG_ODDS), MAX_LOG_ODDS)
    slope = tl.load(slope_ptr)
    intercepts = tl.load(intercepts_ptr + k, mask=mask, other=0.0)
    prediction = tl.sigmoid(slope * feature + intercepts)
    tl.store(features_ptr + row * num_steps + k, feature, mask=mask)
    tl.store(predictions_ptr + row * num_steps + k, prediction, mask=mask)


@triton.jit
def _accumulate_kernel(
    info_ptr,
    grad_ptr,
    counts_ptr,
    features_ptr,
    predictions_ptr,
    num_correct_drafts_ptr,
    verify_lens_ptr,
    num_reqs,
    num_steps,
    BLOCK_R: tl.constexpr,
):
    step = tl.program_id(0)
    info_a = 0.0
    info_b = 0.0
    info_c = 0.0
    grad_0 = 0.0
    grad_1 = 0.0
    count = 0.0
    for start in range(0, num_reqs, BLOCK_R):
        rows = start + tl.arange(0, BLOCK_R)
        mask = rows < num_reqs
        num_correct = tl.load(num_correct_drafts_ptr + rows, mask=mask, other=0)
        num_verified = tl.load(verify_lens_ptr + rows, mask=mask, other=1) - 1
        offsets = rows.to(tl.int64) * num_steps + step
        feature = tl.load(features_ptr + offsets, mask=mask, other=0.0)
        prediction = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        observed = mask & (step <= num_correct) & (step < num_verified)
        label = tl.where(step < num_correct, 1.0, 0.0)
        weight = tl.where(observed, prediction * (1.0 - prediction), 0.0)
        residual = tl.where(observed, label - prediction, 0.0)
        info_a += tl.sum(weight * feature * feature, axis=0)
        info_b += tl.sum(weight * feature, axis=0)
        info_c += tl.sum(weight, axis=0)
        grad_0 += tl.sum(residual * feature, axis=0)
        grad_1 += tl.sum(residual, axis=0)
        count += tl.sum(tl.where(observed, 1.0, 0.0), axis=0)
    tl.store(info_ptr + step * 3 + 0, tl.load(info_ptr + step * 3 + 0) + info_a)
    tl.store(info_ptr + step * 3 + 1, tl.load(info_ptr + step * 3 + 1) + info_b)
    tl.store(info_ptr + step * 3 + 2, tl.load(info_ptr + step * 3 + 2) + info_c)
    tl.store(grad_ptr + step * 2 + 0, tl.load(grad_ptr + step * 2 + 0) + grad_0)
    tl.store(grad_ptr + step * 2 + 1, tl.load(grad_ptr + step * 2 + 1) + grad_1)
    tl.store(counts_ptr + step, tl.load(counts_ptr + step) + count)


@triton.jit
def _refit_kernel(
    slope_ptr,
    intercepts_ptr,
    info_ptr,
    grad_ptr,
    counts_ptr,
    num_steps,
    L2: tl.constexpr,
    DAMPING: tl.constexpr,
    MAX_STEP: tl.constexpr,
    MAX_SLOPE: tl.constexpr,
    MAX_INTERCEPT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    mask = k < num_steps
    a_k = tl.load(info_ptr + k * 3 + 0, mask=mask, other=0.0)
    b_k = tl.load(info_ptr + k * 3 + 1, mask=mask, other=0.0)
    c_k = tl.load(info_ptr + k * 3 + 2, mask=mask, other=0.0) + L2
    g0_k = tl.load(grad_ptr + k * 2 + 0, mask=mask, other=0.0)
    g1_k = tl.load(grad_ptr + k * 2 + 1, mask=mask, other=0.0)
    n_k = tl.load(counts_ptr + k, mask=mask, other=0.0)
    a = tl.sum(a_k, axis=0) + L2
    g0 = tl.sum(g0_k, axis=0)
    slope = tl.load(slope_ptr)
    intercepts = tl.load(intercepts_ptr + k, mask=mask, other=0.0)

    shrink = tl.where(mask, b_k * b_k / c_k, 0.0)
    coupling = tl.where(mask, b_k * g1_k / c_k, 0.0)
    step_slope = (g0 - tl.sum(coupling, axis=0)) / (a - tl.sum(shrink, axis=0))
    total_n = tl.sum(n_k, axis=0)
    step_slope *= total_n / (total_n + DAMPING)
    step_slope = tl.where(step_slope == step_slope, step_slope, 0.0)
    step_slope = tl.minimum(tl.maximum(step_slope, -MAX_STEP), MAX_STEP)
    new_slope = tl.minimum(tl.maximum(slope + step_slope, 0.0), MAX_SLOPE)
    step_intercepts = (g1_k - b_k * (new_slope - slope)) / c_k
    step_intercepts *= n_k / (n_k + DAMPING)
    step_intercepts = tl.where(step_intercepts == step_intercepts, step_intercepts, 0.0)
    step_intercepts = tl.minimum(tl.maximum(step_intercepts, -MAX_STEP), MAX_STEP)
    new_intercepts = tl.minimum(
        tl.maximum(intercepts + step_intercepts, -MAX_INTERCEPT), MAX_INTERCEPT
    )

    tl.store(slope_ptr, new_slope)
    tl.store(intercepts_ptr + k, new_intercepts, mask=mask)
    tl.store(info_ptr + k * 3 + 0, 0.0, mask=mask)
    tl.store(info_ptr + k * 3 + 1, 0.0, mask=mask)
    tl.store(info_ptr + k * 3 + 2, 0.0, mask=mask)
    tl.store(grad_ptr + k * 2 + 0, 0.0, mask=mask)
    tl.store(grad_ptr + k * 2 + 1, 0.0, mask=mask)
    tl.store(counts_ptr + k, 0.0, mask=mask)


class AcceptancePrediction(msgspec.Struct, frozen=True):
    features: torch.Tensor
    predictions: torch.Tensor


class OnlineAcceptanceEstimator:
    """p_k = sigmoid(slope * logit(q_max_k) + intercept_k), fitted online by damped
    IRLS; q_max depends on the draft prefix only, so trimming on it stays lossless."""

    REFIT_INTERVAL = 100
    DAMPING_OBSERVATIONS = 50.0
    L2 = 1e-3

    def __init__(self, *, num_steps: int, device: torch.device) -> None:
        self.num_steps = num_steps
        self.device = device
        self._steps_since_refit = 0
        self.num_refits = 0
        self.slope = torch.full((1,), _INIT_SLOPE, dtype=torch.float32, device=device)
        self.intercepts = torch.full(
            (num_steps,), _INIT_INTERCEPT, dtype=torch.float32, device=device
        )
        self._info = torch.zeros(num_steps, 3, dtype=torch.float32, device=device)
        self._grad = torch.zeros(num_steps, 2, dtype=torch.float32, device=device)
        self._counts = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self._block_k = triton.next_power_of_2(num_steps)

    def predict(self, draft_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = draft_probs.shape[0]
        features = torch.empty(
            (num_reqs, self.num_steps), dtype=torch.float32, device=self.device
        )
        predictions = torch.empty_like(features)
        if num_reqs > 0:
            _predict_kernel[(num_reqs,)](
                draft_probs,
                draft_probs.stride(0),
                features,
                predictions,
                self.slope,
                self.intercepts,
                self.num_steps,
                MAX_LOG_ODDS=_MAX_LOG_ODDS,
                BLOCK_K=self._block_k,
            )
        return features, predictions

    def observe(
        self,
        *,
        features: torch.Tensor,
        predictions: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> None:
        num_reqs = features.shape[0]
        if num_reqs > 0:
            _accumulate_kernel[(self.num_steps,)](
                self._info,
                self._grad,
                self._counts,
                features,
                predictions,
                num_correct_drafts.to(torch.int32),
                verify_lens.to(torch.int32),
                num_reqs,
                self.num_steps,
                BLOCK_R=min(triton.next_power_of_2(num_reqs), 1024),
            )
        self._steps_since_refit += 1
        if self._steps_since_refit < self.REFIT_INTERVAL:
            return
        self._steps_since_refit = 0
        _refit_kernel[(1,)](
            self.slope,
            self.intercepts,
            self._info,
            self._grad,
            self._counts,
            self.num_steps,
            L2=self.L2,
            DAMPING=self.DAMPING_OBSERVATIONS,
            MAX_STEP=_MAX_STEP,
            MAX_SLOPE=_MAX_SLOPE,
            MAX_INTERCEPT=_MAX_INTERCEPT,
            BLOCK_K=self._block_k,
        )
        self.num_refits += 1
