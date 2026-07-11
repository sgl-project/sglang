from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        DSparkScheduleConfig,
    )

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_SCHEDULE_TOPK.get()


class ScheduleVerifyLensTopk:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk(confidence=confidence, budget=budget, cfg=cfg)

    @classmethod
    def triton(
        cls,
        *,
        confidence: torch.Tensor,
        budget: int,
        cfg: DSparkScheduleConfig,
    ) -> torch.Tensor:
        return schedule_verify_lens_topk_triton(
            confidence=confidence, budget=budget, cfg=cfg
        )


def compute_sort_survival(confidence: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    return schedule_verify_lens_topk_from_survival(
        survival_probs=compute_sort_survival(confidence), budget=budget, cfg=cfg
    )


def schedule_verify_lens_topk_from_survival(
    *,
    survival_probs: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    num_requests, _gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len()
    device = survival_probs.device

    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        candidate_window = survival_probs[:, :max_len]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(candidate_window.shape[1], device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = candidate_window >= cfg.survival_eps

            flat_prob = candidate_window.reshape(-1).to(torch.float64)
            flat_request = request_index.reshape(-1)
            flat_position = position_index.reshape(-1)
            flat_valid = valid.reshape(-1)

            order = _value_independent_descending_order(
                probs=flat_prob,
                positions=flat_position,
                requests=flat_request,
                valid=flat_valid,
            )

            take = min(int(budget), num_candidates)
            chosen = order[:take]
            chosen_requests = flat_request[chosen]
            chosen_valid = flat_valid[chosen].to(torch.int64)
            selected_extra.scatter_add_(0, chosen_requests, chosen_valid)

    min_len = torch.full(
        (num_requests,), cfg.min_verify_len, dtype=torch.int64, device=device
    )
    verify_lens = min_len + selected_extra
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = torch.clamp(verify_lens, min=lower_bound, max=max_len)
    return verify_lens.to(torch.int32)


def _value_independent_descending_order(
    *,
    probs: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_prob = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
    num_candidates = masked_prob.numel()
    order = torch.arange(num_candidates, device=probs.device)
    order = order[torch.argsort(requests[order], stable=True)]
    order = order[torch.argsort(positions[order], stable=True)]
    order = order[torch.argsort(-masked_prob[order], stable=True)]
    return order


@triton.jit
def _schedule_topk_prep_kernel(
    confidence_ptr,
    survival_ptr,
    selected_extra_ptr,
    gamma,
    cols,
    G_P2: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.arange(0, G_P2)
    conf = tl.load(
        confidence_ptr + row.to(tl.int64) * gamma + g, mask=g < gamma, other=1.0
    ).to(tl.float32)
    surv = tl.cumprod(conf, axis=0)
    tl.store(survival_ptr + row.to(tl.int64) * cols + g, surv, mask=g < cols)
    tl.store(selected_extra_ptr + row, 0)


@triton.jit
def _schedule_topk_finalize_kernel(
    selected_extra_ptr,
    out_ptr,
    min_verify_len,
    lower_bound,
    max_len,
    bs,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    extra = tl.load(selected_extra_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = min_verify_len + extra
    lens = tl.maximum(lens, lower_bound)
    lens = tl.minimum(lens, max_len)
    tl.store(out_ptr + offs, lens, mask=mask)


@triton.jit
def _schedule_topk_selected_extra_kernel(
    survival_ptr,
    selected_extra_ptr,
    budget,
    cols,
    n,
    survival_eps,
    BLOCK_C: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < n
    r = c // cols
    p = c % cols
    sp = tl.load(survival_ptr + c, mask=cmask, other=0.0)
    valid_c = sp >= survival_eps
    mp = tl.where(valid_c, sp, float("-inf"))
    rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    for cp0 in range(0, n, BLOCK_CP):
        cp = cp0 + tl.arange(0, BLOCK_CP)
        cpmask = cp < n
        rp = cp // cols
        pp = cp % cols
        spp = tl.load(survival_ptr + cp, mask=cpmask, other=0.0)
        validp = spp >= survival_eps
        mpp = tl.where(validp, spp, float("-inf"))
        gt = mpp[None, :] > mp[:, None]
        eq = mpp[None, :] == mp[:, None]
        pos_lt = pp[None, :] < p[:, None]
        pos_eq = pp[None, :] == p[:, None]
        req_lt = rp[None, :] < r[:, None]
        before = gt | (eq & (pos_lt | (pos_eq & req_lt)))
        before = before & cpmask[None, :]
        rank += tl.sum(before.to(tl.int32), axis=1)
    selected = valid_c & (rank < budget)
    tl.atomic_add(selected_extra_ptr + r, selected.to(tl.int32), mask=cmask)


def schedule_verify_lens_topk_triton(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols

    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    _schedule_topk_prep_kernel[(num_requests,)](
        confidence.contiguous(),
        survival,
        selected_extra,
        gamma,
        cols,
        G_P2=triton.next_power_of_2(max(gamma, 1)),
    )
    if budget > 0 and n > 0:
        BLOCK_C = 64
        BLOCK_CP = 256
        grid = (triton.cdiv(n, BLOCK_C),)
        _schedule_topk_selected_extra_kernel[grid](
            survival,
            selected_extra,
            int(budget),
            cols,
            n,
            float(cfg.survival_eps),
            BLOCK_C=BLOCK_C,
            BLOCK_CP=BLOCK_CP,
        )

    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    BLOCK = 256
    _schedule_topk_finalize_kernel[(triton.cdiv(num_requests, BLOCK),)](
        selected_extra,
        verify_lens,
        int(cfg.min_verify_len),
        max(cfg.min_verify_len, 1),
        int(max_len),
        num_requests,
        BLOCK=BLOCK,
    )
    return verify_lens
