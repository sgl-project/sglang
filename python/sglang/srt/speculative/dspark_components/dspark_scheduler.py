from __future__ import annotations

import logging
from typing import Optional, Union

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
    _interp_clamped,
    build_uninitialized_sps_table,
    load_sps_table_from_path,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    _value_independent_descending_order as _value_independent_descending_order,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    schedule_verify_lens_topk as schedule_verify_lens_topk,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    schedule_verify_lens_topk_from_survival as schedule_verify_lens_topk_from_survival,
)

logger = logging.getLogger(__name__)


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self) -> int:
        return self.max_verify_len or (self.gamma + 1)

    def validate(self) -> None:
        max_len = self.resolved_max_verify_len()
        if self.gamma < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")
        if not (0 <= self.min_verify_len <= max_len <= self.gamma + 1):
            raise ValueError(
                "DSpark verify-len config must satisfy 0 <= min <= max <= gamma+1, "
                f"got min={self.min_verify_len}, max={max_len}, gamma={self.gamma}."
            )
        if self.survival_eps < 0:
            raise ValueError(f"survival_eps must be >= 0, got {self.survival_eps}.")


class VerifyBudgetDecision(msgspec.Struct):
    budget: int
    predicted_step_seconds: Optional[float] = None
    predicted_theta: Optional[float] = None


def compute_verify_token_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: Union[SpsCostTable, SpsAdditiveCostTable],
    cfg: DSparkScheduleConfig,
) -> VerifyBudgetDecision:
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()

    candidates = history_survival_probs[:, :max_len].flatten()
    candidates = candidates[candidates >= cfg.survival_eps].to(torch.float64)
    candidates_sorted = torch.sort(candidates, descending=True).values
    prefix_sum = torch.cumsum(candidates_sorted, dim=0)

    tau_star = num_requests + torch.cat(
        [torch.zeros(1, dtype=torch.float64), prefix_sum]
    )
    if isinstance(sps_table, SpsAdditiveCostTable):
        step_time = _additive_step_time_tensor(
            table=sps_table,
            num_requests=int(num_requests),
            num_budgets=int(tau_star.numel()),
        )
        theta = tau_star / step_time
        idx = int(torch.argmax(theta))
        predicted_step_seconds = float(step_time[idx])
    else:
        batch_tokens = num_requests + torch.arange(tau_star.numel(), dtype=torch.int64)
        sps = _lookup_sps_tensor(sps_table=sps_table, batch_tokens=batch_tokens)
        theta = tau_star * sps
        idx = int(torch.argmax(theta))
        sps_at_idx = float(sps[idx])
        predicted_step_seconds = 1.0 / sps_at_idx if sps_at_idx > 0 else None
    return VerifyBudgetDecision(
        budget=idx,
        predicted_step_seconds=predicted_step_seconds,
        predicted_theta=float(theta[idx]),
    )


def _lookup_sps_tensor(
    *, sps_table: SpsCostTable, batch_tokens: torch.Tensor
) -> torch.Tensor:
    probes = torch.tensor(sps_table.sample_batch_tokens, dtype=torch.int64)
    sps = torch.tensor(sps_table.sample_steps_per_sec, dtype=torch.float64)
    idx = torch.bucketize(batch_tokens, probes, right=True) - 1
    idx = idx.clamp_(0, probes.numel() - 1)
    return sps[idx]


def _additive_step_time_tensor(
    *, table: SpsAdditiveCostTable, num_requests: int, num_budgets: int
) -> torch.Tensor:
    floor = table.bias_seconds + _interp_clamped(
        table.bs_probes, table.alpha_seconds, float(num_requests)
    )
    m_probes = torch.tensor(table.m_probes, dtype=torch.float64)
    theta_vals = torch.tensor(table.theta_seconds, dtype=torch.float64)
    m = (num_requests + torch.arange(num_budgets, dtype=torch.float64)).clamp_(
        min=float(table.m_probes[0]), max=float(table.m_probes[-1])
    )
    hi = torch.bucketize(m, m_probes, right=True).clamp_(1, m_probes.numel() - 1)
    lo = hi - 1
    span = (m_probes[hi] - m_probes[lo]).clamp_(min=1e-9)
    frac = (m - m_probes[lo]) / span
    theta_at_m = theta_vals[lo] + frac * (theta_vals[hi] - theta_vals[lo])
    return floor + theta_at_m


class HostConfidenceBudgetPlanner:

    def __init__(
        self,
        *,
        sps_table: SpsCostTable,
        cfg: DSparkScheduleConfig,
        model_runner,
        relay_lag_steps: int = 1,
    ) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg
        self._model_runner = model_runner
        self.forced_budget_frac: Optional[float] = None
        self.last_decision: Optional[VerifyBudgetDecision] = None
        self.lag_steps = max(
            int(envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.get()), 1
        )
        self.carry_steps = max(self.lag_steps - int(relay_lag_steps), 0)
        self._carry_confidence: Optional[torch.Tensor] = None
        self._carry_generation: Optional[torch.Tensor] = None
        self._carry_pos = 0

    def compute_budget(
        self,
        *,
        confidence: torch.Tensor,
        generation: torch.Tensor,
        current_generation: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> int:
        lagged_confidence, lagged_generation = self._shift_to_lag(
            confidence=confidence,
            generation=generation,
            req_pool_indices_cpu=req_pool_indices_cpu,
        )
        survival = self._two_steps_prior_survival(
            lagged_confidence=lagged_confidence,
            lagged_generation=lagged_generation,
            current_generation=current_generation,
        )
        forced_frac = self.forced_budget_frac
        if forced_frac is not None:
            full_budget = int(survival[:, : self.cfg.resolved_max_verify_len()].numel())
            forced_budget = max(0, int(float(forced_frac) * full_budget))
            self.last_decision = VerifyBudgetDecision(budget=forced_budget)
            return forced_budget
        decision = compute_verify_token_budget(
            history_survival_probs=survival,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )
        self.last_decision = decision
        return decision.budget

    def take_last_decision(self) -> Optional[VerifyBudgetDecision]:
        decision = self.last_decision
        self.last_decision = None
        return decision

    def note_non_decode_step(self) -> None:
        self.last_decision = None

    def _shift_to_lag(
        self,
        *,
        confidence: torch.Tensor,
        generation: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.carry_steps == 0:
            return confidence, generation
        self._ensure_carry(gamma=confidence.shape[-1])
        slot = self._carry_pos % self.carry_steps
        rows = req_pool_indices_cpu.to(torch.int64)
        lagged_confidence = self._carry_confidence[slot, rows].clone()
        lagged_generation = self._carry_generation[slot, rows].clone()
        self._carry_confidence[slot, rows] = confidence.to(torch.float32)
        self._carry_generation[slot, rows] = generation.to(torch.int64)
        self._carry_pos += 1
        return lagged_confidence, lagged_generation

    def _two_steps_prior_survival(
        self,
        *,
        lagged_confidence: torch.Tensor,
        lagged_generation: torch.Tensor,
        current_generation: torch.Tensor,
    ) -> torch.Tensor:
        k_survival = torch.cumprod(lagged_confidence.to(torch.float32), dim=1)
        current_gen = current_generation.to(torch.int64)
        fresh = (
            (current_gen >= 1) & (lagged_generation.to(torch.int64) == current_gen)
        ).view(-1, 1)
        return torch.where(fresh, k_survival, torch.ones_like(k_survival))

    def _ensure_carry(self, *, gamma: int) -> None:
        if self._carry_confidence is not None:
            return
        req_pool_size = int(self._model_runner.req_to_token_pool.req_to_token.shape[0])
        self._carry_confidence = torch.zeros(
            (self.carry_steps, req_pool_size, gamma), dtype=torch.float32
        )
        self._carry_generation = torch.zeros(
            (self.carry_steps, req_pool_size),
            dtype=torch.int64,
        )


def build_sps_cost_table(
    *,
    server_args: ServerArgs,
    verify_num_draft_tokens: int,
) -> Union[SpsCostTable, SpsAdditiveCostTable]:
    sps_table_path = server_args.speculative_dspark_sps_table_path
    if sps_table_path:
        return load_sps_table_from_path(sps_table_path)
    max_batch_tokens = max(
        1,
        int(server_args.max_running_requests or 1) * verify_num_draft_tokens,
    )
    return build_uninitialized_sps_table(max_batch_tokens=max_batch_tokens)
