from __future__ import annotations

import bisect
import logging
from collections import deque
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_LAG,
    ResolvedConfidence,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    graph_tier_fill_budget,
    local_verify_tier_num_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    build_uninitialized_sps_table,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, round_up_grid

logger = logging.getLogger(__name__)

_PROFILE_REPLAYS = 5
_RESIDUAL_SYNC_INTERVAL = 64
_MIN_RESIDUAL_SAMPLES = 8
_EMA_WEIGHT = 0.1


class _ResidualCost:
    """Step time outside target verify per (batch size, trimmed), measured on TP
    rank 0; a trimmed step also pays for packing its verify window."""

    def __init__(self, *, tp_group, is_rank0: bool) -> None:
        self._tp_group = tp_group
        self._is_rank0 = is_rank0
        self._local: dict[tuple[int, bool], tuple[float, int]] = {}
        self._synced: dict[tuple[int, bool], float] = {}
        self._inflight: deque = deque()
        self._steps = 0

    def begin(self) -> Optional[torch.cuda.Event]:
        if not self._is_rank0:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(
        self,
        *,
        start: Optional[torch.cuda.Event],
        num_reqs: int,
        trimmed: bool,
        verify_cost_ms: float,
    ) -> None:
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._inflight.append((start, end, (num_reqs, trimmed), verify_cost_ms))
            self._drain()
        self._steps += 1
        if self._steps % _RESIDUAL_SYNC_INTERVAL == 0:
            self._synced = self._tp_group.broadcast_object(
                {
                    key: value
                    for key, (value, count) in self._local.items()
                    if count >= _MIN_RESIDUAL_SAMPLES
                },
                src=0,
            )

    def _drain(self) -> None:
        while self._inflight and self._inflight[0][1].query():
            start, end, key, verify_cost_ms = self._inflight.popleft()
            residual = max(start.elapsed_time(end) - verify_cost_ms, 0.0)
            value, count = self._local.get(key, (residual, 0))
            if count:
                value = (1.0 - _EMA_WEIGHT) * value + _EMA_WEIGHT * residual
            self._local[key] = (value, count + 1)

    def lookup(self, num_reqs: int, *, trimmed: bool) -> Optional[float]:
        known = sorted(bs for bs, flag in self._synced if flag == trimmed)
        if not known:
            return None
        pos = bisect.bisect_left(known, num_reqs)
        if pos < len(known) and known[pos] == num_reqs:
            return self._synced[(num_reqs, trimmed)]
        if pos == 0:
            return self._synced[(known[0], trimmed)]
        if pos == len(known):
            return self._synced[(known[-1], trimmed)]
        lo, hi = known[pos - 1], known[pos]
        frac = (num_reqs - lo) / (hi - lo)
        lo_value, hi_value = self._synced[(lo, trimmed)], self._synced[(hi, trimmed)]
        return lo_value + frac * (hi_value - lo_value)


class AdaptiveVerifyScheduler:
    """Picks each step's verify-token budget from per-draft acceptance confidences
    and the profiled cost of the target verify graphs; independent of the drafter.
    A budget of None verifies every draft through the uniform verify graph."""

    def __init__(
        self,
        *,
        num_steps: int,
        target_model_runner,
        device: torch.device,
        overlap: bool,
    ) -> None:
        runner = target_model_runner.decode_cuda_graph_runner
        if runner is None or not runner.uniform_verify_graphs:
            raise ValueError(
                "--speculative-adaptive-verification needs the target verify CUDA "
                "graphs; do not disable CUDA graphs."
            )
        self.num_steps = num_steps
        self.num_draft_tokens = num_steps + 1
        self.device = device
        self._overlap = overlap
        self._capture_bs = list(runner.capture_bs)
        self._tiers_by_slots = {
            slots: list(tiers) for slots, tiers in runner.ragged_tiers_by_slots.items()
        }
        self.max_num_reqs = runner.max_bs
        self._schedule_cfg = DSparkScheduleConfig(gamma=num_steps)
        self._tp_group = get_parallel().tp_group
        self._tier_costs, self._uniform_costs = self._profile_verify_costs(
            runner=runner
        )
        self._req_generation = target_model_runner.req_to_token_pool.req_generation
        self._budget_planner = HostConfidenceBudgetPlanner(
            sps_table=build_uninitialized_sps_table(
                max_batch_tokens=self.max_num_reqs * self.num_draft_tokens
            ),
            cfg=self._schedule_cfg,
            model_runner=target_model_runner,
            relay_lag_steps=CONFIDENCE_RELAY_RING_LAG if overlap else 0,
        )
        self._residual = _ResidualCost(
            tp_group=self._tp_group, is_rank0=self._tp_group.rank_in_group == 0
        )
        self._forced_tier = envs.SGLANG_TEST_ADAPTIVE_VERIFY_FORCE_TIER.get()
        self._log_interval = envs.SGLANG_ADAPTIVE_VERIFY_LOG_INTERVAL.get()
        self._num_planned = 0
        self._num_scheduled_tokens = 0
        self._num_full_tokens = 0
        if self._tp_group.rank_in_group == 0:
            logger.info(
                "Adaptive verification enabled: verify cost ms by batch size, "
                "uniform graph then ragged graphs by token count: %s",
                {
                    slots: [round(self._uniform_costs[slots], 3)]
                    + [
                        (tier, round(self._tier_costs[(slots, tier)], 3))
                        for tier in self._tiers_by_slots[slots]
                    ]
                    for slots in self._capture_bs
                },
            )

    @property
    def overlap(self) -> bool:
        return self._overlap

    def plan_budget(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        resolved: Optional[ResolvedConfidence],
    ) -> Optional[int]:
        return self._local_budget(
            req_pool_indices_cpu=req_pool_indices_cpu, resolved=resolved
        )

    def plan_budget_now(
        self, *, req_pool_indices_cpu: torch.Tensor, confidence: torch.Tensor
    ) -> Optional[int]:
        rows = req_pool_indices_cpu.to(torch.int64)
        return self.plan_budget(
            req_pool_indices_cpu=req_pool_indices_cpu,
            resolved=ResolvedConfidence(
                confidence=confidence.cpu(),
                generation=self._req_generation[rows].clone(),
            ),
        )

    def note_non_decode_step(self) -> None:
        self._budget_planner.note_non_decode_step()

    def schedule_layout(
        self,
        *,
        confidence: torch.Tensor,
        verify_token_budget: Optional[int],
        num_reqs: int,
    ) -> Optional[RaggedVerifyLayout]:
        self._num_planned += 1
        self._num_full_tokens += num_reqs * self.num_draft_tokens
        if verify_token_budget is None:
            self._num_scheduled_tokens += num_reqs * self.num_draft_tokens
            return None
        tier = num_reqs + verify_token_budget
        self._num_scheduled_tokens += tier
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=confidence, budget=verify_token_budget, cfg=self._schedule_cfg
        )
        return RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens, graph_num_tokens=tier
        )

    def begin_step(self) -> Optional[torch.cuda.Event]:
        return self._residual.begin()

    def end_step(
        self,
        *,
        start: Optional[torch.cuda.Event],
        num_reqs: int,
        layout: Optional[RaggedVerifyLayout],
    ) -> None:
        if num_reqs == 0 or num_reqs > self.max_num_reqs:
            return
        slots = self._slots(num_reqs)
        self._residual.end(
            start=start,
            num_reqs=num_reqs,
            trimmed=layout is not None,
            verify_cost_ms=(
                self._uniform_costs[slots]
                if layout is None
                else self._tier_costs[(slots, layout.graph_num_tokens)]
            ),
        )

    def should_log(self) -> bool:
        return (
            self._tp_group.rank_in_group == 0
            and self._log_interval > 0
            and self._num_planned % self._log_interval == 0
        )

    def log_state(self, *, estimator) -> None:
        logger.info(
            "Adaptive verification: steps=%d verify_token_ratio=%.3f slope=%.4f "
            "intercepts=%s",
            self._num_planned,
            self._num_scheduled_tokens / max(self._num_full_tokens, 1),
            float(estimator.slope.item()),
            [round(v, 4) for v in estimator.intercepts.tolist()],
        )

    def _profile_verify_costs(
        self, *, runner
    ) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
        tier_shapes = [
            (slots, tier)
            for slots, tiers in self._tiers_by_slots.items()
            for tier in tiers
        ]
        tier_costs = [
            runner.measure_graph_replay_ms(
                size=slots, num_tokens=tier, ragged=True, num_replays=_PROFILE_REPLAYS
            )
            for slots, tier in tier_shapes
        ]
        uniform_costs = [
            runner.measure_graph_replay_ms(
                size=bs,
                num_tokens=bs * self.num_draft_tokens,
                ragged=False,
                num_replays=_PROFILE_REPLAYS,
            )
            for bs in self._capture_bs
        ]
        tier_costs, uniform_costs = self._tp_group.broadcast_object(
            (tier_costs, uniform_costs), src=0
        )
        return (
            {shape: float(cost) for shape, cost in zip(tier_shapes, tier_costs)},
            {bs: float(cost) for bs, cost in zip(self._capture_bs, uniform_costs)},
        )

    def _slots(self, num_reqs: int) -> int:
        return self._capture_bs[bisect.bisect_left(self._capture_bs, num_reqs)]

    def _local_budget(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        resolved: Optional[ResolvedConfidence],
    ) -> Optional[int]:
        if resolved is None:
            self._budget_planner.note_non_decode_step()
            return None
        num_reqs = req_pool_indices_cpu.shape[0]
        if num_reqs == 0 or num_reqs > self.max_num_reqs:
            return None
        slots = self._slots(num_reqs)
        full_tokens = num_reqs * self.num_draft_tokens
        tiers = [t for t in self._tiers_by_slots[slots] if num_reqs < t < full_tokens]
        rows = req_pool_indices_cpu.to(torch.int64)
        cost_table = self._cost_table(num_reqs=num_reqs, slots=slots, tiers=tiers)
        budget = self._budget_planner.compute_budget(
            confidence=resolved.confidence,
            generation=resolved.generation,
            current_generation=self._req_generation[rows],
            req_pool_indices_cpu=rows,
            sps_table=cost_table,
        )
        if self._forced_tier > 0 and tiers:
            tier = tiers[min(self._forced_tier, len(tiers)) - 1]
        elif cost_table is None:
            return None
        else:
            total = local_verify_tier_num_tokens(
                bs=num_reqs,
                verify_token_budget=budget,
                verify_num_draft_tokens=self.num_draft_tokens,
                min_verify_len=self._schedule_cfg.min_verify_len,
            )
            if total > tiers[-1]:
                return None
            tier = round_up_grid(total, tiers)
        return graph_tier_fill_budget(
            graph_num_tokens=tier,
            bs=num_reqs,
            verify_num_draft_tokens=self.num_draft_tokens,
            min_verify_len=self._schedule_cfg.min_verify_len,
        )

    def _cost_table(
        self, *, num_reqs: int, slots: int, tiers: list[int]
    ) -> Optional[SpsAdditiveCostTable]:
        residual = self._residual.lookup(num_reqs, trimmed=False)
        if residual is None or not tiers:
            return None
        trimmed_residual = self._residual.lookup(num_reqs, trimmed=True)
        trim_overhead = 0.0 if trimmed_residual is None else trimmed_residual - residual
        return SpsAdditiveCostTable(
            bias_seconds=max(residual, 1e-3) / 1000.0,
            bs_probes=[num_reqs],
            alpha_seconds=[0.0],
            m_probes=tiers + [num_reqs * self.num_draft_tokens],
            theta_seconds=[
                (self._tier_costs[(slots, t)] + trim_overhead) / 1000.0 for t in tiers
            ]
            + [self._uniform_costs[slots] / 1000.0],
        )
