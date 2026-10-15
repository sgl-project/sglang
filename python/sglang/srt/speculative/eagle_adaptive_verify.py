from __future__ import annotations

import bisect
import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildRaggedVerifyWindow,
    RaggedVerifyWindow,
    ScatterCompactToStrided,
)
from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_LAG,
    ResolvedConfidence,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.acceptance_estimator import OnlineAcceptanceEstimator
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

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

logger = logging.getLogger(__name__)

_PROFILE_REPLAYS = 5
_RESIDUAL_SYNC_INTERVAL = 64
_MIN_RESIDUAL_SAMPLES = 8
_EMA_WEIGHT = 0.1


class RaggedVerifyForward(msgspec.Struct, frozen=True):
    window: RaggedVerifyWindow
    mrope_positions: Optional[torch.Tensor]
    layout: RaggedVerifyLayout
    num_draft_tokens: int

    def unpack_output(self, logits_output) -> None:
        logits_output.next_token_logits = self._to_strided(
            logits_output.next_token_logits
        )
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = self._to_strided(logits_output.hidden_states)

    def _to_strided(self, compact: torch.Tensor) -> torch.Tensor:
        return ScatterCompactToStrided.execute(
            compact=compact,
            layout=self.layout,
            fill_value=0.0,
            verify_num_draft_tokens=self.num_draft_tokens,
        )


class AdaptiveVerifyStep(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    features: torch.Tensor
    predictions: torch.Tensor
    num_reqs: int
    verify_cost_ms: float


class _ResidualCost:
    """Step time outside target verify per batch size, measured on TP rank 0."""

    def __init__(self, *, tp_group, is_rank0: bool) -> None:
        self._tp_group = tp_group
        self._is_rank0 = is_rank0
        self._local: dict[int, tuple[float, int]] = {}
        self._synced: dict[int, float] = {}
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
        verify_cost_ms: float,
    ) -> None:
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._inflight.append((start, end, num_reqs, verify_cost_ms))
            self._drain()
        self._steps += 1
        if self._steps % _RESIDUAL_SYNC_INTERVAL == 0:
            self._synced = self._tp_group.broadcast_object(
                {
                    bs: value
                    for bs, (value, count) in self._local.items()
                    if count >= _MIN_RESIDUAL_SAMPLES
                },
                src=0,
            )

    def _drain(self) -> None:
        while self._inflight and self._inflight[0][1].query():
            start, end, num_reqs, verify_cost_ms = self._inflight.popleft()
            residual = max(start.elapsed_time(end) - verify_cost_ms, 0.0)
            value, count = self._local.get(num_reqs, (residual, 0))
            if count:
                value = (1.0 - _EMA_WEIGHT) * value + _EMA_WEIGHT * residual
            self._local[num_reqs] = (value, count + 1)

    def lookup(self, num_reqs: int) -> Optional[float]:
        if num_reqs in self._synced:
            return self._synced[num_reqs]
        known = sorted(self._synced)
        if not known:
            return None
        pos = bisect.bisect_left(known, num_reqs)
        if pos == 0:
            return self._synced[known[0]]
        if pos == len(known):
            return self._synced[known[-1]]
        lo, hi = known[pos - 1], known[pos]
        frac = (num_reqs - lo) / (hi - lo)
        return self._synced[lo] + frac * (self._synced[hi] - self._synced[lo])


class EagleAdaptiveVerifier:
    """Trims EAGLE chain verification per request to maximize tokens per second."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_draft_tokens: int,
        target_model_runner,
        device: torch.device,
        overlap: bool,
    ) -> None:
        assert num_draft_tokens == num_steps + 1
        runner = target_model_runner.decode_cuda_graph_runner
        if runner is None or not runner.uniform_verify_graphs:
            raise ValueError(
                "--speculative-adaptive-verification needs the target verify CUDA "
                "graphs; do not disable CUDA graphs."
            )
        self.num_steps = num_steps
        self.num_draft_tokens = num_draft_tokens
        self.device = device
        self._model_runner = target_model_runner
        self._overlap = overlap
        self._capture_bs = list(runner.capture_bs)
        self._tiers_by_slots = {
            slots: list(tiers) for slots, tiers in runner.ragged_tiers_by_slots.items()
        }
        self._max_slots = runner.max_bs
        self._schedule_cfg = DSparkScheduleConfig(gamma=num_steps)
        self._tp_group = get_parallel().tp_group
        self._tier_costs, self._uniform_costs = self._profile_verify_costs(
            runner=runner
        )
        self.estimator = OnlineAcceptanceEstimator(num_steps=num_steps, device=device)
        self._req_generation = target_model_runner.req_to_token_pool.req_generation
        self._budget_planner = HostConfidenceBudgetPlanner(
            sps_table=build_uninitialized_sps_table(
                max_batch_tokens=self._max_slots * num_draft_tokens
            ),
            cfg=self._schedule_cfg,
            model_runner=target_model_runner,
            relay_lag_steps=CONFIDENCE_RELAY_RING_LAG if overlap else 0,
        )
        self._residual = _ResidualCost(
            tp_group=self._tp_group, is_rank0=self._tp_group.rank_in_group == 0
        )
        self._log_interval = envs.SGLANG_ADAPTIVE_VERIFY_LOG_INTERVAL.get()
        self._forced_tier = envs.SGLANG_TEST_ADAPTIVE_VERIFY_FORCE_TIER.get()
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

    def applies_to(self, batch: ScheduleBatch) -> bool:
        if batch.forward_mode.is_idle() or batch.has_grammar:
            return False
        num_reqs = len(batch.seq_lens)
        if num_reqs == 0 or num_reqs > self._max_slots:
            return False
        if self._model_runner.model_config.model_is_mrope and any(
            mm is not None for mm in batch.multimodal_inputs
        ):
            return False
        return batch.req_pool_indices_cpu is not None

    def prepare_verify_budget(self, batch: ScheduleBatch, future_map: FutureMap):
        """Scheduler hook under overlap: plan this step's budget on the host."""
        draft_input = batch.spec_info
        if (
            draft_input is None
            or batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or not self.applies_to(batch)
        ):
            self._budget_planner.note_non_decode_step()
            return
        draft_input.verify_token_budget = self._plan_budget(
            batch=batch, resolved=future_map.resolve_confidence_cpu(batch)
        )

    def begin_step(self):
        return self._residual.begin()

    def plan(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        draft_probs: torch.Tensor,
        verify_token_budget: Optional[int],
    ) -> AdaptiveVerifyStep:
        num_reqs = len(batch.seq_lens)
        features, predictions = self.estimator.predict(draft_probs)
        if not self._overlap:
            rows = batch.req_pool_indices_cpu.to(torch.int64)
            verify_token_budget = self._plan_budget(
                batch=batch,
                resolved=ResolvedConfidence(
                    confidence=predictions.cpu(),
                    generation=self._req_generation[rows].clone(),
                ),
            )
        self._num_planned += 1
        self._num_full_tokens += num_reqs * self.num_draft_tokens
        if verify_token_budget is None:
            self._num_scheduled_tokens += num_reqs * self.num_draft_tokens
            return AdaptiveVerifyStep(
                verify_lens=torch.full(
                    (num_reqs,),
                    self.num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                features=features,
                predictions=predictions,
                num_reqs=num_reqs,
                verify_cost_ms=self._uniform_costs[self._slots(num_reqs)],
            )
        tier = num_reqs + verify_token_budget
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=predictions, budget=verify_token_budget, cfg=self._schedule_cfg
        )
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens, graph_num_tokens=tier
        )
        verify_input.ragged_verify_layout = layout
        verify_input.ragged_forward = self._build_forward(
            verify_input=verify_input, batch=batch, layout=layout
        )
        verify_input.retrieve_next_token = self._truncate_chain(
            retrieve_next_token=verify_input.retrieve_next_token,
            verify_lens=layout.verify_lens,
        )
        self._num_scheduled_tokens += tier
        return AdaptiveVerifyStep(
            verify_lens=layout.verify_lens,
            features=features,
            predictions=predictions,
            num_reqs=num_reqs,
            verify_cost_ms=self._tier_costs[(self._slots(num_reqs), tier)],
        )

    def finish_step(
        self,
        *,
        step: AdaptiveVerifyStep,
        accept_lens: torch.Tensor,
        timer_start,
    ) -> None:
        self.estimator.observe(
            features=step.features,
            predictions=step.predictions,
            num_correct_drafts=accept_lens - 1,
            verify_lens=step.verify_lens,
        )
        self._residual.end(
            start=timer_start,
            num_reqs=step.num_reqs,
            verify_cost_ms=step.verify_cost_ms,
        )
        if self._log_interval and self._num_planned % self._log_interval == 0:
            self._log_state()

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

    def _plan_budget(
        self, *, batch: ScheduleBatch, resolved: Optional[ResolvedConfidence]
    ) -> Optional[int]:
        budget = self._local_budget(batch=batch, resolved=resolved)
        if self._tp_group.world_size > 1:
            budget = self._tp_group.broadcast_object(budget, src=0)
        return budget

    def _local_budget(
        self, *, batch: ScheduleBatch, resolved: Optional[ResolvedConfidence]
    ) -> Optional[int]:
        if resolved is None:
            self._budget_planner.note_non_decode_step()
            return None
        num_reqs = len(batch.seq_lens)
        slots = self._slots(num_reqs)
        full_tokens = num_reqs * self.num_draft_tokens
        tiers = [t for t in self._tiers_by_slots[slots] if num_reqs < t < full_tokens]
        rows = batch.req_pool_indices_cpu.to(torch.int64)
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
        residual = self._residual.lookup(num_reqs)
        if residual is None or not tiers:
            return None
        return SpsAdditiveCostTable(
            bias_seconds=max(residual, 1e-3) / 1000.0,
            bs_probes=[num_reqs],
            alpha_seconds=[0.0],
            m_probes=tiers + [num_reqs * self.num_draft_tokens],
            theta_seconds=[self._tier_costs[(slots, t)] / 1000.0 for t in tiers]
            + [self._uniform_costs[slots] / 1000.0],
        )

    def _build_forward(
        self,
        *,
        verify_input: EagleVerifyInput,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
    ) -> RaggedVerifyForward:
        bs = layout.bs
        strided = verify_input.draft_token.view(bs, self.num_draft_tokens)
        window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
            draft_block_ids=strided,
            draft_tokens=strided[:, 1:],
            bs=bs,
            device=self.device,
            verify_num_draft_tokens=self.num_draft_tokens,
            model_runner=self._model_runner,
        )
        valid = window.window_index >= 0
        mrope_positions = verify_input.prepared_mrope_positions
        if mrope_positions is not None:
            mrope_positions = torch.where(
                valid, mrope_positions[:, window.window_index.clamp(min=0)], 0
            )
        elif self._model_runner.model_config.model_is_mrope:
            mrope_positions = window.positions.unsqueeze(0).repeat(3, 1)
        return RaggedVerifyForward(
            window=window,
            mrope_positions=mrope_positions,
            layout=layout,
            num_draft_tokens=self.num_draft_tokens,
        )

    def _truncate_chain(
        self, *, retrieve_next_token: torch.Tensor, verify_lens: torch.Tensor
    ) -> torch.Tensor:
        steps = torch.arange(self.num_draft_tokens, device=self.device)
        last = (verify_lens.to(torch.int64) - 1).unsqueeze(1)
        return torch.where(
            steps.unsqueeze(0) >= last,
            torch.full_like(retrieve_next_token, -1),
            retrieve_next_token,
        )

    def _log_state(self) -> None:
        if self._tp_group.rank_in_group != 0:
            return
        logger.info(
            "Adaptive verification: steps=%d refits=%d verify_token_ratio=%.3f "
            "slope=%.4f intercepts=%s",
            self._num_planned,
            self.estimator.num_refits,
            self._num_scheduled_tokens / max(self._num_full_tokens, 1),
            float(self.estimator.slope.item()),
            [round(v, 4) for v in self.estimator.intercepts.tolist()],
        )
