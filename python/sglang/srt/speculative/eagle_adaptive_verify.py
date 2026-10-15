from __future__ import annotations

import bisect
import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    CompactRowIndex,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.acceptance_estimator import OnlineAcceptanceEstimator
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

logger = logging.getLogger(__name__)

_PROFILE_REPLAYS = 5
# Plan from confidences copied two steps ago so every TP rank reads the same data.
_STALE_LAG = 2
_RESIDUAL_SYNC_INTERVAL = 64
_MIN_RESIDUAL_SAMPLES = 8
_EMA_WEIGHT = 0.1
_DEFAULT_ACCEPT_RATE = 0.7


class RaggedVerifyForward(msgspec.Struct, frozen=True):
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    window_cols: torch.Tensor
    mrope_positions: Optional[torch.Tensor]
    strided_rows: torch.Tensor

    def unpack_output(self, logits_output) -> None:
        logits_output.next_token_logits = logits_output.next_token_logits[
            self.strided_rows
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[self.strided_rows]


class AdaptiveVerifyStep(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    features: torch.Tensor
    predictions: torch.Tensor
    num_reqs: int
    verify_cost_ms: float


class _StaleSurvival:
    def __init__(self, *, pool_size: int, num_steps: int, max_bs: int) -> None:
        self._predictions = np.full(
            (pool_size, num_steps), _DEFAULT_ACCEPT_RATE, dtype=np.float32
        )
        self._generation = np.full(pool_size, -1, dtype=np.int64)
        self._default = np.full(num_steps, _DEFAULT_ACCEPT_RATE, dtype=np.float32)
        self._slots = [
            torch.empty((max_bs, num_steps), dtype=torch.float32, pin_memory=True)
            for _ in range(_STALE_LAG + 1)
        ]
        self._next_slot = 0
        self._pending: deque = deque()

    def push(
        self,
        *,
        predictions: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
        generation_cpu: torch.Tensor,
    ) -> None:
        num_reqs = predictions.shape[0]
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        slot[:num_reqs].copy_(predictions, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending.append(
            (
                slot,
                num_reqs,
                event,
                req_pool_indices_cpu.numpy().astype(np.int64),
                generation_cpu.numpy().astype(np.int64),
            )
        )

    def drain(self) -> None:
        while len(self._pending) >= _STALE_LAG:
            slot, num_reqs, event, rows, generation = self._pending.popleft()
            event.synchronize()
            values = slot[:num_reqs].numpy()
            self._predictions[rows] = values
            self._generation[rows] = generation
            self._default = (
                1.0 - _EMA_WEIGHT
            ) * self._default + _EMA_WEIGHT * values.mean(axis=0)

    def survival(
        self, *, req_pool_indices_cpu: np.ndarray, generation_cpu: np.ndarray
    ) -> np.ndarray:
        fresh = self._generation[req_pool_indices_cpu] == generation_cpu
        predictions = np.where(
            fresh[:, None], self._predictions[req_pool_indices_cpu], self._default
        )
        return np.cumprod(predictions, axis=1)


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
        self._capture_bs = list(runner.capture_bs)
        self._tiers_by_slots = {
            slots: list(tiers) for slots, tiers in runner.ragged_tiers_by_slots.items()
        }
        self._max_slots = runner.max_bs
        self._schedule_cfg = DSparkScheduleConfig(gamma=num_steps)
        self._schedule_cfg.validate()
        self._full_verify_lens = torch.full(
            (self._max_slots,), num_draft_tokens, dtype=torch.int32, device=device
        )

        tp_group = get_parallel().tp_group
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
                num_tokens=bs * num_draft_tokens,
                ragged=False,
                num_replays=_PROFILE_REPLAYS,
            )
            for bs in self._capture_bs
        ]
        tier_costs, uniform_costs = tp_group.broadcast_object(
            (tier_costs, uniform_costs), src=0
        )
        self._tier_costs = dict(zip(tier_shapes, tier_costs))
        self._uniform_costs = np.asarray(uniform_costs, dtype=np.float64)
        self.estimator = OnlineAcceptanceEstimator(num_steps=num_steps, device=device)
        pool = target_model_runner.req_to_token_pool
        self._req_generation = pool.req_generation
        self._stale = _StaleSurvival(
            pool_size=int(self._req_generation.shape[0]),
            num_steps=num_steps,
            max_bs=self._max_slots,
        )
        self._residual = _ResidualCost(
            tp_group=tp_group, is_rank0=tp_group.rank_in_group == 0
        )
        self._log_interval = envs.SGLANG_ADAPTIVE_VERIFY_LOG_INTERVAL.get()
        self._forced_tier = envs.SGLANG_TEST_ADAPTIVE_VERIFY_FORCE_TIER.get()
        self._num_planned = 0
        self._num_scheduled_tokens = 0
        self._num_full_tokens = 0
        if tp_group.rank_in_group == 0:
            logger.info(
                "Adaptive verification enabled: verify cost ms by batch size, "
                "uniform graph then ragged graphs by token count: %s",
                {
                    slots: [round(float(cost), 3)]
                    + [
                        (tier, round(float(self._tier_costs[(slots, tier)]), 3))
                        for tier in self._tiers_by_slots[slots]
                    ]
                    for slots, cost in zip(self._capture_bs, self._uniform_costs)
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

    def begin_step(self):
        return self._residual.begin()

    def plan(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        draft_probs: torch.Tensor,
    ) -> AdaptiveVerifyStep:
        num_reqs = len(batch.seq_lens)
        features, predictions = self.estimator.predict(draft_probs)
        self._stale.drain()
        tier = self._choose_tier(batch=batch, num_reqs=num_reqs)
        self._stale.push(
            predictions=predictions,
            req_pool_indices_cpu=batch.req_pool_indices_cpu,
            generation_cpu=self._req_generation[batch.req_pool_indices_cpu],
        )
        self._num_planned += 1
        self._num_full_tokens += num_reqs * self.num_draft_tokens
        if tier is None:
            self._num_scheduled_tokens += num_reqs * self.num_draft_tokens
            return AdaptiveVerifyStep(
                verify_lens=self._full_verify_lens[:num_reqs],
                features=features,
                predictions=predictions,
                num_reqs=num_reqs,
                verify_cost_ms=self._uniform_cost(num_reqs),
            )
        budget = min(tier, num_reqs * self.num_draft_tokens) - num_reqs
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=predictions, budget=budget, cfg=self._schedule_cfg
        )
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=verify_lens, graph_num_tokens=tier
        )
        verify_input.ragged_verify_layout = layout
        verify_input.ragged_forward = self._pack(
            verify_input=verify_input, batch=batch, layout=layout
        )
        verify_input.retrieve_next_token = self._truncate_chain(
            retrieve_next_token=verify_input.retrieve_next_token,
            verify_lens=layout.verify_lens,
        )
        self._num_scheduled_tokens += num_reqs + budget
        return AdaptiveVerifyStep(
            verify_lens=layout.verify_lens,
            features=features,
            predictions=predictions,
            num_reqs=num_reqs,
            verify_cost_ms=float(self._tier_costs[(self._slots(num_reqs), tier)]),
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

    def _slots(self, num_reqs: int) -> int:
        return self._capture_bs[bisect.bisect_left(self._capture_bs, num_reqs)]

    def _uniform_cost(self, num_reqs: int) -> float:
        index = bisect.bisect_left(self._capture_bs, num_reqs)
        return float(self._uniform_costs[index])

    def _choose_tier(self, *, batch: ScheduleBatch, num_reqs: int) -> Optional[int]:
        slots = self._slots(num_reqs)
        full_tokens = num_reqs * self.num_draft_tokens
        tiers = [t for t in self._tiers_by_slots[slots] if num_reqs < t < full_tokens]
        if self._forced_tier > 0:
            return tiers[min(self._forced_tier, len(tiers)) - 1] if tiers else None
        residual = self._residual.lookup(num_reqs)
        if residual is None or not tiers:
            return None
        rows = batch.req_pool_indices_cpu.to(torch.int64)
        survival = self._stale.survival(
            req_pool_indices_cpu=rows.numpy(),
            generation_cpu=self._req_generation[rows].numpy(),
        )
        gains = np.concatenate(
            ([0.0], np.cumsum(np.sort(survival.ravel())[::-1], dtype=np.float64))
        )
        best_tier = None
        best_rate = (num_reqs + gains[-1]) / (residual + self._uniform_cost(num_reqs))
        for tier in tiers:
            rate = (num_reqs + gains[tier - num_reqs]) / (
                residual + self._tier_costs[(slots, tier)]
            )
            if rate > best_rate:
                best_tier, best_rate = tier, rate
        return best_tier

    def _pack(
        self,
        *,
        verify_input: EagleVerifyInput,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
    ) -> RaggedVerifyForward:
        req_ids, within, valid = CompactRowIndex.execute(
            verify_lens=layout.verify_lens,
            padded_total=layout.graph_num_tokens,
            device=self.device,
        )
        safe_req = req_ids.clamp(max=layout.bs - 1)
        cols = safe_req * self.num_draft_tokens + within
        window_cols = torch.where(valid, cols, -1)

        def gather(strided: torch.Tensor) -> torch.Tensor:
            return torch.where(valid, strided[cols], 0)

        out_cache_loc = verify_input.prepared_out_cache_loc
        if out_cache_loc is None:
            from sglang.kernels.ops.speculative.cache_locs import (
                assign_extend_cache_locs_uniform_func,
            )

            out_cache_loc = assign_extend_cache_locs_uniform_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                batch_size=layout.bs,
                draft_token_num=self.num_draft_tokens,
                device=self.device,
            )
        mrope_positions = verify_input.prepared_mrope_positions
        if mrope_positions is not None:
            mrope_positions = torch.where(valid, mrope_positions[:, cols], 0)
        elif self._model_runner.model_config.model_is_mrope:
            positions = gather(verify_input.positions)
            mrope_positions = positions.to(torch.int64).unsqueeze(0).repeat(3, 1)
        return RaggedVerifyForward(
            input_ids=gather(verify_input.draft_token),
            positions=gather(verify_input.positions),
            out_cache_loc=gather(out_cache_loc),
            window_cols=window_cols,
            mrope_positions=mrope_positions,
            strided_rows=self._strided_rows(layout=layout),
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

    def _strided_rows(self, *, layout: RaggedVerifyLayout) -> torch.Tensor:
        steps = torch.arange(self.num_draft_tokens, device=self.device)
        last = (layout.verify_lens.to(torch.int64) - 1).unsqueeze(1)
        starts = layout.qo_indptr_device[:-1].to(torch.int64).unsqueeze(1)
        return (starts + torch.minimum(steps.unsqueeze(0), last)).reshape(-1)

    def _log_state(self) -> None:
        if get_parallel().tp_group.rank_in_group != 0:
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
