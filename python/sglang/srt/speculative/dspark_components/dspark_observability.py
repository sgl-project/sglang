from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Callable, ContextManager, Iterator, Optional, Union

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.kv_canary.runner.future_tensor import FutureTensors
from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus

logger = logging.getLogger(__name__)

_NULL_SEGMENT = nullcontext()

ALL_COMPONENTS_TOKEN = "all"


class InfoComponent(str, Enum):
    CORE = "core"
    STEP_CPU_TIME = "step_cpu_time"
    STEP_GPU_TIME = "step_gpu_time"
    DRAFT_GPU_TIME = "draft_gpu_time"
    TARGET_VERIFY_GPU_TIME = "target_verify_gpu_time"
    REQS = "reqs"


class InfoSegment(str, Enum):
    STEP = "step"
    DRAFT = "draft"
    TARGET_VERIFY = "target_verify"


INFO_DUMP_MAX_RECORDS = 200_000
INFO_DUMP_MAX_STEP_CPU_SECONDS = 1.0


def resolve_enabled_components() -> set[InfoComponent]:
    """Components enabled via env: SGLANG_DSPARK_DEBUG_DUMP tokens, plus the
    published SPS-profiling switch SGLANG_DSPARK_ENABLE_SPS_RECORD=1, which is
    an alias for the core,step_cpu_time components the SPS table fit needs."""
    components = resolve_components(envs.SGLANG_DSPARK_DEBUG_DUMP.get())
    if envs.SGLANG_DSPARK_ENABLE_SPS_RECORD.get():
        components |= {InfoComponent.CORE, InfoComponent.STEP_CPU_TIME}
    return components


def resolve_components(raw: tuple[str, ...]) -> set[InfoComponent]:
    tokens = {token.strip() for token in raw if token.strip()}
    if not tokens:
        return set()
    if ALL_COMPONENTS_TOKEN in tokens:
        return set(InfoComponent)
    try:
        return {InfoComponent(token) for token in tokens}
    except ValueError as exc:
        valid = [component.value for component in InfoComponent]
        raise ValueError(
            f"Invalid SGLANG_DSPARK_DEBUG_DUMP token in {sorted(tokens)}; "
            f"valid: {valid} or '{ALL_COMPONENTS_TOKEN}'."
        ) from exc


class ReqDetail(msgspec.Struct, omit_defaults=True):
    req_pool_index: int
    prefix_len: int
    verify_len: int
    acc_len: int
    correct_drafts: int
    cap_trim: int
    bonus_token: int
    draft_tokens: list[int]
    rid: Optional[str] = None
    confidence: Optional[list[float]] = None
    survival: Optional[list[float]] = None


class DecodeStepRecord(msgspec.Struct, omit_defaults=True):
    forward_ct: int
    bs: int = -1
    mode: str = ""
    budget: Optional[int] = None
    lag_steps: Optional[int] = None
    num_running_reqs: int = -1
    num_verify_tokens: int = -1
    verify_tokens_local: int = -1
    verify_tokens_dp_synced: int = -1
    verify_tokens_graph_key: int = -1
    predicted_step_ms: Optional[float] = None
    predicted_theta: Optional[float] = None
    step_cpu_ms: Optional[float] = None
    step_gpu_ms: Optional[float] = None
    draft_gpu_ms: Optional[float] = None
    target_verify_gpu_ms: Optional[float] = None
    reqs: Optional[list[ReqDetail]] = None


class DecodeStepObservation(msgspec.Struct):
    forward_ct: int
    bs: int
    mode: str
    budget: Optional[int]
    lag_steps: Optional[int]
    num_verify_tokens: int
    verify_tokens_local: int
    verify_tokens_dp_synced: int
    verify_tokens_graph_key: int
    predicted_step_ms: Optional[float]
    predicted_theta: Optional[float]
    verify_lens: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor]
    req_pool_indices: torch.Tensor
    prefix_lens: torch.Tensor
    draft_tokens: torch.Tensor
    bonus_tokens: torch.Tensor
    correct_len: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    rids: Optional[list[str]]


class _PendingStep(msgspec.Struct):
    forward_ct: int
    bs: int
    mode: str
    budget: Optional[int]
    lag_steps: Optional[int]
    num_verify_tokens: int
    verify_tokens_local: int
    verify_tokens_dp_synced: int
    verify_tokens_graph_key: int
    predicted_step_ms: Optional[float]
    predicted_theta: Optional[float]
    step_cpu_ms: Optional[float]
    rids: Optional[list[str]]
    future: Optional[FutureTensors]
    segment_events: dict[InfoSegment, tuple[torch.cuda.Event, torch.cuda.Event]]


class DsparkInfoDumper:
    def __init__(
        self,
        *,
        components: set[Union[InfoComponent, str]],
        gamma: int,
        verify_num_draft_tokens: int,
        tp_rank: int,
        device: torch.device,
        mode_value: str,
        sps_report_interval: int = 0,
        max_records: int = INFO_DUMP_MAX_RECORDS,
        max_step_cpu_seconds: float = INFO_DUMP_MAX_STEP_CPU_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.tp_rank = int(tp_rank)
        self.device = device
        self.mode_value = mode_value
        self._clock = clock
        self._max_step_cpu_seconds = max_step_cpu_seconds

        self._components: set[InfoComponent] = {
            InfoComponent(component) for component in components
        }
        self._sps_report_interval = int(sps_report_interval)
        if self._sps_report_interval > 0:
            self._components.add(InfoComponent.STEP_GPU_TIME)
        self.enabled = bool(self._components) and self.tp_rank == 0
        self._sps_window: list[tuple[float, float]] = []
        self._sps_mismatched = 0

        self._records: deque[DecodeStepRecord] = deque(maxlen=max_records)
        self._pending: Optional[_PendingStep] = None
        self._prev_stamp: Optional[float] = None

        self._d2h_stream: Optional[torch.cuda.Stream] = None
        if self.enabled and InfoComponent.REQS in self._components:
            self._d2h_stream = torch.cuda.Stream(device=device)

        self._current_segments: dict[
            InfoSegment, tuple[torch.cuda.Event, torch.cuda.Event]
        ] = {}
        self._open_segments: dict[InfoSegment, torch.cuda.Event] = {}

    def begin_step(self) -> None:
        if not self.enabled:
            return
        self._current_segments = {}
        self._open_segments = {}
        if InfoComponent.STEP_GPU_TIME in self._components:
            self._open_segment(InfoSegment.STEP)

    def segment(self, name: Union[InfoSegment, str]) -> ContextManager[None]:
        if not self.enabled:
            return _NULL_SEGMENT
        segment = InfoSegment(name)
        if not self._segment_enabled(segment):
            return _NULL_SEGMENT
        return self._active_segment(segment)

    @contextmanager
    def _active_segment(self, segment: InfoSegment) -> Iterator[None]:
        self._open_segment(segment)
        try:
            yield
        finally:
            self._close_segment(segment)

    def observe_decode_step(self, obs: DecodeStepObservation) -> None:
        if not self.enabled:
            return
        if InfoComponent.STEP_GPU_TIME in self._components:
            self._close_segment(InfoSegment.STEP)

        now = self._clock()
        step_cpu_ms = self._step_cpu_ms(now=now)
        self._drain_pending()

        future = (
            self._stage_reqs(obs) if InfoComponent.REQS in self._components else None
        )
        self._pending = _PendingStep(
            forward_ct=int(obs.forward_ct),
            bs=int(obs.bs),
            mode=obs.mode,
            budget=None if obs.budget is None else int(obs.budget),
            lag_steps=None if obs.lag_steps is None else int(obs.lag_steps),
            num_verify_tokens=int(obs.num_verify_tokens),
            verify_tokens_local=int(obs.verify_tokens_local),
            verify_tokens_dp_synced=int(obs.verify_tokens_dp_synced),
            verify_tokens_graph_key=int(obs.verify_tokens_graph_key),
            predicted_step_ms=obs.predicted_step_ms,
            predicted_theta=obs.predicted_theta,
            step_cpu_ms=step_cpu_ms,
            rids=obs.rids,
            future=future,
            segment_events=self._current_segments,
        )
        self._current_segments = {}
        self._prev_stamp = now

    def note_non_decode_step(self) -> None:
        if not self.enabled:
            return
        self._drain_pending()
        self._prev_stamp = None
        self._current_segments = {}
        self._open_segments = {}

    def flush(self) -> None:
        if not self.enabled:
            return
        self._drain_pending()

    def clear(self) -> None:
        self._records.clear()
        self._pending = None
        self._prev_stamp = None
        self._current_segments = {}
        self._open_segments = {}
        self._sps_window = []
        self._sps_mismatched = 0

    def dump(self) -> Optional[dict]:
        if not self.enabled:
            return None
        self.flush()
        return {
            "mode": self.mode_value,
            "gamma": self.gamma,
            "verify_num_draft_tokens": self.verify_num_draft_tokens,
            "components": sorted(component.value for component in self._components),
            "records": [msgspec.to_builtins(record) for record in self._records],
        }

    def _segment_enabled(self, segment: InfoSegment) -> bool:
        if segment is InfoSegment.STEP:
            return InfoComponent.STEP_GPU_TIME in self._components
        if segment is InfoSegment.DRAFT:
            return InfoComponent.DRAFT_GPU_TIME in self._components
        if segment is InfoSegment.TARGET_VERIFY:
            return InfoComponent.TARGET_VERIFY_GPU_TIME in self._components
        return False

    def _open_segment(self, segment: InfoSegment) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._open_segments[segment] = start

    def _close_segment(self, segment: InfoSegment) -> None:
        start = self._open_segments.pop(segment, None)
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._current_segments[segment] = (start, end)

    def _stage_reqs(self, obs: DecodeStepObservation) -> Optional[FutureTensors]:
        tensors: dict[str, torch.Tensor] = {
            "req_pool_indices": obs.req_pool_indices,
            "prefix_lens": obs.prefix_lens,
            "draft_tokens": obs.draft_tokens,
            "bonus_tokens": obs.bonus_tokens,
            "correct_len": obs.correct_len,
            "cap_trim_lens": obs.cap_trim_lens,
            "commit_lens": obs.commit_lens,
        }
        if obs.verify_lens is not None:
            tensors["verify_lens"] = obs.verify_lens
        if obs.confidence is not None:
            tensors["confidence"] = obs.confidence
        return FutureTensors.device_to_host(tensors, d2h_stream=self._d2h_stream)

    def _drain_pending(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return

        record = DecodeStepRecord(forward_ct=pending.forward_ct)
        if InfoComponent.CORE in self._components:
            record.bs = pending.bs
            record.mode = pending.mode
            record.budget = pending.budget
            record.lag_steps = pending.lag_steps
            record.num_running_reqs = pending.bs
            record.num_verify_tokens = pending.num_verify_tokens
            record.verify_tokens_local = pending.verify_tokens_local
            record.verify_tokens_dp_synced = pending.verify_tokens_dp_synced
            record.verify_tokens_graph_key = pending.verify_tokens_graph_key
            record.predicted_step_ms = pending.predicted_step_ms
            record.predicted_theta = pending.predicted_theta
        if InfoComponent.STEP_CPU_TIME in self._components:
            record.step_cpu_ms = pending.step_cpu_ms
        if InfoComponent.STEP_GPU_TIME in self._components:
            record.step_gpu_ms = self._segment_ms(pending, InfoSegment.STEP)
        if InfoComponent.DRAFT_GPU_TIME in self._components:
            record.draft_gpu_ms = self._segment_ms(pending, InfoSegment.DRAFT)
        if InfoComponent.TARGET_VERIFY_GPU_TIME in self._components:
            record.target_verify_gpu_ms = self._segment_ms(
                pending, InfoSegment.TARGET_VERIFY
            )
        if InfoComponent.REQS in self._components and pending.future is not None:
            record.reqs = self._build_reqs(
                host=pending.future.wait(), bs=pending.bs, rids=pending.rids
            )
        elif pending.future is not None:
            pending.future.wait()

        self._records.append(record)
        if self._sps_report_interval > 0:
            self._report_sps_prediction(pending=pending, step_gpu_ms=record.step_gpu_ms)

    def _report_sps_prediction(
        self, *, pending: _PendingStep, step_gpu_ms: Optional[float]
    ) -> None:
        predicted = pending.predicted_step_ms
        if predicted is None or step_gpu_ms is None:
            return
        matched = (
            pending.budget is not None
            and pending.bs + pending.budget == pending.num_verify_tokens
        )
        if not matched:
            self._sps_mismatched += 1
            return
        self._sps_window.append((predicted, step_gpu_ms))
        if len(self._sps_window) < self._sps_report_interval:
            return

        predictions = [p for p, _ in self._sps_window]
        actuals = [a for _, a in self._sps_window]
        abs_err = [abs(p - a) for p, a in self._sps_window]
        rel_err = [abs(p - a) / a * 100 for p, a in self._sps_window if a > 0]
        total = len(self._sps_window) + self._sps_mismatched
        logger.info(
            "DSpark SPS prediction: n=%d  mean predicted=%.3fms  mean actual=%.3fms  "
            "MAE=%.3fms  median rel-err=%.1f%%  mean bias(pred-actual)=%+.3fms  "
            "M_mismatch_rate=%.1f%% (%d/%d)",
            len(self._sps_window),
            statistics.fmean(predictions),
            statistics.fmean(actuals),
            statistics.fmean(abs_err),
            statistics.median(rel_err) if rel_err else float("nan"),
            statistics.fmean([p - a for p, a in self._sps_window]),
            self._sps_mismatched / total * 100 if total else 0.0,
            self._sps_mismatched,
            total,
        )
        self._sps_window = []
        self._sps_mismatched = 0

    def _step_cpu_ms(self, *, now: float) -> Optional[float]:
        prev = self._prev_stamp
        if prev is None:
            return None
        step_cpu = now - prev
        if not (0.0 < step_cpu <= self._max_step_cpu_seconds):
            return None
        return round(step_cpu * 1000.0, 4)

    def _segment_ms(
        self, pending: _PendingStep, segment: InfoSegment
    ) -> Optional[float]:
        events = pending.segment_events.get(segment)
        if events is None:
            return None
        start, end = events
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        if elapsed_ms > self._max_step_cpu_seconds * 1000.0:
            return None
        return round(elapsed_ms, 4)

    def _build_reqs(
        self, *, host: dict, bs: int, rids: Optional[list[str]]
    ) -> list[ReqDetail]:
        req_ids = host["req_pool_indices"].tolist()
        prefixes = host["prefix_lens"].tolist()
        draft_rows = host["draft_tokens"].tolist()
        bonus = host["bonus_tokens"].tolist()
        correct = host["correct_len"].tolist()
        cap_trim = host["cap_trim_lens"].tolist()
        commit = host["commit_lens"].tolist()
        verify_lens = host["verify_lens"].tolist() if "verify_lens" in host else None
        if "confidence" in host:
            conf_host = host["confidence"].float()
            conf_rows = conf_host.tolist()
            survival_rows = torch.cumprod(conf_host, dim=1).tolist()
        else:
            conf_rows = None
            survival_rows = None

        reqs: list[ReqDetail] = []
        for row in range(bs):
            verify_len = (
                self.verify_num_draft_tokens
                if verify_lens is None
                else int(verify_lens[row])
            )
            reqs.append(
                ReqDetail(
                    rid=None if rids is None else rids[row],
                    req_pool_index=int(req_ids[row]),
                    prefix_len=int(prefixes[row]),
                    verify_len=verify_len,
                    acc_len=int(commit[row]),
                    correct_drafts=int(correct[row]),
                    cap_trim=int(cap_trim[row]),
                    bonus_token=int(bonus[row]),
                    draft_tokens=[int(t) for t in draft_rows[row]],
                    confidence=(
                        None
                        if conf_rows is None
                        else [round(float(p), 4) for p in conf_rows[row]]
                    ),
                    survival=(
                        None
                        if survival_rows is None
                        else [round(float(p), 4) for p in survival_rows[row]]
                    ),
                )
            )
        return reqs


EPS_PROB = 1e-8


def _format_float(value: float, digits: int = 4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


class PerPositionConfidenceMetrics:

    def __init__(
        self,
        *,
        gamma: int,
        device: torch.device,
        num_coarse_bins: int = 15,
        num_fine_bins: int = 1024,
    ) -> None:
        self.gamma = int(gamma)
        self.num_coarse_bins = int(num_coarse_bins)
        self.num_fine_bins = int(num_fine_bins)
        self.coarse_count = torch.zeros(
            (self.gamma, self.num_coarse_bins), dtype=torch.float64, device=device
        )
        self.coarse_pred = torch.zeros_like(self.coarse_count)
        self.coarse_target = torch.zeros_like(self.coarse_count)
        self.fine_pos = torch.zeros(
            (self.gamma, self.num_fine_bins), dtype=torch.float64, device=device
        )
        self.fine_neg = torch.zeros_like(self.fine_pos)
        self.brier_num = torch.zeros(self.gamma, dtype=torch.float64, device=device)

    def update(self, *, survival: torch.Tensor, prefix_mask: torch.Tensor) -> None:
        assert survival.shape == prefix_mask.shape
        assert survival.dim() == 2 and survival.shape[1] == self.gamma

        probs = survival.to(torch.float64).clamp(EPS_PROB, 1.0 - EPS_PROB)
        targets = prefix_mask.to(torch.float64)
        bs = probs.shape[0]

        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        weights = torch.ones_like(probs_flat)
        pos_idx = (
            torch.arange(self.gamma, device=probs.device)
            .view(1, -1)
            .expand(bs, self.gamma)
            .reshape(-1)
        )

        coarse_idx = (
            (probs_flat * self.num_coarse_bins)
            .long()
            .clamp_(0, self.num_coarse_bins - 1)
        )
        flat_coarse = pos_idx * self.num_coarse_bins + coarse_idx
        self.coarse_count.view(-1).scatter_add_(0, flat_coarse, weights)
        self.coarse_pred.view(-1).scatter_add_(0, flat_coarse, probs_flat)
        self.coarse_target.view(-1).scatter_add_(0, flat_coarse, targets_flat)

        fine_idx = (
            (probs_flat * self.num_fine_bins).long().clamp_(0, self.num_fine_bins - 1)
        )
        flat_fine = pos_idx * self.num_fine_bins + fine_idx
        self.fine_pos.view(-1).scatter_add_(0, flat_fine, targets_flat)
        self.fine_neg.view(-1).scatter_add_(0, flat_fine, 1.0 - targets_flat)

        self.brier_num.add_((probs - targets).pow(2).sum(dim=0))

    @staticmethod
    def _auroc_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
        total_pos = float(pos_hist.sum())
        total_neg = float(neg_hist.sum())
        if total_pos <= 0.0 or total_neg <= 0.0:
            return float("nan")
        cum_neg = torch.cumsum(neg_hist, dim=0)
        cum_neg_before = cum_neg - neg_hist
        pair = (pos_hist * cum_neg_before).sum() + 0.5 * (pos_hist * neg_hist).sum()
        return float(pair) / (total_pos * total_neg)

    def compute(self) -> list[dict]:
        coarse_count = self.coarse_count.cpu()
        coarse_pred = self.coarse_pred.cpu()
        coarse_target = self.coarse_target.cpu()
        fine_pos = self.fine_pos.cpu()
        fine_neg = self.fine_neg.cpu()
        brier_num = self.brier_num.cpu()

        out: list[dict] = []
        for pos in range(self.gamma):
            weights = coarse_count[pos]
            total = float(weights.sum())
            if total <= 1e-12:
                out.append(
                    {
                        "position": pos,
                        "total_weight": 0.0,
                        "ece": float("nan"),
                        "auc": float("nan"),
                        "brier": float("nan"),
                        "pred_mean": float("nan"),
                        "target_mean": float("nan"),
                        "reliability": [],
                    }
                )
                continue

            denom = weights.clamp_min(1e-12)
            avg_pred = coarse_pred[pos] / denom
            avg_target = coarse_target[pos] / denom
            bin_err = (avg_pred - avg_target).abs()
            ece = float((bin_err * weights).sum()) / total
            auc = self._auroc_from_hist(fine_pos[pos], fine_neg[pos])
            brier = float(brier_num[pos]) / total
            reliability = []
            for bin_idx in range(self.num_coarse_bins):
                weight = float(weights[bin_idx])
                if weight <= 0.0:
                    continue
                reliability.append(
                    {
                        "bin": bin_idx,
                        "range": [
                            bin_idx / self.num_coarse_bins,
                            (bin_idx + 1) / self.num_coarse_bins,
                        ],
                        "avg_pred": float(avg_pred[bin_idx]),
                        "avg_target": float(avg_target[bin_idx]),
                        "weight": weight,
                    }
                )
            out.append(
                {
                    "position": pos,
                    "total_weight": total,
                    "ece": ece,
                    "auc": auc,
                    "brier": brier,
                    "pred_mean": float(coarse_pred[pos].sum()) / total,
                    "target_mean": float(coarse_target[pos].sum()) / total,
                    "reliability": reliability,
                }
            )
        return out

    def format_table(self) -> str:
        rows = self.compute()
        header = (
            f"{'pos':>3} {'count':>12} {'pred':>8} {'target':>8} "
            f"{'ece':>8} {'auc':>8} {'brier':>8}"
        )
        lines = [
            "DSpark confidence-head per-position calibration "
            "(cumprod survival vs leading-correct-prefix)",
            header,
        ]
        for row in rows:
            lines.append(
                f"{row['position']:>3} {row['total_weight']:>12.0f} "
                f"{_format_float(row['pred_mean']):>8} "
                f"{_format_float(row['target_mean']):>8} "
                f"{_format_float(row['ece']):>8} "
                f"{_format_float(row['auc']):>8} "
                f"{_format_float(row['brier']):>8}"
            )
        return "\n".join(lines)


class ConfidenceMetricsProbe:

    def __init__(
        self,
        *,
        gamma: int,
        verify_num_draft_tokens: int,
        tp_rank: int,
        print_every: int = 256,
    ) -> None:
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.tp_rank = int(tp_rank)
        self.print_every = int(print_every)
        self._metrics: Optional[PerPositionConfidenceMetrics] = None
        self._step_ct: int = 0
        self._compact_warned: bool = False

    def maybe_observe(
        self,
        *,
        carries_confidence: bool,
        is_compact_mode: bool,
        confidence_raw: Optional[torch.Tensor],
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor,
        bs: int,
    ) -> None:
        if not envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.get():
            return
        if self.tp_rank != 0:
            return
        if not carries_confidence:
            return
        if is_compact_mode:
            if not self._compact_warned:
                logger.warning(
                    "SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS is ignored under "
                    "SGLANG_RAGGED_VERIFY_MODE=compact (padded verify rows corrupt the "
                    "per-position prefix label); run cap-accept or static to measure it."
                )
                self._compact_warned = True
            return
        if confidence_raw is None:
            return

        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        num_correct_drafts, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=verify_ids_2d,
            target_predict=target_predict,
        )
        positions = torch.arange(self.gamma, device=confidence_raw.device).view(1, -1)
        prefix_mask = (positions < num_correct_drafts.view(-1, 1)).to(torch.float32)
        survival = torch.cumprod(torch.sigmoid(confidence_raw.float()), dim=1)

        if self._metrics is None:
            self._metrics = PerPositionConfidenceMetrics(
                gamma=self.gamma, device=confidence_raw.device
            )
        self._metrics.update(survival=survival, prefix_mask=prefix_mask)
        self._step_ct += 1
        if self._step_ct % self.print_every == 0:
            logger.info("%s", self._metrics.format_table())
