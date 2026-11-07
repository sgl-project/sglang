"""Causal DSPark simulation inputs, progress and staged worker timing.

The scheduler supplies the actual local batches. A cost adapter owns measured
phase costs and a separate progress model owns native committed counts. Neither
acceptance observations from the current call nor a speculative speedup are
accepted as execution inputs. This module imports no SGLang or device runtime.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from sglang_simulator.time_predictor.base import PredictorError

STEP_ATTRIBUTE = "_sglang_simulator_dspark_step"
CONTRACT = "sglang-dspark-step-v1"
_DECODE_PHASES = (
    "worker_preamble",
    "proposal",
    "target_verify",
    "accept_commit_publication",
    "worker_tail",
)
_PREFILL_PHASES = (
    "worker_preamble",
    "target_prefill",
    "draft_kv_inject",
    "worker_tail",
)


def _fail(message, **details):
    raise PredictorError("invalid_dspark_step", message, **details)


def _integer(value, minimum, label):
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")


def _canonical_object(value, label):
    if type(value) is not str:
        _fail(f"{label} must be canonical JSON")
    try:
        parsed = json.loads(value)
        canonical = json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (ValueError, TypeError) as error:
        raise PredictorError("invalid_dspark_step", f"invalid {label}") from error
    if not isinstance(parsed, dict) or not parsed or value != canonical:
        _fail(f"{label} must be a nonempty canonical JSON object")
    return parsed


def _read_canonical_budget(value, label):
    budget = _canonical_object(value, label).get("max_new_tokens")
    return budget if type(budget) is int else None


@lru_cache(maxsize=32)
def _cached_canonical_budget(value, label):
    # Retain only immutable validation metadata, never the parsed object.
    return _read_canonical_budget(value, label)


def _canonical_budget(value, label):
    # The bound limits retention only; larger or non-builtin inputs retain the
    # original validation/error path without hashing user-defined objects.
    if type(value) is str and type(label) is str and len(value) <= 1024 * 1024:
        return _cached_canonical_budget(value, label)
    return _read_canonical_budget(value, label)


@dataclass(frozen=True)
class DSparkTopology:
    tp_size: int
    attention_dp_size: int
    pp_size: int = 1

    def __post_init__(self):
        for label in ("tp_size", "attention_dp_size", "pp_size"):
            _integer(getattr(self, label), 1, label)
        if (self.tp_size, self.attention_dp_size, self.pp_size) not in {
            (2, 1, 1),
            (4, 1, 1),
            (2, 2, 1),
        }:
            raise PredictorError(
                "unsupported_topology", "DSPark layout has no simulator contract"
            )

    @property
    def attention_tp_size(self):
        return self.tp_size // self.attention_dp_size


@dataclass(frozen=True)
class DSparkRequest:
    rid: str
    target_prefix_len: int
    extend_len: int
    draft_valid_from: int
    draft_valid_to: int
    output_tokens_before: int
    max_new_tokens: int
    sampling_params_json: str
    emits_output: bool = True

    def __post_init__(self):
        if type(self.rid) is not str or not self.rid:
            _fail("DSPark request requires its native request ID")
        for name in (
            "target_prefix_len",
            "draft_valid_from",
            "draft_valid_to",
            "output_tokens_before",
        ):
            _integer(getattr(self, name), 0, name)
        _integer(self.extend_len, 1, "extend_len")
        _integer(self.max_new_tokens, 1, "max_new_tokens")
        if (
            self.draft_valid_from > self.draft_valid_to
            or self.draft_valid_to > self.target_prefix_len
            or self.output_tokens_before >= self.max_new_tokens
        ):
            _fail("DSPark request has inconsistent before-state")
        recipe_budget = _canonical_budget(
            self.sampling_params_json, "sampling_params_json"
        )
        if type(self.emits_output) is not bool:
            _fail("emits_output must reflect the native chunk completion flag")
        if recipe_budget is None or recipe_budget != self.max_new_tokens:
            _fail("DSPark output budget differs from its full sampling recipe")

    @property
    def remaining_output_tokens(self):
        return self.max_new_tokens - self.output_tokens_before

    @property
    def draft_history_tokens(self):
        # Available history is a causal range, not a target-prefix proxy.
        return self.draft_valid_to - self.draft_valid_from

    def bounded_draft_history_tokens(self, window_size):
        """Project a source read window from a contiguous causal KV range."""
        _integer(window_size, 1, "draft history window")
        if self.draft_valid_to != self.target_prefix_len:
            raise PredictorError(
                "data_unavailable", "draft read window has an unknown trailing KV gap"
            )
        return min(window_size, self.draft_history_tokens)


@dataclass(frozen=True)
class DSparkGroup:
    attention_dp_rank: int
    forward_mode: str
    requests: tuple[DSparkRequest, ...]
    # Exact source-selected path, including capture key, padding and fallback.
    execution_json: str

    def __post_init__(self):
        _integer(self.attention_dp_rank, 0, "attention_dp_rank")
        if self.forward_mode not in {"EXTEND", "MIXED", "TARGET_VERIFY", "IDLE"}:
            _fail("DSPark group requires its actual worker forward mode")
        if type(self.requests) is not tuple or any(
            type(request) is not DSparkRequest for request in self.requests
        ):
            _fail("DSPark group requests must be an immutable typed tuple")
        if (self.forward_mode == "IDLE") != (not self.requests):
            _fail("only an explicit idle group may have no local requests")
        _canonical_budget(self.execution_json, "execution_json")


@dataclass(frozen=True)
class DSparkStep:
    forward_id: int
    topology: DSparkTopology
    # All attention-DP groups, in rank order, including idle participants.
    groups: tuple[DSparkGroup, ...]
    # Complete pinned model/runtime/source identity, supplied before work.
    runtime_profile_json: str

    def __post_init__(self):
        _integer(self.forward_id, 0, "forward_id")
        if type(self.topology) is not DSparkTopology or type(self.groups) is not tuple:
            _fail("DSPark step requires typed topology and immutable groups")
        if any(type(group) is not DSparkGroup for group in self.groups) or tuple(
            group.attention_dp_rank for group in self.groups
        ) != tuple(range(self.topology.attention_dp_size)):
            _fail("DSPark step must retain every actual attention-DP partition")
        ids = [request.rid for group in self.groups for request in group.requests]
        if len(set(ids)) != len(ids) or not ids:
            _fail("DSPark step requires unique live requests across all groups")
        _canonical_budget(self.runtime_profile_json, "runtime_profile_json")

    @property
    def requests(self):
        return tuple(request for group in self.groups for request in group.requests)

    @property
    def required_phases(self):
        modes = {group.forward_mode for group in self.groups}
        if "MIXED" in modes or ("EXTEND" in modes and "TARGET_VERIFY" in modes):
            raise PredictorError(
                "unsupported_execution_profile",
                "heterogeneous DSPark phases need a source-specific coordination adapter",
            )
        phases = []
        if modes & {"EXTEND", "MIXED"}:
            phases.extend(_PREFILL_PHASES)
        if modes & {"TARGET_VERIFY", "MIXED"}:
            phases.extend(_DECODE_PHASES)
        return tuple(phases)


@dataclass(frozen=True)
class DSparkPhaseCost:
    phase: str
    seconds: float
    # The adapter must attribute every cost to exact SDK/PerfDB provenance.
    evidence_ref: str

    def __post_init__(self):
        if self.phase not in _PREFILL_PHASES + _DECODE_PHASES + ("whole_worker",):
            _fail("unknown DSPark worker cost boundary")
        if (
            type(self.seconds) not in (float, int)
            or not math.isfinite(self.seconds)
            or self.seconds <= 0
            or type(self.evidence_ref) is not str
            or not self.evidence_ref
        ):
            _fail("DSPark cost requires positive time and provenance")


@dataclass(frozen=True)
class DSparkStepTiming:
    forward_id: int
    # Sequential worker boundaries. Each cost is already coordinated over TP/DP;
    # the simulator must never multiply it by ranks or use max(sum(rank costs)).
    phases: tuple[DSparkPhaseCost, ...]

    def validate_for(self, step):
        whole_worker = (
            type(self.phases) is tuple
            and len(self.phases) == 1
            and type(self.phases[0]) is DSparkPhaseCost
            and self.phases[0].phase == "whole_worker"
        )
        if whole_worker:
            modes = {group.forward_mode for group in step.groups} - {"IDLE"}
            if step.topology != DSparkTopology(2, 2) or modes not in (
                {"EXTEND"},
                {"TARGET_VERIFY"},
            ):
                raise PredictorError(
                    "data_unavailable",
                    "whole-worker timing requires complete serialized TP2/ADP2 groups",
                )
        if (
            type(self.forward_id) is not int
            or self.forward_id != step.forward_id
            or type(self.phases) is not tuple
            or any(type(cost) is not DSparkPhaseCost for cost in self.phases)
            or not whole_worker
            and tuple(cost.phase for cost in self.phases) != step.required_phases
        ):
            raise PredictorError(
                "data_unavailable", "DSPark timing omits or reorders worker boundaries"
            )
        try:
            total = self.seconds
        except OverflowError as error:
            raise PredictorError(
                "invalid_dspark_step", "DSPark timing overflow"
            ) from error
        if not math.isfinite(total) or total <= 0:
            _fail("DSPark timing must be finite and positive")
        return self

    @property
    def seconds(self):
        return math.fsum(cost.seconds for cost in self.phases)


@dataclass(frozen=True)
class DSparkProgress:
    forward_id: int
    # Ordered native (request ID, commit count), never current measured counts.
    # The native scheduler owns visible output/finish clipping. A verify commit
    # may cross the output budget and still advances KV by its full count.
    committed_tokens: tuple[tuple[str, int], ...]
    model_ref: str

    def validate_for(self, step):
        if (
            type(self.forward_id) is not int
            or self.forward_id != step.forward_id
            or type(self.model_ref) is not str
            or not self.model_ref
            or type(self.committed_tokens) is not tuple
            or any(
                type(row) is not tuple or len(row) != 2 for row in self.committed_tokens
            )
            or tuple(row[0] for row in self.committed_tokens)
            != tuple(request.rid for request in step.requests)
        ):
            _fail("DSPark progress must match ordered requests and model identity")
        for request, (_, count) in zip(step.requests, self.committed_tokens):
            _integer(
                count, 0 if not request.emits_output else 1, "committed output count"
            )
            if not request.emits_output and count != 0:
                _fail("an intermediate prefill chunk cannot emit output")
            if count > request.extend_len:
                _fail("DSPark progress exceeds executed work")
        return self


class DSparkCostModel(Protocol):
    def predict_step(self, step: DSparkStep) -> DSparkStepTiming: ...


class DSparkProgressModel(Protocol):
    def predict_progress(self, step: DSparkStep) -> DSparkProgress: ...


@dataclass(frozen=True)
class DSparkStepResult:
    step: DSparkStep
    timing: DSparkStepTiming
    progress: DSparkProgress
    local_attention_dp_rank: int = 0
    # CPU simulation work to exclude from the offline scheduler wall term.
    # The native runtime supplies its complete replacement-worker wall cost.
    prediction_wall_seconds: float = 0.0

    def __post_init__(self):
        if type(self.step) is not DSparkStep:
            _fail("DSPark result requires its exact before-step")
        if (
            type(self.timing) is not DSparkStepTiming
            or type(self.progress) is not DSparkProgress
        ):
            _fail("DSPark result requires typed cost and independent progress")
        self.timing.validate_for(self.step)
        self.progress.validate_for(self.step)
        _integer(self.local_attention_dp_rank, 0, "local_attention_dp_rank")
        if self.local_attention_dp_rank >= self.step.topology.attention_dp_size:
            _fail("local attention-DP rank is outside the declared topology")
        if (
            type(self.prediction_wall_seconds) not in (float, int)
            or not math.isfinite(self.prediction_wall_seconds)
            or self.prediction_wall_seconds < 0
        ):
            _fail("invalid DSPark predictor wall time")

    @property
    def local_group(self):
        return self.step.groups[self.local_attention_dp_rank]


@dataclass(frozen=True)
class CommittedHistory:
    target_prefix_len: int
    draft_valid_from: int
    draft_valid_to: int

    def __post_init__(self):
        for label in ("target_prefix_len", "draft_valid_from", "draft_valid_to"):
            _integer(getattr(self, label), 0, label)
        if (
            not 0
            <= self.draft_valid_from
            <= self.draft_valid_to
            <= self.target_prefix_len
        ):
            _fail("invalid committed DSPark KV range")


class DSparkSession:
    """Transactional step execution; injected/committed history is the only state.

    Source-specific prefill/KV injection calls seed explicit ranges using
    ``record_history``. A long target prefix alone cannot seed draft history.
    Instances are per simulation run; request completion calls ``forget``.
    """

    def __init__(
        self, cost_model: DSparkCostModel, progress_model: DSparkProgressModel
    ):
        self.cost_model = cost_model
        self.progress_model = progress_model
        self._history = {}
        self._last_forward_id = -1

    def record_history(self, rid: str, history: CommittedHistory):
        if type(rid) is not str or not rid or type(history) is not CommittedHistory:
            _fail("history requires a request ID and explicit causal KV ranges")
        self._history[rid] = history

    def history_for(self, rid):
        if rid not in self._history:
            raise PredictorError(
                "data_unavailable",
                "DSPark draft history has no prior injection/commit",
                rid=rid,
            )
        return self._history[rid]

    def forget(self, rid):
        self._history.pop(rid, None)

    def predict(
        self, step: DSparkStep, *, local_attention_dp_rank=0
    ) -> DSparkStepResult:
        if type(step) is not DSparkStep or step.forward_id <= self._last_forward_id:
            _fail("DSPark step must advance the run's forward sequence")
        modes = {group.forward_mode for group in step.groups} - {"IDLE"}
        if modes not in ({"TARGET_VERIFY"}, {"EXTEND"}):
            raise PredictorError(
                "unsupported_forward_mode",
                "heterogeneous steps need their source coordination adapter",
            )
        prefill = modes == {"EXTEND"}
        for request in step.requests:
            expected = CommittedHistory(
                request.target_prefix_len,
                request.draft_valid_from,
                request.draft_valid_to,
            )
            if prefill:
                known = self._history.get(request.rid)
                if known is None or known.draft_valid_from > request.target_prefix_len:
                    prior = CommittedHistory(
                        request.target_prefix_len,
                        request.target_prefix_len,
                        request.target_prefix_len,
                    )
                else:
                    prior = CommittedHistory(
                        request.target_prefix_len,
                        known.draft_valid_from,
                        min(known.draft_valid_to, request.target_prefix_len),
                    )
            else:
                prior = self.history_for(request.rid)
            if prior != expected:
                _fail(
                    "DSPark before-work differs from committed causal history",
                    rid=request.rid,
                )
            if not prefill and request.draft_valid_to != request.target_prefix_len:
                raise PredictorError(
                    "data_unavailable",
                    "DSPark commit cannot bridge an unknown draft KV gap",
                    rid=request.rid,
                )
        # A missing cost must fail before generating progress or mutating history.
        started = time.perf_counter()
        timing = self.cost_model.predict_step(step)
        if type(timing) is not DSparkStepTiming:
            _fail("cost adapter must return typed complete DSPark timing")
        timing.validate_for(step)
        progress = (
            DSparkProgress(
                step.forward_id,
                tuple((r.rid, int(r.emits_output)) for r in step.requests),
                "native_prefill_bonus_v1",
            )
            if prefill
            else self.progress_model.predict_progress(step)
        )
        if type(progress) is not DSparkProgress:
            _fail("progress model must return typed DSPark progress")
        result = DSparkStepResult(
            step,
            timing,
            progress,
            local_attention_dp_rank,
            time.perf_counter() - started,
        )
        return result

    def commit(self, result: DSparkStepResult):
        step = result.step
        if step.forward_id <= self._last_forward_id:
            _fail("DSPark result was already committed")
        # Recheck before-state so an outstanding result cannot overwrite a
        # different successful step. predict() does not advance any run state.
        for group in step.groups:
            for request in group.requests:
                if group.forward_mode == "TARGET_VERIFY" and self.history_for(
                    request.rid
                ) != CommittedHistory(
                    request.target_prefix_len,
                    request.draft_valid_from,
                    request.draft_valid_to,
                ):
                    _fail("stale DSPark result differs from committed history")
        updated = dict(self._history)
        modes = {group.forward_mode for group in step.groups}
        prefill = "EXTEND" in modes
        for request, (_, count) in zip(step.requests, result.progress.committed_tokens):
            if not prefill and request.draft_valid_to != request.target_prefix_len:
                raise PredictorError(
                    "data_unavailable",
                    "DSPark commit cannot bridge an unknown draft KV gap",
                    rid=request.rid,
                )
            new_prefix = request.target_prefix_len + (
                request.extend_len if prefill else count
            )
            lower = (
                request.draft_valid_from
                if request.draft_valid_to == request.target_prefix_len
                else request.target_prefix_len
            )
            updated[request.rid] = CommittedHistory(new_prefix, lower, new_prefix)
        self._history = updated
        self._last_forward_id = step.forward_id

    def execute(self, step: DSparkStep, *, local_attention_dp_rank=0):
        result = self.predict(step, local_attention_dp_rank=local_attention_dp_rank)
        self.commit(result)
        return result

    def clear(self):
        self._history.clear()
        # Preserve the forward sequence across a native cache reset.


def dspark_result_from_native(result):
    """Consume only an explicitly attached, immutable simulated step result."""
    value = getattr(result, STEP_ATTRIBUTE, None)
    if value is not None and type(value) is not DSparkStepResult:
        _fail("native result has invalid DSPark simulation metadata")
    return value
