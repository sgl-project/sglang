# SPDX-License-Identifier: Apache-2.0
"""DPCache: calibrated key-timestep caching for training-free acceleration.

Adaptation of DPCache (https://arxiv.org/abs/2602.22654). An offline
calibration run records the final transformer-block feature (the input of the
output norm) at every denoising step. A dynamic program then picks the
``K`` steps that run the full block stack; every other step extrapolates the
final feature from the two latest full steps and runs only the output head.

Differences from the paper and its reference code, on purpose:

* The predictor is first order: ``h_j + (h_j - h_i) / (j - i) * (t - j)``
  from the two latest *full* steps ``i < j``. Predictions never become anchors,
  and the arithmetic runs in the feature dtype (BF16 only, for now) with the
  same operation order in calibration and inference.
* Because the prediction at ``t`` depends on exactly the two retained anchors,
  the calibrated cost of a key triple ``i < j < k`` is
  ``C[i, j, k] = sum_{j < t < k} mean|predict(h_i, h_j, t) - h_t|`` and the
  planner is an exact DP over (previous, current) key pairs. The reference
  planner keeps one predecessor per (budget, current key), which is not
  optimal for triple-dependent costs.
* Only steps that the runtime actually predicts are scored. The terminal
  sentinel ``T`` is an endpoint only; no synthetic feature at ``T`` is
  invented or scored (the paper's proxy includes the endpoint).

The schedule is a JSON artifact (never pickle) with the metadata needed to
reject a mismatched request before inference.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from typing import Any, Callable, Iterable, Mapping, Sequence

import msgspec
import torch

SCHEDULE_SCHEMA = "sglang-dpcache-schedule-v2"
PREDICTOR = {
    "name": "final-feature-linear-extrapolation",
    "order": 1,
    "arithmetic": "feature-dtype-elementwise-v1",
}
OBJECTIVE = "pact-mean-abs-final-feature-predicted-steps-only-v1"
DEFAULT_MANDATORY_FULL_STEPS = (0, 1, 2)


def predict_feature(
    previous: torch.Tensor,
    last: torch.Tensor,
    previous_step: int,
    last_step: int,
    step: int,
) -> torch.Tensor:
    """Extrapolate the final feature at ``step`` from two full steps.

    Every intermediate stays in the feature dtype. Calibration must call this
    same function so that scored and served predictions are identical.
    """
    if not previous_step < last_step < step:
        raise ValueError(
            f"need previous_step < last_step < step, got "
            f"{previous_step}, {last_step}, {step}"
        )
    slope = (last - previous) / (last_step - previous_step)
    return last + slope * (step - last_step)


# --------------------------------------------------------------------------
# Runtime state
# --------------------------------------------------------------------------


class DPCacheState:
    """Anchors for one request and one CFG branch.

    Full steps must call :meth:`record`; predicted steps call :meth:`predict`.
    Steps must arrive in increasing order, and the two latest recorded full
    features are the only inputs to a prediction.
    """

    def __init__(self, full_steps: Iterable[int], num_steps: int):
        self.full_steps = frozenset(full_steps)
        self.num_steps = num_steps
        self._anchors: list[tuple[int, torch.Tensor]] = []
        self._last_step = -1
        self.num_full = 0
        self.num_predicted = 0

    def _advance(self, step: int) -> None:
        if not 0 <= step < self.num_steps:
            raise ValueError(f"DPCache step {step} outside [0, {self.num_steps})")
        if step <= self._last_step:
            raise RuntimeError(
                f"DPCache steps must increase: got {step} after {self._last_step}"
            )
        self._last_step = step

    def is_full_step(self, step: int) -> bool:
        return step in self.full_steps

    def record(self, step: int, feature: torch.Tensor) -> None:
        if feature.dtype != torch.bfloat16:
            raise RuntimeError(
                f"DPCache supports BF16 features only, got {feature.dtype}"
            )
        if not self.is_full_step(step):
            raise RuntimeError(f"DPCache step {step} is not a scheduled full step")
        self._advance(step)
        self._anchors = [*self._anchors[-1:], (step, feature.detach().clone())]
        self.num_full += 1

    def predict(self, step: int) -> torch.Tensor:
        if self.is_full_step(step):
            raise RuntimeError(f"DPCache step {step} is a scheduled full step")
        self._advance(step)
        if len(self._anchors) < 2:
            raise RuntimeError(f"DPCache step {step} has fewer than two full anchors")
        (previous_step, previous), (last_step, last) = self._anchors
        self.num_predicted += 1
        return predict_feature(previous, last, previous_step, last_step, step)


class DPCacheCalibrationState:
    """Scores PACT errors online for one request and one CFG branch.

    Every step is a full step. Each recorded final feature is scored against
    the earlier anchor pairs as soon as it arrives, so a calibration request
    never ships its features out of the worker; only the ``(T, T, T)`` error
    tensor leaves. Features no later step can use are dropped.
    """

    def __init__(self, num_steps: int, max_gap: int | None = None):
        _validate_max_gap(max_gap)
        self.num_steps = num_steps
        self.max_gap = max_gap
        self.errors = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
        self._features: dict[int, torch.Tensor] = {}
        self.num_full = 0
        self.num_predicted = 0

    def is_full_step(self, step: int) -> bool:
        return True

    def record(self, step: int, feature: torch.Tensor) -> None:
        if feature.dtype != torch.bfloat16:
            raise RuntimeError(
                f"DPCache supports BF16 features only, got {feature.dtype}"
            )
        if step != self.num_full:
            raise RuntimeError(
                f"DPCache calibration needs every step in order: got {step} "
                f"after {self.num_full} steps"
            )
        self._features[step] = feature.detach().clone()
        _score_step(self.errors, self._features, step, self.max_gap)
        if self.max_gap is not None:
            # a later step t only uses anchors i > t - 2 * max_gap
            self._features.pop(step - 2 * self.max_gap, None)
        self.num_full += 1

    def predict(self, step: int) -> torch.Tensor:
        raise RuntimeError("DPCache calibration runs every step")

    @property
    def complete(self) -> bool:
        return self.num_full == self.num_steps


class DPCacheMixin:
    """DiT hook that runs or predicts the transformer block stack.

    Wired into ``CachableDiT`` (see ``runtime/models/dits/base.py``). A model
    opts in by setting ``_supports_dpcache = True`` and routing its block stack
    through :meth:`dpcache_blocks`; the output head must stay outside
    ``run_blocks`` so a predicted step still applies it with the current
    timestep. ``DenoisingStage`` rejects DPCache for any other model.
    """

    _supports_dpcache: bool = False

    def dpcache_feature_dtype(self) -> torch.dtype:
        """Dtype of the block-stack output, recorded in the schedule signature.

        Defaults to the output projection's weight dtype; override for a model
        whose head is not ``proj_out``.
        """
        return self.proj_out.weight.dtype

    def dpcache_blocks(self, run_blocks: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run ``run_blocks`` on a full step, or predict its output otherwise."""
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context_or_none,
        )

        context = get_forward_context_or_none()
        batch = None if context is None else context.forward_batch
        states = getattr(batch, "dpcache_states", None)
        if states is None:
            requested = getattr(batch, "dpcache_budget", None) or getattr(
                batch, "dpcache_calibration", None
            )
            if requested and not batch.is_warmup:
                # a requested budget or calibration must never silently run
                # without its state
                raise RuntimeError("DPCache is requested but no state is attached")
            return run_blocks()
        state = states[batch.is_cfg_negative]
        step = context.current_timestep
        if not state.is_full_step(step):
            # no block runs, so layerwise offload streams no block weights
            return state.predict(step)
        features = run_blocks()
        state.record(step, features)
        return features


# --------------------------------------------------------------------------
# Calibration and planning (offline)
# --------------------------------------------------------------------------


def _allowed(i: int, j: int, k: int, max_gap: int | None) -> bool:
    return max_gap is None or (j - i <= max_gap and k - j <= max_gap)


def _score_step(
    errors: torch.Tensor,
    features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
    step: int,
    max_gap: int | None,
) -> None:
    """Fill ``errors[i, j, step]`` for every scorable anchor pair ``i < j``.

    The single scoring routine: offline scoring and the runtime calibration
    state both call it, so they produce identical errors from identical
    features.
    """
    first_j = 1 if max_gap is None else max(1, step - max_gap + 1)
    for j in range(first_j, step):
        first_i = 0 if max_gap is None else max(0, j - max_gap)
        for i in range(first_i, j):
            predicted = predict_feature(features[i], features[j], i, j, step)
            diff = predicted.float() - features[step].float()
            errors[i, j, step] = (
                diff.abs().sum(dtype=torch.float64).item() / diff.numel()
            )


def pact_step_errors(
    features: Sequence[torch.Tensor], max_gap: int | None = None
) -> torch.Tensor:
    """``E[i, j, t] = mean|predict(h_i, h_j, t) - h_t|`` for ``i < j < t``.

    ``features`` are one sample's full-step final features in step order. The
    prediction uses :func:`predict_feature` on the features' own dtype and
    device, so score on the inference device; only the error reduction is
    widened (FP32 per element, FP64 sum), which never feeds back into
    inference. With ``max_gap``, only triples some allowed transition can use
    are scored (``j - i <= max_gap`` and ``t < j + max_gap``). Unscored
    entries are NaN.
    """
    num_steps = len(features)
    errors = torch.full((num_steps,) * 3, math.nan, dtype=torch.float64)
    for step in range(2, num_steps):
        _score_step(errors, features, step, max_gap)
    return errors


def pact_costs(step_errors: torch.Tensor, max_gap: int | None = None) -> torch.Tensor:
    """Transition costs ``C[i, j, k]`` for key triples, ``k`` up to ``T``.

    ``C[i, j, k] = sum_{j < t < k} E[i, j, t]``; ``k == T`` is the terminal
    sentinel, which scores steps ``j + 1 .. T - 1`` and nothing at ``T``.
    Adjacent keys cost zero. Invalid or gap-disallowed triples are +inf.
    """
    num_steps = step_errors.shape[0]
    if step_errors.shape != (num_steps,) * 3:
        raise ValueError("step errors must have shape (T, T, T)")
    costs = torch.full(
        (num_steps, num_steps, num_steps + 1), math.inf, dtype=torch.float64
    )
    for i, j in itertools.combinations(range(num_steps), 2):
        if not _allowed(i, j, j + 1, max_gap):
            continue
        running = 0.0
        costs[i, j, j + 1] = 0.0
        for k in range(j + 2, num_steps + 1):
            if not _allowed(i, j, k, max_gap):
                break
            value = step_errors[i, j, k - 1].item()
            if not math.isfinite(value):
                raise ValueError(f"nonfinite PACT error at ({i}, {j}, {k - 1})")
            running += value
            costs[i, j, k] = running
    return costs


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_mandatory(num_steps: int, mandatory: Sequence[int]) -> tuple[int, ...]:
    mandatory = tuple(mandatory)
    if (
        len(mandatory) < 2
        or not all(_is_int(s) for s in mandatory)
        or mandatory != tuple(range(len(mandatory)))
    ):
        raise ValueError(
            "mandatory full steps must be a leading int range 0..m-1 with m >= 2, "
            f"got {list(mandatory)}"
        )
    if len(mandatory) > num_steps:
        raise ValueError("more mandatory full steps than denoising steps")
    return mandatory


def _validate_max_gap(max_gap: Any) -> None:
    if max_gap is not None and (not _is_int(max_gap) or max_gap < 1):
        raise ValueError(f"max_gap must be a positive int or None, got {max_gap!r}")


def plan_schedule(
    costs: torch.Tensor,
    num_full_steps: int,
    mandatory: Sequence[int] = DEFAULT_MANDATORY_FULL_STEPS,
    force_last_full: bool = False,
    max_gap: int | None = None,
) -> tuple[list[int], float]:
    """Exact minimum-cost set of ``num_full_steps`` full steps.

    ``D[m, i, j]`` is the minimum cost of a prefix with ``m`` full steps whose
    two latest keys are ``i < j``. ``D[len(mandatory), m-2, m-1] = 0`` and
    ``D[m + 1, j, k] = min_i D[m, i, j] + C[i, j, k]``; the answer adds the
    terminal ``C[i, j, T]``. ``max_gap`` bounds every key gap, the terminal one
    included. Ties go to the lexicographically smallest predecessor and final
    pair, so the result is deterministic.
    """
    num_steps = costs.shape[0]
    if costs.shape != (num_steps, num_steps, num_steps + 1):
        raise ValueError("costs must have shape (T, T, T + 1)")
    mandatory = _validate_mandatory(num_steps, mandatory)
    _validate_max_gap(max_gap)
    lead = len(mandatory)
    if not _is_int(num_full_steps):
        raise TypeError("num_full_steps must be an int")
    minimum = lead + (1 if force_last_full and num_steps > lead else 0)
    if not minimum <= num_full_steps <= num_steps:
        raise ValueError(
            f"num_full_steps={num_full_steps} infeasible for T={num_steps} with "
            f"{lead} mandatory steps (force_last_full={force_last_full})"
        )

    c = costs.tolist()
    # best[(i, j)]: minimum cost at the current budget level; back[n][(j, k)]
    # is the predecessor i of pair (j, k) at level lead + n + 1.
    best: dict[tuple[int, int], float] = {(lead - 2, lead - 1): 0.0}
    back: list[dict[tuple[int, int], int]] = []
    for _ in range(lead, num_full_steps):
        level: dict[tuple[int, int], float] = {}
        pointers: dict[tuple[int, int], int] = {}
        for i, j in sorted(best):
            stop = num_steps if max_gap is None else min(num_steps, j + max_gap + 1)
            for k in range(j + 1, stop):
                value = best[(i, j)] + c[i][j][k]
                if value < level.get((j, k), math.inf):
                    level[(j, k)] = value
                    pointers[(j, k)] = i
        best = level
        back.append(pointers)
    chosen, chosen_cost = None, math.inf
    for i, j in sorted(best):
        if force_last_full and j != num_steps - 1:
            continue
        if max_gap is not None and num_steps - j > max_gap:
            continue
        value = best[(i, j)] + c[i][j][num_steps]
        if value < chosen_cost:
            chosen, chosen_cost = (i, j), value
    if chosen is None or not math.isfinite(chosen_cost):
        raise ValueError(
            f"no finite schedule with {num_full_steps} full steps of {num_steps}"
            f" (max_gap={max_gap})"
        )
    keys, pair = set(chosen), chosen
    for pointers in reversed(back):
        pair = (pointers[pair], pair[0])
        keys.add(pair[0])
    schedule = sorted(keys | set(mandatory))
    if len(schedule) != num_full_steps or schedule[:lead] != list(mandatory):
        raise AssertionError(f"planner backtrack produced {schedule}")
    return schedule, chosen_cost


def schedule_cost(
    costs: torch.Tensor, full_steps: Sequence[int], num_steps: int
) -> float:
    """Calibrated cost of an explicit schedule (terminal sentinel included)."""
    keys = list(full_steps) + [num_steps]
    return sum(
        costs[keys[n - 2], keys[n - 1], keys[n]].item() for n in range(2, len(keys))
    )


# --------------------------------------------------------------------------
# Schedule artifact
# --------------------------------------------------------------------------


def checkpoint_identity(model_path: str) -> str:
    """Snapshot revision for Hugging Face cache paths, else the path itself."""
    path = os.path.realpath(model_path) if os.path.exists(model_path) else model_path
    parent, name = os.path.split(path.rstrip("/"))
    if os.path.basename(parent) == "snapshots":
        return name
    return model_path


def config_digest(config: Mapping[str, Any]) -> str:
    """Order-independent digest of a scheduler (or other) config mapping."""
    text = json.dumps(dict(config), sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()


class DPCacheRequestSignature(msgspec.Struct, frozen=True, kw_only=True):
    """What a request must match for a schedule to apply."""

    pipeline: str
    checkpoint: str
    num_inference_steps: int
    height: int
    width: int
    guidance_scale: float
    do_classifier_free_guidance: bool
    quality: str
    attention_backend: str | None
    dtype: str
    scheduler: str
    scheduler_config_sha256: str
    timesteps: tuple[float, ...]
    sigmas: tuple[float, ...]


def _float_list(values: torch.Tensor | Sequence[float]) -> list[float]:
    # canonical FP32 values, so JSON round trips compare exactly with the runtime
    return torch.as_tensor(values).detach().cpu().float().tolist()


def _signature_dict(signature: DPCacheRequestSignature) -> dict[str, Any]:
    request = msgspec.to_builtins(signature)
    request["timesteps"] = _float_list(signature.timesteps)
    request["sigmas"] = _float_list(signature.sigmas)
    return request


def build_schedule_artifact(
    *,
    signature: DPCacheRequestSignature,
    full_steps: Sequence[int],
    mandatory: Sequence[int],
    calibrated_cost: float,
    calibration: Mapping[str, Any],
    source_commit: str,
    force_last_full: bool = False,
    max_gap: int | None = None,
    schedule_method: str = "exact-pair-state-dp",
) -> dict[str, Any]:
    artifact = {
        "schema": SCHEDULE_SCHEMA,
        "schedule_method": schedule_method,
        "predictor": dict(PREDICTOR),
        "objective": OBJECTIVE,
        "source_commit": source_commit,
        "request": _signature_dict(signature),
        "mandatory_full_steps": list(mandatory),
        "force_last_full": force_last_full,
        "max_gap": max_gap,
        "num_full_steps": len(full_steps),
        "full_steps": list(full_steps),
        "calibrated_cost": calibrated_cost,
        "calibration": dict(calibration),
    }
    validate_schedule(artifact)
    return artifact


_REQUIRED = {
    "schema",
    "predictor",
    "objective",
    "source_commit",
    "request",
    "mandatory_full_steps",
    "force_last_full",
    "max_gap",
    "num_full_steps",
    "full_steps",
    "calibrated_cost",
    "calibration",
}
_SIGNATURE_FIELDS = set(DPCacheRequestSignature.__struct_fields__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"DPCache schedule: {message}")


def validate_schedule(schedule: Mapping[str, Any]) -> tuple[int, ...]:
    """Structural and provenance checks independent of any request.

    Returns the full steps. ``source_commit`` and ``calibration`` are
    provenance; compatibility is decided by the schema, predictor and
    ``request`` block, not by the source commit.
    """
    if not isinstance(schedule, Mapping):
        raise TypeError("a DPCache schedule must be a JSON object")
    missing = _REQUIRED - set(schedule)
    _require(not missing, f"missing {sorted(missing)}")
    _require(
        schedule["schema"] == SCHEDULE_SCHEMA,
        f"unsupported schema {schedule['schema']!r}",
    )
    _require(
        dict(schedule["predictor"]) == PREDICTOR,
        f"unsupported predictor {schedule['predictor']!r}",
    )
    _require(
        schedule["objective"] == OBJECTIVE,
        f"unsupported objective {schedule['objective']!r}",
    )
    _require(
        isinstance(schedule["source_commit"], str) and schedule["source_commit"] != "",
        "source_commit must be a nonempty string",
    )
    calibration = schedule["calibration"]
    _require(isinstance(calibration, Mapping), "calibration must be an object")
    digest = calibration.get("manifest_sha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and set(digest) <= set("0123456789abcdef"),
        "calibration.manifest_sha256 must be a sha256 hex digest",
    )
    _require(
        _is_int(calibration.get("num_samples")) and calibration["num_samples"] >= 0,
        "calibration.num_samples must be a nonnegative int",
    )
    cost = schedule["calibrated_cost"]
    _require(
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost),
        "calibrated_cost must be finite",
    )

    request = schedule["request"]
    _require(isinstance(request, Mapping), "request must be an object")
    _require(
        set(request) == _SIGNATURE_FIELDS,
        f"request fields must be {sorted(_SIGNATURE_FIELDS)}",
    )
    num_steps = request["num_inference_steps"]
    _require(
        _is_int(num_steps) and num_steps >= 2, "num_inference_steps must be an int >= 2"
    )
    for name in ("height", "width"):
        _require(
            _is_int(request[name]) and request[name] > 0,
            f"{name} must be a positive int",
        )
    for name, length in (("timesteps", num_steps), ("sigmas", num_steps + 1)):
        values = request[name]
        _require(
            isinstance(values, list)
            and len(values) == length
            and all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in values
            ),
            f"request.{name} must be {length} finite numbers",
        )
    _require(request["dtype"] == "bfloat16", "only bfloat16 schedules are supported")

    mandatory = _validate_mandatory(num_steps, schedule["mandatory_full_steps"])
    max_gap = schedule["max_gap"]
    _validate_max_gap(max_gap)
    steps = schedule["full_steps"]
    _require(
        isinstance(steps, list) and all(_is_int(s) for s in steps),
        "full_steps must be a list of ints",
    )
    _require(
        steps == sorted(set(steps)) and all(0 <= s < num_steps for s in steps),
        "full_steps must be sorted, unique and inside [0, T)",
    )
    _require(
        steps[: len(mandatory)] == list(mandatory),
        "full_steps must start with the mandatory full steps",
    )
    _require(
        schedule["num_full_steps"] == len(steps),
        "num_full_steps does not match full_steps",
    )
    _require(
        not schedule["force_last_full"] or steps[-1] == num_steps - 1,
        "force_last_full schedule does not end with step T-1",
    )
    if max_gap is not None:
        keys = steps + [num_steps]
        _require(
            all(b - a <= max_gap for a, b in zip(keys, keys[1:])),
            f"a key gap (terminal included) exceeds max_gap={max_gap}",
        )
    return tuple(steps)


def check_schedule_matches(
    schedule: Mapping[str, Any], signature: DPCacheRequestSignature
) -> tuple[int, ...]:
    """Reject a schedule calibrated for a different request configuration."""
    steps = validate_schedule(schedule)
    expected = dict(schedule["request"])
    expected["timesteps"] = _float_list(expected["timesteps"])
    expected["sigmas"] = _float_list(expected["sigmas"])
    actual = _signature_dict(signature)
    mismatched = sorted(name for name in actual if expected[name] != actual[name])
    if mismatched:
        details = "; ".join(
            f"{name}: schedule={expected[name]!r} request={actual[name]!r}"
            for name in mismatched
            if name not in ("timesteps", "sigmas")
        )
        raise ValueError(
            f"DPCache schedule does not match this request ({mismatched}). {details}"
        )
    return steps


# --------------------------------------------------------------------------
# Server-side schedule set
# --------------------------------------------------------------------------


def load_schedule_dir(
    path: str, *, pipeline: str, checkpoint: str, attention_backend: str | None
) -> list[dict[str, Any]]:
    """Load and check every ``*.json`` schedule a server will serve.

    Runs at startup, so a schedule built for another pipeline, checkpoint or
    attention backend fails the launch instead of every request. Two schedules
    with the same budget and request configuration would make selection
    ambiguous, so they are rejected too.
    """
    names = sorted(n for n in os.listdir(path) if n.endswith(".json"))
    if not names:
        raise ValueError(f"no DPCache schedules (*.json) in {path}")
    served = {
        "pipeline": pipeline,
        "checkpoint": checkpoint,
        "attention_backend": attention_backend,
    }
    schedules, seen = [], {}
    for name in names:
        with open(os.path.join(path, name)) as f:
            schedule = json.load(f)
        try:
            validate_schedule(schedule)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from exc
        for field, value in served.items():
            if schedule["request"][field] != value:
                raise ValueError(
                    f"{name} was calibrated for {field}="
                    f"{schedule['request'][field]!r}, this server runs {value!r}"
                )
        key = (
            schedule["num_full_steps"],
            json.dumps(schedule["request"], sort_keys=True),
        )
        if key in seen:
            raise ValueError(
                f"{name} and {seen[key]} have the same budget and request configuration"
            )
        seen[key] = name
        schedules.append(schedule)
    return schedules


def _describe(request: Mapping[str, Any]) -> str:
    return (
        f"{request['width']}x{request['height']}, "
        f"{request['num_inference_steps']} steps, "
        f"guidance {request['guidance_scale']}"
    )


def select_schedule(
    schedules: Sequence[Mapping[str, Any]],
    budget: int,
    signature: DPCacheRequestSignature,
) -> tuple[int, ...]:
    """Full steps of the schedule with ``budget`` calibrated for this request."""
    request = _signature_dict(signature)
    for schedule in schedules:
        if schedule["num_full_steps"] == budget:
            try:
                return check_schedule_matches(schedule, signature)
            except ValueError:
                continue
    available = sorted(
        {f"K={s['num_full_steps']} ({_describe(s['request'])})" for s in schedules}
    )
    raise ValueError(
        f"no DPCache schedule with K={budget} for this request "
        f"({_describe(request)}); available: {', '.join(available)}"
    )


# --------------------------------------------------------------------------
# Calibration captures (runtime -> offline planner)
# --------------------------------------------------------------------------

CAPTURE_SCHEMA = "sglang-dpcache-calibration-capture-v1"
_CALIBRATION_KEYS = {"output", "max_gap"}


def validate_calibration_request(calibration: Mapping[str, Any]) -> None:
    """Check a ``dpcache_calibration`` request field."""
    if not isinstance(calibration, Mapping):
        raise TypeError("dpcache_calibration must be a JSON object")
    unknown = set(calibration) - _CALIBRATION_KEYS
    if unknown:
        raise ValueError(f"dpcache_calibration: unknown keys {sorted(unknown)}")
    output = calibration.get("output")
    if not isinstance(output, str) or not output:
        raise ValueError("dpcache_calibration.output must be a nonempty path")
    _validate_max_gap(calibration.get("max_gap"))


def save_calibration_capture(
    path: str,
    *,
    state: DPCacheCalibrationState,
    signature: DPCacheRequestSignature,
    prompt: str,
    seed: int,
) -> None:
    """Write one calibration request's PACT errors and request signature."""
    if not state.complete:
        raise RuntimeError(
            f"DPCache calibration scored {state.num_full} of {state.num_steps} steps"
        )
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "schema": CAPTURE_SCHEMA,
            "signature": _signature_dict(signature),
            "max_gap": state.max_gap,
            "prompt": prompt,
            "seed": seed,
            "errors": state.errors,
        },
        path,
    )


def load_calibration_capture(path: str) -> dict[str, Any]:
    capture = torch.load(path, map_location="cpu", weights_only=True)
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"{path}: not a DPCache calibration capture")
    return capture


def plan_schedules(
    captures: Sequence[Mapping[str, Any]],
    budgets: Iterable[int],
    *,
    source_commit: str,
    mandatory: Sequence[int] = DEFAULT_MANDATORY_FULL_STEPS,
    max_gap: int | None = None,
    force_last_full: bool = False,
) -> dict[int, dict[str, Any]]:
    """Schedule artifacts for each budget from one configuration's captures.

    Errors are averaged over the captures (summed in the given order, then
    divided). Every capture must share one request signature. ``max_gap`` may
    tighten, never loosen, the bound the captures were scored with, because a
    looser planner would need costs that were never computed.
    """
    if not captures:
        raise ValueError("no calibration captures")
    signature = captures[0]["signature"]
    scored_gap = captures[0]["max_gap"]
    for capture in captures[1:]:
        if capture["signature"] != signature:
            raise ValueError("calibration captures come from different requests")
        if capture["max_gap"] != scored_gap:
            raise ValueError("calibration captures were scored with different max_gap")
    if scored_gap is not None and (max_gap is None or max_gap > scored_gap):
        raise ValueError(
            f"max_gap={max_gap} is looser than the scored bound {scored_gap}"
        )
    total = None
    for capture in captures:
        errors = capture["errors"]
        total = errors if total is None else total + errors
    costs = pact_costs(total / len(captures), max_gap=max_gap)
    manifest = sorted([c["prompt"], c["seed"]] for c in captures)
    calibration = {
        "manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        "num_samples": len(captures),
        "scored_max_gap": scored_gap,
    }
    request = msgspec.convert(signature, DPCacheRequestSignature)
    schedules = {}
    for budget in budgets:
        full_steps, cost = plan_schedule(
            costs,
            budget,
            mandatory=mandatory,
            force_last_full=force_last_full,
            max_gap=max_gap,
        )
        schedules[budget] = build_schedule_artifact(
            signature=request,
            full_steps=full_steps,
            mandatory=mandatory,
            calibrated_cost=cost,
            calibration=calibration,
            source_commit=source_commit,
            force_last_full=force_last_full,
            max_gap=max_gap,
        )
    return schedules
