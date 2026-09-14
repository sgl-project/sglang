#!/usr/bin/env python3
"""Operational controller for SGLang PD runtime role switching.

The controller keeps orchestration policy outside SGLang workers:

1. collect router + worker metrics,
2. build an explicit D->P / P->D flip plan,
3. dry-run or execute coordinated role transitions.
"""

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request
from urllib.parse import quote, urljoin, urlparse

try:
    from pd_flip_monitor import ClusterSLOSnapshot, PDFlipSLOMonitor
    from pd_flip_trace_slo import TraceSLOMonitor
except ModuleNotFoundError:
    import importlib.util

    _MONITOR_PATH = Path(__file__).with_name("pd_flip_monitor.py")
    _MONITOR_SPEC = importlib.util.spec_from_file_location(
        "pd_flip_monitor", _MONITOR_PATH
    )
    _MONITOR_MODULE = importlib.util.module_from_spec(_MONITOR_SPEC)
    sys.modules[_MONITOR_SPEC.name] = _MONITOR_MODULE
    _MONITOR_SPEC.loader.exec_module(_MONITOR_MODULE)
    ClusterSLOSnapshot = _MONITOR_MODULE.ClusterSLOSnapshot
    PDFlipSLOMonitor = _MONITOR_MODULE.PDFlipSLOMonitor

    _TRACE_SLO_PATH = Path(__file__).with_name("pd_flip_trace_slo.py")
    _TRACE_SLO_SPEC = importlib.util.spec_from_file_location(
        "pd_flip_trace_slo", _TRACE_SLO_PATH
    )
    _TRACE_SLO_MODULE = importlib.util.module_from_spec(_TRACE_SLO_SPEC)
    sys.modules[_TRACE_SLO_SPEC.name] = _TRACE_SLO_MODULE
    _TRACE_SLO_SPEC.loader.exec_module(_TRACE_SLO_MODULE)
    TraceSLOMonitor = _TRACE_SLO_MODULE.TraceSLOMonitor

try:
    from pd_flip_progressive_policy import (
        ProgressiveDecision,
        RatioSelection,
        RequestCapacity,
        evaluate_slo_decision,
        select_first_batch,
    )
except ModuleNotFoundError:
    import importlib.util

    _PROGRESSIVE_POLICY_PATH = Path(__file__).with_name("pd_flip_progressive_policy.py")
    _PROGRESSIVE_POLICY_SPEC = importlib.util.spec_from_file_location(
        "pd_flip_progressive_policy", _PROGRESSIVE_POLICY_PATH
    )
    _PROGRESSIVE_POLICY_MODULE = importlib.util.module_from_spec(
        _PROGRESSIVE_POLICY_SPEC
    )
    sys.modules[_PROGRESSIVE_POLICY_SPEC.name] = _PROGRESSIVE_POLICY_MODULE
    _PROGRESSIVE_POLICY_SPEC.loader.exec_module(_PROGRESSIVE_POLICY_MODULE)
    ProgressiveDecision = _PROGRESSIVE_POLICY_MODULE.ProgressiveDecision
    RatioSelection = _PROGRESSIVE_POLICY_MODULE.RatioSelection
    RequestCapacity = _PROGRESSIVE_POLICY_MODULE.RequestCapacity
    evaluate_slo_decision = _PROGRESSIVE_POLICY_MODULE.evaluate_slo_decision
    select_first_batch = _PROGRESSIVE_POLICY_MODULE.select_first_batch

try:
    from pd_flip_queue_util_policy import decide_queue_util_flip_direction
except ModuleNotFoundError:
    import importlib.util

    _QUEUE_UTIL_POLICY_PATH = Path(__file__).with_name(
        "pd_flip_queue_util_policy.py"
    )
    _QUEUE_UTIL_POLICY_SPEC = importlib.util.spec_from_file_location(
        "pd_flip_queue_util_policy", _QUEUE_UTIL_POLICY_PATH
    )
    _QUEUE_UTIL_POLICY_MODULE = importlib.util.module_from_spec(
        _QUEUE_UTIL_POLICY_SPEC
    )
    sys.modules[_QUEUE_UTIL_POLICY_SPEC.name] = _QUEUE_UTIL_POLICY_MODULE
    _QUEUE_UTIL_POLICY_SPEC.loader.exec_module(_QUEUE_UTIL_POLICY_MODULE)
    decide_queue_util_flip_direction = (
        _QUEUE_UTIL_POLICY_MODULE.decide_queue_util_flip_direction
    )

try:
    from pd_flip_online_policy import (
        compute_policy_violation_rates,
        decide_decode_first,
        decide_slo_target,
        decide_tpot_capacity,
        estimate_decode_sufficiency,
        estimate_window_batch_size_at_nonattainment,
    )
except ModuleNotFoundError:
    import importlib.util

    _ONLINE_POLICY_PATH = Path(__file__).with_name("pd_flip_online_policy.py")
    _ONLINE_POLICY_SPEC = importlib.util.spec_from_file_location(
        "pd_flip_online_policy", _ONLINE_POLICY_PATH
    )
    _ONLINE_POLICY_MODULE = importlib.util.module_from_spec(_ONLINE_POLICY_SPEC)
    sys.modules[_ONLINE_POLICY_SPEC.name] = _ONLINE_POLICY_MODULE
    _ONLINE_POLICY_SPEC.loader.exec_module(_ONLINE_POLICY_MODULE)
    compute_policy_violation_rates = (
        _ONLINE_POLICY_MODULE.compute_policy_violation_rates
    )
    decide_decode_first = _ONLINE_POLICY_MODULE.decide_decode_first
    decide_slo_target = _ONLINE_POLICY_MODULE.decide_slo_target
    decide_tpot_capacity = _ONLINE_POLICY_MODULE.decide_tpot_capacity
    estimate_decode_sufficiency = (
        _ONLINE_POLICY_MODULE.estimate_decode_sufficiency
    )
    estimate_window_batch_size_at_nonattainment = (
        _ONLINE_POLICY_MODULE.estimate_window_batch_size_at_nonattainment
    )


def _migration_source_start_payload(
    session_id: str,
    target_url: str,
    rids: Optional[List[str]],
    include_waiting: bool = False,
    prefill_donor_mode: bool = False,
    target_decode_dp_rank: Optional[int] = None,
    target_decode_dp_ranks: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    payload = {
        "session_id": session_id,
        "target_url": target_url,
        "rids": None if rids is None else list(rids),
        "include_waiting": include_waiting,
    }
    if prefill_donor_mode:
        payload["prefill_donor_mode"] = True
    if target_decode_dp_rank is not None:
        payload["target_decode_dp_rank"] = int(target_decode_dp_rank)
    if target_decode_dp_ranks is not None:
        payload["target_decode_dp_ranks"] = {
            str(rid): int(rank) for rid, rank in target_decode_dp_ranks.items()
        }
    return payload


JsonDict = Dict[str, Any]
SOURCE_DELTA_QUIESCE_PENDING_MESSAGE = (
    "source batch quiesce pending; retry delta after quiesce"
)
P_TO_D_HANDOFF_RACE_MESSAGE = (
    "P->D bootstrap candidate left the queue before source hold"
)
D_TO_P_REQUEST_COMPLETED_BEFORE_BASE_TRANSFER_MESSAGE = (
    "selected migration requests completed before base transfer"
)


def parse_topology_schedule(value: str, total_workers: int) -> List[JsonDict]:
    """Parse an increasing trace-relative topology target schedule."""

    if total_workers < 2:
        raise ValueError("scheduled topology requires at least two workers")
    try:
        raw = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid topology schedule JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("topology schedule must be a non-empty JSON list")

    schedule: List[JsonDict] = []
    previous_offset = -1.0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"topology schedule item {index} must be an object")
        try:
            offset_seconds = float(item["offset_seconds"])
            prefill_workers = int(item["prefill_workers"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"topology schedule item {index} requires numeric "
                "offset_seconds and prefill_workers"
            ) from exc
        if offset_seconds < 0 or offset_seconds <= previous_offset:
            raise ValueError("topology schedule offsets must be non-negative and increasing")
        if prefill_workers < 1 or prefill_workers >= total_workers:
            raise ValueError(
                f"topology schedule item {index} must retain at least one P and one D"
            )
        decode_workers = total_workers - prefill_workers
        declared_decode = item.get("decode_workers")
        if declared_decode is not None and int(declared_decode) != decode_workers:
            raise ValueError(
                f"topology schedule item {index} P/D counts do not total {total_workers}"
            )
        schedule.append(
            {
                "event_index": index,
                "offset_seconds": offset_seconds,
                "prefill_workers": prefill_workers,
                "decode_workers": decode_workers,
                "topology": f"{prefill_workers}P{decode_workers}D",
                "note": item.get("note"),
            }
        )
        previous_offset = offset_seconds
    return schedule


def expand_topology_schedule(
    initial_prefill_workers: int, schedule: Sequence[JsonDict]
) -> List[JsonDict]:
    """Expand topology targets into ordered single-worker role flips."""

    current = int(initial_prefill_workers)
    expanded: List[JsonDict] = []
    for event in schedule:
        target = int(event["prefill_workers"])
        direction = "d_to_p" if target > current else "p_to_d"
        for subflip_index in range(abs(target - current)):
            current += 1 if direction == "d_to_p" else -1
            row = dict(event)
            row.update(
                {
                    "subflip_index": subflip_index,
                    "direction": direction,
                    "expected_prefill_after": current,
                }
            )
            expanded.append(row)
    return expanded


def wait_for_trace_start_monotonic(
    ledger_path: str, *, timeout_seconds: float, poll_interval_seconds: float
) -> float:
    """Return the first client request start time from the live ledger."""

    deadline = time.monotonic() + timeout_seconds
    path = Path(ledger_path)
    while time.monotonic() < deadline:
        if path.is_file():
            starts = []
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        try:
                            row = json.loads(raw)
                        except ValueError:
                            continue
                        start_time = row.get("start_time")
                        if isinstance(start_time, (int, float)):
                            starts.append(float(start_time))
            except OSError:
                starts = []
            if starts:
                return min(starts)
        time.sleep(max(0.01, poll_interval_seconds))
    raise TimeoutError("timed out waiting for first trace request in SLO ledger")


@dataclass(frozen=True)
class SLODirectionDecision:
    """One symmetric, evidence-carrying PD Flip direction decision."""

    direction: Optional[str]
    reason: str
    prefill_attainment: Optional[float]
    decode_attainment: Optional[float]
    attainment_gap: Optional[float]
    prefill_samples: int
    decode_samples: int
    gap_threshold: Optional[float]

    def to_dict(self) -> JsonDict:
        return asdict(self)


def decide_slo_flip_direction(
    snapshot: ClusterSLOSnapshot,
    *,
    enter_threshold: float,
    gap_threshold: Optional[float],
    min_prefill_samples: int,
    min_decode_samples: int,
) -> SLODirectionDecision:
    """Choose D->P or P->D without assigning priority to either direction.

    Positive ``decode - prefill`` means Prefill attainment is worse and needs
    another P. Negative means Decode attainment is worse and needs another D.
    When a gap threshold is configured it creates a symmetric no-action band.
    The absolute-threshold fallback only acts when exactly one side is at risk.
    """

    prefill = getattr(snapshot, "prefill_slo_attainment", None)
    decode = getattr(snapshot, "decode_slo_attainment", None)
    prefill_counts = getattr(snapshot, "prefill_counts", None)
    decode_counts = getattr(snapshot, "decode_counts", None)
    prefill_samples = int(getattr(prefill_counts, "total", 0) or 0)
    decode_samples = int(getattr(decode_counts, "total", 0) or 0)
    gap = (
        float(decode) - float(prefill)
        if prefill is not None and decode is not None
        else None
    )

    common = {
        "prefill_attainment": prefill,
        "decode_attainment": decode,
        "attainment_gap": gap,
        "prefill_samples": prefill_samples,
        "decode_samples": decode_samples,
        "gap_threshold": gap_threshold,
    }
    if prefill_samples < min_prefill_samples:
        return SLODirectionDecision(
            direction=None, reason="insufficient_prefill_samples", **common
        )
    if decode_samples < min_decode_samples:
        return SLODirectionDecision(
            direction=None, reason="insufficient_decode_samples", **common
        )
    if gap is None:
        return SLODirectionDecision(
            direction=None, reason="missing_slo_attainment", **common
        )

    if gap_threshold is not None:
        if gap >= gap_threshold:
            return SLODirectionDecision(
                direction="d_to_p", reason="prefill_slo_worse", **common
            )
        if gap <= -gap_threshold:
            return SLODirectionDecision(
                direction="p_to_d", reason="decode_slo_worse", **common
            )
        return SLODirectionDecision(
            direction=None, reason="within_slo_gap_deadband", **common
        )

    prefill_risk = float(prefill) < enter_threshold
    decode_risk = float(decode) < enter_threshold
    if prefill_risk and not decode_risk:
        return SLODirectionDecision(
            direction="d_to_p", reason="prefill_only_slo_risk", **common
        )
    if decode_risk and not prefill_risk:
        return SLODirectionDecision(
            direction="p_to_d", reason="decode_only_slo_risk", **common
        )
    return SLODirectionDecision(
        direction=None,
        reason=("both_sides_at_risk" if prefill_risk else "both_sides_healthy"),
        **common,
    )


class HttpLike:
    def get_json(self, base_url: str, path: str) -> Any: ...

    def post_json(self, base_url: str, path: str, payload: JsonDict) -> Any: ...


class HttpClient:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 10.0):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def get_json(self, base_url: str, path: str) -> Any:
        req = self._request(base_url, path, method="GET")
        body = self._open_text(req)
        return json.loads(body) if body else {}

    def get_text(self, base_url: str, path: str) -> str:
        req = self._request(base_url, path, method="GET")
        return self._open_text(req)

    def post_json(self, base_url: str, path: str, payload: JsonDict) -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = self._request(base_url, path, method="POST", data=body)
        req.add_header("Content-Type", "application/json")
        return self._open_json(req)

    def _request(
        self,
        base_url: str,
        path: str,
        method: str,
        data: Optional[bytes] = None,
    ) -> request.Request:
        req = request.Request(_join_url(base_url, path), data=data, method=method)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return req

    def _open_text(self, req: request.Request) -> str:
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{req.full_url} returned HTTP {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"failed to connect to {req.full_url}: {exc}") from exc

    def _open_json(self, req: request.Request) -> Any:
        body = self._open_text(req)
        return json.loads(body) if body else {}


@dataclass(frozen=True)
class PDNode:
    name: str
    worker_url: str
    router_worker_id: str
    bootstrap_port: Optional[int] = None


@dataclass(frozen=True)
class PDClusterConfig:
    router_url: str
    nodes: List[PDNode]
    request_timeout_seconds: float = 10.0
    migration_timeout_seconds: float = 120.0
    migration_poll_interval_seconds: float = 0.5
    observation_quiesce_seconds: float = 0.0
    post_migration_idle_timeout_seconds: float = 2.0
    first_migration_ratio: float = 0.5
    observation_seconds: float = 10.0
    slo_threshold: float = 0.9
    slo_recovery_threshold: float = 0.95
    slo_attainment_gap_threshold: Optional[float] = None
    slo_attainment_gap_recovery_threshold: Optional[float] = None
    force_second_migration_after_observation: bool = False
    min_prefill_slo_samples: int = 20
    min_decode_slo_samples: int = 20
    session_journal_path: str = "pd_flip_session.json"
    session_id_prefix: Optional[str] = None
    prefill_donor_mode: bool = False
    p_to_d_min_handoff_requests: int = 1
    decision_policy: str = "slo_gap"
    decode_first_gap_threshold: float = 0.10
    decode_first_prefill_protect: bool = True
    decode_first_d_to_p_require_prefill_gap: bool = True
    decode_first_bs_estimator: str = "fitted_formula"
    decode_first_fixed_batch_size: float = 10.0
    decode_first_window_target_violation_rate: float = 0.20
    decode_first_window_min_samples: int = 20
    slo_target_gap_threshold: float = 0.20
    tpot_capacity_intercept_ms: float = 8.0
    tpot_capacity_batch_slope_ms: float = 0.4
    queue_window_requests: int = 50
    queue_threshold_seconds: float = 0.010
    queue_overload_ratio: float = 0.10
    queue_scale_in_ratio: float = 0.05
    prefill_scale_in_headroom_workers: float = 1.5
    prefill_min_role_seconds: float = 30.0
    d_to_p_direct_full_drain: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.first_migration_ratio < 1:
            raise ValueError(
                "first_migration_ratio must be greater than 0 and less than 1"
            )
        if not self.observation_seconds >= 0:
            raise ValueError("observation_seconds must be greater than or equal to 0")
        if not 0 <= self.slo_threshold <= 1:
            raise ValueError("slo_threshold must be between 0 and 1 inclusive")
        if not self.slo_threshold <= self.slo_recovery_threshold <= 1:
            raise ValueError(
                "slo_recovery_threshold must be between slo_threshold and 1 inclusive"
            )
        if self.slo_attainment_gap_threshold is not None:
            if not 0 <= self.slo_attainment_gap_threshold <= 1:
                raise ValueError(
                    "slo_attainment_gap_threshold must be between 0 and 1 inclusive"
                )
            recovery = self.slo_attainment_gap_recovery_threshold
            if recovery is None or not 0 <= recovery <= self.slo_attainment_gap_threshold:
                raise ValueError(
                    "slo_attainment_gap_recovery_threshold must be between 0 and "
                    "slo_attainment_gap_threshold inclusive"
                )
        if self.min_prefill_slo_samples <= 0:
            raise ValueError("min_prefill_slo_samples must be greater than 0")
        if self.min_decode_slo_samples <= 0:
            raise ValueError("min_decode_slo_samples must be greater than 0")
        if self.p_to_d_min_handoff_requests <= 0:
            raise ValueError("p_to_d_min_handoff_requests must be greater than 0")
        if self.decision_policy not in (
            "slo_gap",
            "prefill_queue_util",
            "decode_first",
            "slo_target",
            "tpot_capacity",
        ):
            raise ValueError(
                "decision_policy must be slo_gap, prefill_queue_util, "
                "decode_first, slo_target, or tpot_capacity"
            )
        if not 0 <= self.decode_first_gap_threshold <= 1:
            raise ValueError("decode_first_gap_threshold must be between 0 and 1")
        if self.decode_first_bs_estimator not in (
            "fitted_formula",
            "window_p20_nonattainment",
            "fixed_batch_size",
        ):
            raise ValueError(
                "decode_first_bs_estimator must be fitted_formula, "
                "window_p20_nonattainment, or fixed_batch_size"
            )
        if (
            not math.isfinite(self.decode_first_fixed_batch_size)
            or self.decode_first_fixed_batch_size <= 0
        ):
            raise ValueError("decode_first_fixed_batch_size must be positive")
        if not 0 <= self.decode_first_window_target_violation_rate <= 1:
            raise ValueError(
                "decode_first_window_target_violation_rate must be between 0 and 1"
            )
        if self.decode_first_window_min_samples <= 0:
            raise ValueError("decode_first_window_min_samples must be positive")
        if not 0 <= self.slo_target_gap_threshold <= 1:
            raise ValueError("slo_target_gap_threshold must be between 0 and 1")
        if self.tpot_capacity_intercept_ms < 0:
            raise ValueError("tpot_capacity_intercept_ms must be non-negative")
        if self.tpot_capacity_batch_slope_ms <= 0:
            raise ValueError("tpot_capacity_batch_slope_ms must be positive")
        if self.queue_window_requests <= 0:
            raise ValueError("queue_window_requests must be greater than 0")
        if self.queue_threshold_seconds < 0:
            raise ValueError("queue_threshold_seconds must be non-negative")
        if not 0 <= self.queue_overload_ratio <= 1:
            raise ValueError("queue_overload_ratio must be between 0 and 1")
        if not 0 <= self.queue_scale_in_ratio <= 1:
            raise ValueError(
                "queue_scale_in_ratio must be between 0 and 1"
            )
        if self.queue_scale_in_ratio > self.queue_overload_ratio:
            raise ValueError(
                "queue_scale_in_ratio must not exceed queue_overload_ratio"
            )
        if self.prefill_scale_in_headroom_workers <= 0:
            raise ValueError(
                "prefill_scale_in_headroom_workers must be greater than 0"
            )
        if self.prefill_min_role_seconds < 0:
            raise ValueError("prefill_min_role_seconds must be non-negative")

    @staticmethod
    def from_dict(data: JsonDict) -> "PDClusterConfig":
        nodes = [
            PDNode(
                name=str(item["name"]),
                worker_url=str(item["worker_url"]),
                router_worker_id=str(item.get("router_worker_id") or item["name"]),
                bootstrap_port=item.get("bootstrap_port"),
            )
            for item in data["nodes"]
        ]
        return PDClusterConfig(
            router_url=str(data["router_url"]),
            nodes=nodes,
            request_timeout_seconds=float(data.get("request_timeout_seconds", 10.0)),
            migration_timeout_seconds=float(
                data.get("migration_timeout_seconds", 120.0)
            ),
            migration_poll_interval_seconds=float(
                data.get("migration_poll_interval_seconds", 0.5)
            ),
            observation_quiesce_seconds=float(
                data.get(
                    "observation_quiesce_seconds",
                    os.environ.get("PD_FLIP_OBSERVE_QUIESCE_SECONDS", 0.0),
                )
            ),
            post_migration_idle_timeout_seconds=float(
                data.get(
                    "post_migration_idle_timeout_seconds",
                    os.environ.get("PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS", 2.0),
                )
            ),
            first_migration_ratio=float(data.get("first_migration_ratio", 0.5)),
            observation_seconds=float(data.get("observation_seconds", 10.0)),
            slo_threshold=float(data.get("slo_threshold", 0.9)),
            slo_recovery_threshold=float(
                data.get("slo_recovery_threshold", 0.95)
            ),
            slo_attainment_gap_threshold=(
                float(data["slo_attainment_gap_threshold"])
                if data.get("slo_attainment_gap_threshold") is not None
                else None
            ),
            slo_attainment_gap_recovery_threshold=(
                float(data["slo_attainment_gap_recovery_threshold"])
                if data.get("slo_attainment_gap_recovery_threshold") is not None
                else None
            ),
            force_second_migration_after_observation=bool(
                data.get("force_second_migration_after_observation", False)
            ),
            min_prefill_slo_samples=int(data.get("min_prefill_slo_samples", 20)),
            min_decode_slo_samples=int(data.get("min_decode_slo_samples", 20)),
            session_journal_path=str(
                data.get("session_journal_path", "pd_flip_session.json")
            ),
            session_id_prefix=(
                str(data["session_id_prefix"]) if data.get("session_id_prefix") else None
            ),
            prefill_donor_mode=bool(data.get("prefill_donor_mode", False)),
            p_to_d_min_handoff_requests=int(
                data.get("p_to_d_min_handoff_requests", 1)
            ),
            decision_policy=str(data.get("decision_policy", "slo_gap")),
            decode_first_gap_threshold=float(
                data.get("decode_first_gap_threshold", 0.10)
            ),
            decode_first_prefill_protect=bool(
                data.get("decode_first_prefill_protect", True)
            ),
            decode_first_d_to_p_require_prefill_gap=bool(
                data.get("decode_first_d_to_p_require_prefill_gap", True)
            ),
            decode_first_bs_estimator=str(
                data.get("decode_first_bs_estimator", "fitted_formula")
            ),
            decode_first_fixed_batch_size=float(
                data.get("decode_first_fixed_batch_size", 10.0)
            ),
            decode_first_window_target_violation_rate=float(
                data.get("decode_first_window_target_violation_rate", 0.20)
            ),
            decode_first_window_min_samples=int(
                data.get("decode_first_window_min_samples", 20)
            ),
            slo_target_gap_threshold=float(
                data.get("slo_target_gap_threshold", 0.20)
            ),
            tpot_capacity_intercept_ms=float(
                data.get("tpot_capacity_intercept_ms", 8.0)
            ),
            tpot_capacity_batch_slope_ms=float(
                data.get("tpot_capacity_batch_slope_ms", 0.4)
            ),
            queue_window_requests=int(data.get("queue_window_requests", 50)),
            queue_threshold_seconds=float(
                data.get("queue_threshold_seconds", 0.010)
            ),
            queue_overload_ratio=float(data.get("queue_overload_ratio", 0.10)),
            queue_scale_in_ratio=float(
                data.get("queue_scale_in_ratio", 0.05)
            ),
            prefill_scale_in_headroom_workers=float(
                data.get("prefill_scale_in_headroom_workers", 1.5)
            ),
            prefill_min_role_seconds=float(
                data.get("prefill_min_role_seconds", 30.0)
            ),
            d_to_p_direct_full_drain=bool(
                data.get("d_to_p_direct_full_drain", False)
            ),
        )


class PDFlipSessionJournal:
    """Durable single-session ownership journal for controller recovery."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, record: JsonDict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def read(self) -> Optional[JsonDict]:
        if not self.path.exists():
            return None
        record = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("session journal root is not an object")
        return record

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


@dataclass
class NodeMetrics:
    name: str
    worker_url: str
    router_worker_id: str
    router_role: str = "unknown"
    worker_role: str = "unknown"
    draining: bool = False
    router_active_load: int = 0
    bootstrap_port: Optional[int] = None
    is_idle: bool = False
    admission_paused: bool = False
    running_reqs: int = 0
    waiting_reqs: int = 0
    total_tokens: int = 0
    token_usage: Optional[float] = None
    raw_status: JsonDict = field(default_factory=dict)
    dp_statuses: List[JsonDict] = field(default_factory=list)
    raw_loads: List[JsonDict] = field(default_factory=list)

    @property
    def effective_role(self) -> str:
        return self.worker_role if self.worker_role != "unknown" else self.router_role


@dataclass(frozen=True)
class ControllerAction:
    step: str
    target: str
    method: str
    url: str
    payload: Optional[JsonDict] = None


@dataclass
class FlipPlan:
    dry_run: bool
    direction: str
    source: Optional[str]
    target_role: Optional[str]
    migration_target: Optional[str]
    reason: str
    actions: List[ControllerAction]
    metrics: List[NodeMetrics] = field(default_factory=list)


@dataclass
class ActionRecord:
    step: str
    target: str
    method: str
    url: str
    payload: Optional[JsonDict] = None
    response: Any = None
    success: bool = True
    message: str = ""
    elapsed_seconds: float = 0.0
    start_wall: Optional[float] = None
    start_monotonic: Optional[float] = None
    end_wall: Optional[float] = None
    end_monotonic: Optional[float] = None


def _action_timing_fields(
    start_monotonic: float, start_wall: float
) -> JsonDict:
    end_monotonic = time.monotonic()
    return {
        "elapsed_seconds": end_monotonic - start_monotonic,
        "start_wall": start_wall,
        "start_monotonic": start_monotonic,
        "end_wall": time.time(),
        "end_monotonic": end_monotonic,
    }


@dataclass
class FlipExecutionResult:
    success: bool
    message: str
    direction: str
    source: Optional[str]
    target_role: Optional[str]
    migration_target: Optional[str]
    actions: List[ActionRecord] = field(default_factory=list)
    metrics: List[NodeMetrics] = field(default_factory=list)
    total_seconds: float = 0.0
    migration_seconds: float = 0.0


@dataclass
class MonitorLoopResult:
    success: bool
    message: str
    iterations: int
    snapshots: List[JsonDict] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    state_trace: List[JsonDict] = field(default_factory=list)


class MonitorState:
    SAFE = "safe"
    SELECTING = "selecting"
    COOLDOWN = "cooldown"
    PREPARING_KV_TRANSFER = "preparing_kv_transfer"
    OBSERVING_SOURCE_QUIESCE = "observing_source_quiesce"
    PREPARING_DRAIN = "preparing_drain"
    FLIPPING_ROLE = "flipping_role"


class ProgressiveMonitorState:
    SAFE = "safe"
    SELECTING = "selecting"
    FIRST_MIGRATING = "first_migrating"
    OBSERVING = "observing"
    RECOVERING = "recovering"
    SECOND_MIGRATING = "second_migrating"
    FULL_MIGRATING = "full_migrating"
    FLIPPING_ROLE = "flipping_role"


class ProgressiveAtomicBatchError(RuntimeError):
    def __init__(
        self, message: str, *, source_finished: bool, cutover_started: bool = False
    ):
        super().__init__(message)
        self.source_finished = source_finished
        self.cutover_started = cutover_started


class RoleFlipRouterPendingError(RuntimeError):
    """Worker role changed irreversibly; keep the source paused and drained."""


class ForcedRiskSnapshot:
    prefill_slo_attainment = 0.0
    decode_slo_attainment = 1.0

    def to_dict(self) -> JsonDict:
        return {
            "prefill_slo_attainment": self.prefill_slo_attainment,
            "decode_slo_attainment": self.decode_slo_attainment,
            "forced": True,
        }


class ForcedRiskMonitor:
    def collect_cluster(self, monitor_nodes: Any) -> ForcedRiskSnapshot:
        return ForcedRiskSnapshot()


class PDFlipController:
    def __init__(self, config: PDClusterConfig, client: HttpLike):
        if not config.nodes:
            raise ValueError("PDClusterConfig.nodes must not be empty")
        self.config = config
        self.client = client
        self.session_journal = PDFlipSessionJournal(Path(config.session_journal_path))
        self.monitor_journal_path = Path(
            str(config.session_journal_path) + ".monitor.jsonl"
        )
        self._queue_policy_not_before_wall: Optional[float] = None
        self._decode_batch_samples_by_request: Dict[str, JsonDict] = {}

    def _append_monitor_decisions(self, records: Sequence[JsonDict]) -> None:
        """Persist completed monitor decisions while the controller is running.

        ``controller/result.json`` is written only when the monitor exits.  A
        long-lived automatic controller may be stopped after the workload has
        completed, so keep a small append-only decision ledger next to the
        recovery journal.  This contains no credentials or request contents.
        """
        if not records:
            return
        self.monitor_journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.monitor_journal_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def _progressive_session_prefix(
        self, source: NodeMetrics, target: NodeMetrics
    ) -> str:
        if self.config.session_id_prefix:
            return self.config.session_id_prefix
        return f"pd-flip-{source.name}-to-{target.name}-{uuid.uuid4().hex}"

    def _resolve_prefill_donor_groups(
        self, manifests: Sequence[JsonDict]
    ) -> Dict[str, List[JsonDict]]:
        nodes_by_host: Dict[str, List[PDNode]] = {}
        nodes_by_endpoint: Dict[Tuple[str, int], List[PDNode]] = {}
        for node in self.config.nodes:
            hostname = (urlparse(node.worker_url).hostname or "").lower()
            if hostname:
                nodes_by_host.setdefault(hostname, []).append(node)
                if node.bootstrap_port is not None:
                    nodes_by_endpoint.setdefault(
                        (hostname, int(node.bootstrap_port)), []
                    ).append(node)

        groups: Dict[str, List[JsonDict]] = {}
        for manifest in manifests:
            if int(manifest.get("prefill_donor_end") or 0) == 0:
                continue
            donor_host = str(manifest.get("prefill_donor_host") or "").lower()
            raw_donor_port = manifest.get("prefill_donor_port")
            if raw_donor_port is not None:
                try:
                    donor_port = int(raw_donor_port)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "original Prefill donor has invalid bootstrap port: "
                        f"{raw_donor_port!r}"
                    ) from exc
                if donor_port <= 0:
                    raise RuntimeError(
                        "original Prefill donor has invalid bootstrap port: "
                        f"{raw_donor_port!r}"
                    )
                matches = nodes_by_endpoint.get((donor_host, donor_port), [])
                donor_identity = f"{donor_host}:{donor_port}"
            else:
                # Preserve legacy single-instance-per-host manifests. A shared
                # host is intentionally rejected without the per-instance
                # bootstrap port instead of selecting an arbitrary donor.
                matches = nodes_by_host.get(donor_host, [])
                donor_identity = donor_host
            if not matches:
                raise RuntimeError(
                    "original Prefill donor endpoint is not configured: "
                    f"{donor_identity!r}"
                )
            if len(matches) != 1:
                raise RuntimeError(
                    "ambiguous original Prefill donor endpoint: "
                    f"{donor_identity!r} matches {[node.name for node in matches]}"
                )
            groups.setdefault(matches[0].worker_url, []).append(manifest)
        return groups

    @staticmethod
    def _bind_output_relay_urls(
        groups: Dict[str, List[JsonDict]],
    ) -> None:
        """Persist each request's real HTTP stream owner across Decode hops."""
        for donor_url, manifests in groups.items():
            for manifest in manifests:
                manifest.setdefault("pd_flip_output_relay_url", donor_url)

    @staticmethod
    def _journal_record(
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        batch_rids: Sequence[str],
        phase: str,
        source_finished: bool,
        metadata: Optional[JsonDict] = None,
    ) -> JsonDict:
        record = {
            "source_name": source.name,
            "source_url": source.worker_url,
            "target_name": target.name,
            "target_url": target.worker_url,
            "session_id": session_id,
            "batch_rids": [str(rid) for rid in batch_rids],
            "phase": phase,
            "source_finished": source_finished,
        }
        if metadata:
            record.update(metadata)
        return record

    def _write_journal_phase(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        batch_rids: Sequence[str],
        phase: str,
        source_finished: bool = False,
        metadata: Optional[JsonDict] = None,
    ) -> None:
        self.session_journal.write(
            self._journal_record(
                source,
                target,
                session_id,
                batch_rids,
                phase,
                source_finished,
                metadata,
            )
        )

    def _write_p_to_d_journal(
        self,
        *,
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        manifests: Sequence[JsonDict],
        phase: str,
        source_finished: bool = False,
        metadata: Optional[JsonDict] = None,
    ) -> None:
        record: JsonDict = {
            "direction": "p_to_d",
            "source_name": source.name,
            "source_url": source.worker_url,
            "target_name": target.name,
            "target_url": target.worker_url,
            "session_id": session_id,
            "batch_rids": [str(item.get("rid")) for item in manifests],
            "phase": phase,
            "source_finished": source_finished,
            "updated_wall": time.time(),
        }
        if metadata:
            record.update(metadata)
        self.session_journal.write(record)

    def reconcile_session(self, session_id: str) -> FlipExecutionResult:
        try:
            record = self.session_journal.read()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return FlipExecutionResult(
                False,
                "session state requires operator recovery: invalid journal",
                "d_to_p",
                None,
                None,
                None,
            )
        if record is None or str(record.get("session_id")) != str(session_id):
            return FlipExecutionResult(
                False, "session journal not found", "d_to_p", None, None, None
            )
        if record.get("direction") == "p_to_d":
            return FlipExecutionResult(
                False,
                "P->D handoff journal requires forward-only operator reconciliation",
                "p_to_d",
                record.get("source_name"),
                "decode",
                record.get("target_name"),
            )

        source_name = record.get("source_name")
        target_name = record.get("target_name")
        source_url = record.get("source_url")
        target_url = record.get("target_url")
        batch_rids = [str(rid) for rid in record.get("batch_rids") or []]
        waiting_only_pending = (
            record.get("phase") == "source_start_intent"
            and record.get("include_waiting") is True
            and record.get("batch_scope") == "waiting_only_pending_manifest"
        )
        phase = str(record.get("phase") or "")
        fallback_pre_cutover_phases = {
            "source_full_fallback_intent",
            "source_full_fallback_started",
            "target_full_fallback_prepare_intent",
            "target_full_fallback_prepared",
        }
        next_fsm_phase = record.get("next_fsm_phase") or record.get("next_phase")
        current_cutover_phases = {
            "ownership_cutover_intent",
            "observing_activation_pending",
        }
        legacy_cutover_phases = {
            "source_finish_intent",
            "source_finish_complete",
            "target_activate_intent",
        }
        legacy_cutover = phase in legacy_cutover_phases and (
            str(session_id).endswith("-first")
            or str(session_id).endswith("-final")
        )
        ownership_cutover_pending = phase in current_cutover_phases or legacy_cutover
        if ownership_cutover_pending and next_fsm_phase is None:
            if str(session_id).endswith("-first"):
                next_fsm_phase = "observing"
            elif str(session_id).endswith("-final"):
                next_fsm_phase = "role_flip_worker_prefill_intent"
        if (
            not all((source_name, target_name, source_url, target_url))
            or (not batch_rids and not waiting_only_pending)
        ):
            return FlipExecutionResult(
                False,
                "session state requires operator recovery: incomplete journal",
                "d_to_p",
                source_name,
                None,
                target_name,
            )

        configured_source = next(
            (node for node in self.config.nodes if node.name == source_name), None
        )
        source = NodeMetrics(
            str(source_name),
            str(source_url),
            configured_source.router_worker_id
            if configured_source is not None
            else str(source_name),
            bootstrap_port=(
                configured_source.bootstrap_port
                if configured_source is not None
                else None
            ),
        )
        target = NodeMetrics(str(target_name), str(target_url), str(target_name))
        records: List[ActionRecord] = []
        if record.get("phase") == "observing":
            self._resume_decode_source(source, records)
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "observation_recovered_safe",
                True,
                {
                    "next_fsm_phase": "observing",
                    "batch_ordinal": record.get("batch_ordinal", 1),
                    "source_admission_paused": False,
                    "router_drained": False,
                },
            )
            return FlipExecutionResult(
                True,
                "observation crash recovered safely; source remains decode",
                "d_to_p",
                source.name,
                "decode",
                target.name,
                actions=records,
            )
        if record.get("phase") in {
            "role_flip_worker_prefill_intent",
            "role_flip_worker_prefill",
            "role_flip_router_pending",
        }:
            if record.get("phase") == "role_flip_worker_prefill_intent":
                runtime = self.client.get_json(
                    source.worker_url, "/pd_flip/runtime_role/status"
                )
                statuses = list(_index_dp_responses(runtime).values())
                roles = set()
                for status in statuses:
                    role, _, _ = _parse_runtime_status(status)
                    inner = (
                        status.get("status")
                        if isinstance(status.get("status"), dict)
                        else status
                    )
                    active = _normalize_role(inner.get("active_event_loop_role"))
                    roles.add((role, active))
                if roles == {("decode", "decode")}:
                    self._post_worker(
                        records,
                        "set_source_runtime_role",
                        source,
                        "/pd_flip/runtime_role/set",
                        {"role": "prefill", "force": False},
                    )
                elif roles != {("prefill", "prefill")}:
                    return FlipExecutionResult(
                        False,
                        f"session state requires operator recovery: mixed runtime roles {roles}",
                        "d_to_p",
                        source.name,
                        "prefill",
                        target.name,
                        actions=records,
                    )
            self._wait_source_role(records, source, "prefill", "reconcile_prefill_loop")
            self._complete_prefill_router_flip(source, records)
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "role_flip_complete",
                True,
                {
                    "next_fsm_phase": "role_flip_worker_prefill_intent",
                    "batch_ordinal": record.get("batch_ordinal", 2),
                },
            )
            return FlipExecutionResult(
                True,
                "pending prefill router role reconciled",
                "d_to_p",
                source.name,
                "prefill",
                target.name,
                actions=records,
            )

        try:
            source_status_response = self.client.get_json(
                source_url, "/pd_flip/migration/status"
            )
            target_status_response = self.client.get_json(
                target_url, "/pd_flip/migration/status"
            )
            if waiting_only_pending:
                # The intent is durable before source/start. A crash may therefore
                # leave no worker-side session at all; querying both workers is
                # still required, but session-wide abort is safe and idempotent.
                source_statuses = []
                target_statuses = []
            else:
                source_statuses = _strict_migration_statuses(
                    source_status_response, session_id
                )
                target_statuses = _strict_migration_statuses(
                    target_status_response, session_id
                )
        except (TypeError, ValueError):
            return FlipExecutionResult(
                False,
                "session state requires operator recovery: invalid worker status",
                "d_to_p",
                source_name,
                None,
                target_name,
            )

        source_states = {status.get("state") for status in source_statuses}
        target_states = {status.get("state") for status in target_statuses}
        if waiting_only_pending:
            self._write_journal_phase(
                source,
                target,
                session_id,
                [],
                "abort_intent",
                False,
                {
                    "include_waiting": True,
                    "batch_scope": "waiting_only_pending_manifest",
                },
            )
            abort_complete = self._abort_two_phase_migration(
                source,
                target,
                session_id,
                records,
                prefill_donor_urls=tuple(
                    record.get("prefill_donor_urls") or []
                ),
            )
            self._write_journal_phase(
                source,
                target,
                session_id,
                [],
                "aborted" if abort_complete else "abort_incomplete",
                False,
                {
                    "include_waiting": True,
                    "batch_scope": "waiting_only_pending_manifest",
                },
            )
            return FlipExecutionResult(
                abort_complete,
                (
                    "waiting-only session aborted during reconciliation"
                    if abort_complete
                    else "session state requires operator recovery: abort incomplete"
                ),
                "d_to_p",
                source.name,
                None,
                target.name,
                actions=records,
            )
        if ownership_cutover_pending:
            if next_fsm_phase not in {
                "observing",
                "role_flip_worker_prefill_intent",
            }:
                return FlipExecutionResult(
                    False,
                    "session state requires operator recovery: cutover next phase missing",
                    "d_to_p",
                    source.name,
                    None,
                    target.name,
                    actions=records,
                )
            if source_states != {"source_released"}:
                abort_complete = self._abort_two_phase_migration(
                    source,
                    target,
                    session_id,
                    records,
                    prefill_donor_urls=tuple(
                        record.get("prefill_donor_urls") or []
                    ),
                )
                if abort_complete:
                    self._resume_decode_source(source, records)
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    batch_rids,
                    "cutover_aborted_safe" if abort_complete else "abort_incomplete",
                    False,
                    {"next_fsm_phase": next_fsm_phase},
                )
                return FlipExecutionResult(
                    abort_complete,
                    (
                        "pre-release ownership cutover aborted safely"
                        if abort_complete
                        else "session state requires operator recovery: abort incomplete"
                    ),
                    "d_to_p",
                    source.name,
                    "decode" if abort_complete else None,
                    target.name,
                    actions=records,
                )
            if target_states == {"ready_to_activate"}:
                self._post_worker(
                    records,
                    "activate_decode_migration_target",
                    target,
                    "/pd_flip/migration/target/activate",
                    {"session_id": session_id, "rids": batch_rids},
                )
            elif target_states != {"active"}:
                return FlipExecutionResult(
                    False,
                    "session state requires operator recovery: observation target not activatable",
                    "d_to_p",
                    source.name,
                    "decode",
                    target.name,
                    actions=records,
                )
            if next_fsm_phase == "observing":
                self._resume_decode_source(source, records)
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    batch_rids,
                    "observation_recovered_safe",
                    True,
                    {
                        "next_fsm_phase": "observing",
                        "batch_ordinal": record.get("batch_ordinal", 1),
                        "source_admission_paused": False,
                        "router_drained": False,
                    },
                )
                return FlipExecutionResult(
                    True,
                    "observation activation recovered safely; source remains decode",
                    "d_to_p",
                    source.name,
                    "decode",
                    target.name,
                    actions=records,
                )
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "role_flip_worker_prefill_intent",
                True,
                {
                    "next_fsm_phase": next_fsm_phase,
                    "source_admission_paused": True,
                    "router_drained": True,
                },
            )
            reconciled = self.reconcile_session(session_id)
            reconciled.actions = records + reconciled.actions
            return reconciled
        if target_states == {"active"}:
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_active", True
            )
            return FlipExecutionResult(
                True,
                "session already active",
                "d_to_p",
                source.name,
                "decode",
                target.name,
                actions=records,
            )
        if (
            source_states == {"source_aborted"}
            and target_states == {"target_aborted"}
        ):
            if phase == "abort_complete_traffic_recovery_incomplete":
                try:
                    self._resume_decode_source(source, records)
                except Exception as exc:
                    self._write_journal_phase(
                        source,
                        target,
                        session_id,
                        batch_rids,
                        "abort_complete_traffic_recovery_incomplete",
                        False,
                        {"recovery_error": str(exc)},
                    )
                    return FlipExecutionResult(
                        False,
                        "migration aborted but source traffic recovery failed: "
                        + str(exc),
                        "d_to_p",
                        source.name,
                        "decode",
                        target.name,
                        actions=records,
                    )
            self._write_journal_phase(
                source, target, session_id, batch_rids, "aborted", False
            )
            return FlipExecutionResult(
                True,
                "session already aborted",
                "d_to_p",
                source.name,
                None,
                target.name,
                actions=records,
            )
        if (
            source_states == {"source_released"}
            and target_states == {"ready_to_activate"}
        ):
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_activate_intent", True
            )
            self._post_worker(
                records,
                "activate_decode_migration_target",
                target,
                "/pd_flip/migration/target/activate",
                {"session_id": session_id, "rids": batch_rids},
            )
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_active", True
            )
            return FlipExecutionResult(
                True,
                "target activated during reconciliation",
                "d_to_p",
                source.name,
                "decode",
                target.name,
                actions=records,
            )
        safe_source_abort_states = {
            "source_started",
            "source_transferred",
            "source_quiesce_requested",
            "source_delta_started",
            "source_delta_transferred",
            "source_fallback_started",
            "source_failed",
        }
        safe_target_abort_states = {
            "target_prepared",
            "target_transferred_held",
            "target_delta_started",
            "target_delta_transferred",
            "target_fallback_required",
            "target_fallback_prepared",
            "ready_to_activate",
            "target_failed",
        }
        if (
            source_states <= safe_source_abort_states
            and target_states <= safe_target_abort_states
        ):
            self._write_journal_phase(
                source, target, session_id, batch_rids, "abort_intent", False
            )
            abort_complete = self._abort_two_phase_migration(
                source,
                target,
                session_id,
                records,
                prefill_donor_urls=tuple(
                    record.get("prefill_donor_urls") or []
                ),
            )
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "aborted" if abort_complete else "abort_incomplete",
                False,
            )
            if not abort_complete:
                return FlipExecutionResult(
                    False,
                    "session state requires operator recovery: abort incomplete",
                    "d_to_p",
                    source.name,
                    None,
                    target.name,
                    actions=records,
                )
            if phase in fallback_pre_cutover_phases:
                try:
                    self._resume_decode_source(source, records)
                except Exception as exc:
                    self._write_journal_phase(
                        source,
                        target,
                        session_id,
                        batch_rids,
                        "abort_complete_traffic_recovery_incomplete",
                        False,
                        {"recovery_error": str(exc)},
                    )
                    return FlipExecutionResult(
                        False,
                        "migration aborted but source traffic recovery failed: "
                        + str(exc),
                        "d_to_p",
                        source.name,
                        "decode",
                        target.name,
                        actions=records,
                    )
            return FlipExecutionResult(
                True,
                "session aborted during reconciliation",
                "d_to_p",
                source.name,
                None,
                target.name,
                actions=records,
            )
        return FlipExecutionResult(
            False,
            "session state requires operator recovery",
            "d_to_p",
            source.name,
            None,
            target.name,
            actions=records,
        )

    def _select_progressive_first_batch(
        self, source: NodeMetrics, target: NodeMetrics
    ) -> Optional[RatioSelection]:
        source_status = source.raw_status.get("status", source.raw_status)
        target_status = target.raw_status.get("status", target.raw_status)
        if not isinstance(source_status, dict):
            return None
        running_requests = source_status.get("running_requests", [])
        if not isinstance(running_requests, list) or not running_requests:
            return None

        requests = []
        for item in running_requests:
            if not isinstance(item, dict) or item.get("rid") is None:
                return None
            committed_value = item.get("kv_committed_len")
            if committed_value is None:
                return None
            try:
                committed_tokens = int(committed_value)
            except (TypeError, ValueError):
                return None
            if committed_tokens < 0:
                return None
            requests.append(
                RequestCapacity(rid=str(item["rid"]), committed_tokens=committed_tokens)
            )
        return select_first_batch(
            requests,
            self.config.first_migration_ratio,
            target_req_slots=int(target_status.get("free_request_slots", 0)),
            target_kv_tokens=int(target_status.get("available_kv_tokens", 0)),
            reserve_tokens_per_req=int(
                target_status.get("reserved_decode_tokens_per_req", 0)
            ),
        )

    @staticmethod
    def _progressive_full_drain_capacity(
        source: NodeMetrics, target: NodeMetrics
    ) -> JsonDict:
        """Conservatively check that one D can absorb the entire source queue."""

        source_status = source.raw_status.get("status", source.raw_status)
        target_status = target.raw_status.get("status", target.raw_status)
        if not isinstance(source_status, dict) or not isinstance(target_status, dict):
            return {"feasible": False, "reason": "runtime_status_unavailable"}
        running = source_status.get("running_requests", [])
        waiting = source_status.get("waiting_requests", [])
        if not isinstance(running, list) or not isinstance(waiting, list):
            return {"feasible": False, "reason": "source_queue_status_invalid"}
        reserve = max(
            0, int(target_status.get("reserved_decode_tokens_per_req", 0) or 0)
        )
        committed_tokens = 0
        for queue_name, requests in (("running", running), ("waiting", waiting)):
            for item in requests:
                if not isinstance(item, dict) or item.get("kv_committed_len") is None:
                    return {
                        "feasible": False,
                        "reason": "source_{}_committed_length_unavailable".format(
                            queue_name
                        ),
                    }
                try:
                    committed_tokens += max(0, int(item["kv_committed_len"]))
                except (TypeError, ValueError):
                    return {
                        "feasible": False,
                        "reason": "source_{}_committed_length_invalid".format(
                            queue_name
                        ),
                    }
        request_count = len(running) + len(waiting)
        required_kv_tokens = committed_tokens + reserve * request_count
        free_request_slots = int(target_status.get("free_request_slots", 0) or 0)
        available_kv_tokens = int(
            target_status.get("available_kv_tokens", 0) or 0
        )
        return {
            "feasible": (
                request_count <= free_request_slots
                and required_kv_tokens <= available_kv_tokens
            ),
            "reason": "full_source_queue_fits" if (
                request_count <= free_request_slots
                and required_kv_tokens <= available_kv_tokens
            ) else "full_source_queue_exceeds_target_capacity",
            "running_requests": len(running),
            "waiting_requests": len(waiting),
            "required_request_slots": request_count,
            "free_request_slots": free_request_slots,
            "committed_kv_tokens": committed_tokens,
            "reserved_kv_tokens": reserve * request_count,
            "required_kv_tokens": required_kv_tokens,
            "available_kv_tokens": available_kv_tokens,
        }

    @staticmethod
    def _decode_bootstrap_request_count(node: NodeMetrics) -> Optional[int]:
        """Return pending Decode bootstrap receivers, or ``None`` if unknown."""

        status = node.raw_status.get("status", node.raw_status)
        if not isinstance(status, dict):
            return None
        requests = status.get("decode_bootstrap_requests")
        if not isinstance(requests, list):
            return None
        return len(requests)

    @classmethod
    def _decode_source_is_idle_for_role_flip(cls, node: NodeMetrics) -> bool:
        """Require an observable, bootstrap-free Decode idle point."""

        status = node.raw_status.get("status", node.raw_status)
        if not isinstance(status, dict):
            return False
        if cls._decode_bootstrap_request_count(node) != 0:
            return False
        if not bool(status.get("is_idle") or status.get("is_idle_for_flip")):
            return False
        for field in ("running_requests", "waiting_requests"):
            requests = status.get(field)
            if not isinstance(requests, list) or requests:
                return False
        return True

    @classmethod
    def _bootstrap_empty_decode_sources(
        cls, candidates: Sequence[NodeMetrics]
    ) -> List[NodeMetrics]:
        """Preserve load order while excluding bound or unobservable receivers."""

        return [
            item
            for item in candidates
            if cls._decode_bootstrap_request_count(item) == 0
        ]

    def _wait_for_drained_decode_bootstrap_empty(
        self,
        source: NodeMetrics,
        records: List[ActionRecord],
    ) -> JsonDict:
        """Wait for already-dispatched Decode bootstraps after Router drain.

        The Router is drained before this method is called, but worker
        admission intentionally remains open.  That lets Prefill requests
        dispatched just before the routing cut finish their existing
        bootstrap handshakes without allowing new requests to choose this
        Decode worker.
        """

        started = time.monotonic()
        started_wall = time.time()
        deadline = started + self.config.migration_timeout_seconds
        sample_count = 0
        initial_count: Optional[int] = None
        last_count: Optional[int] = None
        count_changes: List[JsonDict] = []
        last_per_dp: JsonDict = {}
        step = "wait_drained_decode_bootstrap_empty"
        try:
            while True:
                response = self.client.get_json(
                    source.worker_url, "/pd_flip/runtime_role/status"
                )
                _raise_if_unsuccessful(response, step)
                indexed_statuses = _index_dp_responses(response)
                if not indexed_statuses:
                    raise RuntimeError(
                        "{}: runtime status did not contain any DP response".format(
                            step
                        )
                    )
                source.dp_statuses = [
                    indexed_statuses[rank] for rank in sorted(indexed_statuses)
                ]
                source.raw_status = _aggregate_dp_runtime_status(
                    source.dp_statuses, source.name
                )
                count = self._decode_bootstrap_request_count(source)
                per_dp: JsonDict = {}
                for rank, item in sorted(indexed_statuses.items()):
                    status = (
                        item.get("status")
                        if isinstance(item, dict)
                        and isinstance(item.get("status"), dict)
                        else item
                    )
                    requests = (
                        status.get("decode_bootstrap_requests")
                        if isinstance(status, dict)
                        else None
                    )
                    per_dp[str(rank)] = (
                        len(requests) if isinstance(requests, list) else None
                    )
                sample_count += 1
                last_per_dp = per_dp
                if count is None or any(value is None for value in per_dp.values()):
                    raise RuntimeError(
                        "{}: decode bootstrap state is not observable".format(step)
                    )
                if initial_count is None:
                    initial_count = count
                if count != last_count:
                    count_changes.append(
                        {
                            "elapsed_seconds": time.monotonic() - started,
                            "total": count,
                            "per_dp": per_dp,
                        }
                    )
                    last_count = count
                if count == 0:
                    summary = {
                        "sample_count": sample_count,
                        "initial_bootstrap_requests": initial_count,
                        "final_bootstrap_requests": 0,
                        "final_per_dp": per_dp,
                        "count_changes": count_changes,
                        "worker_admission_paused_while_waiting": False,
                    }
                    records.append(
                        ActionRecord(
                            step=step,
                            target=source.name,
                            method="GET",
                            url=_join_url(
                                source.worker_url,
                                "/pd_flip/runtime_role/status",
                            ),
                            response=summary,
                            **_action_timing_fields(started, started_wall),
                        )
                    )
                    return summary
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        "{} timed out for {} with {} bootstrap requests".format(
                            step, source.name, count
                        )
                    )
                time.sleep(
                    min(
                        max(
                            0.01,
                            self.config.migration_poll_interval_seconds,
                        ),
                        max(0.0, deadline - now),
                    )
                )
        except Exception as exc:
            records.append(
                ActionRecord(
                    step=step,
                    target=source.name,
                    method="GET",
                    url=_join_url(
                        source.worker_url, "/pd_flip/runtime_role/status"
                    ),
                    response={
                        "sample_count": sample_count,
                        "initial_bootstrap_requests": initial_count,
                        "last_bootstrap_requests": last_count,
                        "last_per_dp": last_per_dp,
                        "count_changes": count_changes,
                        "worker_admission_paused_while_waiting": False,
                    },
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _wait_for_progressive_full_drain_capacity(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        records: List[ActionRecord],
    ) -> JsonDict:
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        last: JsonDict = {}
        while True:
            self._refresh_progressive_runtime_status(
                source, records, "refresh_source_full_drain_capacity"
            )
            self._refresh_progressive_runtime_status(
                target, records, "refresh_target_full_drain_capacity"
            )
            last = self._progressive_full_drain_capacity(source, target)
            if last.get("feasible"):
                return last
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    "full source drain capacity timed out for {} -> {}: {}".format(
                        source.name, target.name, last
                    )
                )
            time.sleep(
                min(
                    self.config.migration_poll_interval_seconds,
                    max(0.0, deadline - now),
                )
            )

    def _refresh_progressive_runtime_status(
        self,
        node: NodeMetrics,
        records: List[ActionRecord],
        step: str,
    ) -> None:
        status_body = self._record_get(
            records,
            step,
            node.name,
            node.worker_url,
            "/pd_flip/runtime_role/status",
        )
        indexed_statuses = _index_dp_responses(status_body)
        if not indexed_statuses:
            raise RuntimeError(
                f"{step}: runtime status did not contain any DP response"
            )
        node.dp_statuses = [
            indexed_statuses[rank] for rank in sorted(indexed_statuses)
        ]
        node.raw_status = _aggregate_dp_runtime_status(
            node.dp_statuses, node.name
        )

    def collect_metrics(self) -> List[NodeMetrics]:
        router_workers = self._fetch_router_workers()
        def collect_node(node: PDNode) -> NodeMetrics:
            router_status = router_workers.get(node.router_worker_id, {})
            status_body = self.client.get_json(
                node.worker_url, "/pd_flip/runtime_role/status"
            )
            loads_body = self.client.get_json(node.worker_url, "/v1/loads?include=all")
            indexed_statuses = _index_dp_responses(status_body)
            dp_statuses = [indexed_statuses[rank] for rank in sorted(indexed_statuses)]
            status = _aggregate_dp_runtime_status(dp_statuses, node.name)
            role, is_idle, admission_paused = _parse_runtime_status(status)
            running_reqs, waiting_reqs, total_tokens, token_usage, raw_loads = (
                _parse_loads(loads_body)
            )
            return NodeMetrics(
                name=node.name,
                worker_url=node.worker_url,
                router_worker_id=node.router_worker_id,
                router_role=_normalize_role(router_status.get("role")),
                worker_role=role,
                draining=bool(router_status.get("draining", False)),
                router_active_load=int(router_status.get("active_load") or 0),
                bootstrap_port=(
                    router_status.get("bootstrap_port")
                    if router_status.get("bootstrap_port") is not None
                    else node.bootstrap_port
                ),
                is_idle=is_idle,
                admission_paused=admission_paused,
                running_reqs=running_reqs,
                waiting_reqs=waiting_reqs,
                total_tokens=total_tokens,
                token_usage=token_usage,
                raw_status=status,
                dp_statuses=dp_statuses,
                raw_loads=raw_loads,
            )

        # P->D owner matching needs the Prefill and Decode bootstrap queues
        # from one narrow observation interval. Fetch all worker snapshots in
        # parallel; preserve configured node order in the returned list.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self.config.nodes))
        ) as pool:
            return list(pool.map(collect_node, self.config.nodes))

    def _reset_queue_util_policy_window(self) -> None:
        """Start a fresh controller-side request window.

        Workers retain their bounded raw sample buffers for evidence.  The
        controller-side wall-clock boundary prevents warm-up, cooldown, or a
        previously completed flip from authorizing a later flip again.
        """

        self._queue_policy_not_before_wall = time.time()

    @staticmethod
    def _dp_runtime_status(item: JsonDict) -> JsonDict:
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        return status if isinstance(status, dict) else {}

    @classmethod
    def _prefill_busy_ratio(cls, metric: NodeMetrics) -> Optional[float]:
        values = []
        for item in metric.dp_statuses:
            status = cls._dp_runtime_status(item)
            value = status.get("prefill_busy_ratio")
            if value is not None:
                values.append(float(value))
        # A multi-DP worker is considered under-utilized only when every rank
        # is under-utilized, so expose its busiest rank to the policy.
        return max(values) if len(values) == len(metric.dp_statuses) and values else None

    @classmethod
    def _prefill_role_seconds(cls, metric: NodeMetrics) -> Optional[float]:
        values = []
        for item in metric.dp_statuses:
            status = cls._dp_runtime_status(item)
            value = status.get("pd_flip_role_uptime_seconds")
            if value is not None:
                values.append(float(value))
        # Require every rank to have aged past the guard interval.
        return min(values) if len(values) == len(metric.dp_statuses) and values else None

    @classmethod
    def _online_policy_queue_evidence(
        cls, metrics: List[NodeMetrics]
    ) -> Dict[str, JsonDict]:
        samples_by_request: Dict[str, JsonDict] = {}
        for metric in metrics:
            if metric.effective_role != "prefill":
                continue
            for dp_rank, item in enumerate(metric.dp_statuses):
                status = cls._dp_runtime_status(item)
                for raw_sample in status.get("prefill_queue_samples") or []:
                    if not isinstance(raw_sample, dict):
                        continue
                    request_id = raw_sample.get("request_id") or raw_sample.get("rid")
                    if request_id is None:
                        continue
                    try:
                        queue_seconds = float(raw_sample["queue_seconds"])
                        event_time = float(raw_sample.get("event_time") or 0.0)
                    except (KeyError, TypeError, ValueError):
                        continue
                    sample = {
                        "request_id": str(request_id),
                        "rid": (
                            str(raw_sample["rid"])
                            if raw_sample.get("rid") is not None
                            else None
                        ),
                        "node": metric.name,
                        "dp_rank": int(
                            item.get("dp_rank")
                            if item.get("dp_rank") is not None
                            else status.get("dp_rank", dp_rank)
                        ),
                        "seq": int(raw_sample.get("seq") or 0),
                        "role_epoch": int(raw_sample.get("role_epoch") or 0),
                        "queue_seconds": max(0.0, queue_seconds),
                        "event_time": event_time,
                    }
                    existing = samples_by_request.get(sample["request_id"])
                    order = (
                        sample["event_time"],
                        sample["node"],
                        sample["dp_rank"],
                        sample["seq"],
                    )
                    if existing is None or order < (
                        existing["event_time"],
                        existing["node"],
                        existing["dp_rank"],
                        existing["seq"],
                    ):
                        samples_by_request[sample["request_id"]] = sample
        return samples_by_request

    def _decode_sufficiency_estimate(
        self,
        metrics: List[NodeMetrics],
        records: List[JsonDict],
        *,
        tpot_intercept_ms: float = 6.8165,
        tpot_per_batch_ms: float = 0.40830,
        round_required_instances_up: bool = False,
        batch_size_estimator: str = "fitted_formula",
    ) -> Tuple[Optional[bool], Optional[bool], JsonDict]:
        decode_metrics = [
            metric for metric in metrics if metric.effective_role == "decode"
        ]
        sampled_at = time.time()
        for metric in decode_metrics:
            for dp_item in metric.dp_statuses:
                status = self._dp_runtime_status(dp_item)
                running_requests = status.get("running_requests")
                if not isinstance(running_requests, list):
                    continue
                for request_record in running_requests:
                    if not isinstance(request_record, dict):
                        continue
                    request_id = request_record.get("request_id")
                    try:
                        batch_size = float(request_record.get("decode_batch_size"))
                        event_time = float(
                            request_record.get("batch_sample_event_time") or sampled_at
                        )
                    except (TypeError, ValueError):
                        continue
                    if (
                        request_id is None
                        or not math.isfinite(batch_size)
                        or batch_size <= 0
                        or not math.isfinite(event_time)
                    ):
                        continue
                    key = str(request_id)
                    source = "{}:{}".format(
                        metric.name,
                        status.get("dp_rank", dp_item.get("dp_rank", 0)),
                    )
                    summary = self._decode_batch_samples_by_request.setdefault(
                        key,
                        {
                            "sum": 0.0,
                            "count": 0,
                            "maximum": 0.0,
                            "last_event_by_source": {},
                            "last_event_time": 0.0,
                        },
                    )
                    last_by_source = summary["last_event_by_source"]
                    if event_time <= float(last_by_source.get(source, 0.0)):
                        continue
                    last_by_source[source] = event_time
                    summary["sum"] += batch_size
                    summary["count"] += 1
                    summary["maximum"] = max(summary["maximum"], batch_size)
                    summary["last_event_time"] = max(
                        summary["last_event_time"], event_time
                    )
        if len(self._decode_batch_samples_by_request) > 20000:
            oldest = sorted(
                self._decode_batch_samples_by_request.items(),
                key=lambda item: float(item[1].get("last_event_time") or 0.0),
            )[: len(self._decode_batch_samples_by_request) - 20000]
            for request_id, _ in oldest:
                self._decode_batch_samples_by_request.pop(request_id, None)

        inflight_decode_requests = sum(
            max(0, int(metric.running_reqs)) for metric in decode_metrics
        )

        running_slos = []
        window_slos = []
        for record in records:
            try:
                slo = float(record.get("tpot_slo_seconds"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(slo) or slo <= 0:
                continue
            window_slos.append(slo)
            if record.get("status") == "running":
                running_slos.append(slo)
        selected_slos = running_slos or window_slos
        selected_slo = min(selected_slos) if selected_slos else None
        window_estimate: Optional[JsonDict] = None
        estimated_batch_size_override: Optional[float] = None
        batch_size_formula_override: Optional[str] = None
        estimation_reason: Optional[str] = None
        if batch_size_estimator == "fixed_batch_size":
            estimated_batch_size_override = (
                self.config.decode_first_fixed_batch_size
            )
            batch_size_formula_override = "fixed_decode_batch_size"
            estimation_reason = "estimated_from_fixed_decode_batch_size"
        elif batch_size_estimator == "window_p20_nonattainment":
            enriched_records = []
            for record in records:
                enriched = dict(record)
                sample = self._decode_batch_samples_by_request.get(
                    str(record.get("request_id"))
                )
                if sample and int(sample.get("count") or 0) > 0:
                    enriched["decode_batch_size_mean"] = (
                        float(sample["sum"]) / int(sample["count"])
                    )
                    enriched["decode_batch_size_observations"] = int(
                        sample["count"]
                    )
                enriched_records.append(enriched)
            window_estimate = estimate_window_batch_size_at_nonattainment(
                enriched_records,
                target_violation_rate=(
                    self.config.decode_first_window_target_violation_rate
                ),
                min_samples=self.config.decode_first_window_min_samples,
            )
            estimated_batch_size_override = window_estimate.get(
                "estimated_batch_size"
            )
            if estimated_batch_size_override is None:
                evidence = {
                    "current_decode_sufficient": None,
                    "decode_sufficient_after_scale_in": None,
                    "inflight_decode_requests": inflight_decode_requests,
                    "current_decode_instances": len(decode_metrics),
                    "decode_instances_after_scale_in": max(
                        0, len(decode_metrics) - 1
                    ),
                    "tpot_slo_seconds": selected_slo,
                    "estimated_batch_size": None,
                    "required_decode_instances": None,
                    "batch_size_formula": "window_p20_request_nonattainment",
                    "rounding_rule": "fractional_boundary_0.2",
                    "reason": window_estimate["reason"],
                    "batch_size_estimator": batch_size_estimator,
                    "window_batch_size_evidence": window_estimate,
                }
                return None, None, evidence
            batch_size_formula_override = "window_p20_request_nonattainment"
            estimation_reason = "estimated_from_window_request_nonattainment"
        estimate = estimate_decode_sufficiency(
            inflight_decode_requests=inflight_decode_requests,
            current_decode_instances=len(decode_metrics),
            tpot_slo_seconds=selected_slo,
            tpot_intercept_ms=tpot_intercept_ms,
            tpot_per_batch_ms=tpot_per_batch_ms,
            round_required_instances_up=round_required_instances_up,
            estimated_batch_size_override=estimated_batch_size_override,
            batch_size_formula_override=batch_size_formula_override,
            estimation_reason=estimation_reason,
        )
        evidence = estimate.to_dict()
        evidence.update(
            {
                "tpot_slo_selection": (
                    "minimum_running_request_tpot_slo"
                    if running_slos
                    else (
                        "minimum_current_window_tpot_slo_fallback"
                        if window_slos
                        else "unavailable"
                    )
                ),
                "running_request_slo_samples": len(running_slos),
                "current_window_slo_samples": len(window_slos),
                "inflight_request_boundary": (
                    "sum_decode_worker_num_running_reqs"
                ),
                "batch_size_estimator": batch_size_estimator,
                "decode_batch_observation_boundary": (
                    "controller_poll_samples_of_scheduler_running_batch"
                ),
                "window_batch_size_evidence": window_estimate,
            }
        )
        return (
            estimate.current_decode_sufficient,
            estimate.decode_sufficient_after_scale_in,
            evidence,
        )

    def _online_policy_direction_decision(
        self, metrics: List[NodeMetrics], slo_monitor: PDFlipSLOMonitor
    ) -> JsonDict:
        latest_records = getattr(slo_monitor, "latest_records", None)
        if not callable(latest_records):
            return {
                "policy": self.config.decision_policy,
                "direction": None,
                "candidate_direction": None,
                "reason": "request_level_ledger_unavailable",
            }

        records = latest_records()
        queue_samples = self._online_policy_queue_evidence(metrics)
        rates = compute_policy_violation_rates(
            records,
            {
                request_id: sample["queue_seconds"]
                for request_id, sample in queue_samples.items()
            },
        )
        if self.config.decision_policy == "decode_first":
            current_sufficient, after_sufficient, evidence = (
                self._decode_sufficiency_estimate(
                    metrics,
                    records,
                    batch_size_estimator=self.config.decode_first_bs_estimator,
                )
            )
            decision = decide_decode_first(
                rates,
                current_decode_sufficient=current_sufficient,
                decode_sufficient_after_scale_in=after_sufficient,
                sufficiency_evidence=evidence,
                gap_threshold=self.config.decode_first_gap_threshold,
                protect_prefill_when_decode_insufficient=(
                    self.config.decode_first_prefill_protect
                ),
                require_prefill_gap_for_d_to_p=(
                    self.config.decode_first_d_to_p_require_prefill_gap
                ),
            ).to_dict()
        elif self.config.decision_policy == "tpot_capacity":
            current_sufficient, after_sufficient, evidence = (
                self._decode_sufficiency_estimate(
                    metrics,
                    records,
                    tpot_intercept_ms=self.config.tpot_capacity_intercept_ms,
                    tpot_per_batch_ms=(
                        self.config.tpot_capacity_batch_slope_ms
                    ),
                    round_required_instances_up=True,
                )
            )
            decision = decide_tpot_capacity(
                current_decode_sufficient=current_sufficient,
                decode_sufficient_after_scale_in=after_sufficient,
                sufficiency_evidence=evidence,
            ).to_dict()
        else:
            decision = decide_slo_target(
                rates, gap_threshold=self.config.slo_target_gap_threshold
            ).to_dict()

        if self.config.decision_policy == "tpot_capacity":
            decision.update(
                {
                    "decode_bad_tpot_intervals": rates.decode_bad_tpot_intervals,
                    "decode_total_tpot_intervals": rates.decode_total_tpot_intervals,
                    "ttft_decision_input": "ignored",
                }
            )
        else:
            decision.update(rates.to_dict())
        decision.update(
            {
                "prefill_violation_definition": (
                    "count(ttft>slo and ttft-scheduler_queue_time<=slo)"
                    "/count(valid_ttft_requests)"
                ),
                "decode_violation_definition": (
                    "bad_inter_token_intervals/all_inter_token_intervals"
                ),
                "queue_time_boundary": (
                    "scheduler_wait_queue_entry_to_first_prefill_forward"
                ),
                "queue_samples_joined": len(queue_samples),
            }
        )
        if (
            self.config.decision_policy != "tpot_capacity"
            and rates.prefill_total_requests < self.config.min_prefill_slo_samples
        ):
            decision.update(
                {
                    "candidate_direction": (
                        decision.get("direction") or decision.get("candidate_direction")
                    ),
                    "direction": None,
                    "reason": "insufficient_prefill_policy_samples",
                }
            )
        elif rates.decode_total_tpot_intervals < self.config.min_decode_slo_samples:
            decision.update(
                {
                    "candidate_direction": (
                        decision.get("direction") or decision.get("candidate_direction")
                    ),
                    "direction": None,
                    "reason": "insufficient_decode_policy_samples",
                }
            )
        elif (
            self.config.decision_policy != "tpot_capacity"
            and rates.prefill_ttft_violations_missing_queue_evidence > 0
        ):
            decision.update(
                {
                    "candidate_direction": (
                        decision.get("direction") or decision.get("candidate_direction")
                    ),
                    "direction": None,
                    "reason": "missing_queue_evidence_for_ttft_violations",
                }
            )
        return decision

    def _queue_util_direction_decision(
        self, metrics: List[NodeMetrics]
    ) -> JsonDict:
        if self._queue_policy_not_before_wall is None:
            self._reset_queue_util_policy_window()
        not_before = float(self._queue_policy_not_before_wall)

        samples_by_rid: Dict[str, JsonDict] = {}
        utilizations: Dict[str, Optional[float]] = {}
        role_seconds: Dict[str, Optional[float]] = {}
        metrics_available: Dict[str, bool] = {}
        for metric in metrics:
            if metric.effective_role != "prefill":
                continue
            utilizations[metric.name] = self._prefill_busy_ratio(metric)
            role_seconds[metric.name] = self._prefill_role_seconds(metric)
            rank_availability = []
            for dp_rank, item in enumerate(metric.dp_statuses):
                status = self._dp_runtime_status(item)
                rank_availability.append(
                    status.get("pd_flip_policy_metrics_available") is True
                )
                for raw_sample in status.get("prefill_queue_samples") or []:
                    if not isinstance(raw_sample, dict) or raw_sample.get("rid") is None:
                        continue
                    event_time = float(raw_sample.get("event_time") or 0.0)
                    if event_time < not_before:
                        continue
                    try:
                        queue_seconds = float(raw_sample["queue_seconds"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    sample = {
                        "rid": str(raw_sample["rid"]),
                        "node": metric.name,
                        "dp_rank": int(
                            item.get("dp_rank")
                            if item.get("dp_rank") is not None
                            else status.get("dp_rank", dp_rank)
                        ),
                        "seq": int(raw_sample.get("seq") or 0),
                        "role_epoch": int(raw_sample.get("role_epoch") or 0),
                        "queue_seconds": queue_seconds,
                        "event_time": event_time,
                    }
                    existing = samples_by_rid.get(sample["rid"])
                    sample_order = (
                        sample["event_time"],
                        sample["node"],
                        sample["dp_rank"],
                        sample["seq"],
                    )
                    if existing is None or sample_order < (
                        existing["event_time"],
                        existing["node"],
                        existing["dp_rank"],
                        existing["seq"],
                    ):
                        samples_by_rid[sample["rid"]] = sample
            metrics_available[metric.name] = bool(rank_availability) and all(
                rank_availability
            )

        # Missing Prefill wall-clock instrumentation blocks scale-in. Queue-based
        # scale-out remains valid because its request timestamps are separate.
        for name, available in metrics_available.items():
            if not available:
                utilizations[name] = None

        ordered_samples = sorted(
            samples_by_rid.values(),
            key=lambda item: (
                item["event_time"],
                item["node"],
                item["dp_rank"],
                item["seq"],
            ),
        )
        decision = decide_queue_util_flip_direction(
            [item["queue_seconds"] for item in ordered_samples],
            prefill_utilizations=utilizations,
            prefill_role_seconds=role_seconds,
            queue_window_size=self.config.queue_window_requests,
            queue_threshold_seconds=self.config.queue_threshold_seconds,
            queue_overload_ratio_threshold=self.config.queue_overload_ratio,
            queue_scale_in_ratio_threshold=self.config.queue_scale_in_ratio,
            prefill_scale_in_headroom_workers=(
                self.config.prefill_scale_in_headroom_workers
            ),
            prefill_min_role_seconds=self.config.prefill_min_role_seconds,
        ).to_dict()
        decision.update(
            {
                "policy": "prefill_queue_util",
                "window_not_before_wall": not_before,
                "prefill_metrics_available": metrics_available,
                "queue_window_samples": ordered_samples[
                    -self.config.queue_window_requests :
                ],
                "queue_unique_samples_since_window_start": len(ordered_samples),
                "queue_time_boundary": (
                    "scheduler_wait_queue_entry_to_first_prefill_forward"
                ),
                "busy_time_boundary": (
                    "scheduler_batch_dispatch_to_result_processing_wall_clock_union"
                ),
            }
        )
        return decision

    def dry_run(
        self,
        direction: str,
        source_name: Optional[str] = None,
        migration_target_name: Optional[str] = None,
    ) -> FlipPlan:
        metrics = self.collect_metrics()
        direction = direction.strip().lower()
        if direction == "d_to_p":
            source = self._select_source(
                metrics,
                source_name=source_name,
                expected_role="decode",
                prefer_high_load=True,
            )
            migration_target = self._select_decode_migration_target(
                metrics, source, target_name=migration_target_name
            )
            target_role = "prefill"
            actions = self._build_d_to_p_actions(source, migration_target)
            reason = (
                f"move decode node {source.name} to prefill after migrating "
                f"active decode state to {migration_target.name}"
            )
        elif direction == "p_to_d":
            source = self._select_p_to_d_source(metrics, source_name=source_name)
            migration_target = self._select_prefill_handoff_target(
                metrics, source, target_name=migration_target_name
            )
            target_role = "decode"
            actions = self._build_p_to_d_actions(source, migration_target)
            reason = (
                f"handoff bootstrap-queued requests from {source.name} to "
                f"{migration_target.name}, then move the drained source to decode"
            )
        else:
            raise ValueError("direction must be d_to_p or p_to_d")

        return FlipPlan(
            dry_run=True,
            direction=direction,
            source=source.name,
            target_role=target_role,
            migration_target=migration_target.name if migration_target else None,
            reason=reason,
            actions=actions,
            metrics=metrics,
        )

    def execute(
        self,
        direction: str,
        source_name: Optional[str] = None,
        migration_target_name: Optional[str] = None,
    ) -> FlipExecutionResult:
        started = time.monotonic()
        records: List[ActionRecord] = []
        metrics: List[NodeMetrics] = []
        source: Optional[NodeMetrics] = None
        target: Optional[NodeMetrics] = None
        target_role: Optional[str] = None
        migration_seconds = 0.0
        direction = direction.strip().lower()
        self._p_to_d_forward_only_failure = False

        try:
            metrics = self.collect_metrics()
            if direction == "d_to_p":
                source = self._select_source(
                    metrics,
                    source_name=source_name,
                    expected_role="decode",
                    prefer_high_load=True,
                )
                target = self._select_decode_migration_target(
                    metrics, source, target_name=migration_target_name
                )
                target_role = "prefill"
                migration_seconds = self._execute_d_to_p(source, target, records)
            elif direction == "p_to_d":
                source = self._select_p_to_d_source(
                    metrics, source_name=source_name
                )
                target = self._select_prefill_handoff_target(
                    metrics, source, target_name=migration_target_name
                )
                target_role = "decode"
                self._execute_p_to_d(source, target, metrics, records)
            else:
                raise ValueError("direction must be d_to_p or p_to_d")

            return FlipExecutionResult(
                success=True,
                message="pd flip executed",
                direction=direction,
                source=source.name if source else None,
                target_role=target_role,
                migration_target=target.name if target else None,
                actions=records,
                metrics=metrics,
                total_seconds=time.monotonic() - started,
                migration_seconds=migration_seconds,
            )
        except Exception as exc:
            if source is not None and not getattr(
                self, "_p_to_d_forward_only_failure", False
            ):
                self._cleanup_source_after_failure(source, records)
            return FlipExecutionResult(
                success=False,
                message=str(exc),
                direction=direction,
                source=source.name if source else source_name,
                target_role=target_role,
                migration_target=target.name if target else None,
                actions=records,
                metrics=metrics,
                total_seconds=time.monotonic() - started,
                migration_seconds=migration_seconds,
            )

    def execute_two_phase(
        self,
        direction: str,
        source_name: Optional[str] = None,
        migration_target_name: Optional[str] = None,
    ) -> MonitorLoopResult:
        direction = direction.strip().lower()
        if direction != "d_to_p":
            raise ValueError("execute-two-phase currently supports d_to_p only")

        metrics = self.collect_metrics()
        source = self._select_source(
            metrics,
            source_name=source_name,
            expected_role="decode",
            prefer_high_load=True,
        )
        target = self._select_decode_migration_target(
            metrics, source, target_name=migration_target_name
        )
        state_trace = [
            _monitor_state_record(
                state=MonitorState.SAFE,
                snapshot_index=0,
                reason="forced_two_phase",
            )
        ]
        result = self._execute_d_to_p_two_phase(
            source=source,
            target=target,
            slo_monitor=ForcedRiskMonitor(),
            enter_threshold=0.9,
            exit_threshold=2.0,
            commit_threshold=0.9,
            state_trace=state_trace,
            snapshot_index=0,
        )
        result.metrics = metrics
        return MonitorLoopResult(
            success=result.success,
            message=result.message,
            iterations=1,
            snapshots=[ForcedRiskSnapshot().to_dict()],
            actions=[result],
            state_trace=state_trace,
        )

    def monitor(
        self,
        *,
        slo_monitor: PDFlipSLOMonitor,
        enter_threshold: float,
        exit_threshold: float,
        commit_threshold: float,
        iterations: int,
        poll_interval_seconds: float,
        forced_direction: Optional[str] = None,
        scheduled_direction: Optional[str] = None,
        forced_action_not_before_seconds: float = 0.0,
        source_name: Optional[str] = None,
        migration_target_name: Optional[str] = None,
        min_prefill_workers: int = 1,
        min_decode_workers: int = 1,
        max_prefill_workers: Optional[int] = None,
        max_decode_workers: Optional[int] = None,
        expected_terminal_requests: int = 0,
        deadline_monotonic: Optional[float] = None,
    ) -> MonitorLoopResult:
        snapshots: List[JsonDict] = []
        actions: List[Any] = []
        state_trace: List[JsonDict] = []
        monitor_started = time.monotonic()
        monitor_started_wall = time.time()
        monitor_invocation_id = uuid.uuid4().hex
        journaled_snapshot_count = 0
        if (
            forced_direction is None
            and self.config.decision_policy == "prefill_queue_util"
            and self._queue_policy_not_before_wall is None
        ):
            self._reset_queue_util_policy_window()
        completed_iterations = 0
        for idx in range(max(1, iterations)):
            if (
                deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic
            ):
                break
            completed_iterations = idx + 1
            if journaled_snapshot_count < len(snapshots):
                self._append_monitor_decisions(
                    snapshots[journaled_snapshot_count:]
                )
                journaled_snapshot_count = len(snapshots)
            metrics = self.collect_metrics()
            snapshot = slo_monitor.collect_cluster(
                (m.name, m.worker_url, m.effective_role) for m in metrics
            )
            snapshot_record = snapshot.to_dict()
            snapshot_record.update(
                {
                    "monitor_invocation_id": monitor_invocation_id,
                    "monitor_iteration": idx,
                    "monitor_started_wall": monitor_started_wall,
                    "observed_wall": time.time(),
                }
            )
            snapshots.append(snapshot_record)
            terminal_count_fn = getattr(
                slo_monitor, "terminal_request_count", None
            )
            terminal_requests = (
                int(terminal_count_fn()) if callable(terminal_count_fn) else 0
            )
            snapshot_record["terminal_requests"] = terminal_requests
            if (
                expected_terminal_requests > 0
                and terminal_requests >= expected_terminal_requests
            ):
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        snapshot_index=len(snapshots) - 1,
                        reason="workload_terminal_request_count_reached",
                    )
                )
                self._append_monitor_decisions(
                    snapshots[journaled_snapshot_count:]
                )
                return MonitorLoopResult(
                    success=True,
                    message="continuous workload completed",
                    iterations=idx + 1,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )
            if not state_trace:
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        snapshot_index=len(snapshots) - 1,
                        reason="monitor_sampled",
                    )
                )
            if forced_direction not in (None, "d_to_p", "p_to_d"):
                raise ValueError("forced_direction must be d_to_p or p_to_d")
            if scheduled_direction not in (None, "d_to_p", "p_to_d"):
                raise ValueError("scheduled_direction must be d_to_p or p_to_d")
            if forced_direction is not None and scheduled_direction is not None:
                raise ValueError("forced_direction and scheduled_direction are exclusive")

            if scheduled_direction is not None:
                direction_decision = {
                    "direction": scheduled_direction,
                    "reason": "scheduled_topology_target",
                    "policy": "scheduled_topology",
                }
            elif (
                forced_direction is None
                and self.config.decision_policy == "prefill_queue_util"
            ):
                direction_decision = self._queue_util_direction_decision(metrics)
            elif (
                forced_direction is None
                and self.config.decision_policy
                in ("decode_first", "slo_target", "tpot_capacity")
            ):
                direction_decision = self._online_policy_direction_decision(
                    metrics, slo_monitor
                )
            elif forced_direction is None:
                direction_decision = decide_slo_flip_direction(
                    snapshot,
                    enter_threshold=enter_threshold,
                    gap_threshold=self.config.slo_attainment_gap_threshold,
                    min_prefill_samples=self.config.min_prefill_slo_samples,
                    min_decode_samples=self.config.min_decode_slo_samples,
                ).to_dict()
            elif forced_direction == "d_to_p":
                direction_decision = {
                    "direction": (
                        "d_to_p" if _prefill_risk(snapshot, enter_threshold) else None
                    ),
                    "reason": (
                        "forced_d_to_p_prefill_risk"
                        if _prefill_risk(snapshot, enter_threshold)
                        else "waiting_for_forced_d_to_p_prefill_risk"
                    ),
                    "policy": "forced",
                }
            else:
                direction_decision = {
                    "direction": None,
                    "reason": "waiting_for_forced_p_to_d_handoff",
                    "policy": "forced",
                }
            snapshot_record["slo_direction_decision"] = direction_decision
            snapshot_record["decision_policy"] = direction_decision.get("policy")
            if direction_decision.get("policy") == "prefill_queue_util":
                snapshot_record["queue_util_direction_decision"] = direction_decision
            elif direction_decision.get("policy") in (
                "decode_first",
                "slo_target",
                "tpot_capacity",
            ):
                snapshot_record["online_policy_direction_decision"] = direction_decision

            prefill_workers = sum(
                metric.effective_role == "prefill" for metric in metrics
            )
            decode_workers = sum(
                metric.effective_role == "decode" for metric in metrics
            )
            direction_decision["topology_before"] = {
                "prefill_workers": prefill_workers,
                "decode_workers": decode_workers,
            }
            requested_direction = direction_decision.get("direction")
            topology_block_reason = None
            if requested_direction == "d_to_p":
                if decode_workers - 1 < min_decode_workers:
                    topology_block_reason = "minimum_decode_workers"
                elif (
                    max_prefill_workers is not None
                    and prefill_workers + 1 > max_prefill_workers
                ):
                    topology_block_reason = "maximum_prefill_workers"
            elif requested_direction == "p_to_d":
                if prefill_workers - 1 < min_prefill_workers:
                    topology_block_reason = "minimum_prefill_workers"
                elif (
                    max_decode_workers is not None
                    and decode_workers + 1 > max_decode_workers
                ):
                    topology_block_reason = "maximum_decode_workers"
            if topology_block_reason is not None:
                direction_decision.update(
                    {
                        "candidate_direction": requested_direction,
                        "direction": None,
                        "feasible": False,
                        "feasibility_reason": topology_block_reason,
                    }
                )

            if direction_decision.get("direction") == "d_to_p":
                source = None
                target = None
                full_drain_capacity = None
                idle_role_flip = False
                bootstrap_drain_required = False
                if (
                    source_name is None
                    and (forced_direction is None or scheduled_direction is not None)
                ):
                    decode_candidates = sorted(
                        (
                            item
                            for item in metrics
                            if item.effective_role == "decode" and not item.draining
                        ),
                        key=_load_sort_key,
                    )
                    rollover_evidence = {
                        item.name: self._migration_rollover_eligibility(item)
                        for item in decode_candidates
                    }
                    direction_decision["migration_rollover"] = rollover_evidence
                    bootstrap_counts = {
                        item.name: self._decode_bootstrap_request_count(item)
                        for item in decode_candidates
                    }
                    direction_decision["decode_bootstrap_request_counts"] = (
                        bootstrap_counts
                    )
                    rollover_source_candidates = [
                        item
                        for item in decode_candidates
                        if rollover_evidence[item.name]["source_feasible"]
                    ]
                    # Pin the lowest-load rollover-safe Decode even when it has
                    # already-dispatched bootstrap receivers.  The executor
                    # drains that worker in the Router first, lets those
                    # existing handshakes finish with worker admission still
                    # open, and only then pauses admission and migrates.  An
                    # unobservable bootstrap state remains unsafe.
                    source_candidates = [
                        item
                        for item in rollover_source_candidates
                        if bootstrap_counts[item.name] is not None
                    ]
                    for candidate_source in source_candidates:
                        candidate_targets = sorted(
                            (
                                item
                                for item in decode_candidates
                                if item.name != candidate_source.name
                                and rollover_evidence[item.name]["target_feasible"]
                                and (
                                    migration_target_name is None
                                    or item.name == migration_target_name
                                    or item.router_worker_id == migration_target_name
                                )
                            ),
                            key=_load_sort_key,
                        )
                        candidate_bootstrap_count = bootstrap_counts[
                            candidate_source.name
                        ]
                        if candidate_targets and candidate_bootstrap_count > 0:
                            source = candidate_source
                            target = candidate_targets[0]
                            bootstrap_drain_required = True
                            full_drain_capacity = {
                                "feasible": None,
                                "reason": (
                                    "deferred_until_router_drained_bootstrap_empty"
                                ),
                                "initial_decode_bootstrap_requests": (
                                    candidate_bootstrap_count
                                ),
                            }
                            break
                        if (
                            candidate_targets
                            and self._decode_source_is_idle_for_role_flip(
                                candidate_source
                            )
                        ):
                            source = candidate_source
                            target = candidate_targets[0]
                            full_drain_capacity = (
                                self._progressive_full_drain_capacity(
                                    candidate_source, target
                                )
                            )
                            idle_role_flip = True
                            break
                        for candidate_target in candidate_targets:
                            candidate_capacity = (
                                self._progressive_full_drain_capacity(
                                    candidate_source, candidate_target
                                )
                            )
                            batch_feasible = (
                                candidate_capacity.get("feasible") is True
                                if self.config.d_to_p_direct_full_drain
                                else self._select_progressive_first_batch(
                                    candidate_source, candidate_target
                                )
                                is not None
                            )
                            if (
                                batch_feasible
                                and candidate_capacity.get("feasible") is True
                            ):
                                source = candidate_source
                                target = candidate_target
                                full_drain_capacity = candidate_capacity
                                break
                        if source is not None:
                            break
                    if source is None:
                        direction_decision["feasible"] = False
                        if not rollover_source_candidates:
                            feasibility_reason = (
                                "waiting_for_rollover_eligible_decode_source"
                            )
                        elif not source_candidates:
                            feasibility_reason = (
                                "waiting_for_observable_decode_bootstrap_state"
                            )
                        else:
                            feasibility_reason = (
                                "waiting_for_full_decode_drain_capacity"
                            )
                        direction_decision["feasibility_reason"] = feasibility_reason
                        if idx + 1 < max(1, iterations):
                            time.sleep(poll_interval_seconds)
                        continue
                else:
                    source = self._select_source(
                        metrics,
                        source_name=source_name,
                        expected_role="decode",
                        prefer_high_load=True,
                    )
                    target = self._select_decode_migration_target(
                        metrics, source, target_name=migration_target_name
                    )
                direction_decision["feasible"] = True
                direction_decision["selected_source"] = source.name
                direction_decision["selected_migration_target"] = target.name
                direction_decision["bootstrap_drain_required"] = (
                    bootstrap_drain_required
                )
                if bootstrap_drain_required:
                    direction_decision["migration_mode"] = (
                        "router_drain_then_bootstrap_quiesce"
                    )
                else:
                    direction_decision["migration_mode"] = (
                        "idle_decode_direct_role_flip"
                        if idle_role_flip
                        else (
                            "direct_full_source_atomic"
                            if self.config.d_to_p_direct_full_drain
                            else "progressive_request_migration"
                        )
                    )
                if full_drain_capacity is not None:
                    direction_decision["full_drain_capacity"] = (
                        full_drain_capacity
                    )
                if forced_direction is None or scheduled_direction is not None:
                    self._append_monitor_decisions([snapshot_record])
                    journaled_snapshot_count = len(snapshots)
                    if self.config.d_to_p_direct_full_drain:
                        return self._execute_direct_full_drain_d_to_p(
                            source=source,
                            target=target,
                            snapshots=snapshots,
                            records=[],
                            state_trace=state_trace,
                            iterations=idx + 1,
                            require_full_drain_capacity=True,
                        )
                    # Compatibility path for experiments that explicitly keep
                    # the historical prefix/observation/final sequence.
                    return self._execute_progressive_d_to_p(
                        source=source,
                        target=target,
                        slo_monitor=slo_monitor,
                        monitor_nodes=[
                            (
                                metric.name,
                                metric.worker_url,
                                metric.effective_role,
                            )
                            for metric in metrics
                        ],
                        snapshots=snapshots,
                        records=[],
                        state_trace=state_trace,
                        iterations=idx + 1,
                        require_full_drain_capacity=True,
                        force_commit=scheduled_direction is not None,
                    )
                result = self._execute_d_to_p_two_phase(
                    source=source,
                    target=target,
                    slo_monitor=slo_monitor,
                    enter_threshold=enter_threshold,
                    exit_threshold=exit_threshold,
                    commit_threshold=commit_threshold,
                    state_trace=state_trace,
                    snapshot_index=len(snapshots) - 1,
                )
                actions.append(result)
                self._append_monitor_decisions([snapshot_record])
                journaled_snapshot_count = len(snapshots)
                return MonitorLoopResult(
                    success=result.success,
                    message=result.message,
                    iterations=idx + 1,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )
            forced_p_to_d_ready = (
                forced_direction == "p_to_d"
                and time.monotonic() - monitor_started
                >= max(0.0, forced_action_not_before_seconds)
                and any(
                    metric.effective_role == "prefill"
                    and not metric.draining
                    and len(metric.dp_statuses) == 1
                    and (source_name is None or metric.name == source_name)
                    and len(
                        self._prefill_handoff_owner_ready_rids(metrics, metric)
                    )
                    >= self.config.p_to_d_min_handoff_requests
                    for metric in metrics
                )
            )
            if forced_p_to_d_ready:
                direction_decision.update(
                    {
                        "direction": "p_to_d",
                        "reason": "forced_p_to_d_handoff_ready",
                    }
                )
            p_to_d_triggered = forced_p_to_d_ready or (
                direction_decision.get("direction") == "p_to_d"
            )
            if p_to_d_triggered:
                if not forced_p_to_d_ready:
                    try:
                        candidate_source = self._select_p_to_d_source(
                            metrics, source_name=source_name
                        )
                        candidate_target = self._select_prefill_handoff_target(
                            metrics,
                            candidate_source,
                            target_name=migration_target_name,
                        )
                        owner_ready = self._prefill_handoff_owner_ready_rids(
                            metrics, candidate_source
                        )
                    except (RuntimeError, ValueError) as exc:
                        direction_decision["feasible"] = False
                        direction_decision["feasibility_reason"] = str(exc)
                        if idx + 1 < max(1, iterations):
                            time.sleep(poll_interval_seconds)
                        continue
                    if len(owner_ready) < self.config.p_to_d_min_handoff_requests:
                        direction_decision["feasible"] = False
                        direction_decision["feasibility_reason"] = (
                            "waiting_for_prefill_bootstrap_handoff_batch"
                        )
                        direction_decision["owner_ready_handoff_requests"] = len(
                            owner_ready
                        )
                        if idx + 1 < max(1, iterations):
                            time.sleep(poll_interval_seconds)
                        continue
                    direction_decision["feasible"] = True
                    direction_decision["selected_source"] = candidate_source.name
                    direction_decision["selected_migration_target"] = (
                        candidate_target.name
                    )
                    direction_decision["owner_ready_handoff_requests"] = len(
                        owner_ready
                    )
                result = self._execute_p_to_d_monitor(
                    metrics=metrics,
                    state_trace=state_trace,
                    snapshot_index=len(snapshots) - 1,
                    source_name=source_name,
                    migration_target_name=migration_target_name,
                    reason=(
                        "forced_p_to_d_handoff_ready"
                        if forced_p_to_d_ready
                        else direction_decision.get("reason", "automatic_p_to_d")
                    ),
                    require_handoff=True,
                )
                actions.append(result)
                self._append_monitor_decisions([snapshot_record])
                journaled_snapshot_count = len(snapshots)
                if (
                    not result.success
                    and result.message == P_TO_D_HANDOFF_RACE_MESSAGE
                ):
                    if idx + 1 < max(1, iterations):
                        time.sleep(poll_interval_seconds)
                    continue
                return MonitorLoopResult(
                    success=result.success,
                    message=result.message,
                    iterations=idx + 1,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )
            time.sleep(poll_interval_seconds)

        if journaled_snapshot_count < len(snapshots):
            self._append_monitor_decisions(snapshots[journaled_snapshot_count:])
        return MonitorLoopResult(
            success=True,
            message="no flip decision",
            iterations=completed_iterations,
            snapshots=snapshots,
            actions=actions,
            state_trace=state_trace,
        )

    def monitor_continuous(
        self,
        *,
        slo_monitor: PDFlipSLOMonitor,
        enter_threshold: float,
        exit_threshold: float,
        commit_threshold: float,
        iterations: int,
        poll_interval_seconds: float,
        cooldown_seconds: float,
        max_flips: int = 0,
        min_prefill_workers: int = 1,
        min_decode_workers: int = 1,
        max_prefill_workers: Optional[int] = None,
        max_decode_workers: Optional[int] = None,
        expected_terminal_requests: int = 0,
    ) -> MonitorLoopResult:
        """Continuously re-evaluate SLOs and commit multiple topology flips.

        ``monitor`` intentionally remains a one-shot compatibility path.  This
        wrapper starts a fresh logical SLO window after each completed flip,
        waits through a bounded cooldown, rediscovers the live topology on the
        next cycle, and aggregates evidence with globally valid snapshot
        indices.  ``max_flips=0`` means that only the iteration budget bounds
        the controller lifetime.
        """
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if max_flips < 0:
            raise ValueError("max_flips must be non-negative")
        if expected_terminal_requests < 0:
            raise ValueError("expected_terminal_requests must be non-negative")
        if min_prefill_workers < 1 or min_decode_workers < 1:
            raise ValueError("minimum P/D worker counts must be positive")
        if (
            max_prefill_workers is not None
            and max_prefill_workers < min_prefill_workers
        ):
            raise ValueError("max_prefill_workers must be >= min_prefill_workers")
        if (
            max_decode_workers is not None
            and max_decode_workers < min_decode_workers
        ):
            raise ValueError("max_decode_workers must be >= min_decode_workers")

        snapshots: List[JsonDict] = []
        actions: List[Any] = []
        state_trace: List[JsonDict] = []
        consumed_iterations = 0
        completed_flips = 0
        cycle_index = 0
        if (
            self.config.decision_policy == "prefill_queue_util"
            and self._queue_policy_not_before_wall is None
        ):
            self._reset_queue_util_policy_window()

        while consumed_iterations < iterations:
            remaining = iterations - consumed_iterations
            result = self.monitor(
                slo_monitor=slo_monitor,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold,
                commit_threshold=commit_threshold,
                iterations=remaining,
                poll_interval_seconds=poll_interval_seconds,
                min_prefill_workers=min_prefill_workers,
                min_decode_workers=min_decode_workers,
                max_prefill_workers=max_prefill_workers,
                max_decode_workers=max_decode_workers,
                expected_terminal_requests=expected_terminal_requests,
            )
            snapshot_offset = len(snapshots)
            for snapshot in result.snapshots:
                record = dict(snapshot)
                record["continuous_cycle"] = cycle_index
                snapshots.append(record)
            for state in result.state_trace:
                record = dict(state)
                if isinstance(record.get("snapshot_index"), int):
                    record["snapshot_index"] += snapshot_offset
                record["continuous_cycle"] = cycle_index
                state_trace.append(record)
            actions.extend(result.actions)
            consumed_iterations += max(1, int(result.iterations))

            if not result.success:
                return MonitorLoopResult(
                    success=False,
                    message=result.message,
                    iterations=consumed_iterations,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )

            cycle_completed = any(
                item.get("reason") == "role_flip_complete"
                for item in result.state_trace
            )
            if not cycle_completed:
                if result.message == "continuous workload completed":
                    return MonitorLoopResult(
                        success=True,
                        message=result.message,
                        iterations=consumed_iterations,
                        snapshots=snapshots,
                        actions=actions,
                        state_trace=state_trace,
                    )
                # A one-shot monitor can return successfully without flipping
                # when the SLO recovers during the source-quiesce check.  That
                # is a transient no-op, not the end of a continuous run.
                if consumed_iterations < iterations:
                    if poll_interval_seconds > 0:
                        time.sleep(poll_interval_seconds)
                    continue
                return MonitorLoopResult(
                    success=True,
                    message="continuous monitor exhausted without another flip",
                    iterations=consumed_iterations,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )

            completed_flips += 1
            if max_flips and completed_flips >= max_flips:
                return MonitorLoopResult(
                    success=True,
                    message="continuous monitor reached max_flips",
                    iterations=consumed_iterations,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )

            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.COOLDOWN,
                    reason="post_flip_cooldown_started",
                )
            )
            state_trace[-1]["continuous_cycle"] = cycle_index
            state_trace[-1]["cooldown_seconds"] = cooldown_seconds
            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)
            reset_window = getattr(slo_monitor, "reset_window", None)
            if callable(reset_window):
                reset_window()
            if self.config.decision_policy == "prefill_queue_util":
                self._reset_queue_util_policy_window()
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.SAFE,
                    reason="post_flip_fresh_window_started",
                )
            )
            state_trace[-1]["continuous_cycle"] = cycle_index + 1
            cycle_index += 1

        return MonitorLoopResult(
            success=True,
            message="continuous monitor exhausted iteration budget",
            iterations=consumed_iterations,
            snapshots=snapshots,
            actions=actions,
            state_trace=state_trace,
        )

    def monitor_scheduled(
        self,
        *,
        slo_monitor: PDFlipSLOMonitor,
        ledger_path: str,
        schedule: Sequence[JsonDict],
        poll_interval_seconds: float,
        start_timeout_seconds: float,
        event_timeout_seconds: float,
    ) -> MonitorLoopResult:
        """Reach trace-relative topology targets using sequential safe flips."""

        if poll_interval_seconds <= 0:
            raise ValueError("scheduled poll interval must be positive")
        if start_timeout_seconds <= 0 or event_timeout_seconds <= 0:
            raise ValueError("scheduled controller timeouts must be positive")

        trace_start = wait_for_trace_start_monotonic(
            ledger_path,
            timeout_seconds=start_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        snapshots: List[JsonDict] = []
        actions: List[Any] = []
        state_trace: List[JsonDict] = []
        consumed_iterations = 0
        completed_subflips = 0
        total_workers = len(self.config.nodes)

        for event in schedule:
            planned_offset = float(event["offset_seconds"])
            planned_deadline = trace_start + planned_offset
            while time.monotonic() < planned_deadline:
                time.sleep(
                    min(
                        poll_interval_seconds,
                        max(0.0, planned_deadline - time.monotonic()),
                    )
                )

            event_started = time.monotonic()
            target_prefill = int(event["prefill_workers"])
            target_decode = total_workers - target_prefill
            subflip_index = 0
            while True:
                metrics = self.collect_metrics()
                current_prefill = sum(
                    metric.effective_role == "prefill" for metric in metrics
                )
                current_decode = sum(
                    metric.effective_role == "decode" for metric in metrics
                )
                if current_prefill + current_decode != total_workers:
                    raise RuntimeError("scheduled controller observed incomplete topology")
                if current_prefill == target_prefill:
                    break
                if time.monotonic() - event_started > event_timeout_seconds:
                    raise TimeoutError(
                        "scheduled topology event {} timed out at {}P{}D; target {}"
                        .format(
                            event["event_index"],
                            current_prefill,
                            current_decode,
                            event["topology"],
                        )
                    )

                direction = "d_to_p" if target_prefill > current_prefill else "p_to_d"
                trigger_monotonic = time.monotonic()
                trigger_wall = time.time()
                trigger_elapsed = trigger_monotonic - trace_start
                schedule_fields = {
                    "schedule_event_index": int(event["event_index"]),
                    "scheduled_offset_seconds": planned_offset,
                    "scheduled_topology": event["topology"],
                    "scheduled_note": event.get("note"),
                    "scheduled_subflip_index": subflip_index,
                    "scheduled_direction": direction,
                    "topology_before": {
                        "prefill_workers": current_prefill,
                        "decode_workers": current_decode,
                    },
                    "actual_trigger_elapsed_seconds": trigger_elapsed,
                    "trigger_lag_seconds": trigger_elapsed - planned_offset,
                }
                trigger_state = _monitor_state_record(
                    state=MonitorState.SELECTING,
                    direction=direction,
                    reason="scheduled_subflip_triggered",
                )
                trigger_state.update(schedule_fields)
                state_trace.append(trigger_state)
                actions.append(
                    ActionRecord(
                        step="scheduled_subflip_trigger",
                        target=event["topology"],
                        method="TIMER",
                        url="",
                        payload=schedule_fields,
                        response=None,
                        elapsed_seconds=0.0,
                        start_wall=trigger_wall,
                        start_monotonic=trigger_monotonic,
                        end_wall=trigger_wall,
                        end_monotonic=trigger_monotonic,
                    )
                )

                remaining = event_timeout_seconds - (
                    time.monotonic() - event_started
                )
                iteration_budget = max(1, int(remaining / poll_interval_seconds) + 1)
                result = self.monitor(
                    slo_monitor=slo_monitor,
                    enter_threshold=0.0,
                    exit_threshold=1.0,
                    commit_threshold=0.0,
                    iterations=iteration_budget,
                    poll_interval_seconds=poll_interval_seconds,
                    scheduled_direction=direction,
                    min_prefill_workers=1,
                    min_decode_workers=1,
                    max_prefill_workers=total_workers - 1,
                    max_decode_workers=total_workers - 1,
                    deadline_monotonic=event_started + event_timeout_seconds,
                )
                snapshot_offset = len(snapshots)
                for snapshot in result.snapshots:
                    row = dict(snapshot)
                    row.update(schedule_fields)
                    snapshots.append(row)
                annotated_states = []
                for state in result.state_trace:
                    row = dict(state)
                    if isinstance(row.get("snapshot_index"), int):
                        row["snapshot_index"] += snapshot_offset
                    row.update(schedule_fields)
                    annotated_states.append(row)
                state_trace.extend(annotated_states)
                actions.extend(result.actions)
                consumed_iterations += max(1, int(result.iterations))
                if not result.success:
                    return MonitorLoopResult(
                        success=False,
                        message=result.message,
                        iterations=consumed_iterations,
                        snapshots=snapshots,
                        actions=actions,
                        state_trace=state_trace,
                    )
                completed = any(
                    row.get("reason") == "role_flip_complete"
                    for row in annotated_states
                )
                if not completed:
                    continue

                after_metrics = self.collect_metrics()
                after_prefill = sum(
                    metric.effective_role == "prefill" for metric in after_metrics
                )
                expected_after = current_prefill + (
                    1 if direction == "d_to_p" else -1
                )
                if after_prefill != expected_after:
                    raise RuntimeError(
                        "scheduled subflip topology mismatch: expected {}P, got {}P"
                        .format(expected_after, after_prefill)
                    )
                completed_subflips += 1
                subflip_index += 1

            completed_monotonic = time.monotonic()
            event_state = _monitor_state_record(
                state=MonitorState.SAFE,
                reason="scheduled_topology_reached",
            )
            event_state.update(
                {
                    "schedule_event_index": int(event["event_index"]),
                    "scheduled_offset_seconds": planned_offset,
                    "scheduled_topology": event["topology"],
                    "scheduled_note": event.get("note"),
                    "actual_event_completed_elapsed_seconds": (
                        completed_monotonic - trace_start
                    ),
                    "event_completion_lag_seconds": (
                        completed_monotonic - trace_start - planned_offset
                    ),
                    "completed_subflips_total": completed_subflips,
                }
            )
            state_trace.append(event_state)

        return MonitorLoopResult(
            success=True,
            message="scheduled topology targets completed",
            iterations=consumed_iterations,
            snapshots=snapshots,
            actions=actions,
            state_trace=state_trace,
        )

    def monitor_progressive(
        self,
        slo_monitor: PDFlipSLOMonitor,
        *,
        iterations: int,
        poll_interval_seconds: Optional[float] = None,
        source_name: Optional[str] = None,
        migration_target_name: Optional[str] = None,
    ) -> MonitorLoopResult:
        snapshots: List[JsonDict] = []
        records: List[ActionRecord] = []
        state_trace: List[JsonDict] = []
        interval = (
            self.config.migration_poll_interval_seconds
            if poll_interval_seconds is None
            else max(0.0, poll_interval_seconds)
        )
        iteration_count = max(1, iterations)
        for idx in range(iteration_count):
            metrics = self.collect_metrics()
            monitor_nodes = [
                (metric.name, metric.worker_url, metric.effective_role)
                for metric in metrics
            ]
            snapshot = slo_monitor.collect_cluster(monitor_nodes)
            snapshots.append(snapshot.to_dict())
            if not state_trace:
                state_trace.append(
                    _monitor_state_record(
                        state=ProgressiveMonitorState.SAFE,
                        reason="monitor_sampled",
                        snapshot_index=len(snapshots) - 1,
                    )
                )
                state_trace[-1].update(
                    self._progressive_observability_fields(snapshot, None)
                )
            decision = self._evaluate_progressive_snapshot(snapshot, observing=False)
            if decision is ProgressiveDecision.START:
                trigger_evidence = getattr(slo_monitor, "trigger_evidence", None)
                if callable(trigger_evidence):
                    evidence = trigger_evidence()
                    if evidence:
                        monotonic_to_wall = time.time() - time.monotonic()
                        for field in (
                            "threshold_crossing_time",
                            "poll_detection_time",
                        ):
                            value = evidence.get(field)
                            if isinstance(value, (int, float)):
                                evidence[field + "_wall"] = (
                                    float(value) + monotonic_to_wall
                                )
                        snapshots[-1]["trigger"] = evidence
                        state_trace[-1]["trigger"] = evidence
                auto_target: Optional[NodeMetrics] = None
                decode_candidates = sorted(
                    (
                        item
                        for item in metrics
                        if item.effective_role == "decode" and not item.draining
                    ),
                    key=_load_sort_key,
                )
                rollover_evidence = {
                    item.name: self._migration_rollover_eligibility(item)
                    for item in decode_candidates
                }
                bootstrap_counts = {
                    item.name: self._decode_bootstrap_request_count(item)
                    for item in decode_candidates
                }
                if source_name is None:
                    rollover_source_candidates = [
                        item
                        for item in decode_candidates
                        if rollover_evidence[item.name]["source_feasible"]
                    ]
                    source_candidates = [
                        item
                        for item in rollover_source_candidates
                        if bootstrap_counts[item.name] is not None
                    ]
                    if not source_candidates:
                        reason = (
                            "waiting_for_rollover_eligible_decode_source"
                            if not rollover_source_candidates
                            else "waiting_for_observable_decode_bootstrap_state"
                        )
                        state_trace[-1]["reason"] = reason
                        state_trace[-1]["migration_rollover"] = rollover_evidence
                        state_trace[-1]["decode_bootstrap_request_counts"] = (
                            bootstrap_counts
                        )
                        state_trace[-1].update(
                            self._progressive_observability_fields(snapshot, None)
                        )
                        if idx + 1 < iteration_count:
                            time.sleep(interval)
                            continue
                        raise RuntimeError(
                            "no bootstrap-observable rollover-eligible decode "
                            "source is available"
                        )
                    feasible_pair = None
                    for candidate_source in source_candidates:
                        candidate_targets = sorted(
                            (
                                item
                                for item in decode_candidates
                                if item.name != candidate_source.name
                                and rollover_evidence[item.name]["target_feasible"]
                                and (
                                    migration_target_name is None
                                    or item.name == migration_target_name
                                    or item.router_worker_id
                                    == migration_target_name
                                )
                            ),
                            key=_load_sort_key,
                        )
                        if (
                            candidate_targets
                            and bootstrap_counts[candidate_source.name] > 0
                        ):
                            feasible_pair = (
                                candidate_source,
                                candidate_targets[0],
                            )
                            break
                        if (
                            candidate_targets
                            and self._decode_source_is_idle_for_role_flip(
                                candidate_source
                            )
                        ):
                            feasible_pair = (
                                candidate_source,
                                candidate_targets[0],
                            )
                            break
                        for candidate_target in candidate_targets:
                            if self.config.d_to_p_direct_full_drain:
                                feasible = self._progressive_full_drain_capacity(
                                    candidate_source, candidate_target
                                ).get("feasible") is True
                            else:
                                feasible = (
                                    self._select_progressive_first_batch(
                                        candidate_source, candidate_target
                                    )
                                    is not None
                                )
                            if feasible:
                                feasible_pair = (
                                    candidate_source,
                                    candidate_target,
                                )
                                break
                        if feasible_pair is not None:
                            break
                    if feasible_pair is None:
                        source = source_candidates[0]
                    else:
                        source, auto_target = feasible_pair
                else:
                    source = self._select_source(
                        metrics,
                        source_name=source_name,
                        expected_role="decode",
                        prefer_high_load=False,
                    )
                selection_evidence = {
                    "policy": (
                        "lowest_current_decode_load_router_drain_before_"
                        "bootstrap_wait"
                    ),
                    "migration_rollover": rollover_evidence,
                    "decode_bootstrap_request_counts": bootstrap_counts,
                    "candidates": [
                        {
                            "name": metric.name,
                            "running_reqs": metric.running_reqs,
                            "router_active_load": metric.router_active_load,
                            "total_tokens": metric.total_tokens,
                            "token_usage": metric.token_usage,
                            "decode_bootstrap_requests": bootstrap_counts[
                                metric.name
                            ],
                            "load_sort_key": list(_load_sort_key(metric)),
                            "source_rollover_feasible": rollover_evidence[
                                metric.name
                            ]["source_feasible"],
                            "target_rollover_feasible": rollover_evidence[
                                metric.name
                            ]["target_feasible"],
                        }
                        for metric in decode_candidates
                    ],
                    "selected_source": source.name,
                    "migration_mode": (
                        "router_drain_then_bootstrap_quiesce"
                        if (
                            self._decode_bootstrap_request_count(source)
                            is not None
                            and self._decode_bootstrap_request_count(source) > 0
                        )
                        else (
                            "idle_decode_direct_role_flip"
                            if self._decode_source_is_idle_for_role_flip(source)
                            else (
                                "direct_full_source_atomic"
                                if self.config.d_to_p_direct_full_drain
                                else "progressive_request_migration"
                            )
                        )
                    ),
                }
                snapshots[-1]["source_selection"] = selection_evidence
                state_trace[-1]["source_selection"] = selection_evidence
                if migration_target_name is None:
                    if auto_target is not None:
                        target = auto_target
                    else:
                        target_candidates = sorted(
                            (
                                item
                                for item in metrics
                                if item.name != source.name
                                and item.effective_role == "decode"
                                and not item.draining
                            ),
                            key=_load_sort_key,
                        )
                        if not target_candidates:
                            raise RuntimeError(
                                "D->P requires another non-draining decode node "
                                "as migration target"
                            )
                        target = target_candidates[0]
                else:
                    target = _find_metric(metrics, migration_target_name)
                    if target is None:
                        raise RuntimeError(
                            f"migration target {migration_target_name!r} was not found"
                        )
                    if target.name == source.name:
                        raise RuntimeError("migration source and target must differ")
                    if target.effective_role != "decode":
                        raise RuntimeError(
                            f"migration target {target.name} is not decode"
                        )
                selection_evidence["selected_migration_target"] = target.name
                if self.config.d_to_p_direct_full_drain:
                    migration_ready = self._progressive_full_drain_capacity(
                        source, target
                    ).get("feasible") is True
                    waiting_reason = "waiting_for_full_decode_drain_capacity"
                else:
                    migration_ready = (
                        self._select_progressive_first_batch(source, target)
                        is not None
                    )
                    waiting_reason = "waiting_for_feasible_first_migration_batch"
                if (
                    not migration_ready
                    and not self._decode_source_is_idle_for_role_flip(source)
                    and self._decode_bootstrap_request_count(source) == 0
                ):
                    # A TTFT breach can be observed before the selected Decode
                    # has received its first running request. Treat that as a
                    # transient monitor state instead of permanently ending
                    # the state machine before measured traffic reaches D.
                    state_trace[-1]["reason"] = waiting_reason
                    state_trace[-1].update(
                        self._progressive_observability_fields(snapshot, None)
                    )
                    if idx + 1 < iteration_count:
                        time.sleep(interval)
                        continue
                if self.config.d_to_p_direct_full_drain:
                    return self._execute_direct_full_drain_d_to_p(
                        source=source,
                        target=target,
                        snapshots=snapshots,
                        records=records,
                        state_trace=state_trace,
                        iterations=idx + 1,
                        require_full_drain_capacity=True,
                    )
                return self._execute_progressive_d_to_p(
                    source=source,
                    target=target,
                    slo_monitor=slo_monitor,
                    monitor_nodes=monitor_nodes,
                    snapshots=snapshots,
                    records=records,
                    state_trace=state_trace,
                    iterations=idx + 1,
                )
            if idx + 1 < iteration_count:
                time.sleep(interval)

        return MonitorLoopResult(
            success=True,
            message="no progressive flip decision",
            iterations=iteration_count,
            snapshots=snapshots,
            actions=records,
            state_trace=state_trace,
        )

    def _execute_direct_full_drain_d_to_p(
        self,
        *,
        source: NodeMetrics,
        target: NodeMetrics,
        snapshots: List[JsonDict],
        records: List[ActionRecord],
        state_trace: List[JsonDict],
        iterations: int,
        require_full_drain_capacity: bool = True,
    ) -> MonitorLoopResult:
        """Quiesce bootstrap, freeze admission, and migrate the whole source once."""

        session_prefix = self._progressive_session_prefix(source, target)
        trigger_snapshot_index = len(snapshots) - 1
        source_finished = False
        migrated_rids: Tuple[str, ...] = ()
        try:
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SELECTING,
                source,
                target,
                "direct_full_drain_waiting_for_bootstrap_quiesce",
                records,
            )
            state_trace[-1]["migration_mode"] = "direct_full_source_atomic"

            self._post_router(
                records,
                "router_drain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": True},
            )
            self._wait_router_dispatch_quiesced(records, source)
            bootstrap_wait = self._wait_for_drained_decode_bootstrap_empty(
                source, records
            )
            state_trace[-1].update(
                {
                    "decode_bootstrap_drain": bootstrap_wait,
                    "router_drained_before_worker_admission_pause": True,
                }
            )

            self._refresh_progressive_runtime_status(
                target, records, "refresh_target_after_bootstrap_drain"
            )
            if require_full_drain_capacity:
                capacity = self._progressive_full_drain_capacity(source, target)
                if not capacity.get("feasible"):
                    capacity = self._wait_for_progressive_full_drain_capacity(
                        source, target, records
                    )
                state_trace[-1]["full_drain_capacity_after_bootstrap_quiesce"] = (
                    capacity
                )

            self._post_worker(
                records,
                "pause_source_admission",
                source,
                "/pd_flip/runtime_role/admission",
                {"paused": True},
            )
            self._refresh_progressive_runtime_status(
                source, records, "refresh_source_after_admission_pause"
            )
            paused_statuses = _index_dp_responses(source.dp_statuses)
            paused_runtime_statuses = [
                item.get("status")
                for item in paused_statuses.values()
                if isinstance(item, dict)
            ]
            pause_status_observable = bool(paused_statuses) and all(
                isinstance(status, dict)
                and isinstance(status.get("running_requests"), list)
                and isinstance(status.get("waiting_requests"), list)
                and isinstance(status.get("decode_bootstrap_requests"), list)
                for status in paused_runtime_statuses
            )
            if not pause_status_observable:
                raise RuntimeError(
                    "source queues are not fully observable after admission pause"
                )
            source.dp_statuses = [
                paused_statuses[rank] for rank in sorted(paused_statuses)
            ]
            source.raw_status = _aggregate_dp_runtime_status(
                source.dp_statuses, source.name
            )
            if self._decode_bootstrap_request_count(source) != 0:
                raise RuntimeError(
                    "Decode bootstrap reappeared after admission pause"
                )

            if require_full_drain_capacity:
                capacity = self._progressive_full_drain_capacity(source, target)
                if not capacity.get("feasible"):
                    capacity = self._wait_for_progressive_full_drain_capacity(
                        source, target, records
                    )
                state_trace[-1]["full_drain_capacity_after_admission_pause"] = (
                    capacity
                )

            running_rids, waiting_count = self._source_pending_requests(
                source, records
            )
            if running_rids or waiting_count:
                self._append_progressive_state(
                    state_trace,
                    ProgressiveMonitorState.FULL_MIGRATING,
                    source,
                    target,
                    "all_running_and_waiting_selected_atomically",
                    records,
                )
                migration_session_id = session_prefix + "-final"
                for selection_attempt in range(4):
                    try:
                        migrated_rids = self._execute_atomic_batch(
                            source,
                            target,
                            migration_session_id,
                            running_rids,
                            True,
                            select_all_running=True,
                            records=records,
                            next_fsm_phase="role_flip_worker_prefill_intent",
                        )
                        source_finished = True
                        break
                    except ProgressiveAtomicBatchError as exc:
                        retryable_completed_request = (
                            not exc.source_finished
                            and not exc.cutover_started
                            and D_TO_P_REQUEST_COMPLETED_BEFORE_BASE_TRANSFER_MESSAGE
                            in str(exc)
                        )
                        if not retryable_completed_request or selection_attempt >= 3:
                            raise

                        self._refresh_progressive_runtime_status(
                            source,
                            records,
                            "refresh_source_after_completed_request",
                        )
                        self._refresh_progressive_runtime_status(
                            target,
                            records,
                            "refresh_target_after_completed_request",
                        )
                        running_rids, waiting_count = self._source_pending_requests(
                            source, records
                        )
                        state_trace[-1].update(
                            {
                                "selection_retry_count": selection_attempt + 1,
                                "selection_retry_reason": "completed_request",
                                "selection_retry_running_requests": len(
                                    running_rids
                                ),
                                "selection_retry_waiting_requests": waiting_count,
                            }
                        )
                        if not running_rids and waiting_count == 0:
                            migrated_rids = ()
                            break
                        if require_full_drain_capacity:
                            capacity = self._progressive_full_drain_capacity(
                                source, target
                            )
                            if not capacity.get("feasible"):
                                capacity = (
                                    self._wait_for_progressive_full_drain_capacity(
                                        source, target, records
                                    )
                                )
                            state_trace[-1][
                                "full_drain_capacity_after_completed_request_retry"
                            ] = capacity
                        migration_session_id = (
                            session_prefix
                            + "-final-completed_request-retry-{}".format(
                                selection_attempt + 1
                            )
                        )

            self._assert_source_idle_after_migration(records, source)
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.FLIPPING_ROLE,
                source,
                target,
                "source_idle_after_direct_full_drain",
                records,
            )
            self._flip_idle_source_to_prefill(
                source,
                target,
                session_prefix + "-role-flip",
                migrated_rids,
                records,
            )
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SAFE,
                source,
                target,
                "role_flip_complete",
                records,
            )
            state_trace[-1].update(
                {
                    "snapshot_index": trigger_snapshot_index,
                    "migration_mode": (
                        "direct_full_source_atomic"
                        if migrated_rids
                        else "idle_decode_direct_role_flip"
                    ),
                    "migrated_request_count": len(migrated_rids),
                    "first_migration_skipped": True,
                    "observation_skipped": True,
                }
            )
            return self._progressive_result(
                True,
                "source switched to prefill after direct full drain",
                iterations,
                snapshots,
                records,
                state_trace,
            )
        except Exception as exc:
            post_finish_error = source_finished or (
                isinstance(exc, ProgressiveAtomicBatchError) and exc.source_finished
            )
            router_pending = isinstance(exc, RoleFlipRouterPendingError)
            cutover_pending = isinstance(
                exc, ProgressiveAtomicBatchError
            ) and exc.cutover_started
            if not router_pending and not cutover_pending:
                self._cleanup_source_after_failure(source, records)
            self._append_progressive_state(
                state_trace,
                (
                    ProgressiveMonitorState.FLIPPING_ROLE
                    if router_pending or cutover_pending
                    else ProgressiveMonitorState.SAFE
                ),
                source,
                target,
                (
                    "role_flip_router_pending"
                    if router_pending
                    else "ownership_cutover_pending"
                    if cutover_pending
                    else "post_finish_error"
                    if post_finish_error
                    else "error_recovered"
                ),
                records,
            )
            return self._progressive_result(
                False,
                str(exc),
                iterations,
                snapshots,
                records,
                state_trace,
            )

    def _execute_progressive_d_to_p(
        self,
        *,
        source: NodeMetrics,
        target: NodeMetrics,
        slo_monitor: PDFlipSLOMonitor,
        monitor_nodes: List[Tuple[str, str, str]],
        snapshots: List[JsonDict],
        records: List[ActionRecord],
        state_trace: List[JsonDict],
        iterations: int,
        require_full_drain_capacity: bool = False,
        force_commit: bool = False,
    ) -> MonitorLoopResult:
        session_prefix = self._progressive_session_prefix(source, target)
        trigger_snapshot_index = len(snapshots) - 1
        source_finished = False
        try:
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SELECTING,
                source,
                target,
                "prefill_risky_decode_healthy",
                records,
            )
            selection = self._select_progressive_first_batch(source, target)
            state_trace[-1].update(
                self._progressive_observability_fields(snapshots[-1], selection)
            )
            initial_bootstrap_count = self._decode_bootstrap_request_count(source)
            if (
                selection is None
                and not self._decode_source_is_idle_for_role_flip(source)
                and not (
                    initial_bootstrap_count is not None
                    and initial_bootstrap_count > 0
                )
            ):
                self._append_progressive_state(
                    state_trace,
                    ProgressiveMonitorState.SAFE,
                    source,
                    target,
                    "first_batch_capacity_insufficient",
                    records,
                )
                return self._progressive_result(
                    True,
                    "no feasible first migration batch",
                    iterations,
                    snapshots,
                    records,
                    state_trace,
                )

            self._post_router(
                records,
                "router_drain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": True},
            )
            self._wait_router_dispatch_quiesced(records, source)
            bootstrap_wait = self._wait_for_drained_decode_bootstrap_empty(
                source, records
            )
            state_trace[-1].update(
                {
                    "decode_bootstrap_drain": bootstrap_wait,
                    "router_drained_before_worker_admission_pause": True,
                }
            )
            self._refresh_progressive_runtime_status(
                target, records, "refresh_target_after_bootstrap_drain"
            )
            if require_full_drain_capacity:
                capacity = self._progressive_full_drain_capacity(source, target)
                if not capacity.get("feasible"):
                    capacity = self._wait_for_progressive_full_drain_capacity(
                        source, target, records
                    )
                state_trace[-1][
                    "full_drain_capacity_after_bootstrap_quiesce"
                ] = capacity
            selection = self._select_progressive_first_batch(source, target)
            state_trace[-1].update(
                self._progressive_observability_fields(
                    snapshots[-1], selection
                )
            )
            if selection is None:
                if self._decode_source_is_idle_for_role_flip(source):
                    pause_source = self._post_worker(
                        records,
                        "pause_source_admission",
                        source,
                        "/pd_flip/runtime_role/admission",
                        {"paused": True},
                    )
                    paused_statuses = _index_dp_responses(pause_source)
                    paused_runtime_statuses = [
                        item.get("status")
                        for item in paused_statuses.values()
                        if isinstance(item, dict)
                    ]
                    pause_status_observable = bool(paused_statuses) and all(
                        isinstance(status, dict)
                        and isinstance(status.get("running_requests"), list)
                        and isinstance(status.get("waiting_requests"), list)
                        and isinstance(
                            status.get("decode_bootstrap_requests"), list
                        )
                        for status in paused_runtime_statuses
                    )
                    if pause_status_observable:
                        source.dp_statuses = [
                            paused_statuses[rank]
                            for rank in sorted(paused_statuses)
                        ]
                        source.raw_status = _aggregate_dp_runtime_status(
                            source.dp_statuses, source.name
                        )
                    if (
                        not pause_status_observable
                        or not self._decode_source_is_idle_for_role_flip(source)
                    ):
                        self._resume_decode_source(source, records)
                        self._append_progressive_state(
                            state_trace,
                            ProgressiveMonitorState.SAFE,
                            source,
                            target,
                            (
                                "idle_source_status_unobservable_after_admission_pause"
                                if not pause_status_observable
                                else "idle_source_became_busy_before_role_flip"
                            ),
                            records,
                        )
                        return self._progressive_result(
                            True,
                            "idle Decode source was not safe after admission pause",
                            iterations,
                            snapshots,
                            records,
                            state_trace,
                        )
                    self._assert_source_idle_after_migration(records, source)
                    self._append_progressive_state(
                        state_trace,
                        ProgressiveMonitorState.FLIPPING_ROLE,
                        source,
                        target,
                        "idle_decode_no_migration_required",
                        records,
                    )
                    self._flip_idle_source_to_prefill(
                        source,
                        target,
                        session_prefix + "-idle-role-flip",
                        (),
                        records,
                    )
                    self._append_progressive_state(
                        state_trace,
                        ProgressiveMonitorState.SAFE,
                        source,
                        target,
                        "role_flip_complete",
                        records,
                    )
                    state_trace[-1].update(
                        {
                            "snapshot_index": trigger_snapshot_index,
                            "migration_mode": "idle_decode_direct_role_flip",
                            "migrated_request_count": 0,
                        }
                    )
                    return self._progressive_result(
                        True,
                        "idle Decode source switched to prefill",
                        iterations,
                        snapshots,
                        records,
                        state_trace,
                    )
                self._resume_decode_source(source, records)
                self._append_progressive_state(
                    state_trace,
                    ProgressiveMonitorState.SAFE,
                    source,
                    target,
                    "no_feasible_batch_after_bootstrap_drain",
                    records,
                )
                return self._progressive_result(
                    True,
                    "no feasible migration batch after bootstrap drain",
                    iterations,
                    snapshots,
                    records,
                    state_trace,
                )

            pause_source = self._post_worker(
                records,
                "pause_source_admission",
                source,
                "/pd_flip/runtime_role/admission",
                {"paused": True},
            )
            paused_statuses = _index_dp_responses(pause_source)
            paused_runtime_statuses = [
                item.get("status")
                for item in paused_statuses.values()
                if isinstance(item, dict)
            ]
            if paused_statuses and all(
                isinstance(status, dict) and "running_requests" in status
                for status in paused_runtime_statuses
            ):
                source.dp_statuses = [
                    paused_statuses[rank] for rank in sorted(paused_statuses)
                ]
                source.raw_status = _aggregate_dp_runtime_status(
                    source.dp_statuses, source.name
                )
                if self._decode_bootstrap_request_count(source) != 0:
                    self._resume_decode_source(source, records)
                    self._append_progressive_state(
                        state_trace,
                        ProgressiveMonitorState.SAFE,
                        source,
                        target,
                        "bootstrap_reappeared_after_admission_pause",
                        records,
                    )
                    return self._progressive_result(
                        True,
                        "Decode bootstrap reappeared after admission pause",
                        iterations,
                        snapshots,
                        records,
                        state_trace,
                    )
                selection = self._select_progressive_first_batch(source, target)
                state_trace[-1].update(
                    self._progressive_observability_fields(
                        snapshots[-1], selection
                    )
                )
                if selection is None:
                    self._resume_decode_source(source, records)
                    self._append_progressive_state(
                        state_trace,
                        ProgressiveMonitorState.SAFE,
                        source,
                        target,
                        "no_feasible_batch_after_admission_pause",
                        records,
                    )
                    return self._progressive_result(
                        True,
                        "no feasible migration batch after admission pause",
                        iterations,
                        snapshots,
                        records,
                        state_trace,
                    )
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.FIRST_MIGRATING,
                source,
                target,
                "first_batch_selected",
                records,
            )
            first_session_id = session_prefix + "-first"
            for selection_attempt in range(4):
                try:
                    self._execute_atomic_batch(
                        source,
                        target,
                        first_session_id,
                        selection.selected_rids,
                        False,
                        records=records,
                        next_fsm_phase="observing",
                    )
                    break
                except ProgressiveAtomicBatchError as exc:
                    retryable_prefix_race = (
                        not exc.source_finished
                        and not exc.cutover_started
                        and "selected rids must be a running-batch prefix"
                        in str(exc)
                    )
                    retryable_completed_request = (
                        not exc.source_finished
                        and not exc.cutover_started
                        and D_TO_P_REQUEST_COMPLETED_BEFORE_BASE_TRANSFER_MESSAGE
                        in str(exc)
                    )
                    if (
                        not retryable_prefix_race
                        and not retryable_completed_request
                    ) or selection_attempt >= 3:
                        raise
                    retry_reason = (
                        "completed_request"
                        if retryable_completed_request
                        else "prefix_race"
                    )
                    self._refresh_progressive_runtime_status(
                        source,
                        records,
                        f"refresh_source_after_{retry_reason}",
                    )
                    self._refresh_progressive_runtime_status(
                        target,
                        records,
                        f"refresh_target_after_{retry_reason}",
                    )
                    selection = self._select_progressive_first_batch(
                        source, target
                    )
                    state_trace[-1]["selection_retry_count"] = (
                        selection_attempt + 1
                    )
                    state_trace[-1]["selection_retry_reason"] = retry_reason
                    if selection is None:
                        self._resume_decode_source(source, records)
                        self._append_progressive_state(
                            state_trace,
                            ProgressiveMonitorState.SAFE,
                            source,
                            target,
                            f"no_feasible_batch_after_{retry_reason}",
                            records,
                        )
                        return self._progressive_result(
                            True,
                            f"no feasible migration batch after {retry_reason}",
                            iterations,
                            snapshots,
                            records,
                            state_trace,
                        )
                    first_session_id = (
                        session_prefix
                        + "-first-{}-retry-{}".format(
                            (
                                retry_reason
                                if retryable_completed_request
                                else "prefix"
                            ),
                            selection_attempt + 1,
                        )
                    )
            source_finished = True

            slo_monitor.reset_window()
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.OBSERVING,
                source,
                target,
                "fresh_slo_window",
                records,
            )
            observation_start_monotonic = time.monotonic()
            observation_start_wall = time.time()
            observation = self._collect_progressive_observation(
                slo_monitor, monitor_nodes
            )
            observation_end_monotonic = time.monotonic()
            observation_end_wall = time.time()
            records.append(
                ActionRecord(
                    step="observe_slo_window",
                    target="cluster",
                    method="MONITOR",
                    url="",
                    payload={
                        "configured_seconds": self.config.observation_seconds,
                        "poll_interval_seconds": (
                            self.config.migration_poll_interval_seconds
                        ),
                    },
                    response=observation.to_dict(),
                    elapsed_seconds=(
                        observation_end_monotonic - observation_start_monotonic
                    ),
                    start_wall=observation_start_wall,
                    start_monotonic=observation_start_monotonic,
                    end_wall=observation_end_wall,
                    end_monotonic=observation_end_monotonic,
                )
            )
            snapshots.append(observation.to_dict())
            state_trace[-1].update(
                self._progressive_observability_fields(observation, selection)
            )
            # The queue/util trigger already authorized moving exactly one D
            # to P. Keep the configured observation interval for migration
            # safety/evidence, but do not re-introduce the legacy SLO-gap gate
            # for the second half of that same one-node migration.
            decision = (
                ProgressiveDecision.COMMIT
                if force_commit or self.config.decision_policy == "prefill_queue_util"
                else self._evaluate_progressive_snapshot(observation, observing=True)
            )
            if decision in (
                ProgressiveDecision.RECOVER,
                ProgressiveDecision.INSUFFICIENT_SAMPLES,
            ):
                self._append_progressive_state(
                    state_trace,
                    ProgressiveMonitorState.RECOVERING,
                    source,
                    target,
                    decision.value,
                    records,
                )
                self._resume_decode_source(source, records)
                self._write_journal_phase(
                    source,
                    target,
                    first_session_id,
                    selection.selected_rids,
                    "observation_recovered_safe",
                    True,
                    {
                        "next_fsm_phase": "observing",
                        "batch_ordinal": 1,
                        "source_admission_paused": False,
                        "router_drained": False,
                    },
                )
                self._append_progressive_state(
                    state_trace,
                    ProgressiveMonitorState.SAFE,
                    source,
                    target,
                    "source_remains_decode",
                    records,
                )
                return self._progressive_result(
                    True,
                    "source remains decode",
                    iterations,
                    snapshots,
                    records,
                    state_trace,
                )

            if decision is not ProgressiveDecision.COMMIT:
                raise RuntimeError(f"unexpected observation decision: {decision}")

            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SECOND_MIGRATING,
                source,
                target,
                "prefill_risk_persisted",
                records,
            )
            if require_full_drain_capacity:
                capacity = self._wait_for_progressive_full_drain_capacity(
                    source, target, records
                )
                state_trace[-1]["full_drain_capacity"] = capacity
            remaining, waiting_count = self._source_pending_requests(source, records)
            if remaining or waiting_count:
                source_finished = False
                self._execute_atomic_batch(
                    source,
                    target,
                    session_prefix + "-final",
                    remaining,
                    True,
                    select_all_running=True,
                    records=records,
                    next_fsm_phase="role_flip_worker_prefill_intent",
                )
                source_finished = True
            self._assert_source_idle_after_migration(records, source)
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.FLIPPING_ROLE,
                source,
                target,
                "source_idle",
                records,
            )
            self._flip_idle_source_to_prefill(
                source,
                target,
                session_prefix + "-role-flip",
                selection.selected_rids,
                records,
            )
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SAFE,
                source,
                target,
                "role_flip_complete",
                records,
            )
            state_trace[-1]["snapshot_index"] = trigger_snapshot_index
            return self._progressive_result(
                True,
                "source switched to prefill",
                iterations,
                snapshots,
                records,
                state_trace,
            )
        except Exception as exc:
            # An atomic batch owns its pre-finish abort. Once source finish has
            # succeeded, ownership must never be rolled back by the controller.
            post_finish_error = source_finished or (
                isinstance(exc, ProgressiveAtomicBatchError) and exc.source_finished
            )
            router_pending = isinstance(exc, RoleFlipRouterPendingError)
            cutover_pending = isinstance(
                exc, ProgressiveAtomicBatchError
            ) and exc.cutover_started
            if not router_pending and not cutover_pending:
                self._cleanup_source_after_failure(source, records)
            self._append_progressive_state(
                state_trace,
                (
                    ProgressiveMonitorState.FLIPPING_ROLE
                    if router_pending or cutover_pending
                    else ProgressiveMonitorState.SAFE
                ),
                source,
                target,
                (
                    "role_flip_router_pending"
                    if router_pending
                    else "ownership_cutover_pending"
                    if cutover_pending
                    else "post_finish_error"
                    if post_finish_error
                    else "error_recovered"
                ),
                records,
            )
            return self._progressive_result(
                False,
                str(exc),
                iterations,
                snapshots,
                records,
                state_trace,
            )

    def _execute_atomic_batch(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        rids: Sequence[str],
        include_waiting: bool,
        *,
        select_all_running: bool = False,
        next_fsm_phase: str,
        records: Optional[List[ActionRecord]] = None,
    ) -> Tuple[str, ...]:
        if next_fsm_phase not in {
            "observing",
            "role_flip_worker_prefill_intent",
        }:
            raise ValueError(f"invalid next_fsm_phase: {next_fsm_phase}")
        batch_ordinal = 1 if next_fsm_phase == "observing" else 2
        records = records if records is not None else []
        requested_rids = tuple(str(rid) for rid in rids)
        if select_all_running and not include_waiting:
            raise ValueError(
                "atomic all-running selection requires include_waiting"
            )
        if not requested_rids and not include_waiting:
            raise ValueError("atomic migration batch must not have empty rids")
        if select_all_running:
            source_start_metadata = {
                "include_waiting": True,
                "batch_scope": "full_source_queue_atomic",
            }
        elif include_waiting and not requested_rids:
            source_start_metadata = {
                "include_waiting": True,
                "batch_scope": "waiting_only_pending_manifest",
            }
        else:
            source_start_metadata = {"include_waiting": include_waiting}
        source_dp_statuses = source.dp_statuses or [source.raw_status]
        target_dp_statuses = target.dp_statuses or [target.raw_status]
        target_decode_dp_rank: Optional[int] = None
        target_decode_dp_ranks: Optional[Dict[str, int]] = None
        if len(_index_dp_responses(target_dp_statuses)) > 1:
            if requested_rids:
                target_decode_dp_ranks = _assign_target_dp_ranks(
                    target_dp_statuses,
                    _required_kv_pages_by_rid(source_dp_statuses, requested_rids),
                )
            if include_waiting:
                # Waiting-only batches do not expose their RIDs until source
                # selection. Use the roomiest rank; target prepare remains the
                # authoritative capacity check for these requests.
                target_decode_dp_rank = select_target_dp_rank(target_dp_statuses, 0)
        source_finished = False
        cutover_started = False
        journal_rids = requested_rids
        prefill_donor_groups: Dict[str, List[JsonDict]] = {}
        prefill_donor_expected_statuses: Dict[str, Any] = {}
        attempted_prefill_donor_urls: List[str] = []
        use_prefill_donor = bool(self.config.prefill_donor_mode)
        prefill_donor_fallback_reason: Optional[str] = None
        try:
            self._write_journal_phase(
                source,
                target,
                session_id,
                requested_rids,
                "source_start_intent",
                metadata=source_start_metadata,
            )
            source_start = self._post_worker(
                records,
                "start_decode_migration_source",
                source,
                "/pd_flip/migration/source/start",
                _migration_source_start_payload(
                    session_id,
                    target.worker_url,
                    None if select_all_running else list(requested_rids),
                    include_waiting=include_waiting,
                    prefill_donor_mode=self.config.prefill_donor_mode,
                    target_decode_dp_rank=target_decode_dp_rank,
                    target_decode_dp_ranks=target_decode_dp_ranks,
                ),
            )
            _require_worker_dp_ranks(
                source_start, source_dp_statuses, "source start"
            )
            manifests = _strict_response_manifests(
                source_start, "invalid source start response manifests"
            )
            _require_request_owners(
                source_start,
                _manifest_rids(manifests),
                "source start",
            )
            manifests = _order_manifests_by_requested_rids(manifests, requested_rids)
            if target_decode_dp_rank is not None and any(
                int(manifest.get("target_decode_dp_rank", -1))
                != target_decode_dp_rank
                for manifest in manifests
            ):
                raise RuntimeError(
                    "source start did not preserve selected target_decode_dp_rank"
                )
            if target_decode_dp_ranks is not None and any(
                int(manifest.get("target_decode_dp_rank", -1))
                != target_decode_dp_ranks.get(
                    str(manifest.get("rid")), target_decode_dp_rank
                )
                for manifest in manifests
            ):
                raise RuntimeError(
                    "source start did not preserve per-request target DP ranks"
                )
            batch_rids = tuple(_manifest_rids(manifests))
            if include_waiting:
                if select_all_running:
                    # The status snapshot used for the capacity gate is
                    # necessarily older than the Scheduler's atomic source
                    # selection. A running request may finish naturally in
                    # that interval. The source/start call with rids=None is
                    # authoritative: it freezes every request that is still
                    # running plus every eligible waiting request while
                    # admission is paused. Preserve the reconciliation as an
                    # explicit controller event instead of rejecting a safely
                    # completed request as an ownership mismatch.
                    batch_set = set(batch_rids)
                    requested_set = set(requested_rids)
                    present_requested = tuple(
                        rid for rid in requested_rids if rid in batch_set
                    )
                    if batch_rids[: len(present_requested)] != present_requested:
                        raise RuntimeError(
                            "invalid source start response manifests: "
                            "surviving requested running RID order was not preserved"
                        )
                    reconciled_at = time.monotonic()
                    reconciled_wall = time.time()
                    records.append(
                        ActionRecord(
                            step="reconcile_full_source_queue_snapshot",
                            target=source.name,
                            method="LOCAL",
                            url="",
                            payload={
                                "session_id": session_id,
                                "requested_snapshot_rids": list(requested_rids),
                                "batch_scope": "full_source_queue_atomic",
                            },
                            response={
                                "atomic_source_rids": list(batch_rids),
                                "completed_before_atomic_source_freeze_rids": [
                                    rid for rid in requested_rids if rid not in batch_set
                                ],
                                "new_active_or_waiting_rids": [
                                    rid for rid in batch_rids if rid not in requested_set
                                ],
                            },
                            message=(
                                "reconciled stale capacity snapshot against the "
                                "source Scheduler atomic all-active selection"
                            ),
                            elapsed_seconds=0.0,
                            start_wall=reconciled_wall,
                            start_monotonic=reconciled_at,
                            end_wall=reconciled_wall,
                            end_monotonic=reconciled_at,
                        )
                    )
                elif batch_rids[: len(requested_rids)] != requested_rids:
                    raise RuntimeError(
                        "invalid source start response manifests: "
                        "requested running RID prefix was not preserved"
                    )
            elif batch_rids != requested_rids:
                raise RuntimeError(
                    "invalid source start response manifests: "
                    "selected first-batch RIDs do not match"
                )

            journal_rids = batch_rids
            if use_prefill_donor:
                prefill_donor_groups = self._resolve_prefill_donor_groups(manifests)
                self._bind_output_relay_urls(prefill_donor_groups)
            self._write_journal_phase(
                source, target, session_id, batch_rids, "source_started"
            )
            if use_prefill_donor:
                donor_metadata = {
                    "prefill_donor_urls": list(prefill_donor_groups),
                }
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    batch_rids,
                    "prefill_donor_start_intent",
                    metadata=donor_metadata,
                )
                nodes_by_url = {
                    node.worker_url: node for node in self.config.nodes
                }
                try:
                    # Probe the real donor cache before preparing a target that
                    # depends on the donor range. The donor restore begins
                    # synchronously and reports an incomplete L3 prefix here.
                    for donor_url, donor_manifests in prefill_donor_groups.items():
                        attempted_prefill_donor_urls.append(donor_url)
                        donor_start = self._post_worker(
                            records,
                            "start_prefill_donor",
                            nodes_by_url[donor_url],
                            "/pd_flip/migration/prefill-donor/start",
                            {
                                "session_id": session_id,
                                "manifests": donor_manifests,
                            },
                        )
                        _index_dp_responses(donor_start)
                        prefill_donor_expected_statuses[donor_url] = donor_start
                        _require_request_owners(
                            donor_start,
                            _manifest_rids(donor_manifests),
                            f"prefill donor start {donor_url}",
                        )
                except Exception as exc:
                    if "prefill_donor_incomplete" not in str(exc):
                        raise
                    fallback_reason = str(exc)
                    prefill_donor_fallback_reason = fallback_reason
                    # A cache miss is not corruption. Abort only donor sessions
                    # attempted for this batch, then make the source Decode
                    # authoritative for the complete KV range.
                    for donor_url in attempted_prefill_donor_urls:
                        self._post_worker(
                            records,
                            "abort_prefill_donor_for_full_fallback",
                            nodes_by_url[donor_url],
                            "/pd_flip/migration/prefill-donor/abort",
                            {
                                "session_id": session_id,
                                "reason": fallback_reason,
                            },
                        )
                    self._write_journal_phase(
                        source,
                        target,
                        session_id,
                        batch_rids,
                        "source_full_fallback_intent",
                        metadata={
                            "fallback_rids": list(batch_rids),
                            "fallback_reason": fallback_reason,
                            "fallback_origin": "prefill_donor_l3_miss",
                        },
                    )
                    self._post_worker(
                        records,
                        "start_decode_migration_source_full_fallback",
                        source,
                        "/pd_flip/migration/source/fallback",
                        {
                            "session_id": session_id,
                            "rids": list(batch_rids),
                            "reason": fallback_reason,
                        },
                    )
                    self._write_journal_phase(
                        source,
                        target,
                        session_id,
                        batch_rids,
                        "source_full_fallback_started",
                        metadata={
                            "fallback_rids": list(batch_rids),
                            "fallback_reason": fallback_reason,
                            "fallback_origin": "prefill_donor_l3_miss",
                        },
                    )
                    use_prefill_donor = False
                    prefill_donor_groups = {}
                    prefill_donor_expected_statuses = {}
                else:
                    self._write_journal_phase(
                        source,
                        target,
                        session_id,
                        batch_rids,
                        "prefill_donor_started",
                        metadata=donor_metadata,
                    )

            # Only commit the target to donor stitching after the donor has
            # proven that the complete prefix exists. A donor miss uses the
            # ordinary full-source target path instead.
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_prepare_intent"
            )
            target_payload = {
                "session_id": session_id,
                "source_url": source.worker_url,
                "manifests": (
                    [
                        dict(
                            manifest,
                            pd_flip_prefill_donor_fallback=True,
                            pd_flip_prefill_donor_fallback_reason=(
                                prefill_donor_fallback_reason
                            ),
                        )
                        for manifest in manifests
                    ]
                    if prefill_donor_fallback_reason is not None
                    else manifests
                ),
                "prepare_only": True,
                "adopt_on_commit": False,
            }
            if use_prefill_donor:
                target_payload["prefill_donor_mode"] = True
            target_prepare = self._post_worker(
                records,
                "prepare_decode_migration_target",
                target,
                "/pd_flip/migration/target/prepare",
                target_payload,
            )
            _require_worker_dp_ranks(
                target_prepare, target_dp_statuses, "target prepare"
            )
            _require_request_owners(target_prepare, batch_rids, "target prepare")
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_prepared"
            )
            self._wait_atomic_initial_transfer(
                records,
                source,
                target,
                session_id,
                batch_rids,
                prefill_donor_groups=prefill_donor_groups,
                prefill_donor_expected_statuses=prefill_donor_expected_statuses,
            )
            if use_prefill_donor:
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    batch_rids,
                    "prefill_donor_transferred",
                    metadata={
                        "prefill_donor_urls": list(prefill_donor_groups),
                    },
                )
            delta_manifests = self._poll_source_delta_manifests(
                records, source, session_id, batch_rids
            )
            delta_rids = tuple(_manifest_rids(delta_manifests))
            if not _same_atomic_rids(delta_rids, batch_rids):
                raise RuntimeError(
                    "source delta manifests do not match atomic batch RIDs"
                )
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_delta_prepare_intent"
            )
            target_delta_prepare = self._post_worker(
                records,
                "prepare_decode_migration_target_delta",
                target,
                "/pd_flip/migration/target/delta/prepare",
                {
                    "session_id": session_id,
                    "source_url": source.worker_url,
                    "manifests": delta_manifests,
                },
            )
            _require_worker_dp_ranks(
                target_delta_prepare, target_dp_statuses, "target delta prepare"
            )
            _require_request_owners(
                target_delta_prepare, batch_rids, "target delta prepare"
            )
            self._wait_migration(records, "wait_decode_migration_source_delta", source)
            self._wait_migration(records, "wait_decode_migration_target_delta", target)
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_delta_ready"
            )
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_commit_intent"
            )
            target_commit = self._post_worker(
                records,
                "commit_decode_migration_target",
                target,
                "/pd_flip/migration/target/commit",
                {"session_id": session_id, "rids": list(batch_rids)},
            )
            _require_worker_dp_ranks(
                target_commit, target_dp_statuses, "target commit"
            )
            _require_request_owners(target_commit, batch_rids, "target commit")
            self._write_journal_phase(
                source, target, session_id, batch_rids, "target_ready"
            )
            cutover_metadata = {
                "next_fsm_phase": next_fsm_phase,
                "batch_ordinal": batch_ordinal,
                "source_admission_paused": True,
                "router_drained": True,
            }
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "ownership_cutover_intent",
                False,
                cutover_metadata,
            )
            cutover_started = True
            source_finish = self._post_worker(
                records,
                "finish_decode_migration_source",
                source,
                "/pd_flip/migration/source/finish",
                {"session_id": session_id, "released_rids": list(batch_rids)},
            )
            _require_worker_dp_ranks(
                source_finish, source_dp_statuses, "source release"
            )
            _require_request_owners(source_finish, batch_rids, "source release")
            source_finished = True
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "ownership_cutover_intent",
                True,
                cutover_metadata,
            )
            target_activate = self._post_worker(
                records,
                "activate_decode_migration_target",
                target,
                "/pd_flip/migration/target/activate",
                {"session_id": session_id, "rids": list(batch_rids)},
            )
            _require_worker_dp_ranks(
                target_activate, target_dp_statuses, "target activate"
            )
            _require_request_owners(target_activate, batch_rids, "target activate")
            if next_fsm_phase == "observing":
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    batch_rids,
                    "observing",
                    True,
                    {
                        "batch_ordinal": batch_ordinal,
                        "next_fsm_phase": next_fsm_phase,
                        "observation_deadline_epoch": time.time()
                        + self.config.observation_seconds,
                        "source_admission_paused": True,
                        "router_drained": True,
                    },
                )
            else:
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    batch_rids,
                    "role_flip_worker_prefill_intent",
                    True,
                    cutover_metadata,
                )
            return batch_rids
        except Exception as exc:
            if not source_finished and not cutover_started:
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    journal_rids,
                    "abort_intent",
                )
                abort_complete = self._abort_two_phase_migration(
                    source,
                    target,
                    session_id,
                    records,
                    prefill_donor_urls=tuple(prefill_donor_groups),
                )
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    journal_rids,
                    "aborted" if abort_complete else "abort_incomplete",
                )
            raise ProgressiveAtomicBatchError(
                str(exc),
                source_finished=source_finished,
                cutover_started=cutover_started,
            ) from exc

    def _wait_atomic_initial_transfer(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        batch_rids: Sequence[str],
        *,
        prefill_donor_groups: Optional[Dict[str, List[JsonDict]]] = None,
        prefill_donor_expected_statuses: Optional[Dict[str, Any]] = None,
    ) -> None:
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        attempted = set()
        last_source = last_target = None
        last_donors: Dict[str, Any] = {}
        completed_before_transfer_confirmations = 0
        prefill_donor_groups = prefill_donor_groups or {}
        prefill_donor_expected_statuses = prefill_donor_expected_statuses or {}
        nodes_by_url = {node.worker_url: node for node in self.config.nodes}
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "atomic initial migration timed out: "
                    f"source={last_source}, target={last_target}"
                )
            last_source = self._record_get(
                records,
                "wait_decode_migration_source",
                source.name,
                source.worker_url,
                "/pd_flip/migration/status",
            )
            last_target = self._record_get(
                records,
                "wait_decode_migration_target",
                target.name,
                target.worker_url,
                "/pd_flip/migration/status",
            )
            _require_worker_dp_ranks(
                last_source,
                source.dp_statuses or [source.raw_status],
                "source base-ready",
            )
            _require_worker_dp_ranks(
                last_target,
                target.dp_statuses or [target.raw_status],
                "target base-ready",
            )
            last_donors = {
                donor_url: self._record_get(
                    records,
                    "wait_prefill_donor",
                    nodes_by_url[donor_url].name,
                    donor_url,
                    "/pd_flip/migration/prefill-donor/status"
                    f"?session_id={quote(session_id)}",
                )
                for donor_url in prefill_donor_groups
            }
            for donor_url, donor_status in last_donors.items():
                _require_worker_dp_ranks(
                    donor_status,
                    prefill_donor_expected_statuses.get(donor_url, donor_status),
                    f"prefill donor base-ready {donor_url}",
                )
            fallback_rids, reason, status_session = _migration_fallback_request(
                last_target
            )
            if fallback_rids:
                if self.config.prefill_donor_mode:
                    raise RuntimeError(
                        "target requested source-full fallback in strict Prefill donor mode"
                    )
                if status_session and status_session != session_id:
                    raise RuntimeError("fallback status session does not match batch")
                unknown = set(fallback_rids).difference(map(str, batch_rids))
                if unknown:
                    raise RuntimeError(
                        "fallback requested unknown RIDs: "
                        + ", ".join(sorted(unknown))
                    )
                repeated = attempted.intersection(fallback_rids)
                if repeated:
                    raise RuntimeError(
                        "full fallback already attempted for RIDs: "
                        + ", ".join(sorted(repeated))
                    )
                attempted.update(fallback_rids)
                metadata = {"fallback_rids": fallback_rids}
                self._write_journal_phase(
                    source, target, session_id, batch_rids,
                    "source_full_fallback_intent", metadata=metadata
                )
                self._post_worker(
                    records,
                    "start_decode_migration_source_full_fallback",
                    source,
                    "/pd_flip/migration/source/fallback",
                    {"session_id": session_id, "rids": fallback_rids, "reason": reason},
                )
                self._write_journal_phase(
                    source, target, session_id, batch_rids,
                    "source_full_fallback_started", metadata=metadata
                )
                self._write_journal_phase(
                    source, target, session_id, batch_rids,
                    "target_full_fallback_prepare_intent", metadata=metadata
                )
                self._post_worker(
                    records,
                    "prepare_decode_migration_target_full_fallback",
                    target,
                    "/pd_flip/migration/target/fallback/prepare",
                    {"session_id": session_id, "rids": fallback_rids},
                )
                self._write_journal_phase(
                    source, target, session_id, batch_rids,
                    "target_full_fallback_prepared", metadata=metadata
                )
                continue
            source_failed = _migration_response_failed(last_source)
            source_abort_req_failure = (
                source_failed
                and _migration_response_failed_only_by_abort_req(last_source)
            )
            failures = []
            if source_failed and not source_abort_req_failure:
                failures.append(f"{source.name}: {_migration_response_error(last_source)}")
            if _migration_response_failed(last_target):
                failures.append(f"{target.name}: {_migration_response_error(last_target)}")
            for donor_url, donor_status in last_donors.items():
                if _migration_response_failed(donor_status):
                    failures.append(
                        f"{nodes_by_url[donor_url].name}: "
                        f"{_migration_response_error(donor_status)}"
                    )
            if failures:
                raise RuntimeError("atomic migration failed: " + "; ".join(failures))
            if (
                _migration_response_complete(last_source)
                and _migration_response_complete(last_target)
                and all(
                    _migration_response_complete(status)
                    for status in last_donors.values()
                )
            ):
                return
            unstarted_rids = (
                tuple(str(rid) for rid in batch_rids)
                if source_abort_req_failure
                else _unstarted_pending_source_rids(
                    last_source, session_id, batch_rids
                )
            )
            if unstarted_rids:
                runtime_status = self._record_get(
                    records,
                    "check_source_after_unstarted_base_transfer",
                    source.name,
                    source.worker_url,
                    "/pd_flip/runtime_role/status",
                )
                _require_worker_dp_ranks(
                    runtime_status,
                    source.dp_statuses or [source.raw_status],
                    "source natural-completion check",
                )
                if _rids_absent_from_runtime_queues(
                    runtime_status, unstarted_rids
                ):
                    completed_before_transfer_confirmations += 1
                else:
                    completed_before_transfer_confirmations = 0
            else:
                completed_before_transfer_confirmations = 0
            if completed_before_transfer_confirmations >= 2:
                raise RuntimeError(
                    D_TO_P_REQUEST_COMPLETED_BEFORE_BASE_TRANSFER_MESSAGE
                    + ": "
                    + ", ".join(unstarted_rids)
                )
            time.sleep(self.config.migration_poll_interval_seconds)

    def _poll_source_delta_manifests(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        session_id: str,
        rids: Sequence[str],
    ) -> List[JsonDict]:
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        payload = {"session_id": session_id, "rids": list(rids)}
        last_response: Any = None
        while True:
            started = time.monotonic()
            started_wall = time.time()
            path = "/pd_flip/migration/source/delta"
            url = _join_url(source.worker_url, path)
            try:
                response = self.client.post_json(source.worker_url, path, payload)
                last_response = response
                if _delta_quiesce_pending(response):
                    records.append(
                        ActionRecord(
                            step="start_decode_migration_source_delta",
                            target=source.name,
                            method="POST",
                            url=url,
                            payload=payload,
                            response=response,
                            message="quiesce pending",
                            **_action_timing_fields(started, started_wall),
                        )
                    )
                else:
                    _raise_if_unsuccessful(
                        response, "start_decode_migration_source_delta"
                    )
                    _require_worker_dp_ranks(
                        response,
                        source.dp_statuses or [source.raw_status],
                        "source delta",
                    )
                    manifests = _strict_response_manifests(
                        response, "invalid source delta response manifests"
                    )
                    _require_request_owners(
                        response, rids, "source delta"
                    )
                    records.append(
                        ActionRecord(
                            step="start_decode_migration_source_delta",
                            target=source.name,
                            method="POST",
                            url=url,
                            payload=payload,
                            response=response,
                            **_action_timing_fields(started, started_wall),
                        )
                    )
                    return manifests
            except Exception as exc:
                records.append(
                    ActionRecord(
                        step="start_decode_migration_source_delta",
                        target=source.name,
                        method="POST",
                        url=url,
                        payload=payload,
                        response=last_response,
                        success=False,
                        message=str(exc),
                        **_action_timing_fields(started, started_wall),
                    )
                )
                raise
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"source delta quiesce timed out for {source.name}: {last_response}"
                )
            time.sleep(
                min(
                    self.config.migration_poll_interval_seconds,
                    max(0.0, deadline - now),
                )
            )

    def _collect_progressive_observation(
        self,
        slo_monitor: PDFlipSLOMonitor,
        monitor_nodes: List[Tuple[str, str, str]],
    ) -> ClusterSLOSnapshot:
        deadline = time.monotonic() + self.config.observation_seconds
        snapshot: Optional[ClusterSLOSnapshot] = None
        while True:
            snapshot = slo_monitor.collect_cluster(monitor_nodes)
            now = time.monotonic()
            if now >= deadline:
                return snapshot
            remaining = max(0.0, deadline - now)
            poll_interval = self.config.migration_poll_interval_seconds
            time.sleep(
                min(poll_interval, remaining) if poll_interval > 0 else remaining
            )

    def _evaluate_progressive_snapshot(
        self, snapshot: ClusterSLOSnapshot, *, observing: bool
    ) -> ProgressiveDecision:
        prefill = snapshot.prefill_counts
        decode = snapshot.decode_counts
        if prefill is None or decode is None:
            return ProgressiveDecision.INSUFFICIENT_SAMPLES
        return evaluate_slo_decision(
            prefill.good,
            prefill.total,
            decode.good,
            decode.total,
            self.config.slo_threshold,
            self.config.min_prefill_slo_samples,
            self.config.min_decode_slo_samples,
            observing=observing,
            recover_threshold=self.config.slo_recovery_threshold,
            force_commit_after_observation=(
                self.config.force_second_migration_after_observation
            ),
            attainment_gap_threshold=self.config.slo_attainment_gap_threshold,
            attainment_gap_recovery_threshold=(
                self.config.slo_attainment_gap_recovery_threshold
            ),
        )

    def _source_pending_requests(
        self, source: NodeMetrics, records: List[ActionRecord]
    ) -> Tuple[Tuple[str, ...], int]:
        response = self._record_get(
            records,
            "get_remaining_source_requests",
            source.name,
            source.worker_url,
            "/pd_flip/runtime_role/status",
        )
        item = _first_successful_response(response)
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        running = status.get("running_requests", [])
        if not isinstance(running, list):
            raise RuntimeError("source running request status is not a list")
        rids: List[str] = []
        for request_status in running:
            if (
                not isinstance(request_status, dict)
                or request_status.get("rid") is None
            ):
                raise RuntimeError("source running request status has no RID")
            rids.append(str(request_status["rid"]))
        if len(set(rids)) != len(rids):
            raise RuntimeError("source running request status contains duplicate RIDs")
        waiting = status.get("waiting_requests", [])
        if not isinstance(waiting, list):
            raise RuntimeError("source waiting request status is not a list")
        return tuple(rids), len(waiting)

    def _resume_decode_source(
        self, source: NodeMetrics, records: List[ActionRecord]
    ) -> None:
        self._post_worker(
            records,
            "resume_source_admission",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": False},
        )
        self._post_router(
            records,
            "router_undrain_source",
            source,
            "/pd_flip/router/worker/drain",
            {"worker_id": source.router_worker_id, "draining": False},
        )

    def _flip_idle_source_to_prefill(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        batch_rids: Sequence[str],
        records: List[ActionRecord],
    ) -> None:
        role_metadata = {
            "next_fsm_phase": "role_flip_worker_prefill_intent",
            "batch_ordinal": 2,
            "source_admission_paused": True,
            "router_drained": True,
        }
        self._write_journal_phase(
            source,
            target,
            session_id,
            batch_rids,
            "role_flip_worker_prefill_intent",
            True,
            role_metadata,
        )
        try:
            self._post_worker(
                records,
                "set_source_runtime_role",
                source,
                "/pd_flip/runtime_role/set",
                {"role": "prefill", "force": False},
            )
            self._wait_source_role(
                records, source, "prefill", "wait_source_prefill_loop"
            )
        except Exception as exc:
            raise RoleFlipRouterPendingError(str(exc)) from exc
        self._write_journal_phase(
            source,
            target,
            session_id,
            batch_rids,
            "role_flip_worker_prefill",
            True,
            role_metadata,
        )
        try:
            self._complete_prefill_router_flip(source, records)
        except Exception as exc:
            self._write_journal_phase(
                source,
                target,
                session_id,
                batch_rids,
                "role_flip_router_pending",
                True,
                role_metadata,
            )
            raise RoleFlipRouterPendingError(str(exc)) from exc
        self._write_journal_phase(
            source,
            target,
            session_id,
            batch_rids,
            "role_flip_complete",
            True,
            role_metadata,
        )

    def _complete_prefill_router_flip(
        self, source: NodeMetrics, records: List[ActionRecord]
    ) -> None:
        self._invalidate_decode_prefill_peer_caches(source, records)
        self._post_router(
            records,
            "refresh_router_source_role",
            source,
            "/pd_flip/router/worker/role",
            {
                "worker_id": source.router_worker_id,
                "role": "prefill",
                "bootstrap_port": source.bootstrap_port,
                "draining": True,
            },
        )
        self._wait_router_role(
            records, source, "prefill", require_drained=True
        )
        self._post_worker(
            records,
            "resume_source_admission",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": False},
        )
        self._post_router(
            records,
            "router_undrain_source",
            source,
            "/pd_flip/router/worker/drain",
            {"worker_id": source.router_worker_id, "draining": False},
        )

    def _wait_router_role(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        expected_role: str,
        *,
        require_drained: bool,
    ) -> None:
        started = time.monotonic()
        started_wall = time.time()
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        last: Optional[JsonDict] = None
        sample_count = 0
        try:
            while True:
                last = self._fetch_router_workers().get(source.router_worker_id)
                sample_count += 1
                if (
                    isinstance(last, dict)
                    and _normalize_role(last.get("role")) == expected_role
                    and (not require_drained or bool(last.get("draining")))
                ):
                    records.append(
                        ActionRecord(
                            step="wait_router_source_role",
                            target=f"router:{source.router_worker_id}",
                            method="GET",
                            url=_join_url(
                                self.config.router_url,
                                "/pd_flip/router/workers",
                            ),
                            response={
                                "sample_count": sample_count,
                                "worker": last,
                            },
                            **_action_timing_fields(started, started_wall),
                        )
                    )
                    return
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        "router role agreement timed out for "
                        f"{source.name}: {last}"
                    )
                time.sleep(
                    min(
                        self.config.migration_poll_interval_seconds,
                        max(0.0, deadline - now),
                    )
                )
        except Exception as exc:
            records.append(
                ActionRecord(
                    step="wait_router_source_role",
                    target=f"router:{source.router_worker_id}",
                    method="GET",
                    url=_join_url(
                        self.config.router_url,
                        "/pd_flip/router/workers",
                    ),
                    response={
                        "sample_count": sample_count,
                        "worker": last,
                    },
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _wait_router_dispatch_quiesced(
        self, records: List[ActionRecord], source: NodeMetrics
    ) -> None:
        """Record that the drained Router dispatch cut has settled.

        Router ``active_load`` covers the full Prefill request lifetime, so it
        cannot be waited to zero here.  P->D calls this after source admission
        is paused so the existing bootstrap batch can be held.  D->P calls it
        before pausing worker admission; requests dispatched just before the
        Router cut must be allowed to finish their already-bound Decode
        bootstrap handshake before the role switch proceeds.
        """
        started = time.monotonic()
        started_wall = time.time()
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        settle_seconds = max(
            0.5, 2.0 * self.config.migration_poll_interval_seconds
        )
        last: Optional[JsonDict] = None
        first_drained: Optional[JsonDict] = None
        drained_observed_at: Optional[float] = None
        sample_count = 0
        try:
            while True:
                last = self._fetch_router_workers().get(source.router_worker_id)
                sample_count += 1
                now = time.monotonic()
                if isinstance(last, dict) and bool(last.get("draining")):
                    if drained_observed_at is None:
                        drained_observed_at = now
                        first_drained = dict(last)
                    if now - drained_observed_at >= settle_seconds:
                        records.append(
                            ActionRecord(
                                step="wait_router_source_quiesced",
                                target=f"router:{source.router_worker_id}",
                                method="GET",
                                url=_join_url(
                                    self.config.router_url,
                                    "/pd_flip/router/workers",
                                ),
                                response={
                                    "sample_count": sample_count,
                                    "settle_seconds": settle_seconds,
                                    "first_drained_worker": first_drained,
                                    "worker": last,
                                },
                                **_action_timing_fields(started, started_wall),
                            )
                        )
                        return
                else:
                    drained_observed_at = None
                    first_drained = None
                if now >= deadline:
                    raise TimeoutError(
                        "router drain acknowledgement timed out for "
                        f"{source.name}: {last}"
                    )
                time.sleep(
                    min(
                        self.config.migration_poll_interval_seconds,
                        max(0.0, deadline - now),
                    )
                )
        except Exception as exc:
            records.append(
                ActionRecord(
                    step="wait_router_source_quiesced",
                    target=f"router:{source.router_worker_id}",
                    method="GET",
                    url=_join_url(
                        self.config.router_url,
                        "/pd_flip/router/workers",
                    ),
                    response={"sample_count": sample_count, "worker": last},
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _wait_source_role(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        expected_role: str,
        step: str,
    ) -> Any:
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        last_response: Any = None
        while True:
            try:
                last_response = self._record_get(
                    records,
                    step,
                    source.name,
                    source.worker_url,
                    "/pd_flip/runtime_role/status",
                )
            except Exception as exc:
                last_response = {"success": False, "message": str(exc)}
            else:
                statuses = (
                    last_response
                    if isinstance(last_response, list)
                    else [last_response]
                )
                _require_worker_dp_ranks(
                    last_response,
                    getattr(source, "dp_statuses", None)
                    or [getattr(source, "raw_status", {})],
                    step,
                )
                all_active = bool(statuses)
                for status in statuses:
                    if (
                        not isinstance(status, dict)
                        or status.get("success") is not True
                    ):
                        all_active = False
                        break
                    role, _, _ = _parse_runtime_status(status)
                    runtime_status = (
                        status.get("status")
                        if isinstance(status.get("status"), dict)
                        else status
                    )
                    active_event_loop_role = _normalize_role(
                        runtime_status.get("active_event_loop_role")
                    )
                    if role != expected_role or active_event_loop_role != expected_role:
                        all_active = False
                        break
                if all_active:
                    return last_response
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"{step} timed out for {source.name}: {last_response}"
                )
            time.sleep(
                min(
                    self.config.migration_poll_interval_seconds,
                    max(0.0, deadline - now),
                )
            )

    def _append_progressive_state(
        self,
        state_trace: List[JsonDict],
        state: str,
        source: NodeMetrics,
        target: NodeMetrics,
        reason: str,
        records: List[ActionRecord],
    ) -> None:
        state_trace.append(
            _monitor_state_record(
                state=state,
                direction="d_to_p",
                source=source.name,
                migration_target=target.name,
                role_before=source.effective_role,
                role_after=(
                    "prefill"
                    if state == ProgressiveMonitorState.FLIPPING_ROLE
                    else source.effective_role
                ),
                reason=reason,
                action_index=len(records),
            )
        )

    @staticmethod
    def _progressive_observability_fields(
        snapshot: Any, selection: Optional[RatioSelection]
    ) -> JsonDict:
        def value(container: Any, name: str) -> Any:
            if isinstance(container, dict):
                return container.get(name)
            return getattr(container, name, None)

        prefill = value(snapshot, "prefill_counts")
        decode = value(snapshot, "decode_counts")
        return {
            "configured_ratio": value(selection, "configured_ratio"),
            "effective_ratio": value(selection, "effective_ratio"),
            "capacity_fallback_count": value(selection, "fallback_count") or 0,
            "prefill_slo_good": value(prefill, "good") or 0,
            "prefill_slo_total": value(prefill, "total") or 0,
            "decode_slo_good": value(decode, "good") or 0,
            "decode_slo_total": value(decode, "total") or 0,
            "prefill_slo_attainment": value(snapshot, "prefill_slo_attainment"),
            "decode_slo_attainment": value(snapshot, "decode_slo_attainment"),
            "decode_minus_prefill_slo_gap": (
                value(snapshot, "decode_slo_attainment")
                - value(snapshot, "prefill_slo_attainment")
                if value(snapshot, "decode_slo_attainment") is not None
                and value(snapshot, "prefill_slo_attainment") is not None
                else None
            ),
        }

    @staticmethod
    def _progressive_result(
        success: bool,
        message: str,
        iterations: int,
        snapshots: List[JsonDict],
        records: List[ActionRecord],
        state_trace: List[JsonDict],
    ) -> MonitorLoopResult:
        return MonitorLoopResult(
            success=success,
            message=message,
            iterations=iterations,
            snapshots=snapshots,
            actions=records,
            state_trace=state_trace,
        )

    def _execute_d_to_p(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        records: List[ActionRecord],
    ) -> float:
        session_id = f"pd-flip-{source.name}-to-{target.name}"
        self._post_router(
            records,
            "router_drain_source",
            source,
            "/pd_flip/router/worker/drain",
            {"worker_id": source.router_worker_id, "draining": True},
        )
        self._post_worker(
            records,
            "pause_source_admission",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": True},
        )
        self._observe_source_quiesce(records, source)

        migration_started = time.monotonic()
        source_start = self._post_worker(
            records,
            "start_decode_migration_source",
            source,
            "/pd_flip/migration/source/start",
            _migration_source_start_payload(
                session_id, target.worker_url, None, include_waiting=True
            ),
        )
        manifests = _response_manifests(source_start)
        self._post_worker(
            records,
            "prepare_decode_migration_target",
            target,
            "/pd_flip/migration/target/prepare",
            {
                "session_id": session_id,
                "source_url": source.worker_url,
                "manifests": manifests,
                "adopt_on_success": True,
            },
        )
        self._wait_migration(records, "wait_decode_migration_source", source)
        target_status = self._wait_migration(
            records, "wait_decode_migration_target", target
        )
        migration_seconds = time.monotonic() - migration_started

        released_rids = _manifest_rids(_response_manifests(target_status) or manifests)
        self._post_worker(
            records,
            "finish_decode_migration_source",
            source,
            "/pd_flip/migration/source/finish",
            {"session_id": session_id, "released_rids": released_rids},
        )
        self._assert_source_idle_after_migration(records, source)
        self._post_worker(
            records,
            "set_source_runtime_role",
            source,
            "/pd_flip/runtime_role/set",
            {"role": "prefill", "force": False},
        )
        self._wait_source_role(records, source, "prefill", "wait_source_prefill_loop")
        self._invalidate_decode_prefill_peer_caches(source, records)
        self._post_router(
            records,
            "refresh_router_source_role",
            source,
            "/pd_flip/router/worker/role",
            {
                "worker_id": source.router_worker_id,
                "role": "prefill",
                "bootstrap_port": source.bootstrap_port,
                "draining": False,
            },
        )
        self._post_worker(
            records,
            "resume_source_admission",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": False},
        )
        self._post_router(
            records,
            "router_undrain_source",
            source,
            "/pd_flip/router/worker/drain",
            {"worker_id": source.router_worker_id, "draining": False},
        )
        return migration_seconds

    def _execute_d_to_p_two_phase(
        self,
        *,
        source: NodeMetrics,
        target: NodeMetrics,
        slo_monitor: PDFlipSLOMonitor,
        enter_threshold: float,
        exit_threshold: float,
        commit_threshold: float,
        state_trace: Optional[List[JsonDict]] = None,
        snapshot_index: Optional[int] = None,
    ) -> FlipExecutionResult:
        started = time.monotonic()
        records: List[ActionRecord] = []
        session_id = f"pd-flip-{source.name}-to-{target.name}"
        migration_seconds = 0.0
        source_finished = False
        released_rids: List[str] = []
        monitor_nodes = [
            (metric.name, metric.worker_url, metric.effective_role)
            for metric in self.collect_metrics()
        ]
        state_trace = state_trace if state_trace is not None else []

        try:
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.PREPARING_KV_TRANSFER,
                    direction="d_to_p",
                    source=source.name,
                    migration_target=target.name,
                    role_before=source.effective_role,
                    role_after=source.effective_role,
                    reason="prefill_slo_risk",
                    snapshot_index=snapshot_index,
                )
            )
            self._post_router(
                records,
                "router_drain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": True},
            )
            self._post_worker(
                records,
                "pause_source_admission",
                source,
                "/pd_flip/runtime_role/admission",
                {"paused": True},
            )
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.OBSERVING_SOURCE_QUIESCE,
                    direction="d_to_p",
                    source=source.name,
                    migration_target=target.name,
                    role_before=source.effective_role,
                    role_after=source.effective_role,
                    reason="source_drained",
                    snapshot_index=snapshot_index,
                    action_index=len(records),
                )
            )
            self._observe_source_quiesce_for_duration(records, source, 0.0)
            snapshot = slo_monitor.collect_cluster(monitor_nodes)
            if _prefill_recovered(snapshot, exit_threshold):
                self._cleanup_source_after_failure(source, records)
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        direction="d_to_p",
                        source=source.name,
                        migration_target=target.name,
                        role_before=source.effective_role,
                        role_after=source.effective_role,
                        reason="slo_recovered_during_source_quiesce",
                        snapshot_index=snapshot_index,
                        action_index=len(records),
                    )
                )
                return FlipExecutionResult(
                    success=True,
                    message="SLO recovered during source quiesce; migration skipped",
                    direction="d_to_p",
                    source=source.name,
                    target_role="decode",
                    migration_target=target.name,
                    actions=records,
                    total_seconds=time.monotonic() - started,
                    migration_seconds=0.0,
                )

            migration_started = time.monotonic()
            self._write_journal_phase(
                source, target, session_id, released_rids, "source_start_intent"
            )
            source_start = self._post_worker(
                records,
                "start_decode_migration_source",
                source,
                "/pd_flip/migration/source/start",
                _migration_source_start_payload(
                    session_id, target.worker_url, None, include_waiting=True
                ),
            )
            manifests = _strict_response_manifests(
                source_start, "invalid source start response manifests"
            )
            released_rids = _manifest_rids(manifests)
            self._write_journal_phase(
                source, target, session_id, released_rids, "source_started"
            )
            self._write_journal_phase(
                source, target, session_id, released_rids, "target_prepare_intent"
            )
            self._post_worker(
                records,
                "prepare_decode_migration_target",
                target,
                "/pd_flip/migration/target/prepare",
                {
                    "session_id": session_id,
                    "source_url": source.worker_url,
                    "manifests": manifests,
                    "prepare_only": True,
                    "adopt_on_commit": False,
                },
            )
            self._write_journal_phase(
                source, target, session_id, released_rids, "target_prepared"
            )

            transfer_result = self._wait_two_phase_migration_or_recovery(
                records=records,
                source=source,
                target=target,
                slo_monitor=slo_monitor,
                monitor_nodes=monitor_nodes,
                exit_threshold=exit_threshold,
                session_id=session_id,
                migration_rids=released_rids,
            )
            migration_seconds = time.monotonic() - migration_started
            if transfer_result == "recovered":
                self._write_journal_phase(
                    source, target, session_id, released_rids, "abort_intent"
                )
                abort_complete = self._abort_two_phase_migration(
                    source, target, session_id, records
                )
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    released_rids,
                    "aborted" if abort_complete else "abort_incomplete",
                )
                if not abort_complete:
                    self._cleanup_source_after_failure(source, records)
                    return FlipExecutionResult(
                        success=False,
                        message="abort incomplete; session requires operator recovery",
                        direction="d_to_p",
                        source=source.name,
                        target_role="decode",
                        migration_target=target.name,
                        actions=records,
                        total_seconds=time.monotonic() - started,
                        migration_seconds=migration_seconds,
                    )
                self._cleanup_source_after_failure(source, records)
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        direction="d_to_p",
                        source=source.name,
                        migration_target=target.name,
                        role_before=source.effective_role,
                        role_after=source.effective_role,
                        reason="slo_recovered",
                        snapshot_index=snapshot_index,
                        action_index=len(records),
                    )
                )
                return FlipExecutionResult(
                    success=True,
                    message="SLO recovered during preparing; migration aborted",
                    direction="d_to_p",
                    source=source.name,
                    target_role="decode",
                    migration_target=target.name,
                    actions=records,
                    total_seconds=time.monotonic() - started,
                    migration_seconds=migration_seconds,
                )

            snapshot = slo_monitor.collect_cluster(monitor_nodes)
            if not _prefill_risk(snapshot, commit_threshold):
                self._write_journal_phase(
                    source, target, session_id, released_rids, "abort_intent"
                )
                abort_complete = self._abort_two_phase_migration(
                    source, target, session_id, records
                )
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    released_rids,
                    "aborted" if abort_complete else "abort_incomplete",
                )
                if not abort_complete:
                    self._cleanup_source_after_failure(source, records)
                    return FlipExecutionResult(
                        success=False,
                        message="abort incomplete; session requires operator recovery",
                        direction="d_to_p",
                        source=source.name,
                        target_role="decode",
                        migration_target=target.name,
                        actions=records,
                        total_seconds=time.monotonic() - started,
                        migration_seconds=migration_seconds,
                    )
                self._cleanup_source_after_failure(source, records)
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        direction="d_to_p",
                        source=source.name,
                        migration_target=target.name,
                        role_before=source.effective_role,
                        role_after=source.effective_role,
                        reason="slo_recovered",
                        snapshot_index=snapshot_index,
                        action_index=len(records),
                    )
                )
                return FlipExecutionResult(
                    success=True,
                    message="SLO recovered before commit; migration aborted",
                    direction="d_to_p",
                    source=source.name,
                    target_role="decode",
                    migration_target=target.name,
                    actions=records,
                    total_seconds=time.monotonic() - started,
                    migration_seconds=migration_seconds,
                )

            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.FLIPPING_ROLE,
                    direction="d_to_p",
                    source=source.name,
                    migration_target=target.name,
                    role_before=source.effective_role,
                    role_after="prefill",
                    reason="kv_pretransfer_complete",
                    snapshot_index=snapshot_index,
                    action_index=len(records),
                )
            )
            self._sync_two_phase_delta_before_commit(
                records=records,
                source=source,
                target=target,
                session_id=session_id,
                released_rids=released_rids,
            )
            migration_seconds = time.monotonic() - migration_started
            self._write_journal_phase(
                source, target, session_id, released_rids, "target_commit_intent"
            )
            self._post_worker(
                records,
                "commit_decode_migration_target",
                target,
                "/pd_flip/migration/target/commit",
                {"session_id": session_id, "rids": released_rids},
            )
            self._write_journal_phase(
                source, target, session_id, released_rids, "target_ready"
            )
            self._write_journal_phase(
                source, target, session_id, released_rids, "source_finish_intent"
            )
            self._post_worker(
                records,
                "finish_decode_migration_source",
                source,
                "/pd_flip/migration/source/finish",
                {"session_id": session_id, "released_rids": released_rids},
            )
            source_finished = True
            self._write_journal_phase(
                source,
                target,
                session_id,
                released_rids,
                "source_finish_complete",
                True,
            )
            self._write_journal_phase(
                source,
                target,
                session_id,
                released_rids,
                "target_activate_intent",
                True,
            )
            self._post_worker(
                records,
                "activate_decode_migration_target",
                target,
                "/pd_flip/migration/target/activate",
                {"session_id": session_id, "rids": released_rids},
            )
            self._write_journal_phase(
                source, target, session_id, released_rids, "target_active", True
            )
            self._assert_source_idle_after_migration(records, source)
            self._post_worker(
                records,
                "set_source_runtime_role",
                source,
                "/pd_flip/runtime_role/set",
                {"role": "prefill", "force": False},
            )
            self._wait_source_role(
                records, source, "prefill", "wait_source_prefill_loop"
            )
            self._invalidate_decode_prefill_peer_caches(source, records)
            self._post_router(
                records,
                "refresh_router_source_role",
                source,
                "/pd_flip/router/worker/role",
                {
                    "worker_id": source.router_worker_id,
                    "role": "prefill",
                    "bootstrap_port": source.bootstrap_port,
                    "draining": False,
                },
            )
            self._post_worker(
                records,
                "resume_source_admission",
                source,
                "/pd_flip/runtime_role/admission",
                {"paused": False},
            )
            self._post_router(
                records,
                "router_undrain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": False},
            )
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.SAFE,
                    direction="d_to_p",
                    source=source.name,
                    migration_target=target.name,
                    role_before=source.effective_role,
                    role_after="prefill",
                    reason="role_flip_complete",
                    snapshot_index=snapshot_index,
                    action_index=len(records),
                )
            )
            return FlipExecutionResult(
                success=True,
                message="pd flip committed after two-phase migration",
                direction="d_to_p",
                source=source.name,
                target_role="prefill",
                migration_target=target.name,
                actions=records,
                total_seconds=time.monotonic() - started,
                migration_seconds=migration_seconds,
            )
        except Exception as exc:
            if not source_finished:
                self._write_journal_phase(
                    source, target, session_id, released_rids, "abort_intent"
                )
                abort_complete = self._abort_two_phase_migration(
                    source, target, session_id, records
                )
                self._write_journal_phase(
                    source,
                    target,
                    session_id,
                    released_rids,
                    "aborted" if abort_complete else "abort_incomplete",
                )
            self._cleanup_source_after_failure(source, records)
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.SAFE,
                    direction="d_to_p",
                    source=source.name,
                    migration_target=target.name,
                    role_before=source.effective_role,
                    role_after=source.effective_role,
                    reason="error_recovered",
                    snapshot_index=snapshot_index,
                    action_index=len(records),
                )
            )
            return FlipExecutionResult(
                success=False,
                message=str(exc),
                direction="d_to_p",
                source=source.name,
                target_role="prefill",
                migration_target=target.name,
                actions=records,
                total_seconds=time.monotonic() - started,
                migration_seconds=migration_seconds,
            )

    def _wait_two_phase_migration_or_recovery(
        self,
        *,
        records: List[ActionRecord],
        source: NodeMetrics,
        target: NodeMetrics,
        slo_monitor: PDFlipSLOMonitor,
        monitor_nodes: List[Tuple[str, str, str]],
        exit_threshold: float,
        session_id: Optional[str] = None,
        migration_rids: Optional[List[str]] = None,
    ) -> str:
        started = time.monotonic()
        transfer_deadline = started + self.config.migration_timeout_seconds
        observe_until = started + max(0.0, self.config.observation_quiesce_seconds)
        transfer_complete = False
        last_source_status: Any = None
        last_target_status: Any = None
        fallback_attempted_rids = set()
        while True:
            now = time.monotonic()
            if not transfer_complete and now > transfer_deadline:
                raise TimeoutError(
                    "two-phase D->P migration timed out: "
                    f"source={last_source_status}, target={last_target_status}"
                )
            if transfer_complete and now >= observe_until:
                return "transferred"

            snapshot = slo_monitor.collect_cluster(monitor_nodes)
            if _prefill_recovered(snapshot, exit_threshold):
                return "recovered"
            source_status = self._record_get(
                records,
                "wait_decode_migration_source",
                source.name,
                source.worker_url,
                "/pd_flip/migration/status",
            )
            target_status = self._record_get(
                records,
                "wait_decode_migration_target",
                target.name,
                target.worker_url,
                "/pd_flip/migration/status",
            )
            last_source_status = source_status
            last_target_status = target_status
            fallback_rids, fallback_reason, status_session_id = (
                _migration_fallback_request(target_status)
            )
            if fallback_rids:
                effective_session_id = session_id or status_session_id
                if not effective_session_id:
                    raise RuntimeError(
                        "target requested full fallback without a migration session id"
                    )
                repeated = fallback_attempted_rids.intersection(fallback_rids)
                if repeated:
                    raise RuntimeError(
                        "full fallback already attempted for RIDs: "
                        + ", ".join(sorted(repeated))
                    )
                fallback_attempted_rids.update(fallback_rids)
                journal_rids = migration_rids or fallback_rids
                fallback_details = {"fallback_rids": fallback_rids}
                self._write_journal_phase(
                    source,
                    target,
                    effective_session_id,
                    journal_rids,
                    "source_full_fallback_intent",
                    metadata=fallback_details,
                )
                self._post_worker(
                    records,
                    "start_decode_migration_source_full_fallback",
                    source,
                    "/pd_flip/migration/source/fallback",
                    {
                        "session_id": effective_session_id,
                        "rids": fallback_rids,
                        "reason": fallback_reason,
                    },
                )
                self._write_journal_phase(
                    source,
                    target,
                    effective_session_id,
                    journal_rids,
                    "source_full_fallback_started",
                    metadata=fallback_details,
                )
                self._write_journal_phase(
                    source,
                    target,
                    effective_session_id,
                    journal_rids,
                    "target_full_fallback_prepare_intent",
                    metadata=fallback_details,
                )
                self._post_worker(
                    records,
                    "prepare_decode_migration_target_full_fallback",
                    target,
                    "/pd_flip/migration/target/fallback/prepare",
                    {"session_id": effective_session_id, "rids": fallback_rids},
                )
                self._write_journal_phase(
                    source,
                    target,
                    effective_session_id,
                    journal_rids,
                    "target_full_fallback_prepared",
                    metadata=fallback_details,
                )
                continue
            if _migration_response_complete(
                source_status
            ) and _migration_response_complete(target_status):
                transfer_complete = True
                if time.monotonic() >= observe_until:
                    return "transferred"
            failures = []
            if _migration_response_failed(source_status):
                failures.append(
                    f"{source.name}: {_migration_response_error(source_status)}"
                )
            if _migration_response_failed(target_status):
                failures.append(
                    f"{target.name}: {_migration_response_error(target_status)}"
                )
            if failures:
                raise RuntimeError(
                    "two-phase D->P migration failed: " + "; ".join(failures)
                )
            sleep_until = observe_until if transfer_complete else transfer_deadline
            time.sleep(
                min(
                    self.config.migration_poll_interval_seconds,
                    max(0.0, sleep_until - time.monotonic()),
                )
            )

    def _sync_two_phase_delta_before_commit(
        self,
        *,
        records: List[ActionRecord],
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        released_rids: List[str],
    ) -> List[JsonDict]:
        delta_manifests = self._poll_source_delta_manifests(
            records,
            source,
            session_id,
            released_rids,
        )

        self._post_worker(
            records,
            "prepare_decode_migration_target_delta",
            target,
            "/pd_flip/migration/target/delta/prepare",
            {
                "session_id": session_id,
                "source_url": source.worker_url,
                "manifests": delta_manifests,
            },
        )
        self._wait_two_phase_delta(
            records=records,
            source=source,
            target=target,
        )
        return delta_manifests

    def _wait_two_phase_delta(
        self,
        *,
        records: List[ActionRecord],
        source: NodeMetrics,
        target: NodeMetrics,
    ) -> None:
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        last_source_status: Any = None
        last_target_status: Any = None
        while time.monotonic() <= deadline:
            source_status = self._record_get(
                records,
                "wait_decode_migration_source_delta",
                source.name,
                source.worker_url,
                "/pd_flip/migration/status",
            )
            target_status = self._record_get(
                records,
                "wait_decode_migration_target_delta",
                target.name,
                target.worker_url,
                "/pd_flip/migration/status",
            )
            last_source_status = source_status
            last_target_status = target_status
            if _migration_response_complete(
                source_status
            ) and _migration_response_complete(target_status):
                return
            failures = []
            if _migration_response_failed(source_status):
                failures.append(
                    f"{source.name}: {_migration_response_error(source_status)}"
                )
            if _migration_response_failed(target_status):
                failures.append(
                    f"{target.name}: {_migration_response_error(target_status)}"
                )
            if failures:
                raise RuntimeError(
                    "two-phase D->P delta migration failed: " + "; ".join(failures)
                )
            time.sleep(self.config.migration_poll_interval_seconds)
        raise TimeoutError(
            "two-phase D->P delta migration timed out: "
            f"source={last_source_status}, target={last_target_status}"
        )

    def _abort_two_phase_migration(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        session_id: str,
        records: List[ActionRecord],
        *,
        prefill_donor_urls: Sequence[str] = (),
    ) -> bool:
        success = True
        nodes_by_url = {node.worker_url: node for node in self.config.nodes}
        abort_targets = [
            (
                nodes_by_url[url],
                "/pd_flip/migration/prefill-donor/abort",
                {"session_id": session_id, "reason": "monitor aborted preparing"},
                "abort_prefill_donor",
            )
            for url in prefill_donor_urls
            if url in nodes_by_url
        ]
        abort_targets.extend(
            [
            (
                target,
                "/pd_flip/migration/target/abort",
                {"session_id": session_id, "reason": "monitor aborted preparing"},
                "abort_decode_migration",
            ),
            (
                source,
                "/pd_flip/migration/abort",
                {"session_id": session_id, "reason": "monitor aborted preparing"},
                "abort_decode_migration",
            ),
            ]
        )
        for node, path, payload, step in abort_targets:
            try:
                self._post_worker_abort_idempotent(
                    records,
                    step,
                    node,
                    path,
                    payload,
                )
            except Exception:
                success = False
        return success

    def _post_worker_abort_idempotent(
        self,
        records: List[ActionRecord],
        step: str,
        node: NodeMetrics,
        path: str,
        payload: JsonDict,
    ) -> Any:
        """Abort this session without disturbing a newer/foreign session.

        A mismatch means the requested session has no live ownership on that
        worker. It is therefore already clean from this batch's perspective;
        the other session must be preserved. Other failures remain fatal.
        """

        started = time.monotonic()
        started_wall = time.time()
        url = _join_url(node.worker_url, path)
        response = None
        try:
            response = self.client.post_json(node.worker_url, path, payload)
            if not _abort_response_is_idempotent(response):
                _raise_if_unsuccessful(response, step)
            records.append(
                ActionRecord(
                    step=step,
                    target=node.name,
                    method="POST",
                    url=url,
                    payload=payload,
                    response=response,
                    message=(
                        "requested session was already absent; foreign session preserved"
                        if _response_has_session_absent(response)
                        else ""
                    ),
                    **_action_timing_fields(started, started_wall),
                )
            )
            return response
        except Exception as exc:
            records.append(
                ActionRecord(
                    step=step,
                    target=node.name,
                    method="POST",
                    url=url,
                    payload=payload,
                    response=response,
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _execute_p_to_d(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        metrics: List[NodeMetrics],
        records: List[ActionRecord],
    ) -> None:
        self._prepare_p_to_d(source, target, metrics, records)
        self._finish_p_to_d(source, records)

    def _prepare_p_to_d(
        self,
        source: NodeMetrics,
        target: NodeMetrics,
        metrics: List[NodeMetrics],
        records: List[ActionRecord],
        *,
        require_handoff: bool = False,
    ) -> None:
        session_id = (
            f"pd-flip-prefill-handoff-{source.name}-to-{target.name}-"
            f"{uuid.uuid4().hex}"
        )
        source_started = False
        target_prepared = False
        target_prepare_attempted = False
        decode_prepare_attempted_nodes: List[NodeMetrics] = []
        manifests: List[JsonDict] = []
        try:
            # The Controller snapshot only proves that this source recently
            # had owner-ready bootstrap work. Arm the Source Scheduler first;
            # while Router drain is in flight it continuously captures newly
            # admitted bootstrap requests before they can begin Prefill.
            # Requests that already entered computation are not captured and
            # continue normally.
            observed_rids = self._prefill_handoff_owner_ready_rids(
                metrics, source
            )
            if require_handoff and not observed_rids:
                raise RuntimeError(P_TO_D_HANDOFF_RACE_MESSAGE)
            if observed_rids or require_handoff:
                source_response = self._post_worker(
                    records,
                    "arm_source_prefill_bootstrap",
                    source,
                    "/pd_flip/prefill_handoff/source/start",
                    {
                        "session_id": session_id,
                        "target_url": target.worker_url,
                        "target_bootstrap_port": target.bootstrap_port,
                        # max_requests records the pre-RPC observation only;
                        # hold_all_eligible makes the Scheduler-captured
                        # manifests the authoritative migration cut.
                        "max_requests": len(observed_rids),
                        "rids": None,
                        "hold_all_eligible": True,
                        "pause_admission": False,
                    },
                )
                source_started = True
                manifests = _response_manifests_all(source_response)
                if require_handoff and len(manifests) < max(
                    1, self.config.p_to_d_min_handoff_requests
                ):
                    manifests = self._wait_prefill_handoff_capture(
                        records,
                        source,
                        session_id,
                        minimum_requests=max(
                            1, self.config.p_to_d_min_handoff_requests
                        ),
                    )

            # Stop new Router assignments while the armed Scheduler captures
            # requests already dispatched to this Prefill worker. Admission
            # remains open during this interval, so no measured request is
            # rejected merely because the Controller is draining the Router.
            self._post_router(
                records,
                "router_drain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": True},
            )
            self._wait_router_dispatch_quiesced(records, source)

            if source_started:
                source_response = self._post_worker(
                    records,
                    "finalize_source_prefill_bootstrap",
                    source,
                    "/pd_flip/prefill_handoff/source/start",
                    {
                        "session_id": session_id,
                        "target_url": target.worker_url,
                        "target_bootstrap_port": target.bootstrap_port,
                        "max_requests": len(observed_rids),
                        "rids": None,
                        "hold_all_eligible": True,
                        "pause_admission": True,
                    },
                )
                manifests = _response_manifests_all(source_response)
            if require_handoff and not manifests:
                raise RuntimeError(P_TO_D_HANDOFF_RACE_MESSAGE)
            if (
                require_handoff
                and len(manifests) < self.config.p_to_d_min_handoff_requests
            ):
                raise RuntimeError(P_TO_D_HANDOFF_RACE_MESSAGE)
            groups: Dict[str, Tuple[NodeMetrics, List[JsonDict]]] = {}
            if manifests:
                try:
                    # The Source Scheduler has frozen the authoritative cut,
                    # so refresh ownership for those returned bootstrap rooms.
                    # Their Decode receivers cannot finish while the matching
                    # Prefill senders are held. Reusing the pre-RPC snapshot
                    # would recreate the stale-RID race this protocol closes.
                    groups = self._wait_decode_handoff_groups(
                        records,
                        manifests,
                        initial_metrics=None,
                        allow_visibility_wait=True,
                    )
                except RuntimeError:
                    if require_handoff:
                        raise RuntimeError(P_TO_D_HANDOFF_RACE_MESSAGE)
                    raise
                def submit_decode_phase(
                    step: str, phase_payload: JsonDict
                ) -> List[Tuple[str, bool, Optional[str]]]:
                    results: List[Tuple[str, bool, Optional[str]]] = []
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max(1, len(groups))
                    ) as pool:
                        futures = []
                        for decode_node, decode_manifests in groups.values():
                            local_records: List[ActionRecord] = []
                            payload = {
                                "session_id": session_id,
                                "manifests": decode_manifests,
                            }
                            payload.update(phase_payload)
                            future = pool.submit(
                                self._post_worker,
                                local_records,
                                step,
                                decode_node,
                                "/pd_flip/prefill_handoff/decode/rebind",
                                payload,
                            )
                            futures.append((decode_node, local_records, future))
                        for decode_node, local_records, future in futures:
                            try:
                                future.result()
                                results.append((decode_node.name, True, None))
                            except Exception as exc:
                                results.append((decode_node.name, False, str(exc)))
                            finally:
                                records.extend(local_records)
                    return results

                # Phase one reserves every local Decode receiver without
                # changing ownership. If any owner rejects the batch, all
                # successful reservations are abortable and no request has
                # been partially migrated.
                prepare_results = submit_decode_phase(
                    "prepare_decode_prefill_bootstrap_rebind",
                    {"prepare_only": True},
                )
                decode_prepare_attempted_nodes = [
                    decode_node for decode_node, _ in groups.values()
                ]
                prepare_failures = [
                    item for item in prepare_results if item[1] is not True
                ]
                if prepare_failures:
                    messages = "; ".join(
                        "{}: {}".format(item[0], item[2])
                        for item in prepare_failures
                    )
                    # Prepare is read-only apart from abortable reservations.
                    # If even one owner reports that its receiver advanced out
                    # of bootstrap, abort every reservation and retry from a
                    # fresh cluster snapshot. A partial prepare acknowledgement
                    # is not a partial ownership commit.
                    if (
                        "expected exactly one decode bootstrap request"
                        in messages
                        and "found 0" in messages
                    ):
                        raise RuntimeError(P_TO_D_HANDOFF_RACE_MESSAGE)
                    raise RuntimeError(
                        "one or more Decode handoff prepares failed: " + messages
                    )

            # Source and Decode ownership are now both frozen for the exact
            # Scheduler-captured cut. Record a fresh post-cut cluster snapshot.
            snapshot_rids = [str(item.get("rid")) for item in manifests]
            paused_metrics = self.collect_metrics()
            self._p_to_d_session_context = {
                "source": source,
                "target": target,
                "session_id": session_id,
                "manifests": manifests,
            }
            self._write_p_to_d_journal(
                source=source,
                target=target,
                session_id=session_id,
                manifests=manifests,
                phase="source_held_decode_prepared",
                metadata={
                    "queue_snapshot_count": len(snapshot_rids),
                    "queue_snapshot_rids": list(snapshot_rids),
                    "pre_hold_observed_rids": list(observed_rids),
                    "decode_owner_groups": {
                        name: [str(item.get("rid")) for item in group]
                        for name, (_, group) in groups.items()
                    },
                    "held_count": len(manifests),
                },
            )
            if manifests:

                # Decode reservations must precede target reconstruction.
                # Otherwise a receiver can leave the bootstrap queue while
                # the target is rebuilding a large batch of request objects.
                target_prepare_attempted = True
                self._post_worker(
                    records,
                    "prepare_prefill_handoff_target",
                    target,
                    "/pd_flip/prefill_handoff/target/prepare",
                    {
                        "session_id": session_id,
                        "source_url": source.worker_url,
                        "manifests": manifests,
                    },
                )
                target_prepared = True
                self._write_p_to_d_journal(
                    source=source,
                    target=target,
                    session_id=session_id,
                    manifests=manifests,
                    phase="decode_prepared",
                    metadata={"decode_nodes": sorted(groups)},
                )

                # Phase two is the ownership cutover. All owners now hold a
                # reservation, so normal Decode progress cannot invalidate a
                # later owner while these concurrent commits are in flight.
                # A lost commit response is ambiguous and therefore marks the
                # rest of the handoff forward-only.
                self._p_to_d_forward_only_failure = True
                self._write_p_to_d_journal(
                    source=source,
                    target=target,
                    session_id=session_id,
                    manifests=manifests,
                    phase="decode_rebind_intent",
                    metadata={"decode_nodes": sorted(groups)},
                )
                rebind_results = submit_decode_phase(
                    "commit_decode_prefill_bootstrap_rebind",
                    {"commit_prepared": True},
                )
                rebind_failures = [
                    item for item in rebind_results if item[1] is not True
                ]
                if rebind_failures:
                    messages = "; ".join(
                        "{}: {}".format(item[0], item[2])
                        for item in rebind_failures
                    )
                    raise RuntimeError(
                        "one or more prepared Decode handoff commits failed: "
                        + messages
                    )
                self._write_p_to_d_journal(
                    source=source,
                    target=target,
                    session_id=session_id,
                    manifests=manifests,
                    phase="decode_rebound",
                    metadata={"decode_nodes": sorted(groups)},
                )
                self._write_p_to_d_journal(
                    source=source,
                    target=target,
                    session_id=session_id,
                    manifests=manifests,
                    phase="source_release_intent",
                )
                self._post_worker(
                    records,
                    "release_source_prefill_bootstrap",
                    source,
                    "/pd_flip/prefill_handoff/source/finish",
                    {
                        "session_id": session_id,
                        "released_rids": [
                            str(manifest.get("rid")) for manifest in manifests
                        ],
                    },
                )
                self._write_p_to_d_journal(
                    source=source,
                    target=target,
                    session_id=session_id,
                    manifests=manifests,
                    phase="source_released",
                    source_finished=True,
                )
                self._post_worker(
                    records,
                    "activate_prefill_handoff_target",
                    target,
                    "/pd_flip/prefill_handoff/target/activate",
                    {
                        "session_id": session_id,
                        "rids": [str(manifest.get("rid")) for manifest in manifests],
                    },
                )
                self._wait_prefill_handoff_state(
                    records,
                    target,
                    session_id,
                    "target_complete",
                    "wait_prefill_handoff_target_complete",
                )
                for decode_node, _ in groups.values():
                    self._wait_prefill_handoff_state(
                        records,
                        decode_node,
                        session_id,
                        "decode_complete",
                        "wait_decode_prefill_handoff_complete",
                    )
                self._write_p_to_d_journal(
                    source=source,
                    target=target,
                    session_id=session_id,
                    manifests=manifests,
                    phase="target_active",
                    source_finished=True,
                )
                self._p_to_d_forward_only_failure = False
        except Exception:
            if not getattr(self, "_p_to_d_forward_only_failure", False):
                abort_payload = {
                    "session_id": session_id,
                    "reason": "P->D preparation failed before decode rebind",
                }
                for decode_node in decode_prepare_attempted_nodes:
                    try:
                        self._post_worker(
                            records,
                            "abort_decode_prefill_handoff_prepare",
                            decode_node,
                            "/pd_flip/prefill_handoff/abort",
                            abort_payload,
                        )
                    except Exception:
                        pass
                if target_prepared or target_prepare_attempted:
                    try:
                        self._post_worker(
                            records,
                            "abort_prefill_handoff_target",
                            target,
                            "/pd_flip/prefill_handoff/abort",
                            abort_payload,
                        )
                    except Exception:
                        pass
                if source_started:
                    try:
                        self._post_worker(
                            records,
                            "abort_prefill_handoff_source",
                            source,
                            "/pd_flip/prefill_handoff/abort",
                            abort_payload,
                        )
                    except Exception:
                        pass
                self.session_journal.clear()
            raise
        # Requests assigned just before Router drain but outside the atomic
        # owner-ready cut were never removed from the source queue.  Let them
        # enter normal Prefill computation, then wait for those already
        # computing requests to finish before changing the worker to Decode.
        self._post_worker(
            records,
            "resume_source_admission_for_drain",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": False},
        )
        self._wait_source_idle(records, source)

    def _wait_prefill_handoff_capture(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        session_id: str,
        *,
        minimum_requests: int,
    ) -> List[JsonDict]:
        """Wait for an armed Source Scheduler to capture bootstrap work."""
        started = time.monotonic()
        started_wall = time.time()
        deadline = started + min(10.0, self.config.migration_timeout_seconds)
        path = "/pd_flip/prefill_handoff/status?session_id={}".format(
            quote(session_id)
        )
        last_response: Any = None
        manifests: List[JsonDict] = []
        sample_count = 0
        while True:
            last_response = self.client.get_json(source.worker_url, path)
            sample_count += 1
            manifests = _response_manifests_all(last_response)
            if len(manifests) >= minimum_requests:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, self.config.migration_poll_interval_seconds))
        success = len(manifests) >= minimum_requests
        records.append(
            ActionRecord(
                step="wait_source_prefill_bootstrap_capture",
                target=source.name,
                method="GET",
                url=_join_url(source.worker_url, path),
                payload={
                    "session_id": session_id,
                    "minimum_requests": minimum_requests,
                },
                response={
                    "sample_count": sample_count,
                    "captured_rids": [
                        str(manifest.get("rid")) for manifest in manifests
                    ],
                    "worker_response": last_response,
                },
                success=success,
                message=(
                    "armed source captured bootstrap work"
                    if success
                    else "armed source captured no bootstrap work before timeout"
                ),
                **_action_timing_fields(started, started_wall),
            )
        )
        if not success:
            raise RuntimeError(P_TO_D_HANDOFF_RACE_MESSAGE)
        return manifests

    def _wait_prefill_handoff_state(
        self,
        records: List[ActionRecord],
        node: NodeMetrics,
        session_id: str,
        expected_state: str,
        step: str,
    ) -> Any:
        """Wait for one worker to expose the terminal P->D handshake state.

        The final response retains scheduler-side monotonic and epoch timing,
        while this action's own timestamps measure the controller observation
        delay and HTTP polling cost.
        """
        started = time.monotonic()
        started_wall = time.time()
        deadline = started + self.config.migration_timeout_seconds
        path = "/pd_flip/prefill_handoff/status?session_id={}".format(
            quote(session_id)
        )
        url = _join_url(node.worker_url, path)
        last_response: Any = None
        sample_count = 0
        try:
            while True:
                last_response = self.client.get_json(node.worker_url, path)
                sample_count += 1
                statuses = (
                    last_response
                    if isinstance(last_response, list)
                    else [last_response]
                )
                complete = bool(statuses) and all(
                    isinstance(item, dict)
                    and item.get("success") is True
                    and isinstance(item.get("status"), dict)
                    and item["status"].get("state") == expected_state
                    for item in statuses
                )
                if complete:
                    records.append(
                        ActionRecord(
                            step=step,
                            target=node.name,
                            method="GET",
                            url=url,
                            payload={
                                "session_id": session_id,
                                "expected_state": expected_state,
                            },
                            response={
                                "sample_count": sample_count,
                                "worker_response": last_response,
                            },
                            **_action_timing_fields(started, started_wall),
                        )
                    )
                    return last_response
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        "{} timed out for {}: {}".format(
                            step, node.name, last_response
                        )
                    )
                time.sleep(
                    min(
                        self.config.migration_poll_interval_seconds,
                        max(0.0, deadline - now),
                    )
                )
        except Exception as exc:
            records.append(
                ActionRecord(
                    step=step,
                    target=node.name,
                    method="GET",
                    url=url,
                    payload={
                        "session_id": session_id,
                        "expected_state": expected_state,
                    },
                    response={
                        "sample_count": sample_count,
                        "worker_response": last_response,
                    },
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _finish_p_to_d(
        self,
        source: NodeMetrics,
        records: List[ActionRecord],
    ) -> None:
        context = getattr(self, "_p_to_d_session_context", None)
        if context:
            self._write_p_to_d_journal(
                source=context["source"],
                target=context["target"],
                session_id=context["session_id"],
                manifests=context["manifests"],
                phase="role_flip_intent",
                source_finished=bool(context["manifests"]),
            )
        # _prepare_p_to_d temporarily resumes admission so requests that were
        # already assigned before Router drain can run to completion.  The
        # Router remains drained after that wait, but the worker-side hot-cache
        # reconfigure contract also requires admission itself to be paused.
        # Re-establish that local barrier immediately before the role-set POST.
        self._post_worker(
            records,
            "pause_source_admission_for_role_flip",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": True},
        )
        # From the role-set POST until Router undrain, a lost response is an
        # ambiguous topology cutover.  Keep the worker drained for explicit
        # reconciliation instead of restoring its old Prefill routing.
        self._p_to_d_forward_only_failure = True
        self._post_worker(
            records,
            "set_source_runtime_role",
            source,
            "/pd_flip/runtime_role/set",
            {"role": "decode", "force": False},
        )
        self._wait_source_role(
            records, source, "decode", "wait_source_decode_event_loop"
        )
        # Hot reconfiguration rebuilds both the Decode and Prefill Mooncake
        # managers.  The public bootstrap address stays stable, but the new
        # Prefill manager has different manager-local rank ports.  Decode
        # peers may still cache the pre-switch ports from requests served by
        # this worker while it was Prefill.  Invalidate those routes while the
        # worker is still drained so a later immediate D->P reversal cannot
        # send migration metadata to the closed manager.
        self._invalidate_decode_prefill_peer_caches(source, records)
        self._post_router(
            records,
            "refresh_router_source_role",
            source,
            "/pd_flip/router/worker/role",
            {
                "worker_id": source.router_worker_id,
                "role": "decode",
                "bootstrap_port": None,
                "draining": True,
            },
        )
        self._post_worker(
            records,
            "resume_source_admission",
            source,
            "/pd_flip/runtime_role/admission",
            {"paused": False},
        )
        self._post_router(
            records,
            "router_undrain_source",
            source,
            "/pd_flip/router/worker/drain",
            {"worker_id": source.router_worker_id, "draining": False},
        )
        self._p_to_d_forward_only_failure = False
        if context:
            self._write_p_to_d_journal(
                source=context["source"],
                target=context["target"],
                session_id=context["session_id"],
                manifests=context["manifests"],
                phase="role_flip_complete",
                source_finished=bool(context["manifests"]),
            )

    def _invalidate_decode_prefill_peer_caches(
        self, source: NodeMetrics, records: List[ActionRecord]
    ) -> None:
        configured_source = next(
            (node for node in self.config.nodes if node.name == source.name), None
        )
        bootstrap_port = source.bootstrap_port
        if bootstrap_port is None and configured_source is not None:
            bootstrap_port = configured_source.bootstrap_port
        if bootstrap_port is None:
            # Legacy dry-run/test configurations predate runtime bootstrap
            # ports. Real runtime-switch manifests declare a port for every
            # node; if any node does, a missing source port is a hard error.
            if not any(node.bootstrap_port is not None for node in self.config.nodes):
                return
            raise RuntimeError(
                f"D->P source {source.name} has no bootstrap port for peer invalidation"
            )
        host = urlparse(source.worker_url).hostname
        if not host:
            raise RuntimeError(
                f"D->P source {source.name} has invalid worker URL {source.worker_url}"
            )
        bootstrap_addr = f"{host}:{int(bootstrap_port)}"
        for node in self.config.nodes:
            if node.name == source.name:
                continue
            self._post_worker(
                records,
                "invalidate_decode_prefill_peer_cache",
                node,
                "/pd_flip/runtime_role/invalidate_prefill_peer",
                {"bootstrap_addr": bootstrap_addr},
            )

    def _execute_p_to_d_monitor(
        self,
        *,
        metrics: List[NodeMetrics],
        state_trace: List[JsonDict],
        snapshot_index: Optional[int],
        source_name: Optional[str] = None,
        migration_target_name: Optional[str] = None,
        reason: str = "decode_slo_risk",
        require_handoff: bool = False,
    ) -> FlipExecutionResult:
        started = time.monotonic()
        records: List[ActionRecord] = []
        source: Optional[NodeMetrics] = None
        target: Optional[NodeMetrics] = None
        self._p_to_d_forward_only_failure = False
        try:
            source = self._select_p_to_d_source(
                metrics, source_name=source_name
            )
            target = self._select_prefill_handoff_target(
                metrics, source, target_name=migration_target_name
            )
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.SELECTING,
                    direction="p_to_d",
                    source=source.name,
                    migration_target=target.name,
                    role_before=source.effective_role,
                    role_after=source.effective_role,
                    reason=reason,
                    snapshot_index=snapshot_index,
                )
            )
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.PREPARING_DRAIN,
                    direction="p_to_d",
                    source=source.name,
                    role_before=source.effective_role,
                    role_after=source.effective_role,
                    reason=reason,
                    snapshot_index=snapshot_index,
                )
            )
            self._prepare_p_to_d(
                source,
                target,
                metrics,
                records,
                require_handoff=require_handoff,
            )
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.FLIPPING_ROLE,
                    direction="p_to_d",
                    source=source.name,
                    role_before=source.effective_role,
                    role_after="decode",
                    reason="source_drained",
                    snapshot_index=snapshot_index,
                    action_index=len(records),
                )
            )
            self._finish_p_to_d(source, records)
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.SAFE,
                    direction="p_to_d",
                    source=source.name,
                    role_before=source.effective_role,
                    role_after="decode",
                    reason="role_flip_complete",
                    snapshot_index=snapshot_index,
                    action_index=len(records),
                )
            )
            return FlipExecutionResult(
                success=True,
                message="pd flip executed",
                direction="p_to_d",
                source=source.name,
                target_role="decode",
                migration_target=target.name,
                actions=records,
                metrics=metrics,
                total_seconds=time.monotonic() - started,
            )
        except Exception as exc:
            if source is not None and not self._p_to_d_forward_only_failure:
                self._cleanup_source_after_failure(source, records)
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        direction="p_to_d",
                        source=source.name,
                        role_before=source.effective_role,
                        role_after=source.effective_role,
                        reason="error_recovered",
                        snapshot_index=snapshot_index,
                        action_index=len(records),
                    )
                )
            return FlipExecutionResult(
                success=False,
                message=str(exc),
                direction="p_to_d",
                source=source.name if source else None,
                target_role="decode",
                migration_target=target.name if target else None,
                actions=records,
                metrics=metrics,
                total_seconds=time.monotonic() - started,
            )

    def _post_worker(
        self,
        records: List[ActionRecord],
        step: str,
        node: NodeMetrics,
        path: str,
        payload: JsonDict,
    ) -> Any:
        return self._record_post(
            records, step, node.name, node.worker_url, path, payload
        )

    def _post_router(
        self,
        records: List[ActionRecord],
        step: str,
        node: NodeMetrics,
        path: str,
        payload: JsonDict,
    ) -> Any:
        return self._record_post(
            records,
            step,
            f"router:{node.router_worker_id}",
            self.config.router_url,
            path,
            payload,
        )

    def _record_post(
        self,
        records: List[ActionRecord],
        step: str,
        target: str,
        base_url: str,
        path: str,
        payload: JsonDict,
    ) -> Any:
        started = time.monotonic()
        started_wall = time.time()
        url = _join_url(base_url, path)
        response = None
        try:
            response = self.client.post_json(base_url, path, payload)
            _raise_if_unsuccessful(response, step)
            records.append(
                ActionRecord(
                    step=step,
                    target=target,
                    method="POST",
                    url=url,
                    payload=payload,
                    response=response,
                    **_action_timing_fields(started, started_wall),
                )
            )
            return response
        except Exception as exc:
            records.append(
                ActionRecord(
                    step=step,
                    target=target,
                    method="POST",
                    url=url,
                    payload=payload,
                    response=response,
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _record_get(
        self,
        records: List[ActionRecord],
        step: str,
        target: str,
        base_url: str,
        path: str,
    ) -> Any:
        started = time.monotonic()
        started_wall = time.time()
        url = _join_url(base_url, path)
        try:
            response = self.client.get_json(base_url, path)
            _raise_if_unsuccessful(response, step)
            records.append(
                ActionRecord(
                    step=step,
                    target=target,
                    method="GET",
                    url=url,
                    response=response,
                    **_action_timing_fields(started, started_wall),
                )
            )
            return response
        except Exception as exc:
            records.append(
                ActionRecord(
                    step=step,
                    target=target,
                    method="GET",
                    url=url,
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _wait_migration(
        self,
        records: List[ActionRecord],
        step: str,
        node: NodeMetrics,
    ) -> Any:
        deadline = time.monotonic() + self.config.migration_timeout_seconds
        last_response: Any = None
        while time.monotonic() <= deadline:
            last_response = self._record_get(
                records,
                step,
                node.name,
                node.worker_url,
                "/pd_flip/migration/status",
            )
            _require_worker_dp_ranks(
                last_response, node.dp_statuses or [node.raw_status], step
            )
            if _migration_response_complete(last_response):
                return last_response
            if _migration_response_failed(last_response):
                raise RuntimeError(
                    f"{step} failed for {node.name}: "
                    f"{_migration_response_error(last_response)}"
                )
            time.sleep(self.config.migration_poll_interval_seconds)
        raise TimeoutError(f"{step} timed out for {node.name}: {last_response}")

    def _observe_source_quiesce(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
    ) -> JsonDict:
        return self._observe_source_quiesce_for_duration(
            records, source, self.config.observation_quiesce_seconds
        )

    def _observe_source_quiesce_for_duration(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        duration_seconds: float,
    ) -> JsonDict:
        started = time.monotonic()
        started_wall = time.time()
        url = _join_url(source.worker_url, "/pd_flip/runtime_role/status")
        requested_duration = duration_seconds
        deadline = started + max(0.0, requested_duration)
        samples: List[JsonDict] = []
        try:
            while True:
                samples.append(self._source_residual_snapshot(source))
                now = time.monotonic()
                if now >= deadline:
                    break
                time.sleep(
                    min(
                        self.config.migration_poll_interval_seconds,
                        max(0.0, deadline - now),
                    )
                )
            response = dict(samples[-1]) if samples else {}
            response.update(
                {
                    "samples": samples,
                    "sample_count": len(samples),
                    "source_quiesce_elapsed_s": time.monotonic() - started,
                    "configured_observation_quiesce_seconds": self.config.observation_quiesce_seconds,
                    "requested_observation_quiesce_seconds": requested_duration,
                }
            )
            records.append(
                ActionRecord(
                    step="observe_source_quiesce",
                    target=source.name,
                    method="GET",
                    url=url,
                    response=response,
                    **_action_timing_fields(started, started_wall),
                )
            )
            return response
        except Exception as exc:
            records.append(
                ActionRecord(
                    step="observe_source_quiesce",
                    target=source.name,
                    method="GET",
                    url=url,
                    success=False,
                    message=str(exc),
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise

    def _source_residual_snapshot(self, source: NodeMetrics) -> JsonDict:
        status_body = self.client.get_json(
            source.worker_url, "/pd_flip/runtime_role/status"
        )
        status = _first_successful_response(status_body)
        role, is_idle, admission_paused = _parse_runtime_status(status)
        loads_body = self.client.get_json(source.worker_url, "/v1/loads?include=all")
        running_reqs, waiting_reqs, total_tokens, token_usage, raw_loads = _parse_loads(
            loads_body
        )
        decode_prealloc_reqs = _sum_load_metric(raw_loads, "decode_prealloc_queue_reqs")
        decode_transfer_reqs = _sum_load_metric(raw_loads, "decode_transfer_queue_reqs")
        decode_retracted_reqs = _sum_load_metric(
            raw_loads, "decode_retracted_queue_reqs"
        )
        prefill_bootstrap_reqs = _sum_load_metric(
            raw_loads, "prefill_bootstrap_queue_reqs"
        )
        prefill_inflight_reqs = _sum_load_metric(
            raw_loads, "prefill_inflight_queue_reqs"
        )
        total_residual_reqs = (
            running_reqs
            + waiting_reqs
            + decode_prealloc_reqs
            + decode_transfer_reqs
            + decode_retracted_reqs
            + prefill_bootstrap_reqs
            + prefill_inflight_reqs
        )
        server_info: JsonDict = {}
        try:
            info = self.client.get_json(source.worker_url, "/server_info")
            server_info = info if isinstance(info, dict) else {"raw": info}
        except Exception as exc:
            server_info = {"error": str(exc)}
        return {
            "source_role": role,
            "source_is_idle": is_idle,
            "source_admission_paused": admission_paused,
            "source_running_reqs": running_reqs,
            "source_waiting_queue_reqs": waiting_reqs,
            "source_decode_prealloc_queue_reqs": decode_prealloc_reqs,
            "source_decode_transfer_queue_reqs": decode_transfer_reqs,
            "source_decode_retracted_queue_reqs": decode_retracted_reqs,
            "source_prefill_bootstrap_queue_reqs": prefill_bootstrap_reqs,
            "source_prefill_inflight_queue_reqs": prefill_inflight_reqs,
            "source_total_residual_reqs": total_residual_reqs,
            "source_total_tokens": total_tokens,
            "source_token_usage": token_usage,
            "raw_runtime_status": status_body,
            "raw_loads": raw_loads,
            "raw_server_info": server_info,
        }

    def _assert_source_idle_after_migration(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
    ) -> Any:
        started = time.monotonic()
        started_wall = time.time()
        step = "post_migration_idle_assertion"
        path = "/pd_flip/runtime_role/status"
        url = _join_url(source.worker_url, path)
        deadline = started + max(0.0, self.config.post_migration_idle_timeout_seconds)
        samples: List[Any] = []
        last_response: Any = None
        try:
            while True:
                last_response = self.client.get_json(source.worker_url, path)
                _raise_if_unsuccessful(last_response, step)
                samples.append(last_response)
                status = _first_successful_response(last_response)
                _, is_idle, _ = _parse_runtime_status(status)
                if is_idle:
                    response = {
                        "source_idle": True,
                        "sample_count": len(samples),
                        "samples": samples,
                        "last_response": last_response,
                    }
                    records.append(
                        ActionRecord(
                            step=step,
                            target=source.name,
                            method="GET",
                            url=url,
                            response=response,
                            **_action_timing_fields(started, started_wall),
                        )
                    )
                    return last_response
                now = time.monotonic()
                if now >= deadline:
                    break
                time.sleep(
                    min(
                        self.config.migration_poll_interval_seconds,
                        max(0.0, deadline - now),
                    )
                )
            message = f"{step} timed out for {source.name}: {last_response}"
            records.append(
                ActionRecord(
                    step=step,
                    target=source.name,
                    method="GET",
                    url=url,
                    response={
                        "source_idle": False,
                        "sample_count": len(samples),
                        "samples": samples,
                        "last_response": last_response,
                    },
                    success=False,
                    message=message,
                    **_action_timing_fields(started, started_wall),
                )
            )
            raise TimeoutError(message)
        except Exception as exc:
            if not records or records[-1].step != step:
                records.append(
                    ActionRecord(
                        step=step,
                        target=source.name,
                        method="GET",
                        url=url,
                        success=False,
                        message=str(exc),
                        **_action_timing_fields(started, started_wall),
                    )
                )
            raise

    def _wait_source_idle(
        self,
        records: List[ActionRecord],
        source: NodeMetrics,
        *,
        step: str = "wait_source_idle",
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        timeout = (
            self.config.migration_timeout_seconds
            if timeout_seconds is None
            else max(0.0, timeout_seconds)
        )
        deadline = time.monotonic() + timeout
        last_response: Any = None
        while True:
            last_response = self._record_get(
                records,
                step,
                source.name,
                source.worker_url,
                "/pd_flip/runtime_role/status",
            )
            status = _first_successful_response(last_response)
            _, is_idle, _ = _parse_runtime_status(status)
            if is_idle:
                return last_response
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(
                min(
                    self.config.migration_poll_interval_seconds,
                    max(0.0, deadline - now),
                )
            )
        raise TimeoutError(f"{step} timed out for {source.name}: {last_response}")

    def _cleanup_source_after_failure(
        self,
        source: NodeMetrics,
        records: List[ActionRecord],
    ) -> None:
        try:
            response = self._record_get(
                records,
                "cleanup_inspect_source_cache_state",
                source.name,
                source.worker_url,
                "/pd_flip/runtime_role/status",
            )
            shards = response if isinstance(response, list) else [response]
            fatal_cache_state = any(
                isinstance(shard, dict)
                and isinstance(shard.get("status"), dict)
                and shard["status"].get("cache_reconfigure_state") == "fatal"
                for shard in shards
            )
        except Exception:
            fatal_cache_state = False
        if fatal_cache_state:
            records.append(
                ActionRecord(
                    step="cleanup_preserve_fatal_source_isolation",
                    target=source.name,
                    method="NONE",
                    url=source.worker_url,
                    success=True,
                    message=(
                        "source admission remains paused and router remains drained "
                        "after fatal cache reconstruction failure"
                    ),
                )
            )
            return
        try:
            self._post_worker(
                records,
                "cleanup_resume_source_admission",
                source,
                "/pd_flip/runtime_role/admission",
                {"paused": False},
            )
        except Exception:
            pass
        try:
            self._post_router(
                records,
                "cleanup_router_undrain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": False},
            )
        except Exception:
            pass

    def _fetch_router_workers(self) -> Dict[str, JsonDict]:
        body = self.client.get_json(self.config.router_url, "/pd_flip/router/workers")
        workers = body.get("workers", []) if isinstance(body, dict) else []
        return {
            str(worker.get("worker_id")): worker
            for worker in workers
            if isinstance(worker, dict) and worker.get("worker_id") is not None
        }

    def _migration_rollover_eligibility(self, node: NodeMetrics) -> JsonDict:
        """Return whether ``node`` can enter a new source/target session."""

        evidence: JsonDict = {
            "source_feasible": True,
            "target_feasible": True,
            "dp_statuses": [],
        }
        try:
            response = self.client.get_json(
                node.worker_url, "/pd_flip/migration/status"
            )
            indexed = _index_dp_responses(response)
        except Exception as exc:
            evidence.update(
                {
                    "source_feasible": False,
                    "target_feasible": False,
                    "reason": "migration_status_unavailable",
                    "message": str(exc),
                }
            )
            return evidence

        for rank in sorted(indexed):
            item = indexed[rank]
            status = (
                item.get("status")
                if isinstance(item.get("status"), dict)
                else item
            )
            row: JsonDict = {
                "dp_rank": rank,
                "success": item.get("success", True) is not False,
                "enabled": bool(status.get("enabled", False)),
                "role": _normalize_role(status.get("role")),
                "state": str(status.get("state") or "none"),
                "session_id": status.get("session_id"),
                "rollover_blockers": list(
                    status.get("rollover_blockers") or []
                ),
            }
            if not row["success"]:
                row["source_feasible"] = False
                row["target_feasible"] = False
            elif not row["enabled"]:
                row["source_feasible"] = True
                row["target_feasible"] = True
            else:
                terminal_session_feasible = not row["rollover_blockers"]
                # The worker validates and archives the old session according
                # to its existing role.  Once terminal, it can enter either
                # role in the next migration generation.
                row["source_feasible"] = terminal_session_feasible
                row["target_feasible"] = terminal_session_feasible
            evidence["dp_statuses"].append(row)

        evidence["source_feasible"] = all(
            row["source_feasible"] for row in evidence["dp_statuses"]
        )
        evidence["target_feasible"] = all(
            row["target_feasible"] for row in evidence["dp_statuses"]
        )
        return evidence

    def _select_source(
        self,
        metrics: List[NodeMetrics],
        *,
        source_name: Optional[str],
        expected_role: str,
        prefer_high_load: bool,
    ) -> NodeMetrics:
        if source_name:
            source = _find_metric(metrics, source_name)
            if source is None:
                raise ValueError(f"unknown source node: {source_name}")
            if source.effective_role != expected_role:
                raise ValueError(
                    f"source node {source.name} has role {source.effective_role}, "
                    f"expected {expected_role}"
                )
            return source

        candidates = [
            metric
            for metric in metrics
            if metric.effective_role == expected_role and not metric.draining
        ]
        if not candidates:
            raise RuntimeError(f"no non-draining {expected_role} source is available")
        candidates.sort(key=_load_sort_key, reverse=prefer_high_load)
        return candidates[0]

    def _select_p_to_d_source(
        self,
        metrics: List[NodeMetrics],
        *,
        source_name: Optional[str],
    ) -> NodeMetrics:
        if source_name:
            return self._select_source(
                metrics,
                source_name=source_name,
                expected_role="prefill",
                prefer_high_load=False,
            )
        candidates = [
            metric
            for metric in metrics
            if metric.effective_role == "prefill"
            and not metric.draining
            and len(metric.dp_statuses) == 1
        ]
        if not candidates:
            raise RuntimeError("no non-draining DP=1 prefill source is available")
        with_owner = [
            metric
            for metric in candidates
            if len(self._prefill_handoff_owner_ready_rids(metrics, metric))
            >= self.config.p_to_d_min_handoff_requests
        ]
        with_bootstrap = [
            metric
            for metric in candidates
            if self._prefill_handoff_candidate_rids(metric)
        ]
        selected_pool = with_owner or with_bootstrap or candidates
        if self.config.decision_policy == "prefill_queue_util":
            selected_pool.sort(
                key=lambda metric: (
                    self._prefill_busy_ratio(metric)
                    if self._prefill_busy_ratio(metric) is not None
                    else float("inf"),
                    -len(self._prefill_handoff_candidate_rids(metric)),
                    _load_sort_key(metric),
                )
            )
        else:
            selected_pool.sort(
                key=lambda metric: (
                    -len(self._prefill_handoff_candidate_rids(metric)),
                    _load_sort_key(metric),
                )
            )
        return selected_pool[0]

    def _select_decode_migration_target(
        self,
        metrics: List[NodeMetrics],
        source: NodeMetrics,
        *,
        target_name: Optional[str] = None,
    ) -> NodeMetrics:
        if target_name:
            target = _find_metric(metrics, target_name)
            if target is None:
                raise ValueError(f"unknown migration target node: {target_name}")
            if target.name == source.name:
                raise ValueError("migration target must be different from source")
            if target.effective_role != "decode":
                raise ValueError(
                    f"migration target {target.name} has role {target.effective_role}, expected decode"
                )
            return target

        candidates = [
            metric
            for metric in metrics
            if metric.name != source.name
            and metric.effective_role == "decode"
            and not metric.draining
        ]
        if not candidates:
            raise RuntimeError(
                "D->P requires another non-draining decode node as migration target"
            )
        candidates.sort(key=_load_sort_key)
        return candidates[0]

    def _select_prefill_handoff_target(
        self,
        metrics: List[NodeMetrics],
        source: NodeMetrics,
        *,
        target_name: Optional[str] = None,
    ) -> NodeMetrics:
        if target_name:
            target = _find_metric(metrics, target_name)
            if target is None:
                raise ValueError(f"unknown handoff target node: {target_name}")
            if target.name == source.name:
                raise ValueError("prefill handoff target must be different from source")
            if target.effective_role != "prefill":
                raise ValueError(
                    f"handoff target {target.name} has role {target.effective_role}, expected prefill"
                )
            if target.draining:
                raise ValueError(f"handoff target {target.name} is draining")
            if target.bootstrap_port is None:
                raise ValueError(f"handoff target {target.name} has no bootstrap port")
            if len(target.dp_statuses) != 1:
                raise ValueError("P->D bootstrap handoff currently requires DP=1 workers")
            return target

        candidates = [
            metric
            for metric in metrics
            if metric.name != source.name
            and metric.effective_role == "prefill"
            and not metric.draining
            and metric.bootstrap_port is not None
            and len(metric.dp_statuses) == 1
        ]
        if not candidates:
            raise RuntimeError(
                "P->D bootstrap handoff requires another non-draining prefill node"
            )
        candidates.sort(key=_load_sort_key)
        return candidates[0]

    @staticmethod
    def _prefill_handoff_candidate_rids(source: NodeMetrics) -> List[str]:
        if len(source.dp_statuses) != 1:
            raise ValueError("P->D bootstrap handoff currently requires DP=1 workers")
        item = source.dp_statuses[0]
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        return [
            str(request.get("rid"))
            for request in status.get("prefill_bootstrap_requests") or []
            if isinstance(request, dict)
            and request.get("rid") is not None
            and request.get("pending_bootstrap") is True
        ]

    @staticmethod
    def _decode_handoff_owner_entries(
        metrics: List[NodeMetrics], room: int
    ) -> List[Tuple[NodeMetrics, str]]:
        matches: List[Tuple[NodeMetrics, str]] = []
        for metric in metrics:
            if metric.effective_role != "decode":
                continue
            for item in metric.dp_statuses:
                status = (
                    item.get("status")
                    if isinstance(item.get("status"), dict)
                    else item
                )
                requests = status.get("decode_bootstrap_requests") or []
                for request in requests:
                    if (
                        isinstance(request, dict)
                        and request.get("rid") is not None
                        and request.get("bootstrap_room") is not None
                        and int(request.get("bootstrap_room")) == room
                    ):
                        matches.append((metric, str(request.get("rid"))))
        return matches

    @classmethod
    def _prefill_handoff_owner_ready_rids(
        cls, metrics: List[NodeMetrics], source: NodeMetrics
    ) -> List[str]:
        groups = cls._prefill_handoff_owner_ready_groups(metrics, source)
        grouped = {rid for rids in groups.values() for rid in rids}
        return [
            rid
            for rid in cls._prefill_handoff_candidate_rids(source)
            if rid in grouped
        ]

    @classmethod
    def _prefill_handoff_owner_ready_groups(
        cls, metrics: List[NodeMetrics], source: NodeMetrics
    ) -> Dict[str, List[str]]:
        if len(source.dp_statuses) != 1:
            raise ValueError("P->D bootstrap handoff currently requires DP=1 workers")
        item = source.dp_statuses[0]
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        groups: Dict[str, List[str]] = {}
        for request in status.get("prefill_bootstrap_requests") or []:
            if (
                not isinstance(request, dict)
                or request.get("rid") is None
                or request.get("bootstrap_room") is None
                or request.get("pending_bootstrap") is not True
            ):
                continue
            rid = str(request["rid"])
            room = int(request["bootstrap_room"])
            matches = cls._decode_handoff_owner_entries(metrics, room)
            if len(matches) == 1:
                groups.setdefault(matches[0][0].name, []).append(rid)
        return groups

    def _wait_decode_handoff_groups(
        self,
        records: List[ActionRecord],
        manifests: Sequence[JsonDict],
        *,
        initial_metrics: Optional[List[NodeMetrics]] = None,
        allow_visibility_wait: bool = True,
    ) -> Dict[str, Tuple[NodeMetrics, List[JsonDict]]]:
        """Wait until every held bootstrap request has one Decode owner."""
        started_monotonic = time.monotonic()
        started_wall = time.time()
        # A complete ownership sample queries every worker. On an eight-worker
        # deployment that collection can itself take close to two seconds, and
        # Decode preallocation may become visible just after the source's
        # atomic hold. Allow several full samples before treating it as a race.
        deadline = started_monotonic + min(
            (10.0 if allow_visibility_wait else 0.0),
            max(0.1, self.config.migration_timeout_seconds),
        )
        samples: List[JsonDict] = []
        groups: Optional[Dict[str, Tuple[NodeMetrics, List[JsonDict]]]] = None
        last_error = "no decode ownership sample was collected"
        while True:
            if initial_metrics is not None:
                decode_metrics = initial_metrics
                initial_metrics = None
            else:
                decode_metrics = self.collect_metrics()
            ownership = []
            all_unique = True
            for manifest in manifests:
                rid = str(manifest.get("rid"))
                room = int(manifest.get("original_bootstrap_room"))
                matches = self._decode_handoff_owner_entries(decode_metrics, room)
                ownership.append(
                    {
                        "rid": rid,
                        "bootstrap_room": room,
                        "matches": [
                            {"node": node.name, "decode_rid": decode_rid}
                            for node, decode_rid in matches
                        ],
                    }
                )
                all_unique &= len(matches) == 1
            samples.append(
                {
                    "held_count": len(manifests),
                    "unique_owner_count": sum(
                        len(item["matches"]) == 1 for item in ownership
                    ),
                    "ownership": ownership,
                }
            )
            if all_unique:
                groups = self._decode_handoff_groups(decode_metrics, manifests)
                break
            last_error = (
                "not every held bootstrap request has exactly one decode owner"
            )
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        records.append(
            ActionRecord(
                step="inspect_prefill_decode_bootstrap_ownership",
                target="cluster",
                method="GET",
                url="/pd_flip/runtime_role/status",
                payload={
                    "held": [
                        {
                            "rid": str(manifest.get("rid")),
                            "bootstrap_room": manifest.get(
                                "original_bootstrap_room"
                            ),
                        }
                        for manifest in manifests
                    ]
                },
                response={
                    "samples": samples,
                    "decode_groups": sorted(groups) if groups is not None else [],
                },
                success=groups is not None,
                message=(
                    "all held requests have unique decode owners"
                    if groups is not None
                    else last_error
                ),
                **_action_timing_fields(started_monotonic, started_wall),
            )
        )
        if groups is None:
            raise RuntimeError(last_error)
        return groups

    @staticmethod
    def _decode_handoff_groups(
        metrics: List[NodeMetrics], manifests: Sequence[JsonDict]
    ) -> Dict[str, Tuple[NodeMetrics, List[JsonDict]]]:
        groups: Dict[str, Tuple[NodeMetrics, List[JsonDict]]] = {}
        for manifest in manifests:
            rid = str(manifest.get("rid"))
            room = int(manifest.get("original_bootstrap_room"))
            matches = PDFlipController._decode_handoff_owner_entries(metrics, room)
            if len(matches) != 1:
                raise RuntimeError(
                    "could not identify exactly one decode owner for bootstrap "
                    f"request prefill_rid={rid} room={room}; "
                    f"matches={[[m.name, decode_rid] for m, decode_rid in matches]}"
                )
            node, decode_rid = matches[0]
            manifest["original_decode_rid"] = decode_rid
            if node.name not in groups:
                groups[node.name] = (node, [])
            groups[node.name][1].append(manifest)
        return groups

    def _build_d_to_p_actions(
        self, source: NodeMetrics, target: NodeMetrics
    ) -> List[ControllerAction]:
        session_id = f"pd-flip-{source.name}-to-{target.name}"
        return [
            self._router_action(
                "router_drain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": True},
            ),
            self._worker_action(
                "pause_source_admission",
                source,
                "POST",
                "/pd_flip/runtime_role/admission",
                {"paused": True},
            ),
            self._worker_action(
                "observe_source_quiesce",
                source,
                "GET",
                "/pd_flip/runtime_role/status",
                {
                    "duration_seconds": 0.0,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                    "also_samples": ["/v1/loads?include=all", "/server_info"],
                },
            ),
            self._worker_action(
                "start_decode_migration_source",
                source,
                "POST",
                "/pd_flip/migration/source/start",
                _migration_source_start_payload(
                    session_id, target.worker_url, None, include_waiting=True
                ),
            ),
            self._worker_action(
                "prepare_decode_migration_target",
                target,
                "POST",
                "/pd_flip/migration/target/prepare",
                {
                    "session_id": session_id,
                    "source_url": source.worker_url,
                    "manifests": "<from start_decode_migration_source>",
                    "prepare_only": True,
                    "adopt_on_commit": True,
                },
            ),
            self._worker_action(
                "wait_decode_migration_source",
                source,
                "GET",
                "/pd_flip/migration/status",
                {
                    "timeout_seconds": self.config.migration_timeout_seconds,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                    "target_url": target.worker_url,
                },
            ),
            self._worker_action(
                "wait_decode_migration_target",
                target,
                "GET",
                "/pd_flip/migration/status",
                {
                    "timeout_seconds": self.config.migration_timeout_seconds,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                    "source_url": source.worker_url,
                },
            ),
            self._worker_action(
                "start_decode_migration_source_delta",
                source,
                "POST",
                "/pd_flip/migration/source/delta",
                {
                    "session_id": session_id,
                    "rids": "<from start_decode_migration_source>",
                },
            ),
            self._worker_action(
                "prepare_decode_migration_target_delta",
                target,
                "POST",
                "/pd_flip/migration/target/delta/prepare",
                {
                    "session_id": session_id,
                    "source_url": source.worker_url,
                    "manifests": "<from start_decode_migration_source_delta>",
                },
            ),
            self._worker_action(
                "wait_decode_migration_source_delta",
                source,
                "GET",
                "/pd_flip/migration/status",
                {
                    "timeout_seconds": self.config.migration_timeout_seconds,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                    "target_url": target.worker_url,
                },
            ),
            self._worker_action(
                "wait_decode_migration_target_delta",
                target,
                "GET",
                "/pd_flip/migration/status",
                {
                    "timeout_seconds": self.config.migration_timeout_seconds,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                    "source_url": source.worker_url,
                },
            ),
            self._worker_action(
                "finish_decode_migration_source",
                source,
                "POST",
                "/pd_flip/migration/source/finish",
                {
                    "session_id": session_id,
                    "released_rids": "<from migration target manifests>",
                },
            ),
            self._worker_action(
                "post_migration_idle_assertion",
                source,
                "GET",
                "/pd_flip/runtime_role/status",
                {
                    "timeout_seconds": self.config.post_migration_idle_timeout_seconds,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                },
            ),
            self._worker_action(
                "set_source_runtime_role",
                source,
                "POST",
                "/pd_flip/runtime_role/set",
                {"role": "prefill", "force": False},
            ),
            ControllerAction(
                step="invalidate_decode_prefill_peer_cache",
                target="<all other runtime-switch workers>",
                method="POST",
                url="/pd_flip/runtime_role/invalidate_prefill_peer",
                payload={"bootstrap_addr": "<D->P source host:bootstrap_port>"},
            ),
            self._router_action(
                "refresh_router_source_role",
                source,
                "/pd_flip/router/worker/role",
                {
                    "worker_id": source.router_worker_id,
                    "role": "prefill",
                    "bootstrap_port": source.bootstrap_port,
                    "draining": False,
                },
            ),
            self._worker_action(
                "resume_source_admission",
                source,
                "POST",
                "/pd_flip/runtime_role/admission",
                {"paused": False},
            ),
            self._router_action(
                "router_undrain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": False},
            ),
        ]

    def _build_p_to_d_actions(
        self, source: NodeMetrics, target: NodeMetrics
    ) -> List[ControllerAction]:
        session_id = f"pd-flip-prefill-handoff-{source.name}-to-{target.name}-<uuid>"
        return [
            self._worker_action(
                "arm_source_prefill_bootstrap",
                source,
                "POST",
                "/pd_flip/prefill_handoff/source/start",
                {
                    "session_id": session_id,
                    "target_url": target.worker_url,
                    "target_bootstrap_port": target.bootstrap_port,
                    "max_requests": "<pre-hold observed owner-ready count>",
                    "rids": None,
                    "hold_all_eligible": True,
                    "pause_admission": False,
                },
            ),
            self._router_action(
                "router_drain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": True},
            ),
            self._worker_action(
                "finalize_source_prefill_bootstrap",
                source,
                "POST",
                "/pd_flip/prefill_handoff/source/start",
                {
                    "session_id": session_id,
                    "target_url": target.worker_url,
                    "target_bootstrap_port": target.bootstrap_port,
                    "max_requests": "<pre-hold observed owner-ready count>",
                    "rids": None,
                    "hold_all_eligible": True,
                    "pause_admission": True,
                },
            ),
            ControllerAction(
                step="prepare_decode_prefill_bootstrap_rebind",
                target="<decode owner of bootstrap room>",
                method="POST",
                url="<decode worker>/pd_flip/prefill_handoff/decode/rebind",
                payload={
                    "session_id": session_id,
                    "manifests": "<Scheduler-captured source manifests>",
                    "prepare_only": True,
                },
            ),
            self._worker_action(
                "prepare_prefill_handoff_target",
                target,
                "POST",
                "/pd_flip/prefill_handoff/target/prepare",
                {
                    "session_id": session_id,
                    "source_url": source.worker_url,
                    "manifests": "<source manifests>",
                },
            ),
            ControllerAction(
                step="commit_decode_prefill_bootstrap_rebind",
                target="<decode owner of bootstrap room>",
                method="POST",
                url="<decode worker>/pd_flip/prefill_handoff/decode/rebind",
                payload={
                    "session_id": session_id,
                    "manifests": "<source manifests for this decode worker>",
                    "commit_prepared": True,
                },
            ),
            self._worker_action(
                "release_source_prefill_bootstrap",
                source,
                "POST",
                "/pd_flip/prefill_handoff/source/finish",
                {
                    "session_id": session_id,
                    "released_rids": "<migrated bootstrap RIDs>",
                },
            ),
            self._worker_action(
                "activate_prefill_handoff_target",
                target,
                "POST",
                "/pd_flip/prefill_handoff/target/activate",
                {
                    "session_id": session_id,
                    "rids": "<migrated bootstrap RIDs>",
                },
            ),
            self._worker_action(
                "wait_source_idle",
                source,
                "GET",
                "/pd_flip/runtime_role/status",
                {
                    "timeout_seconds": self.config.migration_timeout_seconds,
                    "poll_interval_seconds": self.config.migration_poll_interval_seconds,
                },
            ),
            self._worker_action(
                "set_source_runtime_role",
                source,
                "POST",
                "/pd_flip/runtime_role/set",
                {"role": "decode", "force": False},
            ),
            self._worker_action(
                "wait_source_decode_event_loop",
                source,
                "GET",
                "/pd_flip/runtime_role/status",
                {
                    "expected_role": "decode",
                    "expected_active_event_loop_role": "decode",
                },
            ),
            self._router_action(
                "refresh_router_source_role",
                source,
                "/pd_flip/router/worker/role",
                {
                    "worker_id": source.router_worker_id,
                    "role": "decode",
                    "bootstrap_port": None,
                    "draining": True,
                },
            ),
            self._worker_action(
                "resume_source_admission",
                source,
                "POST",
                "/pd_flip/runtime_role/admission",
                {"paused": False},
            ),
            self._router_action(
                "router_undrain_source",
                source,
                "/pd_flip/router/worker/drain",
                {"worker_id": source.router_worker_id, "draining": False},
            ),
        ]

    def _router_action(
        self,
        step: str,
        source: NodeMetrics,
        path: str,
        payload: JsonDict,
    ) -> ControllerAction:
        return ControllerAction(
            step=step,
            target=f"router:{source.router_worker_id}",
            method="POST",
            url=_join_url(self.config.router_url, path),
            payload=payload,
        )

    def _worker_action(
        self,
        step: str,
        node: NodeMetrics,
        method: str,
        path: str,
        payload: Optional[JsonDict],
    ) -> ControllerAction:
        return ControllerAction(
            step=step,
            target=node.name,
            method=method,
            url=_join_url(node.worker_url, path),
            payload=payload,
        )


def _join_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _normalize_role(role: Any) -> str:
    if role is None:
        return "unknown"
    normalized = str(role).strip().lower()
    return normalized or "unknown"


def _first_successful_response(response: Any) -> JsonDict:
    responses = response if isinstance(response, list) else [response]
    for item in responses:
        if isinstance(item, dict) and item.get("success", True):
            return item
    if responses and isinstance(responses[0], dict):
        return responses[0]
    return {}


def _strict_migration_statuses(response: Any, session_id: str) -> List[JsonDict]:
    responses = response if isinstance(response, list) else [response]
    if not responses:
        raise ValueError("migration status response is empty")
    statuses = []
    for item in responses:
        if not isinstance(item, dict):
            raise ValueError("migration status response item is not an object")
        if item.get("success") is not True:
            raise ValueError("migration status response item was unsuccessful")
        status = item.get("status")
        if not isinstance(status, dict):
            raise ValueError("migration status response status is not an object")
        if str(status.get("session_id")) != str(session_id):
            raise ValueError("migration status response session id does not match")
        statuses.append(status)
    return statuses


def _raise_if_unsuccessful(response: Any, step: str) -> None:
    responses = response if isinstance(response, list) else [response]
    for item in responses:
        if isinstance(item, dict) and item.get("success", True) is False:
            status = item.get("status") if isinstance(item.get("status"), dict) else {}
            raise RuntimeError(
                status.get("last_error")
                or item.get("message")
                or f"{step} failed"
            )


_SESSION_ABSENT_ABORT_MARKERS = (
    "session id does not match",
    "migration session not found",
    "session is already absent",
    "session already absent",
    "session not found",
    "no target migration session exists",
    "no source migration session exists",
)


def _response_has_session_absent(response: Any) -> bool:
    responses = response if isinstance(response, list) else [response]
    for item in responses:
        if not isinstance(item, dict) or item.get("success", True) is not False:
            continue
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        detail = str(status.get("last_error") or item.get("message") or "").lower()
        if any(marker in detail for marker in _SESSION_ABSENT_ABORT_MARKERS):
            return True
    return False


def _abort_response_is_idempotent(response: Any) -> bool:
    responses = response if isinstance(response, list) else [response]
    if not responses:
        return False
    for item in responses:
        if not isinstance(item, dict):
            return False
        if item.get("success", True) is not False:
            continue
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        detail = str(status.get("last_error") or item.get("message") or "").lower()
        if not any(marker in detail for marker in _SESSION_ABSENT_ABORT_MARKERS):
            return False
    return True


def _response_manifests(response: Any) -> List[JsonDict]:
    item = _first_successful_response(response)
    manifests = item.get("manifests", [])
    return [manifest for manifest in manifests if isinstance(manifest, dict)]


def _response_manifests_all(response: Any) -> List[JsonDict]:
    responses = response if isinstance(response, list) else [response]
    manifests = []
    seen = set()
    for item in responses:
        if not isinstance(item, dict) or item.get("success", True) is False:
            continue
        for manifest in item.get("manifests") or []:
            if not isinstance(manifest, dict):
                continue
            rid = str(manifest.get("rid"))
            if rid not in seen:
                seen.add(rid)
                manifests.append(manifest)
    return manifests


def _strict_response_manifests(response: Any, error_prefix: str) -> List[JsonDict]:
    responses = response if isinstance(response, list) else [response]
    if not responses:
        raise RuntimeError(f"{error_prefix}: response is empty")
    manifests: List[JsonDict] = []
    seen_rids = set()
    for item in responses:
        if not isinstance(item, dict):
            raise RuntimeError(f"{error_prefix}: response item is not an object")
        item_manifests = item.get("manifests")
        if not isinstance(item_manifests, list):
            raise RuntimeError(f"{error_prefix}: manifests is not a list")
        for manifest in item_manifests:
            if not isinstance(manifest, dict):
                raise RuntimeError(f"{error_prefix}: manifest is not an object")
            rid = manifest.get("rid")
            rid_text = "" if rid is None else str(rid).strip()
            if not rid_text:
                raise RuntimeError(f"{error_prefix}: manifest RID is missing or empty")
            if rid_text in seen_rids:
                raise RuntimeError(f"{error_prefix}: duplicate manifest RID {rid_text}")
            seen_rids.add(rid_text)
            manifests.append(manifest)
    if not manifests:
        raise RuntimeError(f"{error_prefix}: manifests is empty")
    return manifests


def _delta_quiesce_pending(response: Any) -> bool:
    responses = response if isinstance(response, list) else [response]
    if not responses:
        return False
    return all(
        isinstance(item, dict)
        and item.get("success") is False
        and item.get("manifests") == []
        and isinstance(item.get("manifests"), list)
        and item.get("message") == SOURCE_DELTA_QUIESCE_PENDING_MESSAGE
        for item in responses
    )


def _manifest_rids(manifests: List[JsonDict]) -> List[str]:
    return [
        str(manifest["rid"])
        for manifest in manifests
        if isinstance(manifest, dict) and manifest.get("rid") is not None
    ]


def _same_atomic_rids(left: Sequence[str], right: Sequence[str]) -> bool:
    """Match a complete RID batch without imposing manifest response order."""
    return (
        len(left) == len(right)
        and len(set(left)) == len(left)
        and len(set(right)) == len(right)
        and set(left) == set(right)
    )


def _migration_response_complete(response: Any) -> bool:
    responses = response if isinstance(response, list) else [response]
    if not responses:
        return False
    for item in responses:
        if not isinstance(item, dict) or item.get("success", True) is False:
            return False
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        failed = int(status.get("failed_reqs") or 0)
        pending = int(status.get("pending_reqs") or 0)
        state = str(status.get("state") or item.get("state") or "").lower()
        if failed > 0 or pending > 0 or state.endswith("_failed"):
            return False
    return True


def _unstarted_pending_source_rids(
    response: Any, session_id: str, batch_rids: Sequence[str]
) -> Tuple[str, ...]:
    """Return selected RIDs that are pending before any source send began.

    A Decode request can finish naturally while the Mooncake receiver is still
    bootstrapping.  The source session then remains pending forever because
    there is no request left to drive the first send.  Require complete
    request-level timing evidence before classifying that narrow state; a slow
    transfer that has actually started must continue to use the normal timeout.
    """

    requested = tuple(str(rid) for rid in batch_rids)
    if not requested:
        return ()
    responses = response if isinstance(response, list) else [response]
    measurements: Dict[str, JsonDict] = {}
    pending = transferred = released = failed = 0
    saw_matching_session = False
    for item in responses:
        if not isinstance(item, dict) or item.get("success", True) is False:
            return ()
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        status_session = status.get("session_id") or item.get("session_id")
        if status_session not in (None, session_id):
            return ()
        if status_session == session_id:
            saw_matching_session = True
        pending += int(status.get("pending_reqs") or 0)
        transferred += int(status.get("transferred_reqs") or 0)
        released += int(status.get("released_reqs") or 0)
        failed += int(status.get("failed_reqs") or 0)
        rows = status.get("request_measurements")
        if not isinstance(rows, list):
            return ()
        for row in rows:
            if not isinstance(row, dict) or row.get("request_id") is None:
                continue
            measurements[str(row["request_id"])] = row
    if (
        not saw_matching_session
        or pending < len(requested)
        or transferred
        or released
        or failed
        or any(rid not in measurements for rid in requested)
    ):
        return ()
    for rid in requested:
        events = measurements[rid].get("phase_events")
        if not isinstance(events, list):
            return ()
        phases = {
            str(event.get("phase"))
            for event in events
            if isinstance(event, dict) and event.get("phase") is not None
        }
        if phases.intersection(
            {
                "source_transfer_started",
                "source_send",
                "source_sent",
                "source_transfer_completed",
                "source_transferred",
            }
        ):
            return ()
    return requested


def _rids_absent_from_runtime_queues(
    response: Any, rids: Sequence[str]
) -> bool:
    """Prove that RIDs are absent from every observable source request queue."""

    requested = {str(rid) for rid in rids}
    if not requested:
        return False
    responses = response if isinstance(response, list) else [response]
    if not responses:
        return False
    for item in responses:
        if not isinstance(item, dict) or item.get("success", True) is False:
            return False
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        if not isinstance(status, dict):
            return False
        for field in (
            "running_requests",
            "waiting_requests",
            "decode_bootstrap_requests",
        ):
            queue = status.get(field)
            if not isinstance(queue, list):
                return False
            for request_status in queue:
                if (
                    isinstance(request_status, dict)
                    and request_status.get("rid") is not None
                    and str(request_status["rid"]) in requested
                ):
                    return False
    return True


def _migration_response_failed(response: Any) -> bool:
    responses = response if isinstance(response, list) else [response]
    for item in responses:
        if not isinstance(item, dict) or item.get("success", True) is False:
            return True
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        failed = int(status.get("failed_reqs") or 0)
        state = str(status.get("state") or item.get("state") or "").lower()
        if failed > 0 or state.endswith("_failed"):
            return True
    return False


def _migration_response_error(response: Any) -> str:
    responses = response if isinstance(response, list) else [response]
    errors = []
    for item in responses:
        if not isinstance(item, dict):
            errors.append("non-object response")
            continue
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        failed = int(status.get("failed_reqs") or 0)
        state = str(status.get("state") or item.get("state") or "").lower()
        if item.get("success", True) is False or failed > 0 or state.endswith("_failed"):
            rank = item.get("dp_rank", status.get("dp_rank", "?"))
            detail = str(
                status.get("last_error")
                or item.get("message")
                or status.get("state")
                or "migration failed"
            )
            errors.append(f"dp_rank {rank}: {detail}")
    return "; ".join(errors) or "migration failed"


def _migration_response_failed_only_by_abort_req(response: Any) -> bool:
    """Return true only when every reported migration failure is AbortReq.

    A selected Decode request can naturally finish while the target receiver is
    still bootstrapping.  Mooncake then reports the source sender as aborted by
    ``AbortReq``.  The controller may retry another request only after the RID
    is separately proven absent from all source queues; mixed or unrelated
    failures must remain fatal.
    """

    responses = response if isinstance(response, list) else [response]
    saw_failure = False
    for item in responses:
        if not isinstance(item, dict):
            return False
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        failed = int(status.get("failed_reqs") or 0)
        state = str(status.get("state") or item.get("state") or "").lower()
        item_failed = (
            item.get("success", True) is False
            or failed > 0
            or state.endswith("_failed")
        )
        if not item_failed:
            continue
        saw_failure = True
        detail = str(
            status.get("last_error")
            or item.get("message")
            or status.get("state")
            or ""
        ).lower()
        if "aborted by abortreq" not in detail:
            return False
    return saw_failure


def _migration_fallback_request(response: Any) -> Tuple[List[str], str, Optional[str]]:
    responses = response if isinstance(response, list) else [response]
    rids = []
    seen = set()
    reason = ""
    session_id = None
    for item in responses:
        if not isinstance(item, dict):
            continue
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        session_id = session_id or status.get("session_id") or item.get("session_id")
        reason = reason or str(status.get("fallback_reason") or "")
        requested = status.get("fallback_required_rids") or []
        if not isinstance(requested, list):
            raise RuntimeError("fallback_required_rids must be a list")
        for rid in requested:
            rid = str(rid).strip()
            if rid and rid not in seen:
                seen.add(rid)
                rids.append(rid)
    return rids, reason, str(session_id) if session_id is not None else None


def _parse_runtime_status(item: JsonDict) -> Tuple[str, bool, bool]:
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    role = _normalize_role(
        item.get("role") or status.get("role") or status.get("current_role")
    )
    is_idle = bool(status.get("is_idle") or status.get("is_idle_for_flip"))
    admission_paused = bool(
        status.get("admission_paused") or status.get("pd_runtime_admission_paused")
    )
    return role, is_idle, admission_paused


def _aggregate_dp_runtime_status(items: List[JsonDict], node_name: str) -> JsonDict:
    """Build node-level policy metrics while retaining rank-level status separately."""

    if not items:
        raise RuntimeError(f"runtime status is empty for {node_name}")
    inner = [
        item.get("status") if isinstance(item.get("status"), dict) else item
        for item in items
    ]
    roles = {
        _normalize_role(status.get("role") or status.get("current_role"))
        for status in inner
    }
    if len(roles) != 1:
        raise RuntimeError(f"mixed DP runtime roles for {node_name}: {sorted(roles)}")
    aggregate = dict(items[0])
    status = dict(inner[0])
    status.update(
        {
            "role": next(iter(roles)),
            "is_idle": all(bool(value.get("is_idle")) for value in inner),
            "admission_paused": any(
                bool(
                    value.get("admission_paused")
                    or value.get("pd_runtime_admission_paused")
                    or value.get("pd_flip_admission_paused")
                )
                for value in inner
            ),
            "free_request_slots": sum(
                int(value.get("free_request_slots", 0) or 0) for value in inner
            ),
            "available_kv_tokens": sum(
                int(value.get("available_kv_tokens", 0) or 0) for value in inner
            ),
            "running_requests": [
                request
                for value in inner
                for request in value.get("running_requests", [])
                if isinstance(request, dict)
            ],
            "waiting_requests": [
                request
                for value in inner
                for request in value.get("waiting_requests", [])
                if isinstance(request, dict)
            ],
            "decode_bootstrap_requests": [
                request
                for value in inner
                for request in value.get("decode_bootstrap_requests", [])
                if isinstance(request, dict)
            ],
        }
    )
    aggregate["status"] = status
    aggregate["success"] = all(item.get("success", True) is True for item in items)
    return aggregate


def _order_manifests_by_requested_rids(
    manifests: List[JsonDict], requested_rids: Sequence[str]
) -> List[JsonDict]:
    """Restore controller selection order after asynchronous DP fan-out."""
    by_rid = {str(manifest.get("rid")): manifest for manifest in manifests}
    requested = [str(rid) for rid in requested_rids]
    requested_set = set(requested)
    return [by_rid[rid] for rid in requested if rid in by_rid] + [
        manifest
        for manifest in manifests
        if str(manifest.get("rid")) not in requested_set
    ]


def _index_dp_responses(body: Any) -> Dict[int, JsonDict]:
    """Index a fan-out HTTP response without silently dropping a DP rank."""

    items = body if isinstance(body, list) else [body]
    if not items:
        raise RuntimeError("DP response is empty")
    indexed: Dict[int, JsonDict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("DP response item is not an object")
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        rank = item.get("dp_rank")
        if rank is None:
            rank = status.get("dp_rank")
        if rank is None:
            if len(items) == 1:
                rank = 0
            else:
                raise RuntimeError("missing dp_rank in multi-response status")
        try:
            rank = int(rank)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid dp_rank: {rank!r}") from exc
        if rank in indexed:
            raise RuntimeError(f"duplicate dp_rank: {rank}")
        indexed[rank] = item
    return indexed


def _request_owner_map(responses: Any, field: str) -> Dict[str, int]:
    """Return the unique DP owner reported for each request ID."""

    owners: Dict[str, int] = {}
    for rank, item in _index_dp_responses(responses).items():
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        values = item.get(field, status.get(field, []))
        if not isinstance(values, list):
            raise RuntimeError(f"{field} for dp_rank {rank} is not a list")
        for value in values:
            rid = str(value).strip()
            if not rid:
                raise RuntimeError(f"{field} for dp_rank {rank} contains an empty RID")
            if rid in owners:
                raise RuntimeError(
                    f"request {rid} has multiple owners: "
                    f"dp_rank {owners[rid]} and dp_rank {rank}"
                )
            owners[rid] = rank
    return owners


def select_target_dp_rank(
    statuses: Any, required_pages: int, required_request_slots: int = 1
) -> int:
    """Choose one eligible decode rank deterministically by free KV pages."""

    required_pages = int(required_pages)
    required_request_slots = int(required_request_slots)
    if required_pages < 0:
        raise ValueError("required_pages must be non-negative")
    if required_request_slots < 1:
        raise ValueError("required_request_slots must be positive")
    indexed = _index_dp_responses(statuses)
    candidates: List[Tuple[int, int]] = []
    rejected: List[str] = []
    for rank, item in indexed.items():
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        role = _normalize_role(status.get("role") or status.get("current_role"))
        paused = bool(
            status.get("admission_paused")
            or status.get("pd_runtime_admission_paused")
            or status.get("pd_flip_admission_paused")
        )
        free_slots = int(status.get("free_request_slots", 1) or 0)
        if status.get("free_kv_pages") is not None:
            free_pages = int(status["free_kv_pages"])
        else:
            page_size = max(1, int(status.get("page_size", 1) or 1))
            free_pages = int(status.get("available_kv_tokens", 0) or 0) // page_size
        if role not in {"decode", "unknown"}:
            rejected.append(f"rank {rank}: role={role}")
        elif paused:
            rejected.append(f"rank {rank}: admission paused")
        elif free_slots < required_request_slots:
            rejected.append(
                f"rank {rank}: free_request_slots={free_slots} "
                f"< {required_request_slots}"
            )
        elif free_pages < required_pages:
            rejected.append(
                f"rank {rank}: free_kv_pages={free_pages} < {required_pages}"
            )
        else:
            candidates.append((free_pages, rank))
    if not candidates:
        detail = "; ".join(rejected) or "no status candidates"
        raise RuntimeError(
            f"no decode DP rank has capacity for {required_pages} pages: {detail}"
        )
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return candidates[0][1]


def _required_kv_pages_by_rid(
    statuses: Any, rids: Sequence[str]
) -> Dict[str, int]:
    """Estimate each target allocation from source committed KV plus reserve."""

    requested = {str(rid): 0 for rid in rids}
    if not requested:
        return {}
    committed_by_rid: Dict[str, Tuple[int, int, int]] = {}
    for rank, item in _index_dp_responses(statuses).items():
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        page_size = max(1, int(status.get("page_size", 1) or 1))
        reserve = max(0, int(status.get("reserved_decode_tokens_per_req", 0) or 0))
        running = status.get("running_requests", [])
        if not isinstance(running, list):
            raise RuntimeError(f"running_requests for dp_rank {rank} is not a list")
        for request_status in running:
            if not isinstance(request_status, dict) or request_status.get("rid") is None:
                continue
            rid = str(request_status["rid"])
            if rid not in requested:
                continue
            if rid in committed_by_rid:
                raise RuntimeError(f"request {rid} appears on multiple source DP ranks")
            committed_by_rid[rid] = (
                max(0, int(request_status.get("kv_committed_len", 0) or 0)),
                reserve,
                page_size,
            )
    for rid, (committed, reserve, page_size) in committed_by_rid.items():
        requested[rid] = (committed + reserve + page_size - 1) // page_size
    return requested


def _required_kv_pages(statuses: Any, rids: Sequence[str]) -> int:
    return sum(_required_kv_pages_by_rid(statuses, rids).values())


def _assign_target_dp_ranks(
    statuses: Any, required_pages_by_rid: Dict[str, int]
) -> Dict[str, int]:
    """Greedily place requests while accounting for each rank's remaining capacity."""

    capacities: List[JsonDict] = []
    for rank, item in _index_dp_responses(statuses).items():
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        page_size = max(1, int(status.get("page_size", 1) or 1))
        free_pages = (
            int(status["free_kv_pages"])
            if status.get("free_kv_pages") is not None
            else int(status.get("available_kv_tokens", 0) or 0) // page_size
        )
        capacities.append(
            {
                **status,
                "dp_rank": rank,
                "free_kv_pages": free_pages,
                "free_request_slots": int(
                    status.get("free_request_slots", 1) or 0
                ),
            }
        )

    assignments: Dict[str, int] = {}
    for rid, required_pages in required_pages_by_rid.items():
        rank = select_target_dp_rank(capacities, int(required_pages))
        assignments[str(rid)] = rank
        selected = next(item for item in capacities if item["dp_rank"] == rank)
        selected["free_kv_pages"] -= int(required_pages)
        selected["free_request_slots"] -= 1
    return assignments


def _require_request_owners(response: Any, expected_rids: Sequence[str], step: str) -> None:
    """For fan-out responses, require exactly one handling rank for every RID."""

    indexed = _index_dp_responses(response)
    if len(indexed) <= 1:
        return
    owners = _request_owner_map(list(indexed.values()), "handled_rids")
    expected = {str(rid) for rid in expected_rids}
    actual = set(owners)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"{step} DP ownership barrier failed: "
            f"missing_rids={missing}, unexpected_rids={unexpected}"
        )


def _require_worker_dp_ranks(response: Any, expected_statuses: Any, step: str) -> None:
    """Require a fan-out barrier response from every configured worker rank."""

    expected = set(_index_dp_responses(expected_statuses))
    if len(expected) <= 1:
        return
    actual = set(_index_dp_responses(response))
    if actual != expected:
        raise RuntimeError(
            f"{step} DP rank barrier failed: "
            f"missing_dp_ranks={sorted(expected - actual)}, "
            f"unexpected_dp_ranks={sorted(actual - expected)}"
        )


def _parse_loads(body: Any) -> Tuple[int, int, int, Optional[float], List[JsonDict]]:
    if isinstance(body, dict):
        loads = body.get("loads", [])
    elif isinstance(body, list):
        loads = body
    else:
        loads = []
    raw_loads = [item for item in loads if isinstance(item, dict)]
    running_reqs = sum(int(item.get("num_running_reqs") or 0) for item in raw_loads)
    waiting_reqs = sum(int(item.get("num_waiting_reqs") or 0) for item in raw_loads)
    total_tokens = sum(int(item.get("num_total_tokens") or 0) for item in raw_loads)
    usages = [
        float(item["token_usage"])
        for item in raw_loads
        if item.get("token_usage") is not None
    ]
    token_usage = max(usages) if usages else None
    return running_reqs, waiting_reqs, total_tokens, token_usage, raw_loads


def _sum_load_metric(raw_loads: List[JsonDict], field: str) -> int:
    total = 0
    for item in raw_loads:
        value = item.get(field)
        if value is None and isinstance(item.get("disaggregation"), dict):
            value = item["disaggregation"].get(field)
        if value is None:
            continue
        total += int(value)
    return total


def _load_sort_key(metric: NodeMetrics) -> Tuple[int, int, int, float, str]:
    return (
        metric.running_reqs,
        metric.router_active_load,
        metric.total_tokens,
        metric.token_usage or 0.0,
        metric.name,
    )


def _find_metric(
    metrics: List[NodeMetrics], name_or_worker_id: str
) -> Optional[NodeMetrics]:
    for metric in metrics:
        if (
            metric.name == name_or_worker_id
            or metric.router_worker_id == name_or_worker_id
        ):
            return metric
    return None


def _monitor_state_record(
    *,
    state: str,
    direction: Optional[str] = None,
    source: Optional[str] = None,
    migration_target: Optional[str] = None,
    role_before: Optional[str] = None,
    role_after: Optional[str] = None,
    reason: str = "",
    snapshot_index: Optional[int] = None,
    action_index: Optional[int] = None,
) -> JsonDict:
    entered_monotonic = time.monotonic()
    record: JsonDict = {
        "state": state,
        "direction": direction,
        "source": source,
        "migration_target": migration_target,
        "role_before": role_before,
        "role_after": role_after,
        "reason": reason,
        "entered_wall": time.time(),
        "entered_monotonic": entered_monotonic,
    }
    if snapshot_index is not None:
        record["snapshot_index"] = snapshot_index
    if action_index is not None:
        record["action_index"] = action_index
    return record


def _prefill_risk(snapshot: ClusterSLOSnapshot, threshold: float) -> bool:
    attainment = snapshot.prefill_slo_attainment
    return attainment is not None and attainment < threshold


def _decode_risk(snapshot: ClusterSLOSnapshot, threshold: float) -> bool:
    attainment = snapshot.decode_slo_attainment
    return attainment is not None and attainment < threshold


def _prefill_recovered(snapshot: ClusterSLOSnapshot, threshold: float) -> bool:
    attainment = snapshot.prefill_slo_attainment
    return attainment is not None and attainment >= threshold


def load_config(path: str) -> PDClusterConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    journal_path = Path(str(data.get("session_journal_path", "pd_flip_session.json")))
    if not journal_path.is_absolute():
        data["session_journal_path"] = str(config_path.parent / journal_path)
    return PDClusterConfig.from_dict(data)


def _parse_node_spec(value: str) -> PDNode:
    parts = {}
    for item in value.split(","):
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(
                f"invalid --node entry {value!r}; expected key=value pairs"
            )
        parts[key.strip()] = val.strip()
    name = parts["name"]
    return PDNode(
        name=name,
        worker_url=parts["worker_url"],
        router_worker_id=parts.get("router_worker_id", name),
        bootstrap_port=(
            int(parts["bootstrap_port"]) if parts.get("bootstrap_port") else None
        ),
    )


def config_from_args(args: argparse.Namespace) -> PDClusterConfig:
    if args.config:
        return load_config(args.config)
    if not args.router_url:
        raise ValueError("--router-url is required when --config is not provided")
    if not args.node:
        raise ValueError(
            "at least one --node is required when --config is not provided"
        )
    return PDClusterConfig(
        router_url=args.router_url,
        nodes=[_parse_node_spec(value) for value in args.node],
        request_timeout_seconds=args.timeout_seconds,
        # The CLI runner exposes one bounded controller timeout.  Keep the
        # migration loops on that same explicit budget; otherwise they fall
        # back to PDClusterConfig's 120-second default even when the runner
        # records and passes a larger experiment timeout.
        migration_timeout_seconds=args.timeout_seconds,
        observation_quiesce_seconds=float(
            os.environ.get("PD_FLIP_OBSERVE_QUIESCE_SECONDS", 0.0)
        ),
        post_migration_idle_timeout_seconds=float(
            os.environ.get("PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS", 2.0)
        ),
        first_migration_ratio=args.first_migration_ratio,
        observation_seconds=args.observation_seconds,
        slo_threshold=args.slo_threshold,
        slo_recovery_threshold=args.slo_recovery_threshold,
        slo_attainment_gap_threshold=args.slo_attainment_gap_threshold,
        slo_attainment_gap_recovery_threshold=(
            args.slo_attainment_gap_recovery_threshold
        ),
        force_second_migration_after_observation=(
            args.force_second_migration_after_observation
        ),
        min_prefill_slo_samples=args.min_prefill_slo_samples,
        min_decode_slo_samples=args.min_decode_slo_samples,
        session_journal_path=args.session_journal_path,
        session_id_prefix=getattr(args, "session_id_prefix", None),
        prefill_donor_mode=bool(getattr(args, "prefill_donor_mode", False)),
        p_to_d_min_handoff_requests=args.p_to_d_min_handoff_requests,
        decision_policy=args.decision_policy,
        decode_first_gap_threshold=args.decode_first_gap_threshold,
        decode_first_prefill_protect=(
            not args.disable_decode_first_prefill_protect
        ),
        decode_first_d_to_p_require_prefill_gap=(
            not args.disable_decode_first_d_to_p_prefill_gap
        ),
        decode_first_bs_estimator=args.decode_first_bs_estimator,
        decode_first_fixed_batch_size=args.decode_first_fixed_batch_size,
        decode_first_window_target_violation_rate=(
            args.decode_first_window_target_violation_rate
        ),
        decode_first_window_min_samples=args.decode_first_window_min_samples,
        slo_target_gap_threshold=args.slo_target_gap_threshold,
        tpot_capacity_intercept_ms=args.tpot_capacity_intercept_ms,
        tpot_capacity_batch_slope_ms=args.tpot_capacity_batch_slope_ms,
        queue_window_requests=args.queue_window_requests,
        queue_threshold_seconds=args.queue_threshold_ms / 1000.0,
        queue_overload_ratio=args.queue_overload_ratio,
        queue_scale_in_ratio=args.queue_scale_in_ratio,
        prefill_scale_in_headroom_workers=(
            args.prefill_scale_in_headroom_workers
        ),
        prefill_min_role_seconds=args.prefill_min_role_seconds,
        d_to_p_direct_full_drain=bool(args.d_to_p_direct_full_drain),
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PD runtime-role flip controller for SGLang clusters"
    )
    parser.add_argument("--config", help="JSON file with router_url and nodes")
    parser.add_argument("--router-url", help="Router base URL")
    parser.add_argument(
        "--node",
        action="append",
        help=(
            "Node spec: name=node-a,worker_url=http://host:30000,"
            "router_worker_id=node-a,bootstrap_port=8997"
        ),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Read the API key from this environment variable instead of argv.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--first-migration-ratio", type=float, default=0.5)
    parser.add_argument("--observation-seconds", type=float, default=10.0)
    parser.add_argument("--slo-threshold", type=float, default=0.9)
    parser.add_argument("--slo-recovery-threshold", type=float, default=0.95)
    parser.add_argument("--slo-attainment-gap-threshold", type=float, default=None)
    parser.add_argument(
        "--slo-attainment-gap-recovery-threshold", type=float, default=None
    )
    parser.add_argument(
        "--force-second-migration-after-observation",
        action="store_true",
        help="Always migrate the remaining source requests after observation.",
    )
    parser.add_argument("--min-prefill-slo-samples", type=int, default=20)
    parser.add_argument("--min-decode-slo-samples", type=int, default=20)
    parser.add_argument("--session-journal-path", default="pd_flip_session.json")
    parser.add_argument("--session-id-prefix", default=None)
    parser.add_argument("--prefill-donor-mode", action="store_true")
    parser.add_argument("--p-to-d-min-handoff-requests", type=int, default=1)
    parser.add_argument(
        "--decision-policy",
        choices=[
            "slo_gap",
            "prefill_queue_util",
            "decode_first",
            "slo_target",
            "tpot_capacity",
        ],
        default="slo_gap",
        help=(
            "Automatic direction policy; decode_first estimates required "
            "Decode instances from inflight requests and TPOT SLO; "
            "tpot_capacity uses only that capacity estimate."
        ),
    )
    parser.add_argument("--decode-first-gap-threshold", type=float, default=0.10)
    parser.add_argument(
        "--decode-first-bs-estimator",
        choices=[
            "fitted_formula",
            "window_p20_nonattainment",
            "fixed_batch_size",
        ],
        default="fitted_formula",
        help=(
            "Decode-first BS estimator: fitted_formula uses "
            "TPOT(ms)=6.8165+0.40830*BS; window_p20_nonattainment uses the "
            "previous decision window's request-level BS bucket nearest 20%% "
            "TPOT non-attainment; fixed_batch_size uses "
            "--decode-first-fixed-batch-size."
        ),
    )
    parser.add_argument(
        "--decode-first-fixed-batch-size", type=float, default=10.0
    )
    parser.add_argument(
        "--decode-first-window-target-violation-rate",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--decode-first-window-min-samples", type=int, default=20
    )
    parser.add_argument(
        "--disable-decode-first-prefill-protect",
        action="store_true",
        help=(
            "When current Decode capacity is insufficient, allow P-to-D "
            "without applying the Prefill-minus-Decode violation-gap guard."
        ),
    )
    parser.add_argument(
        "--disable-decode-first-d-to-p-prefill-gap",
        action="store_true",
        help=(
            "Allow D-to-P whenever Decode remains sufficient after removing "
            "one Decode worker, without requiring a Prefill-minus-Decode "
            "violation gap above --decode-first-gap-threshold."
        ),
    )
    parser.add_argument("--slo-target-gap-threshold", type=float, default=0.20)
    parser.add_argument("--tpot-capacity-intercept-ms", type=float, default=8.0)
    parser.add_argument("--tpot-capacity-batch-slope-ms", type=float, default=0.4)
    parser.add_argument("--queue-window-requests", type=int, default=50)
    parser.add_argument("--queue-threshold-ms", type=float, default=10.0)
    parser.add_argument("--queue-overload-ratio", type=float, default=0.10)
    parser.add_argument("--queue-scale-in-ratio", type=float, default=0.05)
    parser.add_argument(
        "--prefill-scale-in-headroom-workers", type=float, default=1.5
    )
    parser.add_argument("--prefill-min-role-seconds", type=float, default=30.0)
    parser.add_argument(
        "--d-to-p-direct-full-drain",
        action="store_true",
        help=(
            "After Router/bootstrap quiescence, atomically migrate all source "
            "running and waiting requests without a first batch or SLO observation."
        ),
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("metrics", help="Collect router/worker metrics")

    dry_run = subparsers.add_parser("dry-run", help="Build a flip plan without POSTs")
    dry_run.add_argument("--direction", choices=["d_to_p", "p_to_d"], required=True)
    dry_run.add_argument("--source-name", default=None)
    dry_run.add_argument("--migration-target-name", default=None)

    execute = subparsers.add_parser("execute", help="Execute a PD role flip")
    execute.add_argument("--direction", choices=["d_to_p", "p_to_d"], required=True)
    execute.add_argument("--source-name", default=None)
    execute.add_argument("--migration-target-name", default=None)

    execute_two_phase = subparsers.add_parser(
        "execute-two-phase",
        help="Force the monitor-style two-phase D->P path with prepare_only/commit.",
    )
    execute_two_phase.add_argument("--direction", choices=["d_to_p"], default="d_to_p")
    execute_two_phase.add_argument("--source-name", default=None)
    execute_two_phase.add_argument("--migration-target-name", default=None)

    monitor = subparsers.add_parser("monitor", help="Run monitor-driven PD flip loop")
    monitor.add_argument("--ttft-slo", type=float, required=True)
    monitor.add_argument("--tpot-slo", type=float, required=True)
    monitor.add_argument("--window-seconds", type=float, default=30.0)
    monitor.add_argument("--enter-threshold", type=float, default=0.9)
    monitor.add_argument("--exit-threshold", type=float, default=0.95)
    monitor.add_argument("--commit-threshold", type=float, default=0.9)
    monitor.add_argument("--iterations", type=int, default=1)
    monitor.add_argument("--poll-interval", type=float, default=1.0)
    monitor.add_argument(
        "--trace-slo-ledger",
        default=None,
        help="Use request-level trace SLO JSONL ledger instead of Prometheus histograms.",
    )
    monitor.add_argument(
        "--forced-direction",
        choices=["d_to_p", "p_to_d"],
        default=None,
        help="Ignore risk in the opposite direction while waiting for a flip trigger.",
    )
    monitor.add_argument(
        "--forced-action-not-before-seconds",
        type=float,
        default=0.0,
        help=(
            "For forced P->D chain validation, wait at least this long and "
            "then act only when a bootstrap-queued request is eligible."
        ),
    )
    monitor.add_argument("--source-name", default=None)
    monitor.add_argument("--migration-target-name", default=None)

    continuous = subparsers.add_parser(
        "monitor-continuous",
        help="Continuously re-evaluate SLOs and commit multiple automatic flips.",
    )
    continuous.add_argument("--ttft-slo", type=float, required=True)
    continuous.add_argument("--tpot-slo", type=float, required=True)
    continuous.add_argument("--window-seconds", type=float, default=30.0)
    continuous.add_argument("--enter-threshold", type=float, default=0.9)
    continuous.add_argument("--exit-threshold", type=float, default=0.95)
    continuous.add_argument("--commit-threshold", type=float, default=0.9)
    continuous.add_argument("--iterations", type=int, default=1)
    continuous.add_argument("--poll-interval", type=float, default=1.0)
    continuous.add_argument("--cooldown-seconds", type=float, default=30.0)
    continuous.add_argument(
        "--max-flips",
        type=int,
        default=0,
        help="Stop after this many flips; zero runs until the iteration budget ends.",
    )
    continuous.add_argument("--min-prefill-workers", type=int, default=1)
    continuous.add_argument("--min-decode-workers", type=int, default=1)
    continuous.add_argument("--max-prefill-workers", type=int, default=None)
    continuous.add_argument("--max-decode-workers", type=int, default=None)
    continuous.add_argument(
        "--expected-terminal-requests",
        type=int,
        default=0,
        help=(
            "Exit successfully after this many request IDs have terminal ledger "
            "rows; zero disables workload-aware termination."
        ),
    )
    continuous.add_argument(
        "--trace-slo-ledger",
        default=None,
        help="Use request-level trace SLO JSONL ledger instead of Prometheus histograms.",
    )

    scheduled = subparsers.add_parser(
        "monitor-scheduled",
        help="Apply sequential topology targets at trace-relative times.",
    )
    scheduled.add_argument("--trace-slo-ledger", required=True)
    scheduled.add_argument("--schedule-json", required=True)
    scheduled.add_argument("--poll-interval", type=float, default=0.25)
    scheduled.add_argument("--start-timeout-seconds", type=float, default=600.0)
    scheduled.add_argument("--event-timeout-seconds", type=float, default=600.0)
    scheduled.add_argument("--window-seconds", type=float, default=10.0)

    progressive = subparsers.add_parser(
        "monitor-progressive",
        help="Run request-level SLO progressive D-to-P migration and observation.",
    )
    progressive.add_argument("--trace-slo-ledger", required=True)
    progressive.add_argument("--source-name", default=None)
    progressive.add_argument("--migration-target-name", default=None)
    progressive.add_argument("--iterations", type=int, default=1)
    progressive.add_argument("--poll-interval", type=float, default=1.0)
    progressive.add_argument("--window-seconds", type=float, default=10.0)
    return parser


def resolve_api_key(args: argparse.Namespace) -> Optional[str]:
    if args.api_key and args.api_key_env:
        raise ValueError("use only one of --api-key and --api-key-env")
    if not args.api_key_env:
        return args.api_key
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"API key environment variable is empty or missing: {args.api_key_env}"
        )
    return api_key


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        config = config_from_args(args)
        client = HttpClient(
            api_key=resolve_api_key(args), timeout_seconds=args.timeout_seconds
        )
        controller = PDFlipController(config, client)
        if args.command == "metrics":
            output = controller.collect_metrics()
        elif args.command == "dry-run":
            output = controller.dry_run(
                direction=args.direction,
                source_name=args.source_name,
                migration_target_name=args.migration_target_name,
            )
        elif args.command == "execute":
            output = controller.execute(
                direction=args.direction,
                source_name=args.source_name,
                migration_target_name=args.migration_target_name,
            )
        elif args.command == "execute-two-phase":
            output = controller.execute_two_phase(
                direction=args.direction,
                source_name=args.source_name,
                migration_target_name=args.migration_target_name,
            )
        elif args.command == "monitor":
            if args.trace_slo_ledger:
                slo_monitor = TraceSLOMonitor(
                    ledger_path=args.trace_slo_ledger,
                    window_seconds=args.window_seconds,
                    client=client,
                )
            else:
                slo_monitor = PDFlipSLOMonitor(
                    ttft_slo_seconds=args.ttft_slo,
                    tpot_slo_seconds=args.tpot_slo,
                    window_seconds=args.window_seconds,
                    client=client,
                )
            output = controller.monitor(
                slo_monitor=slo_monitor,
                enter_threshold=args.enter_threshold,
                exit_threshold=args.exit_threshold,
                commit_threshold=args.commit_threshold,
                iterations=args.iterations,
                poll_interval_seconds=args.poll_interval,
                forced_direction=args.forced_direction,
                forced_action_not_before_seconds=(
                    args.forced_action_not_before_seconds
                ),
                source_name=args.source_name,
                migration_target_name=args.migration_target_name,
            )
        elif args.command == "monitor-progressive":
            slo_monitor = TraceSLOMonitor(
                ledger_path=args.trace_slo_ledger,
                window_seconds=args.window_seconds,
                client=client,
            )
            output = controller.monitor_progressive(
                slo_monitor=slo_monitor,
                iterations=args.iterations,
                poll_interval_seconds=args.poll_interval,
                source_name=args.source_name,
                migration_target_name=args.migration_target_name,
            )
        elif args.command == "monitor-continuous":
            if args.trace_slo_ledger:
                slo_monitor = TraceSLOMonitor(
                    ledger_path=args.trace_slo_ledger,
                    window_seconds=args.window_seconds,
                    client=client,
                )
            else:
                slo_monitor = PDFlipSLOMonitor(
                    ttft_slo_seconds=args.ttft_slo,
                    tpot_slo_seconds=args.tpot_slo,
                    window_seconds=args.window_seconds,
                    client=client,
                )
            output = controller.monitor_continuous(
                slo_monitor=slo_monitor,
                enter_threshold=args.enter_threshold,
                exit_threshold=args.exit_threshold,
                commit_threshold=args.commit_threshold,
                iterations=args.iterations,
                poll_interval_seconds=args.poll_interval,
                cooldown_seconds=args.cooldown_seconds,
                max_flips=args.max_flips,
                min_prefill_workers=args.min_prefill_workers,
                min_decode_workers=args.min_decode_workers,
                max_prefill_workers=args.max_prefill_workers,
                max_decode_workers=args.max_decode_workers,
                expected_terminal_requests=args.expected_terminal_requests,
            )
        elif args.command == "monitor-scheduled":
            slo_monitor = TraceSLOMonitor(
                ledger_path=args.trace_slo_ledger,
                window_seconds=args.window_seconds,
                client=client,
            )
            schedule = parse_topology_schedule(
                args.schedule_json, len(config.nodes)
            )
            output = controller.monitor_scheduled(
                slo_monitor=slo_monitor,
                ledger_path=args.trace_slo_ledger,
                schedule=schedule,
                poll_interval_seconds=args.poll_interval,
                start_timeout_seconds=args.start_timeout_seconds,
                event_timeout_seconds=args.event_timeout_seconds,
            )
        else:
            parser.error(f"unknown command {args.command}")
        print(json.dumps(output, default=_json_default, indent=2, sort_keys=True))
        if args.command in (
            "execute",
            "execute-two-phase",
            "monitor",
            "monitor-continuous",
            "monitor-scheduled",
            "monitor-progressive",
        ) and hasattr(
            output, "success"
        ):
            return 0 if output.success else 1
        return 0
    except Exception as exc:
        print(f"pd_flip_controller: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
