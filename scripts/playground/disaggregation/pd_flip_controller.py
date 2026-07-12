#!/usr/bin/env python3
"""Controller skeleton for four-node PD runtime role flip experiments.

The controller keeps orchestration policy outside SGLang workers:

1. collect router + worker metrics,
2. build an explicit D->P / P->D flip plan,
3. dry-run the HTTP actions before later tasks enable execution.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request
from urllib.parse import urljoin

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


def _migration_source_start_payload(
    session_id: str,
    target_url: str,
    rids: Optional[List[str]],
    include_waiting: bool = False,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "target_url": target_url,
        "rids": None if rids is None else list(rids),
        "include_waiting": include_waiting,
    }


JsonDict = Dict[str, Any]
SOURCE_DELTA_QUIESCE_PENDING_MESSAGE = (
    "source batch quiesce pending; retry delta after quiesce"
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
    min_prefill_slo_samples: int = 20
    min_decode_slo_samples: int = 20
    session_journal_path: str = "pd_flip_session.json"

    def __post_init__(self) -> None:
        if not 0 < self.first_migration_ratio < 1:
            raise ValueError(
                "first_migration_ratio must be greater than 0 and less than 1"
            )
        if not self.observation_seconds >= 0:
            raise ValueError("observation_seconds must be greater than or equal to 0")
        if not 0 <= self.slo_threshold <= 1:
            raise ValueError("slo_threshold must be between 0 and 1 inclusive")
        if self.min_prefill_slo_samples <= 0:
            raise ValueError("min_prefill_slo_samples must be greater than 0")
        if self.min_decode_slo_samples <= 0:
            raise ValueError("min_decode_slo_samples must be greater than 0")

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
            min_prefill_slo_samples=int(data.get("min_prefill_slo_samples", 20)),
            min_decode_slo_samples=int(data.get("min_decode_slo_samples", 20)),
            session_journal_path=str(
                data.get("session_journal_path", "pd_flip_session.json")
            ),
        )


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
    FLIPPING_ROLE = "flipping_role"


class ProgressiveAtomicBatchError(RuntimeError):
    def __init__(self, message: str, *, source_finished: bool):
        super().__init__(message)
        self.source_finished = source_finished


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

    def collect_metrics(self) -> List[NodeMetrics]:
        router_workers = self._fetch_router_workers()
        metrics = []
        for node in self.config.nodes:
            router_status = router_workers.get(node.router_worker_id, {})
            status_body = self.client.get_json(
                node.worker_url, "/pd_flip/runtime_role/status"
            )
            loads_body = self.client.get_json(node.worker_url, "/v1/loads?include=all")
            status = _first_successful_response(status_body)
            role, is_idle, admission_paused = _parse_runtime_status(status)
            running_reqs, waiting_reqs, total_tokens, token_usage, raw_loads = (
                _parse_loads(loads_body)
            )
            metrics.append(
                NodeMetrics(
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
                    raw_loads=raw_loads,
                )
            )
        return metrics

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
            source = self._select_source(
                metrics,
                source_name=source_name,
                expected_role="prefill",
                prefer_high_load=False,
            )
            migration_target = None
            target_role = "decode"
            actions = self._build_p_to_d_actions(source)
            reason = f"move prefill node {source.name} to decode after it becomes idle"
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
                source = self._select_source(
                    metrics,
                    source_name=source_name,
                    expected_role="prefill",
                    prefer_high_load=False,
                )
                target_role = "decode"
                self._execute_p_to_d(source, records)
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
            if source is not None:
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
    ) -> MonitorLoopResult:
        snapshots: List[JsonDict] = []
        actions: List[Any] = []
        state_trace: List[JsonDict] = []
        for idx in range(max(1, iterations)):
            metrics = self.collect_metrics()
            snapshot = slo_monitor.collect_cluster(
                (m.name, m.worker_url, m.effective_role) for m in metrics
            )
            snapshots.append(snapshot.to_dict())
            if not state_trace:
                state_trace.append(
                    _monitor_state_record(
                        state=MonitorState.SAFE,
                        snapshot_index=len(snapshots) - 1,
                        reason="monitor_sampled",
                    )
                )
            if _prefill_risk(snapshot, enter_threshold):
                source = self._select_source(
                    metrics,
                    source_name=None,
                    expected_role="decode",
                    prefer_high_load=True,
                )
                target = self._select_decode_migration_target(metrics, source)
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
                return MonitorLoopResult(
                    success=result.success,
                    message=result.message,
                    iterations=idx + 1,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )
            if _decode_risk(snapshot, enter_threshold):
                result = self._execute_p_to_d_monitor(
                    metrics=metrics,
                    state_trace=state_trace,
                    snapshot_index=len(snapshots) - 1,
                )
                actions.append(result)
                return MonitorLoopResult(
                    success=result.success,
                    message=result.message,
                    iterations=idx + 1,
                    snapshots=snapshots,
                    actions=actions,
                    state_trace=state_trace,
                )
            time.sleep(poll_interval_seconds)

        return MonitorLoopResult(
            success=True,
            message="no flip decision",
            iterations=max(1, iterations),
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
            decision = self._evaluate_progressive_snapshot(snapshot, observing=False)
            if decision is ProgressiveDecision.START:
                source = self._select_source(
                    metrics,
                    source_name=None,
                    expected_role="decode",
                    prefer_high_load=True,
                )
                target = self._select_decode_migration_target(metrics, source)
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
    ) -> MonitorLoopResult:
        session_prefix = f"pd-flip-{source.name}-to-{target.name}"
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
            if selection is None:
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
            self._post_worker(
                records,
                "pause_source_admission",
                source,
                "/pd_flip/runtime_role/admission",
                {"paused": True},
            )
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.FIRST_MIGRATING,
                source,
                target,
                "first_batch_selected",
                records,
            )
            self._execute_atomic_batch(
                source,
                target,
                session_prefix + "-first",
                selection.selected_rids,
                False,
                records=records,
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
            observation = self._collect_progressive_observation(
                slo_monitor, monitor_nodes
            )
            snapshots.append(observation.to_dict())
            decision = self._evaluate_progressive_snapshot(observation, observing=True)
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
            remaining = self._source_running_requests(source, records)
            if remaining:
                source_finished = False
                self._execute_atomic_batch(
                    source,
                    target,
                    session_prefix + "-final",
                    remaining,
                    True,
                    records=records,
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
            self._flip_idle_source_to_prefill(source, records)
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SAFE,
                source,
                target,
                "role_flip_complete",
                records,
            )
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
            self._cleanup_source_after_failure(source, records)
            self._append_progressive_state(
                state_trace,
                ProgressiveMonitorState.SAFE,
                source,
                target,
                "post_finish_error" if post_finish_error else "error_recovered",
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
        records: Optional[List[ActionRecord]] = None,
    ) -> Tuple[str, ...]:
        records = records if records is not None else []
        requested_rids = tuple(str(rid) for rid in rids)
        if not requested_rids:
            raise ValueError("atomic migration batch must not have empty rids")
        source_finished = False
        try:
            source_start = self._post_worker(
                records,
                "start_decode_migration_source",
                source,
                "/pd_flip/migration/source/start",
                _migration_source_start_payload(
                    session_id,
                    target.worker_url,
                    list(requested_rids),
                    include_waiting=include_waiting,
                ),
            )
            manifests = _strict_response_manifests(
                source_start, "invalid source start response manifests"
            )
            batch_rids = tuple(_manifest_rids(manifests))
            if include_waiting:
                if batch_rids[: len(requested_rids)] != requested_rids:
                    raise RuntimeError(
                        "invalid source start response manifests: "
                        "requested running RID prefix was not preserved"
                    )
            elif batch_rids != requested_rids:
                raise RuntimeError(
                    "invalid source start response manifests: "
                    "selected first-batch RIDs do not match"
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
            self._wait_migration(records, "wait_decode_migration_source", source)
            self._wait_migration(records, "wait_decode_migration_target", target)
            delta_manifests = self._poll_source_delta_manifests(
                records, source, session_id, batch_rids
            )
            delta_rids = tuple(_manifest_rids(delta_manifests))
            if delta_rids != batch_rids:
                raise RuntimeError(
                    "source delta manifests do not match atomic batch RIDs"
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
            self._wait_migration(records, "wait_decode_migration_source_delta", source)
            self._wait_migration(records, "wait_decode_migration_target_delta", target)
            self._post_worker(
                records,
                "commit_decode_migration_target",
                target,
                "/pd_flip/migration/target/commit",
                {"session_id": session_id, "rids": list(batch_rids)},
            )
            self._post_worker(
                records,
                "finish_decode_migration_source",
                source,
                "/pd_flip/migration/source/finish",
                {"session_id": session_id, "released_rids": list(batch_rids)},
            )
            source_finished = True
            self._post_worker(
                records,
                "activate_decode_migration_target",
                target,
                "/pd_flip/migration/target/activate",
                {"session_id": session_id, "rids": list(batch_rids)},
            )
            return batch_rids
        except Exception as exc:
            if not source_finished:
                self._abort_two_phase_migration(source, target, session_id, records)
            raise ProgressiveAtomicBatchError(
                str(exc), source_finished=source_finished
            ) from exc

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
                            elapsed_seconds=time.monotonic() - started,
                        )
                    )
                else:
                    _raise_if_unsuccessful(
                        response, "start_decode_migration_source_delta"
                    )
                    manifests = _strict_response_manifests(
                        response, "invalid source delta response manifests"
                    )
                    records.append(
                        ActionRecord(
                            step="start_decode_migration_source_delta",
                            target=source.name,
                            method="POST",
                            url=url,
                            payload=payload,
                            response=response,
                            elapsed_seconds=time.monotonic() - started,
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
                        elapsed_seconds=time.monotonic() - started,
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
        )

    def _source_running_requests(
        self, source: NodeMetrics, records: List[ActionRecord]
    ) -> Tuple[str, ...]:
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
        return tuple(rids)

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
        self, source: NodeMetrics, records: List[ActionRecord]
    ) -> None:
        self._post_worker(
            records,
            "set_source_runtime_role",
            source,
            "/pd_flip/runtime_role/set",
            {"role": "prefill", "force": False},
        )
        self._wait_source_role(records, source, "prefill", "wait_source_prefill_loop")
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
            last_response = self._record_get(
                records,
                step,
                source.name,
                source.worker_url,
                "/pd_flip/runtime_role/status",
            )
            status = _first_successful_response(last_response)
            role, _, _ = _parse_runtime_status(status)
            if role == expected_role:
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
                    "prepare_only": True,
                    "adopt_on_commit": False,
                },
            )

            transfer_result = self._wait_two_phase_migration_or_recovery(
                records=records,
                source=source,
                target=target,
                slo_monitor=slo_monitor,
                monitor_nodes=monitor_nodes,
                exit_threshold=exit_threshold,
            )
            migration_seconds = time.monotonic() - migration_started
            if transfer_result == "recovered":
                self._abort_two_phase_migration(source, target, session_id, records)
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
                self._abort_two_phase_migration(source, target, session_id, records)
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
            released_rids = _manifest_rids(manifests)
            self._sync_two_phase_delta_before_commit(
                records=records,
                source=source,
                target=target,
                session_id=session_id,
                released_rids=released_rids,
            )
            migration_seconds = time.monotonic() - migration_started
            self._post_worker(
                records,
                "commit_decode_migration_target",
                target,
                "/pd_flip/migration/target/commit",
                {"session_id": session_id, "rids": released_rids},
            )
            self._post_worker(
                records,
                "finish_decode_migration_source",
                source,
                "/pd_flip/migration/source/finish",
                {"session_id": session_id, "released_rids": released_rids},
            )
            source_finished = True
            self._post_worker(
                records,
                "activate_decode_migration_target",
                target,
                "/pd_flip/migration/target/activate",
                {"session_id": session_id, "rids": released_rids},
            )
            self._assert_source_idle_after_migration(records, source)
            self._post_worker(
                records,
                "set_source_runtime_role",
                source,
                "/pd_flip/runtime_role/set",
                {"role": "prefill", "force": False},
            )
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
                self._abort_two_phase_migration(source, target, session_id, records)
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
    ) -> str:
        started = time.monotonic()
        transfer_deadline = started + self.config.migration_timeout_seconds
        observe_until = started + max(0.0, self.config.observation_quiesce_seconds)
        transfer_complete = False
        last_source_status: Any = None
        last_target_status: Any = None
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
    ) -> None:
        for node, path, payload in (
            (
                target,
                "/pd_flip/migration/target/abort",
                {"session_id": session_id, "reason": "monitor aborted preparing"},
            ),
            (
                source,
                "/pd_flip/migration/abort",
                {"session_id": session_id, "reason": "monitor aborted preparing"},
            ),
        ):
            try:
                self._post_worker(
                    records,
                    "abort_decode_migration",
                    node,
                    path,
                    payload,
                )
            except Exception:
                pass

    def _execute_p_to_d(
        self,
        source: NodeMetrics,
        records: List[ActionRecord],
    ) -> None:
        self._prepare_p_to_d(source, records)
        self._finish_p_to_d(source, records)

    def _prepare_p_to_d(
        self,
        source: NodeMetrics,
        records: List[ActionRecord],
    ) -> None:
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
        self._wait_source_idle(records, source)

    def _finish_p_to_d(
        self,
        source: NodeMetrics,
        records: List[ActionRecord],
    ) -> None:
        self._post_worker(
            records,
            "set_source_runtime_role",
            source,
            "/pd_flip/runtime_role/set",
            {"role": "decode", "force": False},
        )
        self._post_router(
            records,
            "refresh_router_source_role",
            source,
            "/pd_flip/router/worker/role",
            {
                "worker_id": source.router_worker_id,
                "role": "decode",
                "bootstrap_port": None,
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

    def _execute_p_to_d_monitor(
        self,
        *,
        metrics: List[NodeMetrics],
        state_trace: List[JsonDict],
        snapshot_index: Optional[int],
    ) -> FlipExecutionResult:
        started = time.monotonic()
        records: List[ActionRecord] = []
        source: Optional[NodeMetrics] = None
        try:
            source = self._select_source(
                metrics,
                source_name=None,
                expected_role="prefill",
                prefer_high_load=False,
            )
            state_trace.append(
                _monitor_state_record(
                    state=MonitorState.PREPARING_DRAIN,
                    direction="p_to_d",
                    source=source.name,
                    role_before=source.effective_role,
                    role_after=source.effective_role,
                    reason="decode_slo_risk",
                    snapshot_index=snapshot_index,
                )
            )
            self._prepare_p_to_d(source, records)
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
                migration_target=None,
                actions=records,
                metrics=metrics,
                total_seconds=time.monotonic() - started,
            )
        except Exception as exc:
            if source is not None:
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
                migration_target=None,
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
        url = _join_url(base_url, path)
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
                    elapsed_seconds=time.monotonic() - started,
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
                    success=False,
                    message=str(exc),
                    elapsed_seconds=time.monotonic() - started,
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
                    elapsed_seconds=time.monotonic() - started,
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
                    elapsed_seconds=time.monotonic() - started,
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
                    elapsed_seconds=time.monotonic() - started,
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
                    elapsed_seconds=time.monotonic() - started,
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
                            elapsed_seconds=time.monotonic() - started,
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
                    elapsed_seconds=time.monotonic() - started,
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
                        elapsed_seconds=time.monotonic() - started,
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

    def _build_p_to_d_actions(self, source: NodeMetrics) -> List[ControllerAction]:
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
            self._router_action(
                "refresh_router_source_role",
                source,
                "/pd_flip/router/worker/role",
                {
                    "worker_id": source.router_worker_id,
                    "role": "decode",
                    "bootstrap_port": None,
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


def _raise_if_unsuccessful(response: Any, step: str) -> None:
    responses = response if isinstance(response, list) else [response]
    for item in responses:
        if isinstance(item, dict) and item.get("success", True) is False:
            raise RuntimeError(item.get("message") or f"{step} failed")


def _response_manifests(response: Any) -> List[JsonDict]:
    item = _first_successful_response(response)
    manifests = item.get("manifests", [])
    return [manifest for manifest in manifests if isinstance(manifest, dict)]


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


def _migration_response_complete(response: Any) -> bool:
    item = _first_successful_response(response)
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    failed = int(status.get("failed_reqs") or 0)
    pending = int(status.get("pending_reqs") or 0)
    return failed == 0 and pending == 0


def _migration_response_failed(response: Any) -> bool:
    item = _first_successful_response(response)
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    failed = int(status.get("failed_reqs") or 0)
    state = str(status.get("state") or item.get("state") or "").lower()
    return failed > 0 or state.endswith("_failed")


def _migration_response_error(response: Any) -> str:
    item = _first_successful_response(response)
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    return str(
        status.get("last_error")
        or item.get("message")
        or status.get("state")
        or "migration failed"
    )


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
    record: JsonDict = {
        "state": state,
        "direction": direction,
        "source": source,
        "migration_target": migration_target,
        "role_before": role_before,
        "role_after": role_after,
        "reason": reason,
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
    with open(path, "r", encoding="utf-8") as f:
        return PDClusterConfig.from_dict(json.load(f))


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
        observation_quiesce_seconds=float(
            os.environ.get("PD_FLIP_OBSERVE_QUIESCE_SECONDS", 0.0)
        ),
        post_migration_idle_timeout_seconds=float(
            os.environ.get("PD_FLIP_POST_MIGRATION_IDLE_TIMEOUT_SECONDS", 2.0)
        ),
        first_migration_ratio=args.first_migration_ratio,
        observation_seconds=args.observation_seconds,
        slo_threshold=args.slo_threshold,
        min_prefill_slo_samples=args.min_prefill_slo_samples,
        min_decode_slo_samples=args.min_decode_slo_samples,
        session_journal_path=args.session_journal_path,
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PD runtime-role flip controller for four-node experiments"
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
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--first-migration-ratio", type=float, default=0.5)
    parser.add_argument("--observation-seconds", type=float, default=10.0)
    parser.add_argument("--slo-threshold", type=float, default=0.9)
    parser.add_argument("--min-prefill-slo-samples", type=int, default=20)
    parser.add_argument("--min-decode-slo-samples", type=int, default=20)
    parser.add_argument("--session-journal-path", default="pd_flip_session.json")

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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        config = config_from_args(args)
        client = HttpClient(api_key=args.api_key, timeout_seconds=args.timeout_seconds)
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
            )
        else:
            parser.error(f"unknown command {args.command}")
        print(json.dumps(output, default=_json_default, indent=2, sort_keys=True))
        if args.command in ("execute", "execute-two-phase") and hasattr(
            output, "success"
        ):
            return 0 if output.success else 1
        return 0
    except Exception as exc:
        print(f"pd_flip_controller: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
