#!/usr/bin/env python3
"""Controller skeleton for four-node PD runtime role flip experiments.

The controller keeps orchestration policy outside SGLang workers:

1. collect router + worker metrics,
2. build an explicit D->P / P->D flip plan,
3. dry-run the HTTP actions before later tasks enable execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib import error, request
from urllib.parse import urljoin


JsonDict = Dict[str, Any]


class HttpLike(Protocol):
    def get_json(self, base_url: str, path: str) -> Any:
        ...

    def post_json(self, base_url: str, path: str, payload: JsonDict) -> Any:
        ...


class HttpClient:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 10.0):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def get_json(self, base_url: str, path: str) -> Any:
        req = self._request(base_url, path, method="GET")
        return self._open_json(req)

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

    def _open_json(self, req: request.Request) -> Any:
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{req.full_url} returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"failed to connect to {req.full_url}: {exc}") from exc


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


class PDFlipController:
    def __init__(self, config: PDClusterConfig, client: HttpLike):
        if not config.nodes:
            raise ValueError("PDClusterConfig.nodes must not be empty")
        self.config = config
        self.client = client

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
            migration_target = self._select_decode_migration_target(metrics, source)
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
        self, metrics: List[NodeMetrics], source: NodeMetrics
    ) -> NodeMetrics:
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
                "start_decode_migration_source",
                source,
                "POST",
                "/pd_flip/migration/source/start",
                {"session_id": session_id, "target_url": target.worker_url},
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
                    "adopt_on_success": True,
                },
            ),
            self._worker_action(
                "wait_decode_migration",
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


def _load_sort_key(metric: NodeMetrics) -> Tuple[int, int, int, float, str]:
    return (
        metric.running_reqs,
        metric.router_active_load,
        metric.total_tokens,
        metric.token_usage or 0.0,
        metric.name,
    )


def _find_metric(metrics: List[NodeMetrics], name_or_worker_id: str) -> Optional[NodeMetrics]:
    for metric in metrics:
        if metric.name == name_or_worker_id or metric.router_worker_id == name_or_worker_id:
            return metric
    return None


def load_config(path: str) -> PDClusterConfig:
    with open(path, "r", encoding="utf-8") as f:
        return PDClusterConfig.from_dict(json.load(f))


def _parse_node_spec(value: str) -> PDNode:
    parts = {}
    for item in value.split(","):
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"invalid --node entry {value!r}; expected key=value pairs")
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
        raise ValueError("at least one --node is required when --config is not provided")
    return PDClusterConfig(
        router_url=args.router_url,
        nodes=[_parse_node_spec(value) for value in args.node],
        request_timeout_seconds=args.timeout_seconds,
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

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("metrics", help="Collect router/worker metrics")

    dry_run = subparsers.add_parser("dry-run", help="Build a flip plan without POSTs")
    dry_run.add_argument("--direction", choices=["d_to_p", "p_to_d"], required=True)
    dry_run.add_argument("--source-name", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
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
            )
        else:
            parser.error(f"unknown command {args.command}")
        print(json.dumps(output, default=_json_default, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"pd_flip_controller: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
