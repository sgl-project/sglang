#!/usr/bin/env python3
"""Monitor helpers for PD flip SLO-driven experiments.

This module intentionally has no SGLang package imports so it can run from the
Docker controller host even when only the repository checkout is available.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from urllib import error, request
from urllib.parse import urljoin


JsonDict = Dict[str, Any]

TTFT_METRIC = "sglang:time_to_first_token_seconds"
TPOT_METRIC = "sglang:inter_token_latency_seconds"


@dataclass(frozen=True)
class SampleCounts:
    good: int = 0
    total: int = 0

    @property
    def attainment(self) -> Optional[float]:
        if self.total <= 0:
            return None
        return self.good / self.total

    def delta_from(self, previous: Optional["SampleCounts"]) -> "SampleCounts":
        if previous is None:
            return self
        return SampleCounts(
            good=max(0, self.good - previous.good),
            total=max(0, self.total - previous.total),
        )


@dataclass(frozen=True)
class NodeSLOSample:
    timestamp: float
    name: str
    role: str
    ttft: SampleCounts = field(default_factory=SampleCounts)
    tpot: SampleCounts = field(default_factory=SampleCounts)
    running_reqs: int = 0
    waiting_reqs: int = 0
    token_usage: Optional[float] = None
    raw_load: JsonDict = field(default_factory=dict)

    @property
    def role_attainment(self) -> Optional[float]:
        if self.role == "prefill":
            return self.ttft.attainment
        if self.role == "decode":
            return self.tpot.attainment
        return None


@dataclass(frozen=True)
class ClusterSLOSnapshot:
    timestamp: float
    prefill_nodes: int
    decode_nodes: int
    prefill_slo_attainment: Optional[float]
    decode_slo_attainment: Optional[float]
    nodes: List[NodeSLOSample]

    def to_dict(self) -> JsonDict:
        return asdict(self)


class SLOWindow:
    def __init__(self, window_seconds: float):
        self.window_seconds = max(0.0, float(window_seconds))
        self.samples: Deque[NodeSLOSample] = deque()

    def add(self, sample: NodeSLOSample) -> None:
        self.samples.append(sample)
        self._prune(sample.timestamp)

    def snapshot(self, timestamp: Optional[float] = None) -> ClusterSLOSnapshot:
        if timestamp is None:
            timestamp = self.samples[-1].timestamp if self.samples else time.monotonic()
        self._prune(timestamp)

        prefill_nodes = {sample.name for sample in self.samples if sample.role == "prefill"}
        decode_nodes = {sample.name for sample in self.samples if sample.role == "decode"}
        prefill_counts = _sum_counts(
            sample.ttft for sample in self.samples if sample.role == "prefill"
        )
        if prefill_counts.total <= 0 and prefill_nodes:
            prefill_counts = _sum_counts(sample.ttft for sample in self.samples)
        decode_counts = _sum_counts(
            sample.tpot for sample in self.samples if sample.role == "decode"
        )
        return ClusterSLOSnapshot(
            timestamp=timestamp,
            prefill_nodes=len(prefill_nodes),
            decode_nodes=len(decode_nodes),
            prefill_slo_attainment=prefill_counts.attainment,
            decode_slo_attainment=decode_counts.attainment,
            nodes=list(self.samples),
        )

    def _prune(self, now: float) -> None:
        if self.window_seconds <= 0:
            return
        cutoff = now - self.window_seconds
        while self.samples and self.samples[0].timestamp < cutoff:
            self.samples.popleft()


class HttpClient:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 10.0):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def get_text(self, base_url: str, path: str) -> str:
        req = self._request(base_url, path)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{req.full_url} returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"failed to connect to {req.full_url}: {exc}") from exc

    def get_json(self, base_url: str, path: str) -> Any:
        text = self.get_text(base_url, path)
        return json.loads(text) if text else {}

    def _request(self, base_url: str, path: str) -> request.Request:
        req = request.Request(urljoin(base_url.rstrip("/") + "/", path.lstrip("/")))
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return req


class PDFlipSLOMonitor:
    def __init__(
        self,
        *,
        ttft_slo_seconds: float,
        tpot_slo_seconds: float,
        window_seconds: float,
        client: Optional[HttpClient] = None,
        time_fn=time.monotonic,
    ):
        self.ttft_slo_seconds = float(ttft_slo_seconds)
        self.tpot_slo_seconds = float(tpot_slo_seconds)
        self.window = SLOWindow(window_seconds)
        self.client = client or HttpClient()
        self.time_fn = time_fn
        self._previous_counts: Dict[Tuple[str, str], SampleCounts] = {}

    def collect_node(self, name: str, url: str, role: str) -> NodeSLOSample:
        now = self.time_fn()
        metrics_text = self.client.get_text(url, "/metrics")
        loads_body = self.client.get_json(url, "/v1/loads?include=all")
        load = _aggregate_loads(loads_body)
        ttft = parse_histogram_counts(metrics_text, TTFT_METRIC, self.ttft_slo_seconds)
        tpot = parse_histogram_counts(metrics_text, TPOT_METRIC, self.tpot_slo_seconds)
        ttft_delta = ttft.delta_from(self._previous_counts.get((name, "ttft")))
        tpot_delta = tpot.delta_from(self._previous_counts.get((name, "tpot")))
        self._previous_counts[(name, "ttft")] = ttft
        self._previous_counts[(name, "tpot")] = tpot
        return NodeSLOSample(
            timestamp=now,
            name=name,
            role=normalize_role(role),
            ttft=ttft_delta,
            tpot=tpot_delta,
            running_reqs=int(load.get("num_running_reqs") or 0),
            waiting_reqs=int(load.get("num_waiting_reqs") or 0),
            token_usage=load.get("token_usage"),
            raw_load=load,
        )

    def collect_cluster(self, nodes: Iterable[Tuple[str, str, str]]) -> ClusterSLOSnapshot:
        now = self.time_fn()
        for name, url, role in nodes:
            self.window.add(self.collect_node(name, url, role))
        return self.window.snapshot(now)


def parse_histogram_counts(
    metrics_text: str, metric_name: str, slo_seconds: float
) -> SampleCounts:
    buckets = parse_histogram_buckets(metrics_text, metric_name)
    if not buckets:
        return SampleCounts()

    finite_bounds = sorted(bound for bound in buckets if bound != float("inf"))
    good_bound = None
    for bound in finite_bounds:
        if bound >= slo_seconds:
            good_bound = bound
            break
    if good_bound is None and finite_bounds:
        good_bound = finite_bounds[-1]

    total = buckets.get(float("inf"))
    if total is None:
        total = buckets[max(buckets)]
    good = buckets.get(good_bound, 0) if good_bound is not None else 0
    return SampleCounts(good=int(good), total=int(total))


def parse_histogram_buckets(metrics_text: str, metric_name: str) -> Dict[float, int]:
    prefix = f"{metric_name}_bucket"
    buckets: Dict[float, int] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        labels, value = _split_sample(line[len(prefix) :])
        le = labels.get("le")
        if le is None:
            continue
        try:
            bound = float("inf") if le == "+Inf" else float(le)
            buckets[bound] = int(float(value))
        except ValueError:
            continue
    return buckets


def normalize_role(role: Any) -> str:
    value = str(role or "").strip().lower()
    if value in ("prefill", "decode"):
        return value
    return "unknown"


def _split_sample(rest: str) -> Tuple[Dict[str, str], str]:
    labels: Dict[str, str] = {}
    rest = rest.strip()
    if rest.startswith("{"):
        label_text, _, value = rest.partition("}")
        for item in label_text.lstrip("{").split(","):
            if not item or "=" not in item:
                continue
            key, raw_value = item.split("=", 1)
            labels[key.strip()] = raw_value.strip().strip('"')
        return labels, value.strip()
    parts = rest.split(None, 1)
    return labels, parts[0] if parts else "0"


def _aggregate_loads(loads_body: Any) -> JsonDict:
    loads = loads_body.get("loads", []) if isinstance(loads_body, dict) else []
    result: JsonDict = {
        "num_running_reqs": 0,
        "num_waiting_reqs": 0,
        "num_total_tokens": 0,
        "token_usage": None,
    }
    max_token_usage: Optional[float] = None
    for load in loads:
        if not isinstance(load, dict):
            continue
        result["num_running_reqs"] += int(load.get("num_running_reqs") or 0)
        result["num_waiting_reqs"] += int(load.get("num_waiting_reqs") or 0)
        result["num_total_tokens"] += int(load.get("num_total_tokens") or 0)
        usage = load.get("token_usage")
        if usage is not None:
            max_token_usage = max(float(usage), max_token_usage or 0.0)
    result["token_usage"] = max_token_usage
    return result


def _sum_counts(counts: Iterable[SampleCounts]) -> SampleCounts:
    good = 0
    total = 0
    for count in counts:
        good += count.good
        total += count.total
    return SampleCounts(good=good, total=total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttft-slo", type=float, required=True)
    parser.add_argument("--tpot-slo", type=float, required=True)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        help="Node spec name=<name>,url=<url>,role=<prefill|decode>.",
    )
    return parser


def parse_node_spec(value: str) -> Tuple[str, str, str]:
    fields = {}
    for part in value.split(","):
        if "=" not in part:
            raise ValueError(f"bad node spec item: {part}")
        key, item_value = part.split("=", 1)
        fields[key.strip()] = item_value.strip()
    return fields["name"], fields["url"], fields["role"]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    monitor = PDFlipSLOMonitor(
        ttft_slo_seconds=args.ttft_slo,
        tpot_slo_seconds=args.tpot_slo,
        window_seconds=args.window_seconds,
        client=HttpClient(api_key=args.api_key),
    )
    snapshot = monitor.collect_cluster(parse_node_spec(node) for node in args.node)
    print(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
