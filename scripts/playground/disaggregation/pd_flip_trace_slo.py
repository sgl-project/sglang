#!/usr/bin/env python3
"""Request-level trace SLO monitor for PD flip experiments."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pd_flip_monitor import (
        ClusterSLOSnapshot,
        HttpClient,
        NodeSLOSample,
        SampleCounts,
        _aggregate_loads,
        normalize_role,
    )
except ModuleNotFoundError:
    import importlib.util

    _MONITOR_PATH = Path(__file__).with_name("pd_flip_monitor.py")
    _SPEC = importlib.util.spec_from_file_location("pd_flip_monitor", _MONITOR_PATH)
    _MONITOR = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader is not None
    sys.modules[_SPEC.name] = _MONITOR
    _SPEC.loader.exec_module(_MONITOR)
    ClusterSLOSnapshot = _MONITOR.ClusterSLOSnapshot
    HttpClient = _MONITOR.HttpClient
    NodeSLOSample = _MONITOR.NodeSLOSample
    SampleCounts = _MONITOR.SampleCounts
    _aggregate_loads = _MONITOR._aggregate_loads
    normalize_role = _MONITOR.normalize_role


JsonDict = Dict[str, Any]


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def extract_pd_flip_slo(payload: JsonDict) -> JsonDict:
    custom_params = payload.get("custom_params")
    if not isinstance(custom_params, dict):
        return {}
    slo = custom_params.get("pd_flip_slo")
    if not isinstance(slo, dict):
        return {}

    ttft = _positive_float(slo.get("ttft_seconds"))
    tpot = _positive_float(slo.get("tpot_seconds"))
    if ttft is None and tpot is None:
        return {}

    result: JsonDict = {}
    if ttft is not None:
        result["ttft_seconds"] = ttft
    if tpot is not None:
        result["tpot_seconds"] = tpot
    return result


class TraceSLOMonitor:
    def __init__(
        self,
        *,
        ledger_path: str,
        window_seconds: float,
        client: Optional[HttpClient] = None,
        time_fn=time.monotonic,
    ):
        self.ledger_path = Path(ledger_path)
        self.window_seconds = max(0.0, float(window_seconds))
        self.client = client or HttpClient()
        self.time_fn = time_fn
        self.window_start_time: Optional[float] = None

    def reset_window(self) -> None:
        """Start a fresh logical window without mutating the append-only ledger."""
        self.window_start_time = self.time_fn()

    def collect_cluster(self, nodes: Iterable[Tuple[str, str, str]]) -> ClusterSLOSnapshot:
        now = self.time_fn()
        latest = self._read_latest_records(now)
        ttft_counts = self._ttft_counts(latest)
        tpot_counts = self._tpot_counts(latest)
        samples: List[NodeSLOSample] = []

        for name, url, role in nodes:
            load = self._load(url)
            normalized_role = normalize_role(role)
            samples.append(
                NodeSLOSample(
                    timestamp=now,
                    name=name,
                    role=normalized_role,
                    ttft=ttft_counts if normalized_role == "prefill" else SampleCounts(),
                    tpot=tpot_counts if normalized_role == "decode" else SampleCounts(),
                    running_reqs=int(load.get("num_running_reqs") or 0),
                    waiting_reqs=int(load.get("num_waiting_reqs") or 0),
                    token_usage=load.get("token_usage"),
                    raw_load=load,
                )
            )

        prefill_nodes = {sample.name for sample in samples if sample.role == "prefill"}
        decode_nodes = {sample.name for sample in samples if sample.role == "decode"}
        return ClusterSLOSnapshot(
            timestamp=now,
            prefill_nodes=len(prefill_nodes),
            decode_nodes=len(decode_nodes),
            prefill_slo_attainment=ttft_counts.attainment,
            decode_slo_attainment=tpot_counts.attainment,
            nodes=samples,
        )

    def _load(self, url: str) -> JsonDict:
        return _aggregate_loads(self.client.get_json(url, "/v1/loads?include=all"))

    def _read_latest_records(self, now: float) -> List[JsonDict]:
        if not self.ledger_path.exists():
            return []
        rolling_cutoff = (
            now - self.window_seconds if self.window_seconds > 0 else None
        )
        cutoffs = [
            value
            for value in (rolling_cutoff, self.window_start_time)
            if value is not None
        ]
        cutoff = max(cutoffs) if cutoffs else None
        latest: Dict[str, JsonDict] = {}
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                request_id = record.get("request_id")
                event_time = _positive_float(record.get("event_time"))
                if request_id is None or event_time is None:
                    continue
                if cutoff is not None and event_time < cutoff:
                    continue
                latest[str(request_id)] = record
        return list(latest.values())

    @staticmethod
    def _ttft_counts(records: Iterable[JsonDict]) -> SampleCounts:
        good = 0
        total = 0
        for record in records:
            ttft = _positive_float(record.get("ttft_seconds"))
            slo = _positive_float(record.get("ttft_slo_seconds"))
            if ttft is None or slo is None:
                continue
            total += 1
            if ttft <= slo:
                good += 1
        return SampleCounts(good=good, total=total)

    @staticmethod
    def _tpot_counts(records: Iterable[JsonDict]) -> SampleCounts:
        good = 0
        total = 0
        for record in records:
            try:
                record_total = int(record.get("total_tpot_intervals") or 0)
                record_good = int(record.get("good_tpot_intervals") or 0)
            except (TypeError, ValueError):
                continue
            if record_total <= 0:
                continue
            total += record_total
            good += max(0, min(record_good, record_total))
        return SampleCounts(good=good, total=total)
