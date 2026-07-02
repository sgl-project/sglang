# Request-Level PD Flip SLO Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in trace SLO monitor that computes PD flip TTFT/TPOT attainment from per-request `custom_params.pd_flip_slo` instead of global Prometheus thresholds.

**Architecture:** Keep the existing Prometheus monitor as the default path. Add a focused trace SLO module that reads JSONL request events, computes `prefill_slo_attainment` from user-observed TTFT and `decode_slo_attainment` from token-weighted TPOT intervals, and returns the existing `ClusterSLOSnapshot` shape. The trace runner will start a streaming SLO probe for measurement while preserving the existing non-streaming long request for KV handoff validation.

**Tech Stack:** Python stdlib, unittest, Bash Docker harness, existing `pd_flip_controller.py`, existing `pd_flip_monitor.py`, remote Linux Docker verification.

---

## File Structure

- Create `scripts/playground/disaggregation/pd_flip_trace_slo.py`
  - Parses `custom_params.pd_flip_slo`.
  - Maintains latest request records from a JSONL ledger.
  - Computes request-level TTFT and token-weighted TPOT attainment.
  - Implements `TraceSLOMonitor.collect_cluster(nodes)` with the same snapshot shape as `PDFlipSLOMonitor`.
- Modify `scripts/playground/disaggregation/pd_flip_controller.py`
  - Add an optional `--trace-slo-ledger` flag to the `monitor` command.
  - Use `TraceSLOMonitor` only when the flag is provided.
- Modify `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
  - Pass `--trace-slo-ledger` when `PD_FLIP_TRACE_SLO_LEDGER` is set.
- Modify `scripts/playground/disaggregation/pd_flip_docker/run_trace_handoff.sh`
  - Add opt-in `TRACE_REQUEST_SLO_MODE=1`.
  - Start a streaming SLO probe request that writes `trace_slo_ledger.jsonl`.
  - Keep the existing non-streaming long request for transparent KV handoff.
- Modify `scripts/playground/disaggregation/pd_flip_docker/README.md`
  - Document request-level SLO mode and artifacts.
- Test `test/srt/test_pd_flip_trace_slo_monitor.py`
  - Unit coverage for request SLO parsing, ledger aggregation, windowing, and missing data.
- Modify `test/srt/test_pd_flip_controller.py`
  - CLI coverage that trace monitor mode is selected when `--trace-slo-ledger` is passed.
- Modify `test/srt/test_pd_flip_trace_handoff_runner.py`
  - Static safety coverage for runner/controller opt-in variables and no destructive Docker commands.

## Task 1: Add Request-Level Trace SLO Unit Tests

**Files:**
- Create: `test/srt/test_pd_flip_trace_slo_monitor.py`

- [ ] **Step 1: Write the failing test file**

Create `test/srt/test_pd_flip_trace_slo_monitor.py` with:

```python
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "playground" / "disaggregation" / "pd_flip_trace_slo.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pd_flip_trace_slo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self):
        self.loads = {
            "http://node0": {"loads": [{"num_running_reqs": 1, "num_waiting_reqs": 0, "num_total_tokens": 11, "token_usage": 0.1}]},
            "http://node2": {"loads": [{"num_running_reqs": 2, "num_waiting_reqs": 1, "num_total_tokens": 22, "token_usage": 0.2}]},
        }

    def get_json(self, base_url, path):
        if path != "/v1/loads?include=all":
            raise AssertionError(path)
        return self.loads[base_url]


class TestTraceSLOMonitor(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_extracts_pd_flip_slo_from_custom_params(self):
        payload = {
            "custom_params": {
                "pd_flip_slo": {"ttft_seconds": "3.5", "tpot_seconds": "0.02"}
            }
        }

        slo = self.mod.extract_pd_flip_slo(payload)

        self.assertEqual(slo, {"ttft_seconds": 3.5, "tpot_seconds": 0.02})

    def test_extract_returns_empty_for_missing_or_bad_values(self):
        self.assertEqual(self.mod.extract_pd_flip_slo({}), {})
        self.assertEqual(
            self.mod.extract_pd_flip_slo({"custom_params": {"pd_flip_slo": {"ttft_seconds": -1}}}),
            {},
        )

    def test_snapshot_uses_request_level_ttft_and_token_weighted_tpot(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            path = Path(f.name)
            f.write(json.dumps({
                "request_id": "r1",
                "event_time": 10.0,
                "start_time": 9.0,
                "first_token_time": 10.0,
                "ttft_seconds": 1.0,
                "ttft_slo_seconds": 2.0,
                "good_tpot_intervals": 3,
                "total_tpot_intervals": 4,
                "tpot_slo_seconds": 0.02,
                "status": "running",
            }) + "\n")
            f.write(json.dumps({
                "request_id": "r2",
                "event_time": 11.0,
                "start_time": 9.0,
                "first_token_time": 11.5,
                "ttft_seconds": 2.5,
                "ttft_slo_seconds": 2.0,
                "good_tpot_intervals": 1,
                "total_tpot_intervals": 4,
                "tpot_slo_seconds": 0.02,
                "status": "running",
            }) + "\n")

        monitor = self.mod.TraceSLOMonitor(
            ledger_path=str(path),
            window_seconds=30.0,
            client=FakeClient(),
            time_fn=lambda: 12.0,
        )

        snapshot = monitor.collect_cluster(
            [("node0", "http://node0", "prefill"), ("node2", "http://node2", "decode")]
        )

        self.assertEqual(snapshot.prefill_slo_attainment, 0.5)
        self.assertEqual(snapshot.decode_slo_attainment, 0.5)
        by_name = {sample.name: sample for sample in snapshot.nodes}
        self.assertEqual(by_name["node2"].running_reqs, 2)
        self.assertEqual(by_name["node2"].waiting_reqs, 1)
        self.assertEqual(by_name["node2"].token_usage, 0.2)

    def test_missing_samples_return_none_not_zero(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            path = Path(f.name)
            f.write(json.dumps({
                "request_id": "r1",
                "event_time": 10.0,
                "ttft_slo_seconds": 2.0,
                "tpot_slo_seconds": 0.02,
                "status": "running",
            }) + "\n")

        monitor = self.mod.TraceSLOMonitor(
            ledger_path=str(path),
            window_seconds=30.0,
            client=FakeClient(),
            time_fn=lambda: 12.0,
        )

        snapshot = monitor.collect_cluster(
            [("node0", "http://node0", "prefill"), ("node2", "http://node2", "decode")]
        )

        self.assertIsNone(snapshot.prefill_slo_attainment)
        self.assertIsNone(snapshot.decode_slo_attainment)

    def test_window_uses_latest_record_per_request_and_prunes_old_records(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            path = Path(f.name)
            f.write(json.dumps({
                "request_id": "old",
                "event_time": 1.0,
                "ttft_seconds": 9.0,
                "ttft_slo_seconds": 1.0,
                "good_tpot_intervals": 0,
                "total_tpot_intervals": 10,
                "tpot_slo_seconds": 0.01,
            }) + "\n")
            f.write(json.dumps({
                "request_id": "r1",
                "event_time": 10.0,
                "ttft_seconds": 4.0,
                "ttft_slo_seconds": 1.0,
                "good_tpot_intervals": 0,
                "total_tpot_intervals": 1,
                "tpot_slo_seconds": 0.01,
            }) + "\n")
            f.write(json.dumps({
                "request_id": "r1",
                "event_time": 11.0,
                "ttft_seconds": 0.5,
                "ttft_slo_seconds": 1.0,
                "good_tpot_intervals": 1,
                "total_tpot_intervals": 1,
                "tpot_slo_seconds": 0.01,
            }) + "\n")

        monitor = self.mod.TraceSLOMonitor(
            ledger_path=str(path),
            window_seconds=5.0,
            client=FakeClient(),
            time_fn=lambda: 12.0,
        )

        snapshot = monitor.collect_cluster(
            [("node0", "http://node0", "prefill"), ("node2", "http://node2", "decode")]
        )

        self.assertEqual(snapshot.prefill_slo_attainment, 1.0)
        self.assertEqual(snapshot.decode_slo_attainment, 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and verify RED**

Run on `cloud-099` after syncing the test file:

```bash
docker run --rm --network none -e PYTHONPATH=/workspace/python:/workspace -v /root/sglang:/workspace -w /workspace sglang-pd-switch:tianciJ python3 -m unittest discover -s test/srt -p test_pd_flip_trace_slo_monitor.py
```

Expected: `ImportError` or file-not-found failure because `pd_flip_trace_slo.py` does not exist yet.

## Task 2: Implement Trace SLO Monitor Module

**Files:**
- Create: `scripts/playground/disaggregation/pd_flip_trace_slo.py`
- Test: `test/srt/test_pd_flip_trace_slo_monitor.py`

- [ ] **Step 1: Create the monitor module with minimal data parsing**

Create `scripts/playground/disaggregation/pd_flip_trace_slo.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
        _sum_counts,
        normalize_role,
    )
except ModuleNotFoundError:
    import importlib.util

    _MONITOR_PATH = Path(__file__).with_name("pd_flip_monitor.py")
    _SPEC = importlib.util.spec_from_file_location("pd_flip_monitor", _MONITOR_PATH)
    _MONITOR = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader is not None
    _SPEC.loader.exec_module(_MONITOR)
    ClusterSLOSnapshot = _MONITOR.ClusterSLOSnapshot
    HttpClient = _MONITOR.HttpClient
    NodeSLOSample = _MONITOR.NodeSLOSample
    SampleCounts = _MONITOR.SampleCounts
    _aggregate_loads = _MONITOR._aggregate_loads
    _sum_counts = _MONITOR._sum_counts
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
        cutoff = now - self.window_seconds if self.window_seconds > 0 else None
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
```

- [ ] **Step 2: Run the trace SLO tests and verify GREEN**

Run:

```bash
python3 -m unittest discover -s test/srt -p test_pd_flip_trace_slo_monitor.py
```

Expected: all 5 tests pass.

## Task 3: Add Controller Opt-In Trace Monitor Mode

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `test/srt/test_pd_flip_controller.py`

- [ ] **Step 1: Add failing controller CLI test**

Append to `TestPDFlipController` in `test/srt/test_pd_flip_controller.py`:

```python
    def test_main_monitor_uses_trace_slo_ledger_when_provided(self):
        client = MonitorFakeClient()
        self.script.HttpClient = lambda api_key=None, timeout_seconds=10.0: client

        created = {}

        class FakeTraceSLOMonitor:
            def __init__(self, *, ledger_path, window_seconds, client):
                created["ledger_path"] = ledger_path
                created["window_seconds"] = window_seconds
                created["client"] = client

            def collect_cluster(self, nodes):
                return self_script.ClusterSLOSnapshot(
                    timestamp=1.0,
                    prefill_nodes=1,
                    decode_nodes=2,
                    prefill_slo_attainment=1.0,
                    decode_slo_attainment=1.0,
                    nodes=[],
                )

        self_script = self.script
        self.script.TraceSLOMonitor = FakeTraceSLOMonitor

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = self.script.main(
                [
                    "--router-url",
                    "http://router",
                    "--node",
                    "name=node-a,worker_url=http://node-a:30000,router_worker_id=node-a,bootstrap_port=8997",
                    "--node",
                    "name=node-b,worker_url=http://node-b:30000,router_worker_id=node-b,bootstrap_port=8997",
                    "--node",
                    "name=node-c,worker_url=http://node-c:30000,router_worker_id=node-c,bootstrap_port=8997",
                    "monitor",
                    "--ttft-slo",
                    "0.2",
                    "--tpot-slo",
                    "0.02",
                    "--trace-slo-ledger",
                    "/tmp/trace_slo_ledger.jsonl",
                    "--iterations",
                    "1",
                    "--poll-interval",
                    "0",
                ]
            )

        self.assertEqual(rc, 0)
        self.assertEqual(created["ledger_path"], "/tmp/trace_slo_ledger.jsonl")
        self.assertEqual(created["window_seconds"], 30.0)
        self.assertIs(created["client"], client)
        self.assertIn("no flip decision", stdout.getvalue())
```

- [ ] **Step 2: Run the controller test and verify RED**

Run:

```bash
python3 -m unittest discover -s test/srt -p test_pd_flip_controller.py
```

Expected: failure because `--trace-slo-ledger` is unknown or `TraceSLOMonitor` is absent.

- [ ] **Step 3: Import trace monitor with fallback**

In `pd_flip_controller.py`, near the existing `pd_flip_monitor` import fallback, add:

```python
    from pd_flip_trace_slo import TraceSLOMonitor
```

and in the file-path fallback block load `pd_flip_trace_slo.py` the same way as `pd_flip_monitor.py`.

- [ ] **Step 4: Add parser argument**

In `build_arg_parser()`, add to the `monitor` subparser:

```python
    monitor.add_argument(
        "--trace-slo-ledger",
        default=None,
        help="Use request-level trace SLO JSONL ledger instead of Prometheus histograms.",
    )
```

- [ ] **Step 5: Select monitor implementation in `main()`**

Replace the existing `slo_monitor = PDFlipSLOMonitor(` creation inside the `elif args.command == "monitor":` branch with:

```python
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
```

- [ ] **Step 6: Run controller tests and verify GREEN**

Run:

```bash
python3 -m unittest discover -s test/srt -p test_pd_flip_controller.py
```

Expected: all controller tests pass.

## Task 4: Wire Docker Controller Runner Flag

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Modify: `test/srt/test_pd_flip_trace_handoff_runner.py`

- [ ] **Step 1: Add failing static test assertions**

In `test_pd_flip_trace_handoff_runner.py`, extend `test_controller_runner_supports_safe_node_subset`:

```python
        self.assertIn("PD_FLIP_TRACE_SLO_LEDGER", text)
        self.assertIn("--trace-slo-ledger", text)
```

- [ ] **Step 2: Run static test and verify RED**

Run:

```bash
python3 -m unittest discover -s test/srt -p test_pd_flip_trace_handoff_runner.py
```

Expected: failure because `run_controller.sh` does not yet pass `--trace-slo-ledger`.

- [ ] **Step 3: Add optional monitor args in `run_controller.sh`**

Before the `case` statement, add:

```bash
monitor_extra_args=()
if [[ -n "${PD_FLIP_TRACE_SLO_LEDGER:-}" ]]; then
  monitor_extra_args+=(--trace-slo-ledger "${PD_FLIP_TRACE_SLO_LEDGER}")
fi
```

In the `monitor)` command, append before the final line:

```bash
      "${monitor_extra_args[@]}"
```

The monitor invocation should end as:

```bash
      --iterations "${MONITOR_ITERATIONS}" \
      --poll-interval "${MONITOR_POLL_INTERVAL}" \
      "${monitor_extra_args[@]}"
```

- [ ] **Step 4: Run syntax and static tests**

Run:

```bash
bash -n scripts/playground/disaggregation/pd_flip_docker/run_controller.sh
python3 -m unittest discover -s test/srt -p test_pd_flip_trace_handoff_runner.py
```

Expected: shell syntax passes and test passes.

## Task 5: Add Streaming SLO Probe To Trace Runner

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_trace_handoff.sh`
- Modify: `test/srt/test_pd_flip_trace_handoff_runner.py`

- [ ] **Step 1: Add failing static test assertions**

Extend `test_trace_runner_exists_with_safe_experiment_contract`:

```python
        self.assertIn("TRACE_REQUEST_SLO_MODE", text)
        self.assertIn("trace_slo_ledger.jsonl", text)
        self.assertIn("custom_params", text)
        self.assertIn("pd_flip_slo", text)
        self.assertIn("stream", text)
        self.assertIn("PD_FLIP_TRACE_SLO_LEDGER", text)
```

- [ ] **Step 2: Run static test and verify RED**

Run:

```bash
python3 -m unittest discover -s test/srt -p test_pd_flip_trace_handoff_runner.py
```

Expected: failure because runner lacks request-level SLO mode.

- [ ] **Step 3: Add runner configuration**

Near the other `TRACE_*` variables in `run_trace_handoff.sh`, add:

```bash
TRACE_REQUEST_SLO_MODE="${TRACE_REQUEST_SLO_MODE:-0}"
TRACE_SLO_LEDGER="${TRACE_SLO_LEDGER:-${TRACE_ARTIFACT_DIR}/trace_slo_ledger.jsonl}"
TRACE_SLO_PROBE_MAX_TOKENS="${TRACE_SLO_PROBE_MAX_TOKENS:-128}"
TRACE_SLO_PROBE_TTFT_SLO_SECONDS="${TRACE_SLO_PROBE_TTFT_SLO_SECONDS:-0.001}"
TRACE_SLO_PROBE_TPOT_SLO_SECONDS="${TRACE_SLO_PROBE_TPOT_SLO_SECONDS:-0.02}"
TRACE_SLO_PROBE_PROMPT="${TRACE_SLO_PROBE_PROMPT:-Write a concise numbered list of distributed serving checks.}"
```

Extend the export line:

```bash
export TRACE_PROMPT TRACE_MAX_TOKENS TRACE_TEMPERATURE TRACE_CLIENT_TIMEOUT_SECONDS
export TRACE_SLO_PROBE_MAX_TOKENS TRACE_SLO_PROBE_TTFT_SLO_SECONDS TRACE_SLO_PROBE_TPOT_SLO_SECONDS TRACE_SLO_PROBE_PROMPT TRACE_SLO_LEDGER
```

- [ ] **Step 4: Add `start_slo_probe_client()`**

Add this Bash function before `run_monitor()`:

```bash
start_slo_probe_client() {
  if [[ "${TRACE_REQUEST_SLO_MODE}" != "1" ]]; then
    return
  fi
  : >"${TRACE_SLO_LEDGER}"
  python3 - \
    "${TRACE_ROUTER_URL}" \
    "${TRACE_MODEL_ID}" \
    "${TRACE_ARTIFACT_DIR}/trace_slo_probe_response.json" \
    "${TRACE_ARTIFACT_DIR}/trace_slo_probe_error.json" <<'PY' &
import json
import os
import sys
import time
import traceback
import urllib.request

router_url, model_id, response_path, error_path = sys.argv[1:5]
request_id = "slo-probe-0001"
ledger_path = os.environ["TRACE_SLO_LEDGER"]
ttft_slo = float(os.environ["TRACE_SLO_PROBE_TTFT_SLO_SECONDS"])
tpot_slo = float(os.environ["TRACE_SLO_PROBE_TPOT_SLO_SECONDS"])
record = {
    "request_id": request_id,
    "status": "running",
    "start_time": time.monotonic(),
    "event_time": time.monotonic(),
    "first_token_time": None,
    "end_time": None,
    "ttft_slo_seconds": ttft_slo,
    "tpot_slo_seconds": tpot_slo,
    "ttft_seconds": None,
    "ttft_met": None,
    "good_tpot_intervals": 0,
    "total_tpot_intervals": 0,
    "last_token_time": None,
    "completion_tokens": 0,
    "error": None,
}

def write_event():
    record["event_time"] = time.monotonic()
    with open(ledger_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

write_event()
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": os.environ["TRACE_SLO_PROBE_PROMPT"]}],
    "max_tokens": int(os.environ["TRACE_SLO_PROBE_MAX_TOKENS"]),
    "temperature": 0,
    "stream": True,
    "custom_params": {
        "pd_flip_slo": {
            "ttft_seconds": ttft_slo,
            "tpot_seconds": tpot_slo,
        }
    },
}
request = urllib.request.Request(
    router_url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
chunks = []
try:
    with urllib.request.urlopen(request, timeout=int(os.environ["TRACE_CLIENT_TIMEOUT_SECONDS"])) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            chunks.append(chunk)
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            token_text = delta.get("content")
            if not token_text:
                continue
            now = time.monotonic()
            if record["first_token_time"] is None:
                record["first_token_time"] = now
                record["ttft_seconds"] = now - record["start_time"]
                record["ttft_met"] = record["ttft_seconds"] <= ttft_slo
            else:
                interval = now - record["last_token_time"]
                record["total_tpot_intervals"] += 1
                if interval <= tpot_slo:
                    record["good_tpot_intervals"] += 1
            record["last_token_time"] = now
            record["completion_tokens"] += 1
            write_event()
    record["status"] = "completed"
    record["end_time"] = time.monotonic()
    write_event()
    with open(response_path, "w") as f:
        json.dump({"chunks": len(chunks), "record": record}, f, indent=2, sort_keys=True)
except Exception as exc:
    record["status"] = "error"
    record["error"] = repr(exc)
    record["end_time"] = time.monotonic()
    write_event()
    with open(error_path, "w") as f:
        json.dump({"error": repr(exc), "traceback": traceback.format_exc(), "record": record}, f, indent=2, sort_keys=True)
    raise
PY
  SLO_PROBE_PID=$!
}
```

- [ ] **Step 5: Invoke probe before monitor**

Add `SLO_PROBE_PID=""` near `CLIENT_PID=""`.

Call `start_slo_probe_client` after `start_long_client` and before `sleep "${TRACE_CLIENT_WARMUP_SECONDS}"`:

```bash
start_long_client
start_slo_probe_client
sleep "${TRACE_CLIENT_WARMUP_SECONDS}"
```

After the main client wait, wait for the probe if started:

```bash
if [[ -n "${SLO_PROBE_PID}" ]]; then
  set +e
  wait "${SLO_PROBE_PID}"
  SLO_PROBE_EXIT=$?
  set -e
  echo "${SLO_PROBE_EXIT}" >"${TRACE_ARTIFACT_DIR}/trace_slo_probe.exit"
fi
```

- [ ] **Step 6: Pass ledger to controller monitor**

In `run_monitor()`, add to the environment before `"${SCRIPT_DIR}/run_controller.sh"`:

```bash
  PD_FLIP_TRACE_SLO_LEDGER="${TRACE_SLO_LEDGER}" \
```

Only set it when `TRACE_REQUEST_SLO_MODE=1` by building a local `trace_monitor_env` array:

```bash
  trace_monitor_env=()
  if [[ "${TRACE_REQUEST_SLO_MODE}" == "1" ]]; then
    trace_monitor_env+=(PD_FLIP_TRACE_SLO_LEDGER="${TRACE_SLO_LEDGER}")
  fi
```

Then invoke:

```bash
  env \
  ENV_FILE="${ENV_PATH}" \
  PD_FLIP_NODE_NAMES="${TRACE_NODE_NAMES}" \
  PD_FLIP_TTFT_SLO_SECONDS="${TRACE_TTFT_SLO_SECONDS}" \
  PD_FLIP_TPOT_SLO_SECONDS="${TRACE_TPOT_SLO_SECONDS}" \
  PD_FLIP_WINDOW_SECONDS_OVERRIDE="${TRACE_WINDOW_SECONDS}" \
  PD_FLIP_ENTER_THRESHOLD_OVERRIDE="${TRACE_ENTER_THRESHOLD}" \
  PD_FLIP_EXIT_THRESHOLD_OVERRIDE="${TRACE_EXIT_THRESHOLD}" \
  PD_FLIP_COMMIT_THRESHOLD_OVERRIDE="${TRACE_COMMIT_THRESHOLD}" \
  PD_FLIP_MONITOR_ITERATIONS_OVERRIDE="${TRACE_MONITOR_ITERATIONS}" \
  PD_FLIP_MONITOR_POLL_INTERVAL_OVERRIDE="${TRACE_MONITOR_POLL_INTERVAL}" \
  "${trace_monitor_env[@]}" \
    "${SCRIPT_DIR}/run_controller.sh" monitor >"${raw}" 2>&1
```

- [ ] **Step 7: Include SLO summary**

In `write_summary()`, load `trace_slo_probe_response.json` and include:

```python
trace_slo_probe = load("trace_slo_probe_response.json")
summary = {
    "success": success,
    "artifact_dir": artifact_dir,
    "monitor_exit": exit_code("monitor.exit"),
    "restore_exit": exit_code("restore_source.exit"),
    "monitor_message": monitor.get("message"),
    "state_trace": state_trace,
    "kv_transferred_reqs_observed": transferred,
    "kv_failed_reqs_observed": failed,
    "client_finish_reason": finish_reason,
    "client_completion_tokens": (client.get("usage") or {}).get("completion_tokens")
    if isinstance(client, dict)
    else None,
    "client_content_len": len(content),
    "client_has_abort_literal": "abort" in content.lower(),
    "client_error": client_error,
    "trace_slo_probe": trace_slo_probe,
    "trace_slo_ledger": os.path.join(artifact_dir, "trace_slo_ledger.jsonl"),
    "post_workers": post_workers.get("workers") if isinstance(post_workers, dict) else None,
}
```

- [ ] **Step 8: Run syntax and static tests**

Run:

```bash
bash -n scripts/playground/disaggregation/pd_flip_docker/run_trace_handoff.sh
python3 -m unittest discover -s test/srt -p test_pd_flip_trace_handoff_runner.py
```

Expected: pass.

## Task 6: Document Request-Level Trace Mode

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/README.md`

- [ ] **Step 1: Add README section**

Add under the trace handoff section:

```markdown
Request-level TTFT/TPOT SLO mode:

```bash
ENV_FILE=$PWD/env.local \
TRACE_REQUEST_SLO_MODE=1 \
TRACE_SOURCE_NAME=node2 \
TRACE_MAX_TOKENS=4096 \
bash ./run_trace_handoff.sh
```

This mode writes `trace_slo_ledger.jsonl` and passes it to the controller via
`PD_FLIP_TRACE_SLO_LEDGER`. Each SLO probe request sends:

```json
"custom_params": {
  "pd_flip_slo": {
    "ttft_seconds": 0.001,
    "tpot_seconds": 0.02
  }
}
```

The migrated handoff request remains non-streaming; the streaming probe is only
used to measure request-level TTFT/TPOT for monitor decisions.
```
```

- [ ] **Step 2: Review README formatting**

Run:

```bash
rg -n "Request-level TTFT|trace_slo_ledger|PD_FLIP_TRACE_SLO_LEDGER" scripts/playground/disaggregation/pd_flip_docker/README.md
```

Expected: all three terms appear.

## Task 7: Unit Verification

**Files:**
- No new files.

- [ ] **Step 1: Sync changed files to cloud-099**

Run from Windows workspace:

```powershell
scp scripts/playground/disaggregation/pd_flip_trace_slo.py cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_trace_slo.py
scp scripts/playground/disaggregation/pd_flip_controller.py cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_controller.py
scp scripts/playground/disaggregation/pd_flip_docker/run_controller.sh cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_controller.sh
scp scripts/playground/disaggregation/pd_flip_docker/run_trace_handoff.sh cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/run_trace_handoff.sh
scp scripts/playground/disaggregation/pd_flip_docker/README.md cloud-099:/root/sglang/scripts/playground/disaggregation/pd_flip_docker/README.md
scp test/srt/test_pd_flip_trace_slo_monitor.py cloud-099:/root/sglang/test/srt/test_pd_flip_trace_slo_monitor.py
scp test/srt/test_pd_flip_controller.py cloud-099:/root/sglang/test/srt/test_pd_flip_controller.py
scp test/srt/test_pd_flip_trace_handoff_runner.py cloud-099:/root/sglang/test/srt/test_pd_flip_trace_handoff_runner.py
```

- [ ] **Step 2: Run targeted unit tests in Docker**

Run:

```bash
docker run --rm --network none -e PYTHONPATH=/workspace/python:/workspace -v /root/sglang:/workspace -w /workspace sglang-pd-switch:tianciJ bash -lc 'python3 -m unittest discover -s test/srt -p test_pd_flip_trace_slo_monitor.py && python3 -m unittest discover -s test/srt -p test_pd_flip_controller.py && python3 -m unittest discover -s test/srt -p test_pd_flip_trace_handoff_runner.py'
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run full PD flip discovery**

Run:

```bash
docker run --rm --network none -e PYTHONPATH=/workspace/python:/workspace -v /root/sglang:/workspace -w /workspace sglang-pd-switch:tianciJ python3 -m unittest discover -s test/srt -p 'test_pd_flip*.py'
```

Expected: all tests pass.

## Task 8: Safe Docker Trace Verification

**Files:**
- No new files.

- [ ] **Step 1: Confirm safe topology**

Run:

```bash
curl -s http://127.0.0.1:18001/pd_flip/router/workers
```

Expected: only Codex topology is used for this trace:

```text
node0 prefill active_load=0
node2 decode active_load=0
node3 decode active_load=0
```

- [ ] **Step 2: Run request-level SLO trace**

Run on `cloud-099`:

```bash
cd /root/sglang/scripts/playground/disaggregation/pd_flip_docker
ENV_FILE=$PWD/env.local \
TRACE_REQUEST_SLO_MODE=1 \
TRACE_SOURCE_NAME=node2 \
TRACE_MAX_TOKENS=4096 \
TRACE_CLIENT_WARMUP_SECONDS=8 \
TRACE_MONITOR_ITERATIONS=60 \
bash ./run_trace_handoff.sh
```

Expected `summary.json`:

```text
success = true
monitor_message = "pd flip committed after two-phase migration"
state_trace = ["safe", "preparing_kv_transfer", "flipping_role", "safe"]
kv_transferred_reqs_observed includes 1
client_finish_reason != "abort"
trace_slo_probe.record.ttft_seconds is present
trace_slo_probe.record.total_tpot_intervals > 0
```

- [ ] **Step 3: Confirm topology restored**

Run:

```bash
curl -s http://127.0.0.1:18001/pd_flip/router/workers
```

Expected:

```text
node0 prefill draining=false active_load=0
node2 decode draining=false active_load=0
node3 decode draining=false active_load=0
```

- [ ] **Step 4: Run smoke request**

Run:

```bash
python3 - <<'PY'
import json, urllib.request
body = {
    "model": "deepseek_v3.1_terminus",
    "messages": [{"role": "user", "content": "Reply exactly: OK"}],
    "max_tokens": 2,
    "temperature": 0,
}
req = urllib.request.Request(
    "http://127.0.0.1:18001/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
r = json.load(urllib.request.urlopen(req, timeout=60))
print(json.dumps({
    "finish_reason": r["choices"][0]["finish_reason"],
    "content": r["choices"][0]["message"]["content"],
    "usage": r.get("usage"),
}, ensure_ascii=False))
PY
```

Expected: `finish_reason` is `stop` or `length`, response content contains `OK`.

## Task 9: Final Report

**Files:**
- No new files.

- [ ] **Step 1: Summarize exact behavior**

Report:

```text
Default Prometheus monitor remains unchanged.
Request-level trace mode is opt-in via TRACE_REQUEST_SLO_MODE=1.
Requests carry custom_params.pd_flip_slo.
TraceSLOMonitor computes TTFT request-weighted and TPOT token-weighted attainment.
Controller FSM and KV migration path are unchanged.
```

- [ ] **Step 2: Report verification evidence**

Include:

```text
Targeted unittest results
Full test_pd_flip*.py result
Docker trace artifact directory
summary.json key fields
final router topology
smoke request result
```
