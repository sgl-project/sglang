# KV Transfer Latency Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Docker-friendly two-node benchmark that measures Mooncake/SGLang GPU-buffer KV transfer latency as a function of data size.

**Architecture:** The benchmark has one Python entrypoint with two roles: `target` allocates and registers a destination GPU buffer, writes a JSON endpoint file, and stays alive; `initiator` reads that endpoint file, allocates and registers a source GPU buffer, sweeps configured byte sizes, calls SGLang's `MooncakeTransferEngine.transfer_sync()`, and writes CSV/JSONL results. Small shell wrappers run the Python entrypoint inside the existing `sglang-pd-switch:tianciJ` Docker image with host networking and GPU access.

**Tech Stack:** Python 3, PyTorch CUDA tensors, SGLang MooncakeTransferEngine, Docker `--network host`, shell wrappers, pytest unit tests for pure parsing/statistics/command generation.

---

### File Structure

- Create: `scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py`
  - Owns CLI parsing, size parsing, latency statistics, target endpoint file writing, target loop, initiator sweep, CSV/JSONL output.
- Create: `scripts/playground/disaggregation/kv_transfer_bench/run_target.sh`
  - Runs the target role in Docker on one physical node.
- Create: `scripts/playground/disaggregation/kv_transfer_bench/run_initiator.sh`
  - Runs the initiator role in Docker on another physical node.
- Create: `scripts/playground/disaggregation/kv_transfer_bench/README.md`
  - Four-node operational guide using 099/100/101/102, including first pair 099 -> 102.
- Create: `test/srt/test_kv_transfer_latency_bench.py`
  - Unit tests for CLI-safe pure helpers.

### Task 1: Pure Helpers With TDD

**Files:**
- Create: `test/srt/test_kv_transfer_latency_bench.py`
- Create: `scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

```python
from scripts.playground.disaggregation.kv_transfer_bench.kv_transfer_latency import (
    format_bytes,
    parse_size,
    parse_size_list,
    summarize_latencies_ms,
)

def test_parse_size_accepts_binary_units():
    assert parse_size("1KB") == 1024
    assert parse_size("2MiB") == 2 * 1024 * 1024
    assert parse_size("3g") == 3 * 1024**3

def test_parse_size_list_expands_ranges_and_csv():
    assert parse_size_list("1MB,2MB") == [1024**2, 2 * 1024**2]
    assert parse_size_list("1MB:8MB:x2") == [1024**2, 2 * 1024**2, 4 * 1024**2, 8 * 1024**2]

def test_summarize_latencies_ms_reports_percentiles_and_bandwidth():
    summary = summarize_latencies_ms([1.0, 2.0, 3.0, 4.0], num_bytes=1024**3)
    assert summary["latency_ms_mean"] == 2.5
    assert summary["latency_ms_p50"] == 2.5
    assert summary["latency_ms_p90"] == 3.7
    assert round(summary["bandwidth_GBps_p50"], 3) == 400.0

def test_format_bytes_uses_readable_units():
    assert format_bytes(1024) == "1.00KiB"
    assert format_bytes(1024**2) == "1.00MiB"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
```

Expected: import failure because `kv_transfer_latency.py` does not exist yet.

- [ ] **Step 3: Implement minimal helper code**

Implement the pure functions only. No CUDA/Mooncake runtime code in this step.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
```

Expected: all helper tests pass.

### Task 2: Target Metadata and Result Output

**Files:**
- Modify: `test/srt/test_kv_transfer_latency_bench.py`
- Modify: `scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

```python
import csv
import json

from scripts.playground.disaggregation.kv_transfer_bench.kv_transfer_latency import (
    TargetInfo,
    load_target_info,
    write_csv_summary,
    write_jsonl_samples,
    write_target_info,
)

def test_target_info_roundtrip(tmp_path):
    path = tmp_path / "target.json"
    info = TargetInfo(
        session_id="192.168.0.42:12345",
        host="192.168.0.42",
        gpu_id=0,
        ptr=123456,
        bytes=1024,
        ib_device="mlx5_0",
        protocol="rdma",
    )
    write_target_info(path, info)
    assert load_target_info(path) == info

def test_result_writers_create_csv_and_jsonl(tmp_path):
    csv_path = tmp_path / "summary.csv"
    jsonl_path = tmp_path / "samples.jsonl"
    rows = [{"bytes": 1024, "latency_ms_p50": 1.5, "error_count": 0}]
    samples = [{"bytes": 1024, "iteration": 0, "latency_ms": 1.5, "ret": 0}]
    write_csv_summary(csv_path, rows)
    write_jsonl_samples(jsonl_path, samples)
    with csv_path.open() as f:
        assert list(csv.DictReader(f))[0]["bytes"] == "1024"
    with jsonl_path.open() as f:
        assert json.loads(f.readline())["latency_ms"] == 1.5
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
```

Expected: failures for missing dataclass/writer functions.

- [ ] **Step 3: Implement metadata and output helpers**

Add `TargetInfo`, JSON read/write, CSV summary writer, JSONL sample writer.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
```

Expected: all tests pass.

### Task 3: Runtime Roles

**Files:**
- Modify: `scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py`
- Modify: `test/srt/test_kv_transfer_latency_bench.py`

- [ ] **Step 1: Write failing tests for CLI construction**

Add parser tests that verify:

```python
from scripts.playground.disaggregation.kv_transfer_bench.kv_transfer_latency import build_parser

def test_parser_accepts_target_role():
    args = build_parser().parse_args([
        "--role", "target",
        "--host", "192.168.0.42",
        "--max-bytes", "1GB",
        "--target-info-file", "/tmp/target.json",
    ])
    assert args.role == "target"
    assert args.max_bytes == "1GB"

def test_parser_accepts_initiator_role():
    args = build_parser().parse_args([
        "--role", "initiator",
        "--host", "192.168.0.41",
        "--target-info-file", "/tmp/target.json",
        "--sizes", "1MB:8MB:x2",
    ])
    assert args.role == "initiator"
    assert args.sizes == "1MB:8MB:x2"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
```

Expected: parser test fails until `build_parser()` exists.

- [ ] **Step 3: Implement CLI and runtime code**

Add:

```text
run_target(args): initialize MooncakeTransferEngine, allocate torch.empty(max_bytes, dtype=torch.uint8, device=cuda:gpu_id), register pointer, write TargetInfo, sleep until interrupted.
run_initiator(args): initialize engine, allocate source tensor at max target size, register pointer, read TargetInfo, warm up, repeat transfer_sync per size, write summary CSV and sample JSONL.
main(): parse args and dispatch role.
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
python scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py --help
```

Expected: tests pass and help text prints both roles/options.

### Task 4: Docker Wrappers and Four-Node Guide

**Files:**
- Create: `scripts/playground/disaggregation/kv_transfer_bench/run_target.sh`
- Create: `scripts/playground/disaggregation/kv_transfer_bench/run_initiator.sh`
- Create: `scripts/playground/disaggregation/kv_transfer_bench/README.md`

- [ ] **Step 1: Add shell wrappers**

The wrappers use:

```bash
docker run --rm --network host --ipc host --gpus all \
  -v "${SGLANG_REPO:-/home/tiancij/sglang}:/workspace/sglang" \
  -v "${OUTPUT_DIR:-/tmp/kv-transfer-bench}:/tmp/kv-transfer-bench" \
  "${SGLANG_IMAGE:-sglang-pd-switch:tianciJ}" \
  python /workspace/sglang/scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py ...
```

- [ ] **Step 2: Add README runbook**

Document first accepted pair:

```text
target: lingjun-102, host IP from `hostname -I`
initiator: lingjun-099
```

Then document pair matrix:

```text
099 -> 100
099 -> 101
099 -> 102
100 -> 101
100 -> 102
101 -> 102
```

- [ ] **Step 3: Verify shell syntax**

Run:

```bash
bash -n scripts/playground/disaggregation/kv_transfer_bench/run_target.sh
bash -n scripts/playground/disaggregation/kv_transfer_bench/run_initiator.sh
```

Expected: no output and exit code 0.

### Task 5: Final Verification

**Files:**
- All files above.

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest test/srt/test_kv_transfer_latency_bench.py -q
```

Expected: pass.

- [ ] **Step 2: Run syntax checks**

```bash
python -m py_compile scripts/playground/disaggregation/kv_transfer_bench/kv_transfer_latency.py
bash -n scripts/playground/disaggregation/kv_transfer_bench/run_target.sh
bash -n scripts/playground/disaggregation/kv_transfer_bench/run_initiator.sh
```

Expected: all pass.

- [ ] **Step 3: Confirm Docker feasibility**

The generated commands must satisfy:

```text
uses existing image sglang-pd-switch:tianciJ
uses --network host so Mooncake session ports are reachable across nodes
uses --gpus all so each node can select GPU 0
mounts repo read-only enough for script execution
mounts /tmp/kv-transfer-bench for target JSON and result files
does not require starting full SGLang model servers
```

