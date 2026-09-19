"""True end-to-end: run_sweep.main() with real subprocesses against a stub
`sglang` package (fake launch_server serving /health, fake bench_serving
writing metrics).  No GPU, no torch — but everything else is the real
production path: RealProcs popen/kill, health polling, manifest, report.
"""

import json
from pathlib import Path

import run_sweep

LAUNCH_SERVER_STUB = """
import argparse, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="")
parser.add_argument("--port", type=int, default=30000)
parser.add_argument("--dtype", default="")
parser.add_argument("--device", default="")
parser.add_argument("--attention-backend", default="")
parser.add_argument("--tp-size", type=int, default=1)
parser.add_argument("--mem-fraction-static", type=float, default=0.8)
parser.add_argument("--trust-remote-code", action="store_true")
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *a):
        pass

server = HTTPServer(("127.0.0.1", args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print("stub server up", flush=True)
while True:
    time.sleep(1)
"""

BENCH_STUB = """
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("--output-file", required=True)
parser.add_argument("--tag", default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset-name", default="")
parser.add_argument("--backend", default="")
parser.add_argument("--host", default="")
parser.add_argument("--port", type=int, default=0)
parser.add_argument("--num-prompts", type=int, default=0)
args, _ = parser.parse_known_args()

record = {
    "output_throughput": 1234.5,
    "p99_ttft_ms": 456.7,
    "p99_tpot_ms": 30.0,
    "tag": args.tag,
    "seed": args.seed,
}
with open(args.output_file, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\\n")
print("stub bench done", flush=True)
"""

CONFIG = """
name: e2e
model:
  path: Qwen/Qwen3-8B
server:
  args:
    --port: 34567
  env:
    PYTHONPATH: "{stubdir}"
workload:
  dataset_name: random
  axes:
    --max-concurrency: [2]
  num_prompts_mult: 2
sla:
  thresholds:
    p99_ttft_ms: 5000
run:
  repeats: 1
  health_timeout_s: 30
  bench_timeout_s: 60
  hbm_timeout_s: 5
  hbm_poll_s: 0.5
"""


def _make_stub_pkg(tmp_path: Path) -> Path:
    stubdir = tmp_path / "stubsite"
    pkg = stubdir / "sglang"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "launch_server.py").write_text(LAUNCH_SERVER_STUB, encoding="utf-8")
    (pkg / "bench_serving.py").write_text(BENCH_STUB, encoding="utf-8")
    return stubdir


def _write_cfg(tmp_path: Path, stubdir: Path) -> Path:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        CONFIG.replace("{stubdir}", str(stubdir).replace("\\", "/")),
        encoding="utf-8",
    )
    return cfg_path


def test_end_to_end_with_stub_server_and_bench(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sweep, "capture", lambda *a, **k: {"git_sha": "x"})
    stubdir = _make_stub_pkg(tmp_path)
    cfg = _write_cfg(tmp_path, stubdir)
    workdir = tmp_path / "runs"

    code = run_sweep.main(
        [
            "--config",
            str(cfg),
            "--workdir",
            str(workdir),
            "--run-id",
            "e2e",
            "--skip-hbm-gate",
        ]
    )
    run_dir = workdir / "e2e"
    assert code == 0, (
        open(run_dir / "cells" / "qwen3-8b__s00__w00__r0" / "server.log").read()
        if (workdir / "e2e").exists()
        else "no run dir"
    )

    manifest_lines = (
        (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    )
    statuses = [
        json.loads(line)["status"]
        for line in manifest_lines
        if line.strip() and "status" in json.loads(line)
    ]
    assert statuses[0] == "launching"
    assert "healthy" in statuses and "benching" in statuses
    assert statuses[-1] == "done"

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["done"] == 1
    assert report["ranked"][0]["mean"]["output_throughput"] == 1234.5

    # resume with the same run-id: no new bench output, exit still 0
    bench_count_before = _count_benches(run_dir)
    code = run_sweep.main(
        [
            "--config",
            str(cfg),
            "--workdir",
            str(workdir),
            "--run-id",
            "e2e",
            "--skip-hbm-gate",
        ]
    )
    assert code == 0
    assert _count_benches(run_dir) == bench_count_before


def _count_benches(run_dir: Path) -> int:
    cell = run_dir / "cells" / "qwen3-8b__s00__w00__r0" / "bench.jsonl"
    return len(cell.read_text(encoding="utf-8").splitlines())
