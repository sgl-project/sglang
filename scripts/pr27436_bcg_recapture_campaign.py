#!/usr/bin/env python3
"""Drive the BCG no-recapture validation across many models, deleting each
model's HF cache after its run so disk does not fill up.

For every model it runs scripts/pr27436_validate_diffusion_bcg_service.py in BCG
mode (server-based warmup captures all graphs; two different-length prompts must
add zero new captures), records the capture counts, then removes the model's
Hugging Face hub snapshot.

Usage:
  PYTHONPATH=python python scripts/pr27436_bcg_recapture_campaign.py \
      --models glm-image sana-1.5-1.6b ... --gpu 2 --port-start 47000 \
      --result-root /tmp/bcg_campaign
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def _port_free(port: int) -> bool:
    # No SO_REUSEADDR: match the server's strict check so a TIME_WAIT port (which
    # SO_REUSEADDR would let us bind) is correctly treated as busy.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def find_free_base_port(start: int) -> int:
    """A base port P where P, P+1000 (scheduler) and P+2000 (master) are all
    free, avoiding the contended-box port collisions the validate script's
    --strict-ports would otherwise crash on."""
    p = start
    while p + 2000 <= 65000:
        if all(_port_free(p + off) for off in (0, 1000, 2000)):
            return p
        p += 100
    raise RuntimeError("no free port window found")

ROOT = Path(__file__).resolve().parent.parent
VALIDATE = ROOT / "scripts" / "pr27436_validate_diffusion_bcg_service.py"

sys.path.insert(0, str(ROOT / "scripts"))
import pr27436_validate_diffusion_bcg_service as V  # noqa: E402


def hub_dir_for(model_path: str) -> Path:
    cache = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub",
            ),
        )
    )
    return cache / ("models--" + model_path.replace("/", "--"))


PORT_ERROR_MARKERS = (
    "is unavailable and --strict-ports",
    "EADDRINUSE",
    "address already in use",
    "DistNetworkError",
)


def _server_log_text(result_dir: Path, model_key: str) -> str:
    log = result_dir / model_key / "server.log"
    try:
        return log.read_text(errors="replace")
    except OSError:
        return ""


def run_model_once(model_key: str, args, port: int) -> dict:
    result_dir = Path(args.result_root) / model_key
    cmd = [
        sys.executable,
        str(VALIDATE),
        "--models",
        model_key,
        "--gpu-pool",
        args.gpu,
        "--result-dir",
        str(result_dir),
        "--port-start",
        str(port),
        "--startup-timeout",
        str(args.startup_timeout),
        "--performance-mode",
        args.performance_mode,
    ]
    env = dict(os.environ, PYTHONPATH="python", FLASHINFER_DISABLE_VERSION_CHECK="1")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = round(time.time() - t0, 1)

    results_file = result_dir / "results.json"
    summary = {"model": model_key, "elapsed_s": elapsed, "rc": proc.returncode}
    if results_file.exists():
        data = json.loads(results_file.read_text())
        row = data[0] if isinstance(data, list) and data else data
        for key in (
            "status",
            "captures_after_warmup",
            "captures_after_first",
            "captures_after_second",
            "reason",
            "model_path",
        ):
            if key in row:
                summary[key] = row[key]
    else:
        summary["status"] = "no_results"
    return summary


def run_model(model_key: str, args) -> dict:
    """Run a model, retrying on transient port collisions (the box is shared and
    a probed-free port can be grabbed before the server binds it)."""
    result_dir = Path(args.result_root) / model_key
    attempts = 4
    port_hint = args.port_start
    for attempt in range(1, attempts + 1):
        port = find_free_base_port(port_hint)
        summary = run_model_once(model_key, args, port)
        if summary.get("status") == "passed":
            return summary
        log = _server_log_text(result_dir, model_key)
        if not any(marker in log for marker in PORT_ERROR_MARKERS):
            return summary  # genuine (non-port) outcome; do not retry
        print(
            f"[campaign] {model_key}: port collision on {port} "
            f"(attempt {attempt}/{attempts}); retrying on a higher port",
            flush=True,
        )
        port_hint = port + 200
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--gpu", default="2", help="--gpu-pool value (e.g. 2 or 2,3)")
    parser.add_argument("--port-start", type=int, default=47000)
    parser.add_argument("--startup-timeout", type=int, default=5400)
    parser.add_argument("--performance-mode", default="speed")
    parser.add_argument("--result-root", default="/tmp/bcg_campaign")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="do not delete the HF snapshot after each model",
    )
    args = parser.parse_args()

    Path(args.result_root).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.result_root) / "campaign_summary.json"
    summaries: list[dict] = []
    if summary_path.exists():
        summaries = json.loads(summary_path.read_text())

    done = {s["model"] for s in summaries}
    for model_key in args.models:
        if model_key in done:
            print(f"[campaign] skip already-done {model_key}", flush=True)
            continue
        if model_key not in V.MODELS:
            print(f"[campaign] unknown model {model_key}; skipping", flush=True)
            continue
        print(f"[campaign] === {model_key} ===", flush=True)
        summary = run_model(model_key, args)
        summaries.append(summary)
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(f"[campaign] {model_key}: {summary}", flush=True)

        if not args.keep_cache:
            model_path = V.MODELS[model_key]["path"]
            hub = hub_dir_for(model_path)
            if hub.exists():
                shutil.rmtree(hub, ignore_errors=True)
                print(f"[campaign] deleted HF cache {hub}", flush=True)

    passed = [s for s in summaries if s.get("status") == "passed"]
    print(
        f"[campaign] DONE: {len(passed)}/{len(summaries)} passed",
        flush=True,
    )
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
