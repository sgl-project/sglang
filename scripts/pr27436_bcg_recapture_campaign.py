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
import subprocess
import sys
import time
from pathlib import Path

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


def run_model(model_key: str, args) -> dict:
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
        str(args.port_start),
        "--startup-timeout",
        str(args.startup_timeout),
        "--performance-mode",
        args.performance_mode,
        "--prefer-local-cache",
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
    for i, model_key in enumerate(args.models):
        if model_key in done:
            print(f"[campaign] skip already-done {model_key}", flush=True)
            continue
        if model_key not in V.MODELS:
            print(f"[campaign] unknown model {model_key}; skipping", flush=True)
            continue
        args.port_start += 50 * i
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
