"""Prefill-only correctness test for CP KV-resharding.

With ``max_new_tokens=1``, SGLang samples the first generated token from
the prefill's last-position logits — no separate decode forward. This
isolates the PREFILL path, which under reshard is correct (each rank
has full canonical K, V via CP allgather). Decode on the same node is
out of scope for v1 (the feature is for disaggregated prefill nodes).

Sends each prompt once, compares baseline vs reshard token-by-token with
deterministic inference enabled.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

MODEL = "/home/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-30B-A3B-FP8"
HOST = "127.0.0.1"
PORT = 30000

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Continue: a, b, c, d,",
    "The Pythagorean theorem: a^2 + b^2 =",
    "The largest planet is",
]


def launch(reshard: bool, log_path: Path):
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--trust-remote-code",
        "--tp-size",
        "4",
        "--moe-dp-size",
        "1",
        "--ep-size",
        "4",
        "--attn-cp-size",
        "2",
        "--enable-prefill-context-parallel",
        "--enable-deterministic-inference",
        "--cuda-graph-max-bs",
        "32",
        "--max-running-requests",
        "1",
        "--disable-piecewise-cuda-graph",
    ]
    env = os.environ.copy()
    env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    if reshard:
        cmd.append("--enable-cp-kv-reshard")
        env["SGLANG_TEST_CP_KV_RESHARD_ALLOW_NULL"] = "1"
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_ready(timeout=240):
    url = f"http://{HOST}:{PORT}/health_generate"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"   ready after {time.time()-t0:.1f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def shutdown(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(1)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def query(prompt: str):
    r = requests.post(
        f"http://{HOST}:{PORT}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 1,
            },
            "stream": False,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def run_pass(label, reshard, log_dir):
    proc = launch(reshard, log_dir / f"prefill_only_{label}.log")
    out = []
    try:
        if not wait_ready():
            raise RuntimeError(f"[{label}] not ready")
        for i, p in enumerate(PROMPTS):
            r = query(p)
            print(f"  [{label}] {i}: text={r['text']!r}")
            out.append(r["text"])
    finally:
        shutdown(proc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="test/kv_reshard_logs")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== baseline pass ===")
    base = run_pass("baseline", False, log_dir)
    print("\n=== reshard pass ===")
    resh = run_pass("reshard", True, log_dir)

    print("\n=== prefill-only first-token comparison ===")
    matches = 0
    for i, (b, r) in enumerate(zip(base, resh)):
        ok = b == r
        matches += int(ok)
        print(
            f"  [{'OK ' if ok else 'DIFF'}] prompt {i}: "
            f"baseline={b!r}  reshard={r!r}"
        )
    print(f"\nResult: {matches}/{len(PROMPTS)} prompts match")
    sys.exit(0 if matches == len(PROMPTS) else 1)


if __name__ == "__main__":
    main()
