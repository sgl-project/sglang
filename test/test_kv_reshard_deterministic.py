"""Deterministic-mode accuracy test for CP KV-cache resharding.

Uses --enable-deterministic-inference so that two launches of the SAME
config produce bit-exact outputs. With deterministic baseline, a strict
token-by-token comparison of baseline vs reshard becomes a definitive
correctness check.

Procedure:
  pass A : baseline #1 (cp=2, deterministic, no reshard)
  pass B : baseline #2 (cp=2, deterministic, no reshard)  -> must equal A
  pass C : reshard     (cp=2, deterministic, --enable-cp-kv-reshard)

  expected: A == B (bit-exact)   -> baseline is reproducible
            A == C (bit-exact)   -> resharding preserves accuracy
"""

from __future__ import annotations

import argparse
import json
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
    "Hello, world! Hello,",
    "Continue: a, b, c, d,",
    "The Pythagorean theorem: a^2 + b^2 =",
    "The largest planet in our solar system is",
    "Water boils at",
    "Mount Everest is located in",
    "The author of Hamlet is",
    "Speed of light in vacuum is approximately",
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
        "32",
        "--disable-piecewise-cuda-graph",
    ]
    env = os.environ.copy()
    # Disable strict idle-leak check so we can isolate the accuracy issue
    # (the reshard path has a small known leak that I'll fix separately).
    env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    if reshard:
        cmd.append("--enable-cp-kv-reshard")
        env["SGLANG_TEST_CP_KV_RESHARD_ALLOW_NULL"] = "1"
    print(f">>> launch reshard={reshard} -> {log_path}", flush=True)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_ready(timeout=300):
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
        for _ in range(30):
            if proc.poll() is not None:
                return
            time.sleep(1)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def query(prompt: str, max_new_tokens: int):
    r = requests.post(
        f"http://{HOST}:{PORT}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def run_pass(label: str, reshard: bool, max_new_tokens: int, log_dir: Path):
    print(f"\n=== pass {label} (reshard={reshard}, deterministic=True) ===")
    proc = launch(reshard, log_dir / f"det_{label}.log")
    out = []
    try:
        if not wait_ready():
            raise RuntimeError(f"[{label}] not ready; see log")
        for i, p in enumerate(PROMPTS):
            r = query(p, max_new_tokens)
            otl = r["meta_info"].get("output_token_logprobs") or []
            tids = [int(e[1]) for e in otl if e[1] is not None]
            logps = [float(e[0]) for e in otl]
            print(
                f"  [{label}] {i:>2}: text={r['text']!r:<32} "
                f"ids={tids[:8]}  logp[0]={logps[0]:.6f}"
                if logps
                else f"  [{label}] {i:>2}: empty"
            )
            out.append(
                {
                    "prompt": p,
                    "text": r["text"],
                    "token_ids": tids,
                    "logprobs": logps,
                }
            )
        return out
    finally:
        shutdown(proc)


def pairwise(x, y):
    n = len(x)
    tok_match = sum(1 for a, b in zip(x, y) if a["token_ids"] == b["token_ids"])
    text_match = sum(1 for a, b in zip(x, y) if a["text"] == b["text"])
    max_dlp = 0.0
    for a, b in zip(x, y):
        for la, lb in zip(a["logprobs"], b["logprobs"]):
            max_dlp = max(max_dlp, abs(la - lb))
    return tok_match, text_match, max_dlp, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--log-dir", default="test/kv_reshard_logs")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    A = run_pass(
        "A_base1", reshard=False, max_new_tokens=args.max_new_tokens, log_dir=log_dir
    )
    B = run_pass(
        "B_base2", reshard=False, max_new_tokens=args.max_new_tokens, log_dir=log_dir
    )
    C = run_pass(
        "C_reshd", reshard=True, max_new_tokens=args.max_new_tokens, log_dir=log_dir
    )

    print("\n=== DETERMINISTIC PAIRWISE COMPARISON ===")
    for label, x, y in [
        ("A vs B (baseline self-consistency)", A, B),
        ("A vs C (baseline #1 vs reshard)", A, C),
        ("B vs C (baseline #2 vs reshard)", B, C),
    ]:
        tm, txm, dlp, n = pairwise(x, y)
        marker = "✓" if tm == n else "✗"
        print(
            f"  {marker} {label}: tokens {tm}/{n}, text {txm}/{n}, max |Δlogp| {dlp:.6f}"
        )

    (log_dir / "deterministic_results.json").write_text(
        json.dumps({"A": A, "B": B, "C": C}, indent=2)
    )

    ab_tm, _, _, n = pairwise(A, B)
    ac_tm, _, _, _ = pairwise(A, C)
    if ab_tm != n:
        print(
            f"\n!! baseline is NOT bit-exact reproducible: {ab_tm}/{n} match. "
            "Deterministic-inference doesn't fully cover this config. "
            "Cannot conclude accuracy from strict equality."
        )
        sys.exit(2)
    if ac_tm == n:
        print(
            f"\n=== RESULT: PASS — baseline is bit-exact AND reshard matches "
            f"baseline ({ac_tm}/{n} tokens). Accuracy verified."
        )
        sys.exit(0)
    else:
        print(
            f"\n=== RESULT: FAIL — baseline reproduces but reshard diverges "
            f"({ac_tm}/{n} match). Real correctness issue."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
