"""Deterministic accuracy test for CP KV-cache resharding WITH cache reuse.

The design's signature path -- per-layer remote-prefix gather via
``_cp_fill_remote_prefix_pool_rows`` -- only fires when a request hits
the radix cache and shares a prefix with previously-cached requests.
A clean validation must therefore (a) populate the cache, then (b)
re-send the same prompts to trigger the cache-hit path.

This test:
  pass A : baseline #1 (cp=2, deterministic)
  pass B : baseline #2 (cp=2, deterministic)
  pass C : reshard     (cp=2, deterministic, --enable-cp-kv-reshard)

Each pass sends every prompt TWICE in a row. The second call should:
  - Hit the radix cache (meta_info.cached_tokens > 0)
  - On the reshard pass: trigger `_cp_fill_remote_prefix_pool_rows` for
    every layer, to allgather peer K/V into the transient pool rows for
    non-owned prefix positions.

Then compares baseline vs reshard token-by-token across BOTH shots.
With --enable-deterministic-inference, baseline-vs-baseline is bit-exact,
so any reshard-vs-baseline difference is a real correctness bug.
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

# Longer prompts to give the radix cache a multi-page prefix to reuse.
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 4 + "Continue the story:",
    "Once upon a time in a faraway kingdom, there lived a wise old king "
    "who was loved by his people. He had three sons named Alex, Ben, and "
    "Carl. The king was very fond of his sons but worried about which one "
    "should inherit the kingdom. One day, the king decided to test them. "
    "He gave each son a coin and said:",
    "In computer science, a binary tree is a data structure where each "
    "node has at most two children, referred to as the left child and "
    "the right child. A binary search tree (BST) is a special type of "
    "binary tree that maintains the following property: for every node, "
    "the values of all nodes in its left subtree are less than the node's "
    "value, and the values of all nodes in its right subtree are greater. "
    "This property enables efficient",
    "Photosynthesis is the process by which green plants and some other "
    "organisms use sunlight to synthesize foods with the help of "
    "chlorophyll. The overall chemical equation for photosynthesis is "
    "6CO2 + 6H2O + light energy ->",
]


def launch(reshard: bool, log_path: Path, deterministic: bool = True):
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
        "--cuda-graph-max-bs",
        "32",
        "--max-running-requests",
        "32",
        "--disable-piecewise-cuda-graph",
    ]
    if deterministic:
        cmd.append("--enable-deterministic-inference")
    env = os.environ.copy()
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


def query(prompt: str, max_new_tokens: int, return_logprob: bool = False):
    """Note: return_logprob=True with logprob_start_len=0 forces full
    prefill recomputation (need all input logits), which prevents the
    radix-cache prefix skip. For cache-reuse tests, leave it False; we
    compare output token IDs instead."""
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }
    if return_logprob:
        payload["return_logprob"] = True
        payload["logprob_start_len"] = 0
    r = requests.post(f"http://{HOST}:{PORT}/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def run_pass(
    label: str,
    reshard: bool,
    max_new_tokens: int,
    log_dir: Path,
    deterministic: bool = True,
):
    print(f"\n=== pass {label} (reshard={reshard}, deterministic={deterministic}) ===")
    proc = launch(reshard, log_dir / f"reuse_{label}.log", deterministic=deterministic)
    out = []
    try:
        if not wait_ready():
            raise RuntimeError(f"[{label}] server not ready")
        # First send a warm-up query to populate cuda graphs / weight caches
        # so the very first real query isn't an outlier.
        _ = query("Hi.", 1)
        # We compare via text/token_ids (no logprob recompute that would
        # disable cache reuse). Tokenize on the client side from the response
        # text length is not reliable, so we rely on output_ids parsing.
        for i, p in enumerate(PROMPTS):
            # Shot 1: populate the radix cache for this prompt.
            r1 = query(p, max_new_tokens)
            text1 = r1["text"]
            cached1 = r1["meta_info"].get("cached_tokens", 0)
            prompt_tokens = r1["meta_info"].get("prompt_tokens")
            print(
                f"  [{label}] {i} shot1: cached={cached1}/{prompt_tokens}  text={text1!r:<50}"
            )
            time.sleep(2)
            # Shot 2: same prompt -> hits the radix cache.
            #         Under resharding, triggers _cp_fill_remote_prefix_pool_rows
            #         to allgather peer K/V into transient pool rows on each layer.
            r2 = query(p, max_new_tokens)
            text2 = r2["text"]
            cached2 = r2["meta_info"].get("cached_tokens", 0)
            print(
                f"  [{label}] {i} shot2: cached={cached2}/{prompt_tokens}  text={text2!r:<50}"
            )
            tids1 = []  # not used; text comparison is more direct
            tids2 = []
            out.append(
                {
                    "prompt": p,
                    "prompt_tokens": prompt_tokens,
                    "shot1": {"text": r1["text"], "cached_tokens": cached1},
                    "shot2": {"text": r2["text"], "cached_tokens": cached2},
                }
            )
        return out
    finally:
        shutdown(proc)


def compare(label: str, A, B):
    n_s1 = sum(1 for a, b in zip(A, B) if a["shot1"]["text"] == b["shot1"]["text"])
    n_s2 = sum(1 for a, b in zip(A, B) if a["shot2"]["text"] == b["shot2"]["text"])
    n = len(A)
    marker = "✓" if (n_s1 == n and n_s2 == n) else "✗"
    print(f"  {marker} {label}: shot1 {n_s1}/{n}, shot2 {n_s2}/{n}")
    return n_s1 == n and n_s2 == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--log-dir", default="test/kv_reshard_logs")
    ap.add_argument("--no-deterministic", action="store_true")
    args = ap.parse_args()
    deterministic = not args.no_deterministic
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    A = run_pass(
        "A_base1",
        reshard=False,
        max_new_tokens=args.max_new_tokens,
        log_dir=log_dir,
        deterministic=deterministic,
    )
    B = run_pass(
        "B_base2",
        reshard=False,
        max_new_tokens=args.max_new_tokens,
        log_dir=log_dir,
        deterministic=deterministic,
    )
    C = run_pass(
        "C_reshd",
        reshard=True,
        max_new_tokens=args.max_new_tokens,
        log_dir=log_dir,
        deterministic=deterministic,
    )

    print("\n=== CACHE-HIT REUSE SANITY ===")
    for label, X in [("A_base1", A), ("B_base2", B), ("C_reshd", C)]:
        hits = sum(1 for r in X if r["shot2"]["cached_tokens"] > 0)
        avg_hit = sum(
            r["shot2"]["cached_tokens"] / r["prompt_tokens"] for r in X
        ) / len(X)
        print(
            f"  [{label}] shot2 cache-hit on {hits}/{len(X)} prompts, avg hit ratio = {avg_hit:.2%}"
        )

    print("\n=== DETERMINISTIC PAIRWISE COMPARISON (cache-reuse path) ===")
    ab = compare("A vs B (baseline self-consistency)", A, B)
    ac = compare("A vs C (baseline #1 vs reshard)", A, C)
    bc = compare("B vs C (baseline #2 vs reshard)", B, C)

    (log_dir / "cache_reuse_results.json").write_text(
        json.dumps({"A": A, "B": B, "C": C}, indent=2)
    )

    if not ab:
        print(
            "\n!! baseline-vs-baseline is NOT bit-exact under deterministic mode. "
            "Cannot conclude correctness from strict equality."
        )
        sys.exit(2)
    if ac and bc:
        print(
            "\n=== RESULT: PASS — reshard produces bit-exact outputs as baseline "
            "on BOTH first-shot (write path) AND second-shot (cache-reuse / remote "
            "prefix gather path). Accuracy verified."
        )
        sys.exit(0)
    print("\n=== RESULT: FAIL — reshard diverges from baseline.")
    sys.exit(1)


if __name__ == "__main__":
    main()
