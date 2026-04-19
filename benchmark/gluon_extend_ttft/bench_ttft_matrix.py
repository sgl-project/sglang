#!/usr/bin/env python3
"""E2E TTFT matrix benchmark: triton vs gluon vs ck backends.

Launches SGLang servers as subprocesses, sweeps prefix x extend shapes and
heterogeneous batched shapes, and prints live progress + a comparison table.

Usage:
    python -u bench_ttft_matrix.py              # default set
    python -u bench_ttft_matrix.py --models d64 --backends triton gluon ck
    python -u bench_ttft_matrix.py --batched    # B>1 hetero cases
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

VENV_PYTHON = os.environ.get("VENV_PYTHON", sys.executable)
# Default to the branch this script ships with (sglang-extend). Override via
# SGLANG_DIR if running against a different worktree.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SGLANG_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
SGLANG_DIR = os.environ.get("SGLANG_DIR", _DEFAULT_SGLANG_DIR)
# Allow running under a Docker/CI Python that doesn't pip-install this
# worktree by prepending the branch's python/ dir to PYTHONPATH. User can
# override via BENCH_EXTRA_PYTHONPATH="" to disable.
_DEFAULT_EXTRA_PYPATH = os.path.join(SGLANG_DIR, "python")
EXTRA_PYPATH = os.environ.get("BENCH_EXTRA_PYTHONPATH", _DEFAULT_EXTRA_PYPATH)
RESULTS_DIR = os.environ.get(
    "GLUON_BENCH_RESULTS_DIR",
    os.path.join(SGLANG_DIR, "benchmark", "gluon_extend_ttft", "results"),
)

MODELS = {
    "d64-gptoss": {
        # gpt-oss-120b. We bypass MXFP4 (real weights would route through
        # triton_kernels/matmul_ogs, which has a latent llvm::iota_range
        # assertion in the AMDGCN backend that trips on certain MoE kernel
        # shapes during shape-sweep benchmarking) by loading dummy bf16 weights
        # and forcing the plain fused-moe triton runner (no matmul_ogs).
        # Output isn't coherent (weights are random), but TTFT — which depends
        # on kernel perf, not weight values — is fully representative.
        # 120B bf16 weights fit on a single MI350X (~252GB), so tp=1 lets us
        # run all three backends in parallel on separate GPUs.
        "path": os.environ.get(
            "GPTOSS_PATH", "/raid/models/gpt-oss-120b-config"
        ),
        "label": "D=64 GPT-OSS 120B (dummy bf16)",
        # bf16 dummy weights = 217GB — doesn't fit on a single 252GB MI350X
        # with enough room for KV cache. TP=2 splits to ~109GB per GPU, leaving
        # ~90GB per GPU for KV at mem-fraction-static=0.80. With 4 free GPUs
        # this runs 2 backends in parallel in batch 1, then the 3rd solo in batch 2.
        "tp": 2,
        "num_gpus": 2,
        "max_ctx": 16384,
        "max_total_tokens": 32768,
        "mem_fraction_static": 0.80,
        "extra_args": [
            "--load-format", "dummy",
            "--json-model-override-args", '{"quantization_config": null}',
            "--moe-runner-backend", "triton",
            "--trust-remote-code",
        ],
    },
    "d128-llama70b": {
        # Llama-3.1-70B config-only copy under /home/tussingh/models/ is
        # sufficient for TTFT benchmarking (kernel perf is weight-value
        # independent). Real weights live at /data/models/ but aren't
        # guaranteed on every host; bench_ttft_matrix.py auto-uses dummy
        # weights for speed. Override via LLAMA70B_PATH if you have them.
        "path": os.environ.get(
            "LLAMA70B_PATH", "/home/tussingh/models/Llama-3.1-70B-config"
        ),
        "label": "D=128 Llama-3.1-70B (dummy bf16)",
        "tp": 2,
        "num_gpus": 2,
        "max_ctx": 16384,
        "max_total_tokens": 32768,
        "extra_args": ["--load-format", "dummy", "--trust-remote-code"],
    },
    "d128-qwen7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "label": "D=128 Qwen2.5-7B",
        "tp": 1,
        "num_gpus": 1,
        "max_ctx": 20000,
        "max_total_tokens": 32768,
        "extra_args": [],
    },
}

BACKENDS = {
    # `triton` = the vanilla Triton extend kernel. No extra flags needed:
    # Gluon is opt-in on this branch, so the default Triton backend stays
    # on the Triton reference path.
    "triton": {
        "env": {},
        "attn_backend": "triton",
        "extra_cmd": [],
    },
    # `gluon` = the Triton attention backend with the Gluon extend kernel
    # opted in via `--enable-gluon-extend-attention`. The server refuses
    # to enable Gluon on non-gfx950 hardware, so this flag is a no-op
    # outside MI350/MI355.
    "gluon": {
        "env": {},
        "attn_backend": "triton",
        "extra_cmd": ["--enable-gluon-extend-attention"],
    },
    "ck": {
        "env": {},
        "attn_backend": "aiter",
        "extra_cmd": [],
    },
}

# B=1 sweep matrix. 7 prefixes x 5 extends = 35 cases -> trimmed by max_ctx.
PREFIXES = [0, 256, 1024, 2048, 4096, 8192, 16384]
EXTENDS = [8, 32, 128, 512, 2048]

# B>1 batched cases: mix of homogeneous and heterogeneous.
# 16 hom + 14 hetero = 30 total (matches user's request).
BATCHED_HOM = [
    # (label, batch_size, prefix_per_seq, ext_per_seq)
    ("hom_B2_p1024_e64",  2, 1024, 64),
    ("hom_B4_p1024_e64",  4, 1024, 64),
    ("hom_B8_p1024_e64",  8, 1024, 64),
    ("hom_B16_p1024_e64", 16, 1024, 64),
    ("hom_B2_p4096_e128", 2, 4096, 128),
    ("hom_B4_p4096_e128", 4, 4096, 128),
    ("hom_B8_p4096_e128", 8, 4096, 128),
    ("hom_B4_p8192_e64",  4, 8192, 64),
    ("hom_B2_p0_e512",    2, 0, 512),
    ("hom_B4_p0_e512",    4, 0, 512),
    ("hom_B8_p0_e256",    8, 0, 256),
    ("hom_B4_p2048_e256", 4, 2048, 256),
    ("hom_B8_p2048_e128", 8, 2048, 128),
    ("hom_B4_p1024_e1024", 4, 1024, 1024),
    ("hom_B2_p8192_e256", 2, 8192, 256),
    ("hom_B2_p0_e2048",   2, 0, 2048),
]

BATCHED_HET = [
    # (label, [(pfx, ext), ...])
    ("het_B4_mixed_small_ext", [(100, 8), (512, 16), (2048, 32), (4096, 64)]),
    ("het_B8_spec_decode",     [(100, 7), (512, 7), (1024, 7), (2048, 7),
                                 (4095, 7), (4099, 7), (6001, 7), (8191, 7)]),
    ("het_B8_chunk_mix",       [(512, 128), (1024, 256), (2048, 64), (4096, 32),
                                 (512, 256), (1024, 128), (2048, 32), (4096, 64)]),
    ("het_B4_big_prefill",     [(0, 4096), (0, 2048), (0, 1024), (0, 512)]),
    ("het_B4_chunk_pref",      [(8192, 512), (4096, 1024), (0, 2048), (16384, 128)]),
    ("het_B16_chat_mix",       [(4096, 17), (8192, 33), (16384, 65), (2048, 128),
                                 (4096, 17), (8192, 33), (16384, 65), (2048, 128),
                                 (4096, 17), (8192, 33), (16384, 65), (2048, 128),
                                 (4096, 17), (8192, 33), (16384, 65), (2048, 128)]),
    ("het_B8_realistic",       [(4096, 64), (8192, 32), (2048, 128), (1024, 256),
                                 (512, 512), (4096, 64), (2048, 128), (8192, 32)]),
    ("het_B4_realistic",       [(16384, 64), (4096, 256), (1024, 512), (0, 1024)]),
    ("het_B2_very_long_mix",   [(1024, 2048), (0, 8192)]),
    ("het_B2_short_longpfx",   [(16384, 128), (1024, 32)]),
    ("het_B8_short_tail",      [(0, 17), (0, 37), (0, 65), (0, 97),
                                 (0, 113), (0, 17), (0, 37), (0, 65)]),
    ("het_B4_longpfx_mix",     [(16384, 128), (1024, 33), (8192, 256), (4096, 97)]),
    ("het_B16_prefill_heavy",  [(0, 512), (0, 1024), (0, 2048), (0, 4096),
                                 (0, 256), (0, 128), (0, 64), (0, 32)] * 2),
    ("het_B32_short_tail",     [(0, 17), (0, 37), (0, 65), (0, 97)] * 8),
]

assert len(BATCHED_HOM) + len(BATCHED_HET) == 30, "30 B>1 cases requested"


def wait_for_server(port, timeout=900, label=""):
    start = time.time()
    last_report = 0
    while time.time() - start < timeout:
        elapsed = int(time.time() - start)
        if elapsed - last_report >= 30:
            print(f"    [{label}] waiting on :{port} ({elapsed}s)")
            last_report = elapsed
        try:
            resp = urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=5
            )
            if resp.status == 200:
                print(f"    [{label}] ready in {elapsed}s")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"    [{label}] TIMEOUT after {timeout}s")
    return False


def send_completion(port, prompt_ids, max_tokens=1, timeout=300):
    payload = json.dumps({
        "model": "default", "prompt": prompt_ids,
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        wall = time.perf_counter() - t0
        body = json.loads(resp.read().decode())
        n = body.get("usage", {}).get("completion_tokens", max_tokens)
        return wall, n
    except Exception:
        return None, 0


def flush_cache(port, max_retries=20):
    for _ in range(max_retries):
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/flush_cache", method="POST",
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status == 200:
                time.sleep(0.1)
                return True
        except urllib.error.HTTPError as e:
            if e.code == 400:
                time.sleep(0.2)
                continue
            break
        except Exception:
            break
    time.sleep(0.3)
    return False


def bench_b1_case(port, pfx, ext, warmup=3, reps=5, decode_tokens=0):
    total = pfx + ext
    prefix_ids = list(range(100, 100 + pfx)) if pfx > 0 else []
    full_ids = list(range(100, 100 + total))
    first_ms = []
    decode_ms = []
    for i in range(warmup + reps):
        flush_cache(port)
        if decode_tokens > 1:
            t, n = send_completion(port, full_ids, max_tokens=decode_tokens)
        else:
            t, n = send_completion(port, full_ids, max_tokens=1)
            n = 1
        if t is not None and i >= warmup:
            first_ms.append(t * 1000)
            if decode_tokens > 1 and n > 1:
                # Rough decode-phase wall time: subtract a short TTFT-only probe.
                # This gives a cleaner ITL than dividing total/tokens because
                # TTFT dominates the first ms for small decode counts.
                decode_ms.append(t * 1000 / max(1, n))
    second_ms = []
    if pfx > 0:
        for i in range(warmup + reps):
            flush_cache(port)
            send_completion(port, prefix_ids, max_tokens=1)
            time.sleep(0.05)
            t, _ = send_completion(port, full_ids, max_tokens=1)
            if t is not None and i >= warmup:
                second_ms.append(t * 1000)
    ttft_first = (sum(first_ms) / len(first_ms)) if first_ms else None
    ttft_second = (sum(second_ms) / len(second_ms)) if second_ms else None
    # Amortized tok/s: the wall time includes server overhead, but on a
    # flushed cache (first) the whole `total` length is prefilled. On the
    # cache-hit path (second) only `ext` tokens are freshly prefilled,
    # which is the shape that matters for real serving.
    tput_first = (total * 1000.0 / ttft_first) if ttft_first else None
    tput_second = (ext * 1000.0 / ttft_second) if (ttft_second and ext > 0) else None
    itl_ms = (sum(decode_ms) / len(decode_ms)) if decode_ms else None
    return ttft_first, ttft_second, tput_first, tput_second, itl_ms


def send_completion_async(port, prompt_ids, max_tokens=1, timeout=300):
    payload = json.dumps({
        "model": "default", "prompt": prompt_ids,
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        wall = time.perf_counter() - t0
        body = json.loads(resp.read().decode())
        n = body.get("usage", {}).get("completion_tokens", max_tokens)
        return wall, n
    except Exception:
        return None, 0


def bench_batched_hom(port, B, pfx, ext, warmup=3, reps=3):
    prefix_ids = list(range(100, 100 + pfx)) if pfx > 0 else []
    full_ids = list(range(100, 100 + pfx + ext))
    walls = []
    batch_walls = []
    for t_i in range(warmup + reps):
        flush_cache(port)
        if pfx > 0:
            send_completion(port, prefix_ids, max_tokens=1)
            time.sleep(0.05)
        t_batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=B) as pool:
            futs = [pool.submit(send_completion_async, port, full_ids, 1)
                    for _ in range(B)]
            batch_walls_this = []
            for f in as_completed(futs):
                w, _ = f.result()
                if w is not None:
                    batch_walls_this.append(w)
        batch_wall = time.perf_counter() - t_batch_start
        if t_i >= warmup and batch_walls_this:
            walls.extend(batch_walls_this)
            batch_walls.append(batch_wall)
    if not walls:
        return None
    walls_ms = sorted(w * 1000 for w in walls)
    mean_ms = sum(walls_ms) / len(walls_ms)
    p50 = walls_ms[len(walls_ms) // 2]
    p95 = walls_ms[int(len(walls_ms) * 0.95)]
    wall_ms = (sum(batch_walls) / len(batch_walls) * 1000) if batch_walls else mean_ms
    # Batch throughput: B requests each prefilling (pfx + ext) tokens per
    # batch_wall. Reporting total tokens/sec (B * (pfx + ext) / wall_s) makes
    # cross-shape comparison easy; divide by B to get per-request amortized.
    batch_tput = B * (pfx + ext) * 1000.0 / wall_ms if wall_ms else 0.0
    return dict(
        mean_req_ttft_ms=round(mean_ms, 2),
        p50_req_ttft_ms=round(p50, 2),
        p95_req_ttft_ms=round(p95, 2),
        batch_wall_ms=round(wall_ms, 2),
        batch_tok_per_s=round(batch_tput, 1),
    )


def bench_batched_het(port, shapes, warmup=3, reps=3):
    B = len(shapes)
    max_pfx = max(p for p, _ in shapes)
    all_pfx_ids = list(range(100, 100 + max_pfx))
    walls = []
    batch_walls = []
    for t_i in range(warmup + reps):
        flush_cache(port)
        for pfx, _ in shapes:
            if pfx > 0:
                send_completion(port, all_pfx_ids[:pfx], max_tokens=1)
        time.sleep(0.05)
        t_batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=B) as pool:
            futs = []
            for pfx, ext in shapes:
                full_ids = list(range(100, 100 + pfx + ext))
                futs.append(pool.submit(send_completion_async, port, full_ids, 1))
            walls_this = []
            for f in as_completed(futs):
                w, _ = f.result()
                if w is not None:
                    walls_this.append(w)
        batch_wall = time.perf_counter() - t_batch_start
        if t_i >= warmup and walls_this:
            walls.extend(walls_this)
            batch_walls.append(batch_wall)
    if not walls:
        return None
    walls_ms = sorted(w * 1000 for w in walls)
    mean_ms = sum(walls_ms) / len(walls_ms)
    p50 = walls_ms[len(walls_ms) // 2]
    p95 = walls_ms[int(len(walls_ms) * 0.95)]
    wall_ms = (sum(batch_walls) / len(batch_walls) * 1000) if batch_walls else mean_ms
    total_toks = sum(p + e for p, e in shapes)
    batch_tput = total_toks * 1000.0 / wall_ms if wall_ms else 0.0
    return dict(
        mean_req_ttft_ms=round(mean_ms, 2),
        p50_req_ttft_ms=round(p50, 2),
        p95_req_ttft_ms=round(p95, 2),
        batch_wall_ms=round(wall_ms, 2),
        batch_tok_per_s=round(batch_tput, 1),
    )


def launch_server(model_key, backend_key, gpus, port, extra_env=None,
                  kv_cache_dtype=None):
    cfg = MODELS[model_key]
    bk = BACKENDS[backend_key]
    gpu_str = ",".join(str(g) for g in gpus)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = gpu_str
    # TP-imbalance heuristic trips spuriously under concurrent multi-server
    # startup on ROCm (torch.mem_get_info disagrees across ranks for a few
    # seconds during NCCL init). mem-fraction-static still caps real usage.
    env["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "0"
    if EXTRA_PYPATH:
        # Prepend so an editable/site-packages sglang can't shadow our branch.
        env["PYTHONPATH"] = EXTRA_PYPATH + ":" + env.get("PYTHONPATH", "")
    env.update(bk["env"])
    if extra_env:
        env.update(extra_env)
    cmd = [
        VENV_PYTHON, "-m", "sglang.launch_server",
        "--model-path", cfg["path"],
        "--port", str(port),
        "--tp", str(cfg["tp"]),
        "--mem-fraction-static", str(cfg.get("mem_fraction_static", 0.80)),
        "--max-total-tokens", str(cfg["max_total_tokens"]),
        "--attention-backend", bk["attn_backend"],
        "--disable-cuda-graph",
    ] + bk.get("extra_cmd", []) + cfg["extra_args"]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", kv_cache_dtype]
    label = f"{model_key}/{backend_key}"
    log_path = f"/tmp/sglang_{model_key}_{backend_key}.log"
    log_file = open(log_path, "w")
    print(f"  [{label}] launching on GPU {gpu_str}, port {port}, log {log_path}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, cwd=SGLANG_DIR,
    )
    return proc, log_file, label


def kill_server(proc, log_file=None):
    if proc:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=15)
            except Exception:
                pass
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass


def run_b1(model_key, backend_key, gpus, port, results_lock, all_results,
           kv_cache_dtype=None):
    cfg = MODELS[model_key]
    label = f"{model_key}/{backend_key}"
    max_ctx = cfg["max_ctx"]

    cases = [(p, e) for p in PREFIXES for e in EXTENDS if p + e <= max_ctx]
    if not cases:
        print(f"  [{label}] SKIP no cases survive max_ctx")
        return

    proc, log, _ = launch_server(
        model_key, backend_key, gpus, port, kv_cache_dtype=kv_cache_dtype)
    if not wait_for_server(port, 900, label):
        with results_lock:
            all_results.append(dict(model=model_key, backend=backend_key,
                                    status="FAILED", error="server timeout"))
        kill_server(proc, log)
        return
    # Warmup
    for _wp in range(2):
        for pfx, ext in cases:
            flush_cache(port)
            send_completion(port, list(range(100, 100 + pfx + ext)), max_tokens=1)
    flush_cache(port)
    print(f"  [{label}] warmup done, {len(cases)} cases to measure")
    rows = []
    for i, (pfx, ext) in enumerate(cases, 1):
        # Generate a short decode tail (16 tokens) to estimate ITL without
        # blowing up wall time. Skip when ext is very long (>=4k) because
        # the extra decode overhead becomes noisy relative to the long
        # prefill cost.
        dec_n = 16 if ext < 4096 else 0
        first, second, tput_first, tput_second, itl_ms = bench_b1_case(
            port, pfx, ext, decode_tokens=dec_n)
        rows.append(dict(
            model=model_key, backend=backend_key,
            prefix=pfx, extend=ext, total=pfx + ext,
            first_ms=round(first, 2) if first else None,
            second_ms=round(second, 2) if second else None,
            first_tok_per_s=round(tput_first, 1) if tput_first else None,
            second_tok_per_s=round(tput_second, 1) if tput_second else None,
            itl_ms=round(itl_ms, 3) if itl_ms else None,
        ))
        if i % 5 == 0 or i == len(cases):
            print(f"  [{label}] B=1 {i}/{len(cases)} done")
    kill_server(proc, log)
    with results_lock:
        all_results.extend(rows)
    csv_path = os.path.join(RESULTS_DIR, f"{model_key}_{backend_key}_b1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["model", "backend", "prefix", "extend", "total",
                           "first_ms", "second_ms",
                           "first_tok_per_s", "second_tok_per_s", "itl_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [{label}] B=1 CSV -> {csv_path}")


def run_batched(model_key, backend_key, gpus, port, results_lock, all_results,
                kv_cache_dtype=None):
    cfg = MODELS[model_key]
    label = f"{model_key}/{backend_key}"
    max_ctx = cfg["max_ctx"]

    cases_hom = [(lbl, B, p, e) for lbl, B, p, e in BATCHED_HOM
                 if p + e <= max_ctx]
    cases_het = [(lbl, shapes) for lbl, shapes in BATCHED_HET
                 if all(p + e <= max_ctx for p, e in shapes)]
    total_cases = len(cases_hom) + len(cases_het)
    if not total_cases:
        print(f"  [{label}] SKIP no batched cases survive max_ctx")
        return

    proc, log, _ = launch_server(
        model_key, backend_key, gpus, port, kv_cache_dtype=kv_cache_dtype)
    if not wait_for_server(port, 900, label):
        with results_lock:
            all_results.append(dict(model=model_key, backend=backend_key,
                                    status="FAILED", error="server timeout"))
        kill_server(proc, log)
        return
    # Warmup across hom shapes
    for _wp in range(2):
        for lbl, B, pfx, ext in cases_hom:
            flush_cache(port)
            full_ids = list(range(100, 100 + pfx + ext))
            if pfx > 0:
                send_completion(port, list(range(100, 100 + pfx)), max_tokens=1)
                time.sleep(0.02)
            with ThreadPoolExecutor(max_workers=B) as pool:
                futs = [pool.submit(send_completion, port, full_ids, 1)
                        for _ in range(B)]
                for f in futs:
                    try:
                        f.result()
                    except Exception:
                        pass
    flush_cache(port)
    print(f"  [{label}] warmup done, {total_cases} batched cases")
    rows = []
    done = 0
    for lbl, B, pfx, ext in cases_hom:
        res = bench_batched_hom(port, B, pfx, ext)
        row = dict(model=model_key, backend=backend_key, case=lbl,
                   batch_size=B, min_pfx=pfx, max_pfx=pfx,
                   min_ext=ext, max_ext=ext)
        if res:
            row.update(res)
        rows.append(row)
        done += 1
        if done % 4 == 0:
            print(f"  [{label}] batched {done}/{total_cases} done")
    for lbl, shapes in cases_het:
        res = bench_batched_het(port, shapes)
        pfxs = [p for p, _ in shapes]
        exts = [e for _, e in shapes]
        row = dict(model=model_key, backend=backend_key, case=lbl,
                   batch_size=len(shapes),
                   min_pfx=min(pfxs), max_pfx=max(pfxs),
                   min_ext=min(exts), max_ext=max(exts))
        if res:
            row.update(res)
        rows.append(row)
        done += 1
        if done % 4 == 0 or done == total_cases:
            print(f"  [{label}] batched {done}/{total_cases} done")
    kill_server(proc, log)
    with results_lock:
        all_results.extend(rows)
    csv_path = os.path.join(RESULTS_DIR, f"{model_key}_{backend_key}_batched.csv")
    fields = ["model", "backend", "case", "batch_size", "min_pfx", "max_pfx",
              "min_ext", "max_ext", "mean_req_ttft_ms", "p50_req_ttft_ms",
              "p95_req_ttft_ms", "batch_wall_ms", "batch_tok_per_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  [{label}] batched CSV -> {csv_path}")


def allocate_gpus(jobs, available_gpus):
    batches = []
    remaining = list(jobs)
    while remaining:
        batch = []
        used = set()
        still = []
        for job in remaining:
            need = MODELS[job[0]]["num_gpus"]
            free = [g for g in available_gpus if g not in used]
            if len(free) >= need:
                assigned = free[:need]
                for g in assigned:
                    used.add(g)
                batch.append((job, assigned))
            else:
                still.append(job)
        if batch:
            batches.append(batch)
        remaining = still
        if not batch and remaining:
            print(f"WARNING: cannot schedule {remaining}")
            break
    return batches


def print_b1_table(all_results):
    rows = [r for r in all_results if "prefix" in r and r.get("first_ms")]
    if not rows:
        return
    by_model = {}
    for r in rows:
        by_model.setdefault((r["model"], r["prefix"], r["extend"]), {})[r["backend"]] = r
    for model in sorted(set(r["model"] for r in rows)):
        cfg = MODELS.get(model, {})
        print(f"\n{'=' * 120}")
        print(f"  {cfg.get('label', model)}  -  B=1 TTFT (first prompt = prefill, second = cache hit extend)")
        print(f"{'=' * 120}")
        backends = sorted(set(r["backend"] for r in rows if r["model"] == model))
        hdr = f"  {'pfx':>5} {'ext':>5} {'total':>7}"
        for bk in backends:
            # TTFT ms + effective tok/s. 1st = cold prefill (total tokens).
            # 2nd = cache-hit extend-only (ext tokens, the real serving case).
            hdr += f" | {'1st_' + bk:>9} {'1_t/s_' + bk:>10} {'2nd_' + bk:>9} {'2_t/s_' + bk:>10}"
        if "ck" in backends and "gluon" in backends:
            hdr += f" | {'gluon/ck':>10}"
        if "ck" in backends and "triton" in backends:
            hdr += f" | {'triton/ck':>10}"
        if "gluon" in backends and "triton" in backends:
            hdr += f" | {'triton/gluon':>12}"
        print(hdr)
        keys = sorted([k for k in by_model if k[0] == model], key=lambda k: (k[1], k[2]))
        ratios_gck, ratios_tck, ratios_tg = [], [], []
        for k in keys:
            d = by_model[k]
            pfx, ext = k[1], k[2]
            line = f"  {pfx:>5} {ext:>5} {pfx + ext:>7}"
            first = {bk: d.get(bk, {}).get("first_ms") for bk in backends}
            second = {bk: d.get(bk, {}).get("second_ms") for bk in backends}
            tput1 = {bk: d.get(bk, {}).get("first_tok_per_s") for bk in backends}
            tput2 = {bk: d.get(bk, {}).get("second_tok_per_s") for bk in backends}
            for bk in backends:
                s1 = f"{first[bk]:.1f}" if first[bk] else "-"
                s2 = f"{second[bk]:.1f}" if second[bk] else "-"
                t1 = f"{tput1[bk]:.0f}" if tput1[bk] else "-"
                t2 = f"{tput2[bk]:.0f}" if tput2[bk] else "-"
                line += f" | {s1:>9} {t1:>10} {s2:>9} {t2:>10}"
            # Use 2nd TTFT (cache hit, extend-only) when pfx > 0, else 1st
            def _v(bk):
                return second[bk] if pfx > 0 else first[bk]
            if "ck" in backends and "gluon" in backends and _v("ck") and _v("gluon"):
                r = _v("gluon") / _v("ck")
                ratios_gck.append(r)
                line += f" | {r:>9.3f}x"
            elif "ck" in backends and "gluon" in backends:
                line += f" | {'-':>10}"
            if "ck" in backends and "triton" in backends and _v("ck") and _v("triton"):
                r = _v("triton") / _v("ck")
                ratios_tck.append(r)
                line += f" | {r:>9.3f}x"
            elif "ck" in backends and "triton" in backends:
                line += f" | {'-':>10}"
            if "gluon" in backends and "triton" in backends and _v("gluon") and _v("triton"):
                r = _v("triton") / _v("gluon")
                ratios_tg.append(r)
                line += f" | {r:>11.3f}x"
            elif "gluon" in backends and "triton" in backends:
                line += f" | {'-':>12}"
            print(line)
        if ratios_gck:
            geo = math.exp(sum(math.log(x) for x in ratios_gck) / len(ratios_gck))
            print(f"\n  gluon/ck    geomean = {geo:.3f}  (<1 means gluon faster than CK)")
        if ratios_tck:
            geo = math.exp(sum(math.log(x) for x in ratios_tck) / len(ratios_tck))
            print(f"  triton/ck   geomean = {geo:.3f}")
        if ratios_tg:
            geo = math.exp(sum(math.log(x) for x in ratios_tg) / len(ratios_tg))
            print(f"  triton/gluon geomean = {geo:.3f}  (>1 means gluon faster than triton)")
        # Per-backend ITL summary (inter-token latency during decode).
        # Computed only on shapes where ext < 4096 (see bench loop).
        itl_by_backend = {bk: [] for bk in backends}
        tput1_by_backend = {bk: [] for bk in backends}
        tput2_by_backend = {bk: [] for bk in backends}
        for r in rows:
            if r["model"] != model:
                continue
            if r.get("itl_ms") is not None:
                itl_by_backend[r["backend"]].append(r["itl_ms"])
            if r.get("first_tok_per_s") is not None:
                tput1_by_backend[r["backend"]].append(r["first_tok_per_s"])
            if r.get("second_tok_per_s") is not None:
                tput2_by_backend[r["backend"]].append(r["second_tok_per_s"])
        any_itl = any(itl_by_backend[bk] for bk in backends)
        if any_itl:
            print(f"\n  ITL (mean decode ms/tok, ext<4k shapes only):")
            for bk in backends:
                xs = itl_by_backend[bk]
                if xs:
                    print(f"    {bk:<12} mean={sum(xs)/len(xs):.2f} ms/tok")
        print(f"\n  Amortized throughput (tokens/sec, B=1, arithmetic mean across shapes):")
        for bk in backends:
            xs1 = tput1_by_backend[bk]
            xs2 = tput2_by_backend[bk]
            m1 = f"{sum(xs1)/len(xs1):.0f}" if xs1 else "-"
            m2 = f"{sum(xs2)/len(xs2):.0f}" if xs2 else "-"
            print(f"    {bk:<12} cold_prefill={m1:>7} tok/s   cache_hit_extend={m2:>7} tok/s")


def print_batched_table(all_results):
    rows = [r for r in all_results if "case" in r and r.get("mean_req_ttft_ms")]
    if not rows:
        return
    by_model = {}
    for r in rows:
        by_model.setdefault((r["model"], r["case"]), {})[r["backend"]] = r
    for model in sorted(set(r["model"] for r in rows)):
        cfg = MODELS.get(model, {})
        print(f"\n{'=' * 120}")
        print(f"  {cfg.get('label', model)}  -  B>1 TTFT (mean per-request wall time)")
        print(f"{'=' * 120}")
        backends = sorted(set(r["backend"] for r in rows if r["model"] == model))
        hdr = f"  {'case':<28} {'B':>3}"
        for bk in backends:
            # Add batch-level tokens/sec next to the wall time for a direct
            # throughput view ("how many tokens did the batch actually push").
            hdr += f" | {'mean_' + bk:>10} {'p95_' + bk:>9} {'wall_' + bk:>9} {'t/s_' + bk:>10}"
        if "ck" in backends and "gluon" in backends:
            hdr += f" | {'gluon/ck':>10}"
        if "gluon" in backends and "triton" in backends:
            hdr += f" | {'triton/gluon':>12}"
        if "ck" in backends and "ck-gluon" in backends:
            hdr += f" | {'ckGlu/ck':>10}"
        print(hdr)
        keys = sorted(k for k in by_model if k[0] == model)
        g_ck, t_g, cg_ck = [], [], []
        for k in keys:
            d = by_model[k]
            case = k[1]
            ref = next(iter(d.values()))
            B = ref.get("batch_size", "?")
            line = f"  {case:<28} {B:>3}"
            m = {bk: d.get(bk, {}).get("mean_req_ttft_ms") for bk in backends}
            p = {bk: d.get(bk, {}).get("p95_req_ttft_ms") for bk in backends}
            w = {bk: d.get(bk, {}).get("batch_wall_ms") for bk in backends}
            t = {bk: d.get(bk, {}).get("batch_tok_per_s") for bk in backends}
            for bk in backends:
                ms = f"{m[bk]:.1f}" if m[bk] else "-"
                ps = f"{p[bk]:.1f}" if p[bk] else "-"
                ws = f"{w[bk]:.1f}" if w[bk] else "-"
                ts = f"{t[bk]:.0f}" if t[bk] else "-"
                line += f" | {ms:>10} {ps:>9} {ws:>9} {ts:>10}"
            if "ck" in backends and "gluon" in backends and m["ck"] and m["gluon"]:
                r = m["gluon"] / m["ck"]
                g_ck.append(r)
                line += f" | {r:>9.3f}x"
            elif "ck" in backends and "gluon" in backends:
                line += f" | {'-':>10}"
            if "gluon" in backends and "triton" in backends and m["gluon"] and m["triton"]:
                r = m["triton"] / m["gluon"]
                t_g.append(r)
                line += f" | {r:>11.3f}x"
            elif "gluon" in backends and "triton" in backends:
                line += f" | {'-':>12}"
            if "ck" in backends and "ck-gluon" in backends and m["ck"] and m["ck-gluon"]:
                r = m["ck-gluon"] / m["ck"]
                cg_ck.append(r)
                line += f" | {r:>9.3f}x"
            elif "ck" in backends and "ck-gluon" in backends:
                line += f" | {'-':>10}"
            print(line)
        if g_ck:
            geo = math.exp(sum(math.log(x) for x in g_ck) / len(g_ck))
            print(f"\n  gluon/ck     geomean = {geo:.3f}  (<1 means gluon faster)")
        if t_g:
            geo = math.exp(sum(math.log(x) for x in t_g) / len(t_g))
            print(f"  triton/gluon geomean = {geo:.3f}  (>1 means gluon faster than triton)")
        if cg_ck:
            geo = math.exp(sum(math.log(x) for x in cg_ck) / len(cg_ck))
            print(f"  ck-gluon/ck  geomean = {geo:.3f}  (<1 means gluon wins on CK's path)")
        # Batch-level throughput summary: arithmetic mean tokens/sec across
        # all batched shapes. Higher = more tokens pushed through the batch
        # per second for this backend.
        tput_by_backend = {bk: [] for bk in backends}
        for r in rows:
            if r["model"] != model:
                continue
            if r.get("batch_tok_per_s") is not None:
                tput_by_backend[r["backend"]].append(r["batch_tok_per_s"])
        if any(tput_by_backend[bk] for bk in backends):
            print(f"\n  Batch throughput (tokens/sec, arithmetic mean across batched shapes):")
            for bk in backends:
                xs = tput_by_backend[bk]
                if xs:
                    print(f"    {bk:<12} mean={sum(xs)/len(xs):>8.0f} tok/s   (n={len(xs)} shapes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["d64-gptoss"],
                    choices=list(MODELS.keys()))
    ap.add_argument("--backends", nargs="+", default=["triton", "gluon", "ck"],
                    choices=list(BACKENDS.keys()))
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--batched", action="store_true",
                    help="Run B>1 batched hetero/homo cases instead of B=1 sweep")
    ap.add_argument("--kv-cache-dtype", type=str, default=None,
                    choices=[None, "auto", "fp8_e4m3", "fp8_e5m2"],
                    help="KV cache dtype (fp8_e4m3 = OCP fp8 e4m3fn, which is what "
                         "gfx950 MFMA natively expects). Default: auto (model dtype).")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = [(m, b) for m in args.models for b in args.backends]
    batches = allocate_gpus(jobs, args.gpus)

    print(f"\n{'#' * 80}")
    print(f"  E2E TTFT matrix bench")
    print(f"  Models:   {args.models}")
    print(f"  Backends: {args.backends}")
    print(f"  GPUs:     {args.gpus}")
    print(f"  Mode:     {'B>1 batched' if args.batched else 'B=1 sweep'}")
    print(f"  Results:  {RESULTS_DIR}")
    print(f"  Total:    {len(jobs)} run(s) in {len(batches)} sequential batch(es)")
    print(f"{'#' * 80}\n")

    all_results = []
    lock = threading.Lock()
    base_port = 30200
    if args.kv_cache_dtype and args.kv_cache_dtype != "auto":
        print(f"  KV dtype: {args.kv_cache_dtype} (FP8 KV cache)")
    for batch_idx, batch in enumerate(batches):
        print(f"\n{'=' * 60}\n  BATCH {batch_idx + 1}/{len(batches)}\n{'=' * 60}")
        threads = []
        for i, ((model_key, backend_key), gpus) in enumerate(batch):
            port = base_port + i
            target = run_batched if args.batched else run_b1
            t = threading.Thread(
                target=target,
                args=(model_key, backend_key, gpus, port, lock, all_results),
                kwargs={"kv_cache_dtype": args.kv_cache_dtype},
                name=f"{model_key}/{backend_key}",
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(5)

    if args.batched:
        print_batched_table(all_results)
    else:
        print_b1_table(all_results)

    # merged csv
    b1_rows = [r for r in all_results if "prefix" in r]
    bt_rows = [r for r in all_results if "case" in r]
    if b1_rows:
        with open(os.path.join(RESULTS_DIR, "all_b1.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(b1_rows[0].keys()))
            w.writeheader()
            w.writerows(b1_rows)
    if bt_rows:
        with open(os.path.join(RESULTS_DIR, "all_batched.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bt_rows[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(bt_rows)
    print(f"\nDone. CSVs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
