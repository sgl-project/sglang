# PIN POC Benchmark Review & Rerun Instructions

**Reviewer**: Ishan Dhanani
**Date**: 2026-02-09
**Purpose**: Audit of the PIN benchmark results from Session 3 (2026-02-07), with instructions for re-running with proper controls.

---

## Context

An earlier Claude instance implemented the PIN mechanism and ran benchmarks to prove it works. Three benchmark attempts were made:

1. **Session 1 (aiperf-based)**: Used `benchmark_pin_poc.py` with aiperf. Produced the "76% TTFT reduction" headline number. **INVALID** -- aiperf's `--shared-system-prompt-length` generates synthetic tokens that differ from the warmup system prompt. The pinned blocks were never matched by aiperf's requests. This number does not measure PIN.

2. **Session 2 (aiperf ablation)**: Used `ablation_pin_poc.py`. The `extract_metric()` function failed to parse aiperf's JSON output. All metrics in `/tmp/pin_ablation_results/ablation_summary.json` are `null`. **ZERO usable data.**

3. **Session 3 (custom client)**: Used `pin_benchmark.py` -- a custom asyncio+aiohttp benchmark client that uses the exact same `SYSTEM_PROMPT` for warmup and load testing. This fixes the aiperf prompt mismatch. **This is the valid benchmark**, but it has a systematic ordering bias that inflates results.

The custom benchmark scripts live in `~/sglang/docs/design/`:
- `pin_benchmark.py` -- the custom client (Session 3, valid prompt handling)
- `benchmark_pin_poc.py` -- the aiperf-based benchmark (Session 1, invalid prompt handling)
- `ablation_pin_poc.py` -- the aiperf ablation runner (Session 2, broken metric extraction)

Results live in `/tmp/`:
- `/tmp/pin_benchmark_results/` -- Session 3 results (memory_pressure_sweep.json, concurrency_sweep.json, single_run.json)
- `/tmp/pin_ablation_results/` -- Session 2 results (all nulls, but per-run aiperf JSONs exist in subdirs)

The full end-to-end PIN integration (HTTP endpoint, scheduler, communicator) lives in `~/sglang` on branch `idhanani/llm-83-sglang-emit-kv-events-for-l1l2-tier-transitions`. The POC worktree at `~/sglang-poc-pin` only has the RadixCache-level implementation (no HTTP endpoint).

---

## What the Custom Benchmark Does

`~/sglang/docs/design/pin_benchmark.py` -- `run_comparison()` flow:

```
BASELINE:
  1. POST /flush_cache
  2. sleep(1)
  3. Send 2 warmup requests with SYSTEM_PROMPT (synchronous, max_tokens=16)
  4. sleep(1)
  5. Run 128 concurrent requests with SYSTEM_PROMPT (asyncio, streaming, measure TTFT)

PINNED:
  1. POST /flush_cache
  2. sleep(1)
  3. Send 2 warmup requests with SYSTEM_PROMPT (synchronous, max_tokens=16)
  4. sleep(2)  <-- extra second vs baseline
  5. Collect GPU block hashes from ZMQ KV events
  6. POST /hicache/pin_blocks with all discovered hashes
  7. Run 128 concurrent requests with SYSTEM_PROMPT (asyncio, streaming, measure TTFT)
```

For sweeps, the server is restarted between configurations (good), but baseline and pinned share the same server process within each configuration (problematic).

---

## The Results

### Memory Pressure Sweep (concurrency=32, requests=128, mem_fraction varies)

| mem_fraction | Capacity | BL TTFT avg | PIN TTFT avg | Delta | BL p99 | PIN p99 | Delta |
|-------------|----------|-------------|--------------|-------|--------|---------|-------|
| 0.05 | ~13K tok | 67.09ms | 52.07ms | -22.4% | 117.33ms | 64.41ms | -45.1% |
| 0.08 | ~22K tok | 69.89ms | 55.65ms | -20.4% | 118.08ms | 74.64ms | -36.8% |
| 0.12 | ~32K tok | 68.98ms | 51.02ms | -26.0% | 110.44ms | 65.99ms | -40.2% |
| 0.20 | ~54K tok | 70.96ms | 49.29ms | -30.5% | 123.66ms | 63.12ms | -49.0% |
| 0.40 | ~108K tok | 72.23ms | 54.46ms | -24.6% | 114.93ms | 71.71ms | -37.6% |

### Concurrency Sweep (mem_fraction=0.08, requests=128, concurrency varies)

| Conc | BL TTFT avg | PIN TTFT avg | Delta | BL p50 | PIN p50 | p50 Delta | BL p99 | PIN p99 | p99 Delta |
|------|-------------|--------------|-------|--------|---------|-----------|--------|---------|-----------|
| 4 | 28.63ms | 26.99ms | -5.7% | 26.89ms | 26.78ms | -0.4% | 94.57ms | 29.92ms | -68.4% |
| 8 | 32.85ms | 29.27ms | -10.9% | 28.12ms | 28.57ms | +1.6% | 108.43ms | 33.78ms | -68.8% |
| 16 | 47.63ms | 37.72ms | -20.8% | 41.30ms | 39.69ms | -3.9% | 100.80ms | 41.67ms | -58.7% |
| 32 | 68.39ms | 54.52ms | -20.3% | 55.69ms | **62.56ms** | **+12.3%** | 126.73ms | 72.74ms | -42.6% |
| 64 | 232.01ms | 77.04ms | -66.8% | 117.68ms | 79.58ms | -32.4% | 391.87ms | 104.89ms | -73.2% |

---

## Problems Found

### Problem 1: Systematic Ordering Bias

Baseline ALWAYS runs first, pinned ALWAYS runs second on the same server. No randomization, no repetition, no interleaving. The pinned phase also has extra delay (extra 1s sleep + hash collection + PIN API roundtrip) between warmup and load.

This matters because:
- The first load test on a server hits "cold" conditions (CUDA JIT, scheduler settling, memory allocator patterns not established)
- The second run benefits from all of this being warm

**Evidence -- the baseline has a consistent ~95-130ms p99 spike at ALL concurrencies from 4 to 32:**

```
conc=4:  BL p99=94.57ms   (but p50=26.89ms -- a 3.5x spike)
conc=8:  BL p99=108.43ms  (but p50=28.12ms -- a 3.9x spike)
conc=16: BL p99=100.80ms  (but p50=41.30ms -- a 2.4x spike)
conc=32: BL p99=126.73ms  (but p50=55.69ms -- a 2.3x spike)
```

This spike is suspiciously consistent (~100-130ms) regardless of concurrency. If it were caused by cache eviction, it would scale with concurrency and memory pressure. Instead it looks like a fixed first-run cost that the pinned run (running second) avoids entirely.

The pinned run p99 values are always tight around p50 (std dev 1-16ms vs baseline 10-33ms). This is consistent with the second run simply not having the cold-start spike.

### Problem 2: Memory Pressure Sweep Shows No Eviction Effect

The PIN thesis: "protect blocks from eviction under memory pressure." If correct:
- Baseline TTFT should **increase** as mem_fraction decreases (less memory = more eviction = more re-prefill)
- PIN benefit should **decrease** as mem_fraction increases (more memory = less eviction = nothing to protect)

Actual data:
- Baseline TTFT avg is **flat at 67-72ms** across all memory fractions (0.05 to 0.40)
- PIN benefit is **flat at 20-30%** across all memory fractions

At mem_fraction=0.40, the server has ~108K tokens of capacity for a workload that needs at most ~25K tokens (128 requests * ~200 unique tokens each, with system prompt shared). There is no eviction happening. Yet PIN still shows 25% improvement. This proves the improvement at concurrency=32 is NOT from PIN preventing eviction.

### Problem 3: Concurrency=64 Is Likely Real But Magnitude Is Inflated

At concurrency=64, the baseline enters a qualitatively different regime -- TTFT avg jumps from ~68ms to 232ms, p99 from ~127ms to 392ms. With 64 concurrent requests at ~448 tokens each and only ~22K token capacity, the system genuinely exceeds capacity. Eviction storms cascade: each evicted system prompt block forces re-prefill for the next request that needs it, which evicts something else, etc.

PIN prevents this cascade by keeping the system prompt blocks locked. The pinned TTFT avg (77ms) is in line with pinned values at lower concurrencies, suggesting genuine cache hits.

However, the ordering bias still contributes ~15-20ms of the improvement (based on the consistent baseline-vs-pinned delta at low concurrencies where no eviction occurs). The true PIN effect at concurrency=64 is probably 40-55% rather than 67%.

---

## What You Need to Do

### Step 0: Verify the environment

The PIN integration is in `~/sglang` on branch `idhanani/llm-83-sglang-emit-kv-events-for-l1l2-tier-transitions`. Verify:

```bash
cd ~/sglang
git branch --show-current  # should be idhanani/llm-83-sglang-emit-kv-events-for-l1l2-tier-transitions

# Verify PIN endpoint exists
grep -n "pin_blocks" python/sglang/srt/entrypoints/http_server.py
# Should show: @app.api_route("/hicache/pin_blocks", methods=["POST"])

# Verify sglang is installed from this branch
python -c "from sglang.srt.mem_cache.hiradix_cache import HiRadixCache; print('OK')"
```

The benchmark script is at `~/sglang/docs/design/pin_benchmark.py`. It imports `aiohttp`, `zmq`, `msgspec` -- make sure these are installed.

### Step 1: A/A Control Test (Quantify the Ordering Bias)

Run baseline twice with NO PIN on the same server. This measures how much improvement comes purely from being the second run.

Modify `run_comparison()` in `pin_benchmark.py` to run baseline twice instead of baseline+pinned. Or create a new function:

```python
def run_aa_control(base_url, model, concurrency, num_requests, max_tokens=64):
    """Run baseline TWICE with no PIN. Measures ordering bias."""
    # ---- FIRST RUN ----
    flush_cache(base_url)
    time.sleep(1)
    for i in range(2):
        requests.post(f"{base_url}/v1/chat/completions", json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Warmup {i}. Reply briefly."},
            ],
            "max_tokens": 16, "temperature": 0.0,
        }, timeout=60)
    time.sleep(2)  # match the pinned phase timing

    logger.info(f"RUN A (first): concurrency={concurrency}, requests={num_requests}")
    a_results, a_elapsed = asyncio.run(
        run_load(base_url, model, concurrency, num_requests, max_tokens))
    a_metrics = compute_metrics(a_results, a_elapsed)

    # ---- SECOND RUN (still no PIN) ----
    flush_cache(base_url)
    time.sleep(1)
    for i in range(2):
        requests.post(f"{base_url}/v1/chat/completions", json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Warmup {i}. Reply briefly."},
            ],
            "max_tokens": 16, "temperature": 0.0,
        }, timeout=60)
    time.sleep(2)

    logger.info(f"RUN B (second): concurrency={concurrency}, requests={num_requests}")
    b_results, b_elapsed = asyncio.run(
        run_load(base_url, model, concurrency, num_requests, max_tokens))
    b_metrics = compute_metrics(b_results, b_elapsed)

    return {"run_a": a_metrics, "run_b": b_metrics}
```

Run this at concurrency=32 and concurrency=64. If run B consistently beats run A by 15-25%, that confirms the ordering bias and quantifies it.

### Step 2: Reversed Order Test (PIN First, Baseline Second)

If PIN is the real driver, the pinned run should win regardless of order.

Modify `run_comparison()` to run PINNED first, then BASELINE:

```python
def run_comparison_reversed(base_url, model, concurrency, num_requests, max_tokens=64):
    """Run PINNED first, then BASELINE. Tests if PIN benefit survives order reversal."""
    collector = KVEventCollector()
    collector.start()

    try:
        # ---- PINNED (runs FIRST now) ----
        flush_cache(base_url)
        collector.clear()
        time.sleep(1)
        for i in range(2):
            requests.post(f"{base_url}/v1/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Warmup {i}. Reply briefly."},
                ],
                "max_tokens": 16, "temperature": 0.0,
            }, timeout=60)
        time.sleep(2)

        gpu_hashes = collector.get_gpu_block_hashes()
        if gpu_hashes:
            pin_blocks(base_url, gpu_hashes)

        logger.info(f"PINNED (first): concurrency={concurrency}")
        pin_results, pin_elapsed = asyncio.run(
            run_load(base_url, model, concurrency, num_requests, max_tokens))
        pin_metrics = compute_metrics(pin_results, pin_elapsed)

        if gpu_hashes:
            unpin_blocks(base_url, gpu_hashes)

        # ---- BASELINE (runs SECOND now) ----
        flush_cache(base_url)
        collector.clear()
        time.sleep(1)
        for i in range(2):
            requests.post(f"{base_url}/v1/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Warmup {i}. Reply briefly."},
                ],
                "max_tokens": 16, "temperature": 0.0,
            }, timeout=60)
        time.sleep(2)

        logger.info(f"BASELINE (second): concurrency={concurrency}")
        bl_results, bl_elapsed = asyncio.run(
            run_load(base_url, model, concurrency, num_requests, max_tokens))
        bl_metrics = compute_metrics(bl_results, bl_elapsed)

    finally:
        collector.stop()

    return {"baseline": bl_metrics, "pinned": pin_metrics, "blocks_pinned": len(gpu_hashes)}
```

Run at concurrency=32 and concurrency=64. Expected outcomes:
- At concurrency=32: if ordering bias dominates, the baseline (now second) should actually be faster than pinned (now first). The delta should flip or go to ~0%.
- At concurrency=64: if PIN is real, pinned should still win even running first, though by a smaller margin.

### Step 3: Multi-Repetition with Server Restart

Run each comparison 3 times with full server restart between each pair. This controls for per-server-instance variance.

```python
def run_comparison_multi(model, port, mem_fraction, tp_size, concurrency, num_requests,
                         max_tokens, num_reps=3):
    """Run comparison num_reps times with server restart each time."""
    all_results = []
    for rep in range(num_reps):
        logger.info(f"\n--- REPETITION {rep+1}/{num_reps} ---")
        proc = start_server(model, port, mem_fraction, tp_size)
        try:
            wait_for_server(f"http://localhost:{port}")
            result = run_comparison(
                f"http://localhost:{port}", model,
                concurrency, num_requests, max_tokens)
            result["repetition"] = rep
            all_results.append(result)
            print_comparison(f"rep={rep+1}", result)
        finally:
            stop_server(proc)
            time.sleep(5)

    # Compute mean/std across repetitions
    bl_ttfts = [r["baseline"]["ttft_avg"] for r in all_results]
    pin_ttfts = [r["pinned"]["ttft_avg"] for r in all_results]
    logger.info(f"\nACROSS {num_reps} REPS:")
    logger.info(f"  BL TTFT avg: mean={sum(bl_ttfts)/len(bl_ttfts):.2f}ms, "
                f"std={std_dev(bl_ttfts):.2f}ms")
    logger.info(f"  PIN TTFT avg: mean={sum(pin_ttfts)/len(pin_ttfts):.2f}ms, "
                f"std={std_dev(pin_ttfts):.2f}ms")
    return all_results

def std_dev(data):
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return (sum((x - mean) ** 2 for x in data) / (len(data) - 1)) ** 0.5
```

### Step 4: Server-Side Cache Hit Verification

Add logging to prove that PIN actually increases cache hits. The SGLang scheduler already logs `#cached-token` in prefill batches. Parse `/tmp/sglang_server_30000.log` after each run:

```python
def parse_cache_hits(log_path):
    """Parse cached token counts from SGLang scheduler logs."""
    import re
    cached_tokens = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'#cached-token:\s*(\d+)', line)
            if m:
                cached_tokens.append(int(m.group(1)))
    return {
        "total_cached": sum(cached_tokens),
        "batches": len(cached_tokens),
        "avg_cached_per_batch": sum(cached_tokens) / len(cached_tokens) if cached_tokens else 0,
    }
```

After each baseline and pinned run, parse the server log and compare `total_cached` tokens. If PIN is working, the pinned run should have significantly more cached tokens (the system prompt is always cached).

### Recommended Run Plan

Run all tests at **two concurrency levels**: 32 (where we suspect ordering artifact) and 64 (where we expect real PIN benefit).

```
1. A/A Control (conc=32): quantify ordering bias           ~2 min
2. A/A Control (conc=64): quantify ordering bias           ~2 min
3. Normal order (conc=32): baseline-first, pinned-second   ~2 min
4. Normal order (conc=64): baseline-first, pinned-second   ~2 min
5. Reversed order (conc=32): pinned-first, baseline-second ~2 min
6. Reversed order (conc=64): pinned-first, baseline-second ~2 min
7. Multi-rep normal (conc=64, 3 reps): confidence interval ~6 min
```

Total: ~18 minutes of benchmark time plus server start/stop overhead.

All runs should use: `--model Qwen/Qwen3-0.6B --mem-fraction 0.08 --requests 128`

### Expected Outcomes

If PIN is genuinely effective at concurrency=64:
- A/A control at conc=64: second run is faster, but by much less than 67%
- Reversed order at conc=64: pinned (first) still beats baseline (second)
- Multi-rep at conc=64: consistent improvement with reasonable confidence interval
- Server log shows more `#cached-token` in pinned runs

If the results are entirely ordering artifact:
- A/A control shows ~20-25% improvement for second run
- Reversed order shows baseline (second) beating pinned (first)
- The delta magnitude matches the A/A control

### Output Format

Save all results to `/tmp/pin_benchmark_v2/` as JSON:
- `aa_control_conc32.json`
- `aa_control_conc64.json`
- `normal_order_conc32.json`
- `normal_order_conc64.json`
- `reversed_order_conc32.json`
- `reversed_order_conc64.json`
- `multi_rep_conc64.json`
- `cache_hit_analysis.json`

---

## Summary of What We Know So Far

| Claim | Status | Evidence |
|-------|--------|----------|
| PIN mechanism (lock_ref, evictable_leaves) works correctly | **Confirmed** | 11 unit tests pass, code review shows correct logic |
| PIN HTTP endpoint works end-to-end | **Confirmed** | Server logs show `POST /hicache/pin_blocks 200 OK`, `pinned_count: 213` |
| PIN provides 76% TTFT reduction (aiperf Run 2) | **Invalid** | aiperf used different prompt tokens than pinned blocks |
| PIN provides 20-30% TTFT improvement at conc=32 | **Suspect** | Same improvement at mem_fraction=0.40 (no eviction) indicates ordering artifact |
| PIN provides 67% TTFT improvement at conc=64 | **Directionally correct, magnitude uncertain** | Genuine capacity pressure, but ordering bias inflates by ~15-20ms |
| PIN reduces TTFT variability (std dev) | **Likely real at high concurrency** | Consistent with preventing stochastic eviction events |

The PIN mechanism works. The question is whether the benchmark numbers accurately measure its benefit. Re-running with the controls above will give us clean numbers we can cite with confidence.
