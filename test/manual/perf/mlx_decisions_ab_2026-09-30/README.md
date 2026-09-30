# Bounded `/v1/decisions` A/B: MLX prefill-only request cleanup

## Result

In this single-machine test, the baseline's native MLX worker retained more request caches after each prefill-only completion. The fix kept the cache count at one through 40 measured requests.

| Measure | Baseline `57c81258d` | Fix `021ea7ae0` |
|---|---:|---:|
| Measured `/v1/decisions` responses | 20 HTTP 200 | 40 HTTP 200 |
| Worker request caches after prefill, matched requests 1–20 | 6 → 25 | 1 throughout |
| MLX active memory, matched requests 1–20 | 2.920 → 11.233 GiB | 0.733–0.801 GiB |
| Available system memory | 13.74 → 8.163 GiB through baseline request 19; request 20 unavailable | 17.874–17.913 GiB over 40 requests |
| p50 latency | 124.2 ms (19 client rows) | 109.9 ms (20 matched rows) |

Each run had three warmup requests before measurement. The first 20 measured inputs were the same in both runs. The latency values are descriptive samples, not evidence of a speedup. The worker cache count and active memory are the outcomes this experiment was designed to observe.

The baseline's twentieth measured request returned HTTP 200. Its worker event records 25 caches and 11.233 GiB of MLX active memory. The client sampler did not retain that row's latency or system-available-memory value. The operator reported that the safety guard stopped the run after request 20, but the guard's console output was not saved, so the exact final system-available-memory sample cannot be verified. The test did not run the baseline to OOM.

The base starts at six caches because startup/health checks and warmups had already left five entries. In the fix, completed state is removed before a subsequent non-idle launch; one tail request remains until another launch.

## Method

- **Machine:** Mac16,8; 51,539,607,552 bytes unified memory; macOS 26.5.2.
- **Software:** Python 3.11.12, PyTorch 2.13.0, Transformers 5.12.1, MLX 0.32.3; MLX Metal available.
- **Model:** `Qwen/Qwen3-0.6B-MLX-4bit`, cached snapshot `173234aa840d113125e9f2271100ddbaf16c9620` for both runs.
- **Code:** baseline `57c81258d493a08953407ceabf2cf73b1307ff8b`; fix `021ea7ae0bb08ce34ac15fb5332a66507c3d3c1b`.
- **Server flags:** `SGLANG_USE_MLX=1`, `SGLANG_MLX_CACHE_LIMIT_GB=2`, `--mlx-enable-sampling`, `--context-length 512`, `--max-running-requests 1`, `--disable-radix-cache`; default overlap scheduling; loopback HTTP on port 31237.
- **Workload:** sequential requests, one binary-choice question and 216 prompt tokens per request. The first 20 measured inputs matched between runs. Three warmups preceded each measured run.
- **Counters:** the same temporary worker hook recorded `_req_caches`, `mx.get_active_memory()`, and `mx.get_cache_memory()` after prefill and on request removal. Client-side `psutil` samples recorded server RSS and system-available memory.
- **Safety guard:** configured to stop below 8 GiB system-available memory or above 10 GiB aggregate server-process RSS. The operator reported that the baseline stopped after request 20; the console output was not retained.

Both runs used `--mlx-enable-sampling` because Decisions requires candidate-token log probabilities. An earlier smoke attempt without this flag returned HTTP 500 (`output_logprobs is empty`) and is excluded. The measured Decisions requests used `max_new_tokens=0`.

## Effective-capacity caveat

The flags matched, but automatic token-pool sizing differed with startup memory: **149,630 tokens on the baseline and 168,005 on the fix**. This is a bounded lifecycle check, not a controlled comparison of serving capacity, available-memory headroom, or latency. It covers one quantized model, one prompt length, and concurrency one. It does not measure GPU utilization, other models, concurrent requests, or decode cleanup.

## Included evidence

- `memory_cache_trend.png` — cache count, MLX active memory, and available memory; the baseline available-memory line ends at its last retained sample (request 19).
- `matched_ab_metrics.csv` — aligned first-20 request window. Baseline request 20 has an empty latency and system-memory value; its HTTP status and worker counters are from the HTTP/worker records.
- `baseline_client_requests.csv`, `patched_client_requests.csv` — retained client-side per-request samples (19 baseline rows and 40 fix rows).
- `baseline_worker_events.jsonl`, `patched_worker_events.jsonl` — worker cache and MLX memory events.
- `summary.json` — derived counts, ranges, and latency summaries.
- `startup_pool_sizing.txt` — the two startup log lines confirming the auto-sized token-pool capacities.

The full server logs and the safety-guard console output are not included. The selected pool-sizing lines are included. Accordingly, request 20 has no claimed system-available-memory value.
