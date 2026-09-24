// GigaChat 3.5 per-cell benchmark numbers, keyed by the same `match` tuple as gigachat3.5.jsx cells.
// See _deployment.jsx for the speed/accuracy schema. H100 entries are measured; H200 entries are
// bare stubs until run.

export const benchmarks = [
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "9d58189c12 (main, 2026-09-22)",
    // 8×H100 SXM, TP8/EP8, FP8. GSM8K: sglang.test.run_eval --eval-name gsm8k --api chat
    // (5-shot, 1314 scored questions, max_tokens 8192) — 0.959. Speed: bench_serving random
    // ISL 8192 / OSL 1024, --random-range-ratio 1.0, --warmup-requests 4, --flush-cache; P50.
    accuracy: { gsm8k_pct: 95.9 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 540, tpot_ms: 14.24, tokens_per_sec_per_gpu: 76 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 4930, tpot_ms: 27.85, tokens_per_sec_per_gpu: 551 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 18836, tpot_ms: 48.0, tokens_per_sec_per_gpu: 913 },
    ] },
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "9d58189c12 (main, 2026-09-22)",
    // 8×H100 SXM, TP8/EP8, FP8, MTP 2-1-3, --max-mamba-cache-size 240, --max-running-requests 80,
    // mem-fraction 0.8 (KV pool 185k tokens). GSM8K 0.959 on the full 1314-question chat harness,
    // identical to the non-speculative cell. Speed: bench_serving random ISL 8192 / OSL 1024,
    // --random-range-ratio 1.0, --warmup-requests 4, --flush-cache; P50.
    accuracy: { gsm8k_pct: 95.9 },
    // Accept length 2.82 / 2.85 / 2.78 of 3 at concurrency 1 / 16 / 64.
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 583, tpot_ms: 6.43, tokens_per_sec_per_gpu: 161 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1416, tpot_ms: 17.12, tokens_per_sec_per_gpu: 850 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 52262, tpot_ms: 22.78, tokens_per_sec_per_gpu: 900 },
    ] },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
];
