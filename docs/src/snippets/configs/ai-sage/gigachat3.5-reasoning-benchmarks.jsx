// GigaChat 3.5 Reasoning per-cell benchmark numbers, keyed by the same `match` tuple as
// gigachat3.5-reasoning.jsx cells. See _deployment.jsx for the speed/accuracy schema.
// H100 entries are measured; H200 entries are bare stubs (render "pending") until run.

export const benchmarks = [
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "b63f8416b3 (main, PR #29189 merged 2026-09-21)",
    // Measured at the PR head 744df7de07 (2026-09-16/17); its GigaChat model files are
    // byte-identical to b63f8416b3, so the numbers carry over.
    // 8×H100 SXM, TP8/EP8, FP8. GSM8K: sglang.test.run_eval --eval-name gsm8k --api chat
    // (5-shot, 1314 scored questions, max_tokens 8192) — 0.963. Speed: bench_serving random
    // ISL 8192 / OSL 1024, --random-range-ratio 1.0, --warmup-requests 4, --flush-cache; P50.
    accuracy: { gsm8k_pct: 96.3 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 529, tpot_ms: 14.3, tokens_per_sec_per_gpu: 76 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 4357, tpot_ms: 29.1, tokens_per_sec_per_gpu: 540 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 17400, tpot_ms: 50.12, tokens_per_sec_per_gpu: 900 },
    ] },
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "b63f8416b3 (main, PR #29189 merged 2026-09-21)",
    // Measured at the PR head 744df7de07 (2026-09-16/17); its GigaChat model files are
    // byte-identical to b63f8416b3, so the numbers carry over.
    // 8×H100 SXM, TP8/EP8, FP8, MTP 3-1-4, --max-mamba-cache-size 240, --max-running-requests 80,
    // mem-fraction 0.83 (KV pool 199k tokens). GSM8K 0.963 on the full 1314-question chat harness
    // (measured at mem-fraction 0.8, same recipe otherwise; 0.970 on a 300-question re-check at 0.83);
    // mean accept length 3.41 of 4 on GSM8K, 2.98–3.20 on random-token prompts.
    accuracy: { gsm8k_pct: 96.3 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 587, tpot_ms: 6.99, tokens_per_sec_per_gpu: 156 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1133, tpot_ms: 17.69, tokens_per_sec_per_gpu: 803 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 50029, tpot_ms: 23.45, tokens_per_sec_per_gpu: 947 },
    ] },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
];
