// Command A+ per-cell benchmark numbers, keyed by the same `match` tuple as
// command-a-plus.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Source: SGLang SOTA benchmark campaign on NVIDIA B300 (SM103, ~275 GB), stock
// `main`, image lmsysorg/sglang:dev. Text via sglang.bench_serving (random dataset,
// random-range-ratio 1.0, greedy, ignore_eos, request-rate inf); warm, GPU-idle,
// medians of multiple reps. `tokens_per_sec_per_gpu` = measured total output tok/s
// divided by the cell's TP degree (BF16 TP=4, FP8 TP=2, W4A4 TP=1). GSM8K = 5-shot
// (BF16/W4A4 full test set 1314; FP8 200-question subset).
export const benchmarks = [
  // ====================================================================
  // B300 + BF16 (TP=4)
  // ====================================================================
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "7cead0f",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 64, num_prompts: 80 },
        ttft_ms: 842, tpot_ms: 17.51, tokens_per_sec_per_gpu: 645 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1405, tpot_ms: 13.06, tokens_per_sec_per_gpu: 276 },
    ],
    accuracy: { gsm8k_pct: 89.27 },
  },
  // ====================================================================
  // B300 + FP8 (TP=2)
  // ====================================================================
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "7cead0f",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 80, num_prompts: 80 },
        ttft_ms: 181, tpot_ms: 19.57, tokens_per_sec_per_gpu: 2026 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 80, num_prompts: 80 },
        ttft_ms: 870, tpot_ms: 22.34, tokens_per_sec_per_gpu: 1724 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 256, num_prompts: 256 },
        ttft_ms: 570, tpot_ms: 27.24, tokens_per_sec_per_gpu: 4592 },
    ],
    accuracy: { gsm8k_pct: 91.0 },
  },
  // ====================================================================
  // B300 + W4A4 (TP=1, NVFP4)
  // ====================================================================
  {
    match: { hw: "b300", variant: "default", quant: "w4a4", strategy: "balanced", nodes: "single" },
    sglang_version: "7e63fee",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 80, num_prompts: 80 },
        ttft_ms: 375, tpot_ms: 21.9, tokens_per_sec_per_gpu: 3600 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 80, num_prompts: 80 },
        ttft_ms: 810, tpot_ms: 27.3, tokens_per_sec_per_gpu: 2840 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 256, num_prompts: 256 },
        ttft_ms: null, tpot_ms: 32.4, tokens_per_sec_per_gpu: 7650 },
    ],
    accuracy: { gsm8k_pct: 87.6 },
  },
];
