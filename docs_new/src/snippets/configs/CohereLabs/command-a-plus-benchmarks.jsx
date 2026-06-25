// Command A+ per-cell benchmark numbers, keyed by the same `match` tuple as
// command-a-plus.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Source: SGLang SOTA benchmark campaign on NVIDIA B300 (SM103, ~275 GB), stock
// upstream `main` (commit pinned per entry's sglang_version). Text via sglang.bench_serving (random dataset,
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
    ],
    accuracy: { gsm8k_pct: 87.6 },
  },
  // ====================================================================
  // H200 (Hopper) — verified on an 8x H200 devbox, sglang main @ 20b2817.
  // Plain TP + default Triton MoE, launched with --cuda-graph-backend-prefill
  // disabled (the tc_piecewise prefill-graph compile crashes on current main —
  // see the page §2 note). Decode CUDA graph stays full so throughput is
  // representative; the disabled prefill graph inflates TTFT (esp. long-input).
  // tokens/sec/GPU = total output tok/s ÷ TP (BF16 TP=8, FP8 TP=4). GSM8K 5-shot, 200Q.
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "20b2817",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 64, num_prompts: 80 },
        ttft_ms: 806, tpot_ms: 15.22, tokens_per_sec_per_gpu: 368 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1359, tpot_ms: 11.79, tokens_per_sec_per_gpu: 151 },
    ],
    accuracy: { gsm8k_pct: 90.5 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "20b2817",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 80, num_prompts: 80 },
        ttft_ms: 1088, tpot_ms: 17.60, tokens_per_sec_per_gpu: 1070 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 80, num_prompts: 80 },
        ttft_ms: 6495, tpot_ms: 24.78, tokens_per_sec_per_gpu: 640 },
    ],
    accuracy: { gsm8k_pct: 90.5 },
  },
  // ====================================================================
  // B200 (Blackwell) — same recipe as B300, not yet benchmarked on B200 (pending).
  // ====================================================================
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "fp8",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "w4a4", strategy: "balanced", nodes: "single" } },
  // ====================================================================
  // H100 (Hopper) — same recipe as H200, not yet benchmarked (pending).
  // ====================================================================
  { match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h100", variant: "default", quant: "fp8",  strategy: "balanced", nodes: "single" } },
];
