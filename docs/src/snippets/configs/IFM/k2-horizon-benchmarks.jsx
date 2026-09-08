// Direct NVIDIA H200 measurements. Each displayed value is the mean of two
// independent server launches; methodology and commands are in the cookbook.

export const benchmarks = [
  {
    match: { hw: "h200", variant: "0.9b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #37654",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 47.55,
        tpot_ms: 1.74,
        tokens_per_sec_per_gpu: 5025.24,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 1418.67,
        tpot_ms: 11.18,
        tokens_per_sec_per_gpu: 45773.21,
      },
    ],
    accuracy: { gsm8k_pct: 85.25 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 6.90% in both launches.",
  },
  {
    match: { hw: "h200", variant: "3.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #37654",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 158.71,
        tpot_ms: 5.10,
        tokens_per_sec_per_gpu: 1712.15,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 5174.71,
        tpot_ms: 28.83,
        tokens_per_sec_per_gpu: 16998.94,
      },
    ],
    accuracy: { gsm8k_pct: 92.00 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 4.09% and 4.70%.",
  },
  {
    match: { hw: "h200", variant: "7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #37654",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 249.31,
        tpot_ms: 6.74,
        tokens_per_sec_per_gpu: 1289.51,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 8175.68,
        tpot_ms: 33.81,
        tokens_per_sec_per_gpu: 13784.03,
      },
    ],
    accuracy: { gsm8k_pct: 94.88 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.91% and 1.14%.",
  },
  {
    match: { hw: "h200", variant: "32b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #37654",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 590.81,
        tpot_ms: 13.18,
        tokens_per_sec_per_gpu: 327.51,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 256 },
        ttft_ms: 10370.53,
        tpot_ms: 30.79,
        tokens_per_sec_per_gpu: 3503.00,
      },
    ],
    accuracy: { gsm8k_pct: 95.98 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.45% and 0.53%.",
  },
  {
    match: { hw: "h200", variant: "36b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #37654",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 218.54,
        tpot_ms: 10.77,
        tokens_per_sec_per_gpu: 410.17,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 256 },
        ttft_ms: 3639.21,
        tpot_ms: 27.67,
        tokens_per_sec_per_gpu: 4613.54,
      },
    ],
    accuracy: { gsm8k_pct: 95.15 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.83% and 0.99%.",
  },
  {
    match: { hw: "h200", variant: "375b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #37654",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 299.65,
        tpot_ms: 9.65,
        tokens_per_sec_per_gpu: 113.25,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 8, num_prompts: 256 },
        ttft_ms: 1408.10,
        tpot_ms: 15.68,
        tokens_per_sec_per_gpu: 524.00,
      },
    ],
    accuracy: { gsm8k_pct: 95.56 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.45% and 0.38%.",
  },
];
