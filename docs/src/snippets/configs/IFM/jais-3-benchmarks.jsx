// Jais 3 GSM8K and speed on NVIDIA H100, served from each repository's main
// branch. Speed values come from a single run; null means not measured yet.
// A value stays null until both of its full evaluation runs are in; each
// filled value is the mean of the two. Per-run scores are in the MDX.

export const benchmarks = [
  {
    match: { hw: "h100", variant: "0.9b", quant: "bf16", strategy: "balanced", nodes: "single" },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 52.35,
        tpot_ms: 1.87,
        tokens_per_sec_per_gpu: 4673.93,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 1433.89,
        tpot_ms: 13.92,
        tokens_per_sec_per_gpu: 37488.99,
      },
    ],
    accuracy: { gsm8k_pct: 83.70 },
    notes: "Speed is from a single run. GSM8K is the mean of two full evaluation runs (83.62% and 83.78%). GSM8K truncation was 0.68% and 0.61%.",
  },
  {
    match: { hw: "h100", variant: "3.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 167.43,
        tpot_ms: 6.11,
        tokens_per_sec_per_gpu: 1436.60,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 6754.27,
        tpot_ms: 28.09,
        tokens_per_sec_per_gpu: 12192.66,
      },
    ],
    accuracy: { gsm8k_pct: 96.25 },
    notes: "Speed is from a single run. GSM8K is the mean of two full evaluation runs (96.36% and 96.13%). GSM8K truncation was 0.00% and 0.15%.",
  },
  {
    match: { hw: "h100", variant: "7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 254.89,
        tpot_ms: 8.54,
        tokens_per_sec_per_gpu: 1024.76,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 36797.25,
        tpot_ms: 30.64,
        tokens_per_sec_per_gpu: 9692.60,
      },
    ],
    accuracy: { gsm8k_pct: 95.87 },
    notes: "Speed is from a single run. GSM8K is the mean of two full evaluation runs (95.91% and 95.83%). GSM8K truncation was 0.00% and 0.08%.",
  },
  {
    match: { hw: "h100", variant: "32b", quant: "bf16", strategy: "balanced", nodes: "single" },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 608.07,
        tpot_ms: 17.93,
        tokens_per_sec_per_gpu: 243.09,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 256 },
        ttft_ms: 11442.35,
        tpot_ms: 38.27,
        tokens_per_sec_per_gpu: 2711.51,
      },
    ],
    accuracy: { gsm8k_pct: 96.63 },
    notes: "Speed is from a single run. GSM8K is the mean of two full evaluation runs (96.51% and 96.74%). GSM8K truncation was 0.00% in both.",
  },
  {
    match: { hw: "h100", variant: "36b", quant: "bf16", strategy: "balanced", nodes: "single" },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 252.09,
        tpot_ms: 12.23,
        tokens_per_sec_per_gpu: 360.99,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 256 },
        ttft_ms: 4265.27,
        tpot_ms: 35.24,
        tokens_per_sec_per_gpu: 3657.62,
      },
    ],
    accuracy: { gsm8k_pct: 96.21 },
    notes: "Speed is from a single run. GSM8K is the mean of two full evaluation runs (95.98% and 96.44%). GSM8K truncation was 0.00% in both.",
  },
  {
    match: { hw: "h100", variant: "375b", quant: "bf16", strategy: "balanced", nodes: "multi-2" },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 322.76,
        tpot_ms: 13.19,
        tokens_per_sec_per_gpu: 41.67,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 8, num_prompts: 256 },
        ttft_ms: 1395.83,
        tpot_ms: 19.29,
        tokens_per_sec_per_gpu: 216.73,
      },
    ],
    accuracy: { gsm8k_pct: 96.32 },
    notes: "Served on two nodes (16 GPUs). Speed is from a single run. GSM8K is the mean of two full evaluation runs (96.44% and 96.21%). GSM8K truncation was 0.00% in both.",
  },
];
