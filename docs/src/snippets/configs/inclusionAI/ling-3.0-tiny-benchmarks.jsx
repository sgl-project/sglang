// TTFT/TPOT are P50. INT4 uses 80 exact ISL 8192 / OSL 1024 requests with
// --flush-cache; BF16/FP8 retain their original published measurements.
// Accuracy is full GSM8K (1319).
export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 69.95, tpot_ms: 2.87, tokens_per_sec_per_gpu: 3002 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 78.31, tpot_ms: 5.96, tokens_per_sec_per_gpu: 22446 },
    ],
    accuracy: { gsm8k_pct: 94.01 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 81.78, tpot_ms: 2.82, tokens_per_sec_per_gpu: 3072 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 79.50, tpot_ms: 5.57, tokens_per_sec_per_gpu: 23738 },
    ],
    accuracy: { gsm8k_pct: 94.69 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ 8ba213fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 90.31, tpot_ms: 1.96, tokens_per_sec_per_gpu: 4398 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 840.04, tpot_ms: 3.55, tokens_per_sec_per_gpu: 32958 },
    ],
    accuracy: { gsm8k_pct: 94.54 },
    notes: "Full GSM8K stop rate 100%; default decode CUDA Graph captured 36 shapes through batch 256.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ 8ba213fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 305.67, tpot_ms: 6.33, tokens_per_sec_per_gpu: 1359 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2634.12, tpot_ms: 16.04, tokens_per_sec_per_gpu: 7730 },
    ],
    accuracy: { gsm8k_pct: 94.54 },
    notes: "Full GSM8K stop rate 100%; default decode CUDA Graph captured 52 shapes through batch 512. Triton WNA16 used untuned default E=128,N=256 configs.",
  },
];
