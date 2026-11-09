// Mistral Large 4 per-cell benchmark numbers, keyed by the same `match` tuple
// as mistral-large-4.jsx cells. Speed convention: TTFT/TPOT are medians (P50);
// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU = output/GPU × (isl+osl)/osl.
export const benchmarks = [
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "lmsysorg/sglang:mistral-large-4",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 1 },
        ttft_ms: 72.7, tpot_ms: 10.49, tokens_per_sec_per_gpu: 35.4 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 4 },
        ttft_ms: 176.4, tpot_ms: 13.34, tokens_per_sec_per_gpu: 109.8 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 16 },
        ttft_ms: 471.8, tpot_ms: 18.65, tokens_per_sec_per_gpu: 306.2 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 64 },
        ttft_ms: 1306.9, tpot_ms: 28.42, tokens_per_sec_per_gpu: 774.1 },
    ],
    accuracy: { gsm8k_pct: 96.74 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "lmsysorg/sglang:mistral-large-4",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 1 },
        ttft_ms: 77.4, tpot_ms: 3.85, tokens_per_sec_per_gpu: 90.6 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 4 },
        ttft_ms: 96.1, tpot_ms: 5.36, tokens_per_sec_per_gpu: 258.6 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 16 },
        ttft_ms: 337.4, tpot_ms: 8.17, tokens_per_sec_per_gpu: 620.0 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 64 },
        ttft_ms: 1348.6, tpot_ms: 9.81, tokens_per_sec_per_gpu: 1017.6 },
    ],
    accuracy: { dspark_accept_len: "3.71" },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "lmsysorg/sglang:mistral-large-4",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 1 },
        ttft_ms: 63.4, tpot_ms: 10.13, tokens_per_sec_per_gpu: 73.3 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 4 },
        ttft_ms: 163.8, tpot_ms: 12.36, tokens_per_sec_per_gpu: 237.2 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 16 },
        ttft_ms: 424.8, tpot_ms: 17.79, tokens_per_sec_per_gpu: 645.6 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 64 },
        ttft_ms: 1392.0, tpot_ms: 28.01, tokens_per_sec_per_gpu: 1559.9 },
    ],
    accuracy: { gsm8k_100_pct: 98, mmmu_pro_pct: 72.43 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "lmsysorg/sglang:mistral-large-4",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 1 },
        ttft_ms: 74.6, tpot_ms: 3.79, tokens_per_sec_per_gpu: 177.9 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 4 },
        ttft_ms: 91.7, tpot_ms: 5.37, tokens_per_sec_per_gpu: 495.4 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 16 },
        ttft_ms: 382.9, tpot_ms: 8.66, tokens_per_sec_per_gpu: 1116.8 },
      { workload: { dataset: "random", isl: 1024, osl: 512, max_concurrency: 64 },
        ttft_ms: 1420.9, tpot_ms: 11.35, tokens_per_sec_per_gpu: 2811.6 },
    ],
    accuracy: { dspark_accept_len: 3.69 },
  },
  {
    match: { hw: "vera-rubin", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
    sglang_version: "lmsysorg/sglang:mistral-large-4",
    accuracy: { gsm8k_pct: 96.64, gsm8k_100_pct: 98, mmlu_pct: 91.70, mmmu_pro_pct: 72.22 },
  },
  {
    match: { hw: "vera-rubin", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "lmsysorg/sglang:mistral-large-4",
    accuracy: { gsm8k_pct: 96.74, gsm8k_100_pct: 98, mmlu_pct: 91.64, mmmu_pro_pct: 72.66 },
  },
];
