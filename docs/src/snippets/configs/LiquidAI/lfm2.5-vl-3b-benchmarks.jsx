// BF16 results for LFM2.5-VL-3B.
// Speed uses the existing LFM2.5 cookbook methodology exactly:
// `python3 -m sglang.bench_serving`, random input/output caps 1024/1024,
// 10 prompts at C=1 and 1000 prompts at C=100. CUDA feature IPC and the
// pool-handle cache were enabled; MM splitting was unset.
// `tokens_per_sec_per_gpu` is total input+output throughput.

export const benchmarks = [
  {
    match: { hw: "h100", variant: "default", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "nightly-dev-20260729-16a52bff",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 36.17, tpot_ms: 3.01, tokens_per_sec_per_gpu: 611.37 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 152.89, tpot_ms: 7.15, tokens_per_sec_per_gpu: 25911.87 },
    ],
    accuracy: { countbench_pct: 85.63, docvqa_pct: 90.73, mmmu_pro_pct: 30.06 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "nightly-dev-20260729-16a52bff",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 28.11, tpot_ms: 2.51, tokens_per_sec_per_gpu: 732.86 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 361.56, tpot_ms: 5.86, tokens_per_sec_per_gpu: 29634.41 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "nightly-dev-20260729-16a52bff + PR #33744",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 471.10, tpot_ms: 2.88, tokens_per_sec_per_gpu: 456.46 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 180.51, tpot_ms: 7.35, tokens_per_sec_per_gpu: 24931.73 },
    ],
    accuracy: { countbench_pct: 85.22 },
  },
];
