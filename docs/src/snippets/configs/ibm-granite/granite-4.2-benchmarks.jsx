export const benchmarks = [
  {
    match: { hw: "h200", variant: "3b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "d59c1ddf7",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 131.17,
        tpot_ms: 3.51,
        tokens_per_sec_per_gpu: 2472.96,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1092.54,
        tpot_ms: 7.23,
        tokens_per_sec_per_gpu: 17416.41,
      },
    ],
  },
  {
    match: { hw: "h200", variant: "8b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "d59c1ddf7",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 260.59,
        tpot_ms: 6.28,
        tokens_per_sec_per_gpu: 1378.86,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2168.05,
        tpot_ms: 13.06,
        tokens_per_sec_per_gpu: 9484.18,
      },
    ],
  },
  {
    match: { hw: "h200", variant: "30b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "d59c1ddf7",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 807.38,
        tpot_ms: 17.14,
        tokens_per_sec_per_gpu: 502.74,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 6860.18,
        tpot_ms: 31.19,
        tokens_per_sec_per_gpu: 3803.83,
      },
    ],
  },
];
