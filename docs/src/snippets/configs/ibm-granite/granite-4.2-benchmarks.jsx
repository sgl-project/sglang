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
  {
    match: { hw: "b200", variant: "3b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "d10a656ad8",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 74.93,
        tpot_ms: 2.76,
        tokens_per_sec_per_gpu: 3179.78,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 559.10,
        tpot_ms: 4.66,
        tokens_per_sec_per_gpu: 27563.95,
      },
    ],
  },
  {
    match: { hw: "b200", variant: "8b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "d10a656ad8",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 132.85,
        tpot_ms: 4.44,
        tokens_per_sec_per_gpu: 1969.35,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1060.55,
        tpot_ms: 8.08,
        tokens_per_sec_per_gpu: 15800.59,
      },
    ],
  },
  {
    match: { hw: "b200", variant: "30b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "d10a656ad8",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 386.03,
        tpot_ms: 12.01,
        tokens_per_sec_per_gpu: 727.24,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3391.21,
        tpot_ms: 18.78,
        tokens_per_sec_per_gpu: 6522.86,
      },
    ],
  },
];
