// One entry per cell `match` tuple (same 5 keys as the config cells). Speed
// measured with python3 -m sglang.bench_serving on Modal cloud GPUs (one GPU,
// TP=1): latency = 10 prompts at concurrency 1, throughput = 1000 prompts at
// concurrency 100 (`random` dataset, 1024/1024 token caps). Accuracy comes
// from the config's `defaultAccuracy` (Liquid-AI-reported GPQA / AIME25).

export const benchmarks = [
  // ====================================================================
  // H100
  // ====================================================================
  {
    match: { hw: "h100", variant: "8b-a1b", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 287.24, tpot_ms: 2.4, tokens_per_sec_per_gpu: 650 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 171.72, tpot_ms: 11.87, tokens_per_sec_per_gpu: 15751 },
    ],
  },
  {
    match: { hw: "h100", variant: "instruct", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 18.9, tpot_ms: 2.08, tokens_per_sec_per_gpu: 943 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 180.55, tpot_ms: 7, tokens_per_sec_per_gpu: 26099 },
    ],
  },
  {
    match: { hw: "h100", variant: "thinking", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 16.23, tpot_ms: 2.19, tokens_per_sec_per_gpu: 898 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 127.45, tpot_ms: 5.31, tokens_per_sec_per_gpu: 34862 },
    ],
  },
  {
    match: { hw: "h100", variant: "350m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 18.8, tpot_ms: 1.65, tokens_per_sec_per_gpu: 1181 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 476.85, tpot_ms: 4.26, tokens_per_sec_per_gpu: 37491 },
    ],
  },
  {
    match: { hw: "h100", variant: "230m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 23.74, tpot_ms: 1.77, tokens_per_sec_per_gpu: 1092 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1128.14, tpot_ms: 4.54, tokens_per_sec_per_gpu: 28561 },
    ],
  },
  {
    match: { hw: "h100", variant: "jp", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 17.04, tpot_ms: 2.1, tokens_per_sec_per_gpu: 937 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 195.67, tpot_ms: 5.05, tokens_per_sec_per_gpu: 35389 },
    ],
  },
  {
    match: { hw: "h100", variant: "vl", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 22.01, tpot_ms: 1.54, tokens_per_sec_per_gpu: 1260 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1676.37, tpot_ms: 3.38, tokens_per_sec_per_gpu: 28967 },
    ],
  },
  {
    match: { hw: "h100", variant: "vl-450m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 26.01, tpot_ms: 1.34, tokens_per_sec_per_gpu: 1427 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1604.2, tpot_ms: 3.38, tokens_per_sec_per_gpu: 29704 },
    ],
  },
  // ====================================================================
  // H200
  // ====================================================================
  {
    match: { hw: "h200", variant: "8b-a1b", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 48.8, tpot_ms: 2.23, tokens_per_sec_per_gpu: 853 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 119.9, tpot_ms: 11.96, tokens_per_sec_per_gpu: 15826 },
    ],
  },
  {
    match: { hw: "h200", variant: "instruct", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 20.97, tpot_ms: 2.2, tokens_per_sec_per_gpu: 891 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 601.53, tpot_ms: 5.37, tokens_per_sec_per_gpu: 29748 },
    ],
  },
  {
    match: { hw: "h200", variant: "thinking", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 21.39, tpot_ms: 2.22, tokens_per_sec_per_gpu: 880 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 398.87, tpot_ms: 5.58, tokens_per_sec_per_gpu: 30426 },
    ],
  },
  {
    match: { hw: "h200", variant: "350m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 22.51, tpot_ms: 1.72, tokens_per_sec_per_gpu: 1129 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 880.23, tpot_ms: 4.37, tokens_per_sec_per_gpu: 31530 },
    ],
  },
  {
    match: { hw: "h200", variant: "230m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 18.74, tpot_ms: 1.74, tokens_per_sec_per_gpu: 1123 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 458.07, tpot_ms: 4.47, tokens_per_sec_per_gpu: 35785 },
    ],
  },
  {
    match: { hw: "h200", variant: "jp", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 20.85, tpot_ms: 2.09, tokens_per_sec_per_gpu: 938 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 781.82, tpot_ms: 5.23, tokens_per_sec_per_gpu: 28985 },
    ],
  },
  {
    match: { hw: "h200", variant: "vl", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 20.88, tpot_ms: 1.32, tokens_per_sec_per_gpu: 1465 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1550.27, tpot_ms: 3.26, tokens_per_sec_per_gpu: 30944 },
    ],
  },
  {
    match: { hw: "h200", variant: "vl-450m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 23.48, tpot_ms: 1.2, tokens_per_sec_per_gpu: 1597 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1544.41, tpot_ms: 3.14, tokens_per_sec_per_gpu: 31235 },
    ],
  },
  // ====================================================================
  // B200
  // ====================================================================
  {
    match: { hw: "b200", variant: "8b-a1b", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 124.36, tpot_ms: 2, tokens_per_sec_per_gpu: 873 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 154.77, tpot_ms: 7.54, tokens_per_sec_per_gpu: 24688 },
    ],
  },
  {
    match: { hw: "b200", variant: "instruct", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 11.22, tpot_ms: 1.19, tokens_per_sec_per_gpu: 1637 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1223.9, tpot_ms: 2.19, tokens_per_sec_per_gpu: 42274 },
    ],
  },
  {
    match: { hw: "b200", variant: "thinking", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 11.12, tpot_ms: 1.19, tokens_per_sec_per_gpu: 1637 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1230.34, tpot_ms: 2.18, tokens_per_sec_per_gpu: 42243 },
    ],
  },
  {
    match: { hw: "b200", variant: "350m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 12.18, tpot_ms: 0.91, tokens_per_sec_per_gpu: 2131 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1177.6, tpot_ms: 1.92, tokens_per_sec_per_gpu: 45273 },
    ],
  },
  {
    match: { hw: "b200", variant: "230m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 12.28, tpot_ms: 0.84, tokens_per_sec_per_gpu: 2316 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1550.61, tpot_ms: 1.93, tokens_per_sec_per_gpu: 38411 },
    ],
  },
  {
    match: { hw: "b200", variant: "jp", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 11.98, tpot_ms: 1.19, tokens_per_sec_per_gpu: 1635 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 1367.79, tpot_ms: 2.27, tokens_per_sec_per_gpu: 39589 },
    ],
  },
  {
    match: { hw: "b200", variant: "vl", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 11.55, tpot_ms: 1.22, tokens_per_sec_per_gpu: 1614 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 935.24, tpot_ms: 2.34, tokens_per_sec_per_gpu: 46272 },
    ],
  },
  {
    match: { hw: "b200", variant: "vl-450m", quant: "bf16", strategy: "default", nodes: "single" },
    sglang_version: "0.0.0.dev1+g631db6c75",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 12.09, tpot_ms: 0.92, tokens_per_sec_per_gpu: 2106 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 939.41, tpot_ms: 2.25, tokens_per_sec_per_gpu: 47761 },
    ],
  },
];
