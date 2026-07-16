// DeepSeek-V4 per-cell benchmark numbers, keyed by the same `match` tuple as
// deepseek-v4.jsx cells. See _deployment.jsx for the speed/accuracy schema.
// Measured on sglang v0.5.15 (b300 cells on v0.5.15.post1; b200-pro on clean v0.5.15).
// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU: fp4/fp8 = measured
// output/GPU × (isl+osl)/osl; nvfp4 was measured as total already.
export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 227, tpot_ms: 2.88, tokens_per_sec_per_gpu: 575 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 248, tpot_ms: 5.44, tokens_per_sec_per_gpu: 4767 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1477, tpot_ms: 41.54, tokens_per_sec_per_gpu: 3229 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2874, tpot_ms: 66.2, tokens_per_sec_per_gpu: 8401 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 104331, tpot_ms: 71.27, tokens_per_sec_per_gpu: 7997 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 277454, tpot_ms: 71.44, tokens_per_sec_per_gpu: 8117 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 333, tpot_ms: 4.26, tokens_per_sec_per_gpu: 214 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 405, tpot_ms: 9.95, tokens_per_sec_per_gpu: 1259 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 3615, tpot_ms: 53.04, tokens_per_sec_per_gpu: 1014 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3534, tpot_ms: 101.38, tokens_per_sec_per_gpu: 2708 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 112146, tpot_ms: 45.49, tokens_per_sec_per_gpu: 3956 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 263860, tpot_ms: 43.05, tokens_per_sec_per_gpu: 4148 },
    ],
  },
  // ====================================================================
  // B200 + NVFP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 337, tpot_ms: 2.86, tokens_per_sec_per_gpu: 541 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 408, tpot_ms: 6.82, tokens_per_sec_per_gpu: 4133 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 230, tpot_ms: 4.17, tokens_per_sec_per_gpu: 245 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 462, tpot_ms: 10.07, tokens_per_sec_per_gpu: 1380 },
    ],
  },
  // ====================================================================
  // B300 + FP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 191, tpot_ms: 2.85, tokens_per_sec_per_gpu: 592 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 171, tpot_ms: 5.41, tokens_per_sec_per_gpu: 5032 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1388, tpot_ms: 34.65, tokens_per_sec_per_gpu: 3812 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2172, tpot_ms: 52.44, tokens_per_sec_per_gpu: 10337 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 89642, tpot_ms: 61.18, tokens_per_sec_per_gpu: 9274 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 127395, tpot_ms: 42.36, tokens_per_sec_per_gpu: 12098 },
    ],
  },
  {
    // Not re-verified on 0.5.15: server hung during load/init on the only available
    // b300 devbox (dev image); no clean v0.5.15 b300 capacity to confirm. Prior
    // 0.5.12.post1 numbers removed to avoid implying a 0.5.15 result.
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // B300 + NVFP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 190, tpot_ms: 2.83, tokens_per_sec_per_gpu: 731 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 205, tpot_ms: 5.42, tokens_per_sec_per_gpu: 5173 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
  },
  // ====================================================================
  // GB200 + FP4
  // ====================================================================
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    // Not re-benched on 0.5.15 (no gb200 fleet hardware). Prior 0.5.12.post1 numbers
    // were Mean-latency; removed so the global latencyPercentile:"P50" doesn't mislabel
    // them. gb200 nvfp4 cells below keep their explicit "PR #25820" version tag.
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
  },
  // ====================================================================
  // GB200 + NVFP4
  // ====================================================================
  {
    match: { hw: "gb200", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #25820",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 323.85, tpot_ms: 3.62, tokens_per_sec_per_gpu: 496 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 397.31, tpot_ms: 8.11, tokens_per_sec_per_gpu: 3663 },
    ],
    accuracy: { gsm8k_pct: 96.66 },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "PR #25820",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 338.20, tpot_ms: 6.25, tokens_per_sec_per_gpu: 161 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 524.85, tpot_ms: 14.45, tokens_per_sec_per_gpu: 1015 },
    ],
    accuracy: { gsm8k_pct: 95.98 },
  },
  // ====================================================================
  // GB300 + FP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 481, tpot_ms: 3.61, tokens_per_sec_per_gpu: 429 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 557, tpot_ms: 7.02, tokens_per_sec_per_gpu: 3710 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 821, tpot_ms: 30.22, tokens_per_sec_per_gpu: 4094 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2873, tpot_ms: 85.87, tokens_per_sec_per_gpu: 6703 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 136259, tpot_ms: 91.93, tokens_per_sec_per_gpu: 6434 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 261789, tpot_ms: 76.04, tokens_per_sec_per_gpu: 7160 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 416, tpot_ms: 4.47, tokens_per_sec_per_gpu: 411 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 662, tpot_ms: 12.64, tokens_per_sec_per_gpu: 2026 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 993, tpot_ms: 51.37, tokens_per_sec_per_gpu: 2403 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3794, tpot_ms: 160.68, tokens_per_sec_per_gpu: 3458 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 297124, tpot_ms: 184.15, tokens_per_sec_per_gpu: 2810 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 767287, tpot_ms: 190.21, tokens_per_sec_per_gpu: 2777 },
    ],
  },
  // ====================================================================
  // GB300 + NVFP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 424, tpot_ms: 3.51, tokens_per_sec_per_gpu: 467 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 533, tpot_ms: 7.44, tokens_per_sec_per_gpu: 3665 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 326, tpot_ms: 4.58, tokens_per_sec_per_gpu: 439 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 669, tpot_ms: 12.48, tokens_per_sec_per_gpu: 2165 },
    ],
  },
  // ====================================================================
  // H200 + FP8
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 178, tpot_ms: 3.28, tokens_per_sec_per_gpu: 631 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 263, tpot_ms: 8.99, tokens_per_sec_per_gpu: 3302 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 897, tpot_ms: 41.27, tokens_per_sec_per_gpu: 3098 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 44577, tpot_ms: 83.49, tokens_per_sec_per_gpu: 3423 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 218999, tpot_ms: 147.51, tokens_per_sec_per_gpu: 3958 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 574457, tpot_ms: 147.07, tokens_per_sec_per_gpu: 3940 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "multi-2" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
  },
  // ====================================================================
  // H200 + FP4
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 237, tpot_ms: 3.37, tokens_per_sec_per_gpu: 607 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 394, tpot_ms: 8.42, tokens_per_sec_per_gpu: 3264 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 859, tpot_ms: 37.62, tokens_per_sec_per_gpu: 2758 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3688, tpot_ms: 127.26, tokens_per_sec_per_gpu: 4207 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 193474, tpot_ms: 126.81, tokens_per_sec_per_gpu: 4473 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 496152, tpot_ms: 124.37, tokens_per_sec_per_gpu: 4550 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 623, tpot_ms: 5.64, tokens_per_sec_per_gpu: 171 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 684, tpot_ms: 25.83, tokens_per_sec_per_gpu: 591 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 45971, tpot_ms: 29.03, tokens_per_sec_per_gpu: 535 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 233894, tpot_ms: 30.1, tokens_per_sec_per_gpu: 544 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 925656, tpot_ms: 69.39, tokens_per_sec_per_gpu: 572 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1899200, tpot_ms: 68.99, tokens_per_sec_per_gpu: 570 },
    ],
  },
  // ====================================================================
  // H100 + FP4
  // ====================================================================
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 200, tpot_ms: 3.22, tokens_per_sec_per_gpu: 320 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 308, tpot_ms: 6.76, tokens_per_sec_per_gpu: 2009 },
    ],
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 644, tpot_ms: 25.94, tokens_per_sec_per_gpu: 1987 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 36270, tpot_ms: 47.12, tokens_per_sec_per_gpu: 2429 },
    ],
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 212091, tpot_ms: 67.32, tokens_per_sec_per_gpu: 2203 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 473971, tpot_ms: 66.23, tokens_per_sec_per_gpu: 2260 },
    ],
  },
  {
    match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
  },
  {
    match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
  },
  {
    match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
  },
  // ====================================================================
  // MI300X + FP8 (Flash)
  { match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi300x", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  // MI355X + FP4 (Flash)
  { match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" } },
  // MI355X + FP8 (Flash)
  { match: { hw: "mi355x", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "mi355x", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  // MI355X + FP4 (Pro)
  { match: { hw: "mi355x", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "mi355x", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" } },
  // MI355X + FP8 (Pro)
  { match: { hw: "mi355x", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "mi355x", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "pro", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
];
