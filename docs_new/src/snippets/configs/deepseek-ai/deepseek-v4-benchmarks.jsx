// DeepSeek-V4 per-cell benchmark numbers, keyed by the same `match` tuple as
// deepseek-v4.jsx cells. See _deployment.jsx for the speed/accuracy schema.
// Measured on sglang v0.5.15 / v0.5.15.post1 (per-cell sglang_version).
// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU = output/GPU × (isl+osl)/osl.
export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 302, tpot_ms: 2.91, tokens_per_sec_per_gpu: 677 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 454, tpot_ms: 8.76, tokens_per_sec_per_gpu: 3059 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 642, tpot_ms: 23.2, tokens_per_sec_per_gpu: 5222 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3147, tpot_ms: 64.0, tokens_per_sec_per_gpu: 8399 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 104109, tpot_ms: 70.25, tokens_per_sec_per_gpu: 8345 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 273808, tpot_ms: 71.34, tokens_per_sec_per_gpu: 8156 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 230, tpot_ms: 4.25, tokens_per_sec_per_gpu: 243 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 446, tpot_ms: 11.56, tokens_per_sec_per_gpu: 1165 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1081, tpot_ms: 36.23, tokens_per_sec_per_gpu: 1696 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4330, tpot_ms: 97.59, tokens_per_sec_per_gpu: 2721 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 107158, tpot_ms: 44.45, tokens_per_sec_per_gpu: 4169 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 265159, tpot_ms: 44.12, tokens_per_sec_per_gpu: 4252 },
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
        ttft_ms: 308, tpot_ms: 2.88, tokens_per_sec_per_gpu: 682 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 466, tpot_ms: 8.67, tokens_per_sec_per_gpu: 3059 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 223, tpot_ms: 4.19, tokens_per_sec_per_gpu: 245 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 509, tpot_ms: 11.13, tokens_per_sec_per_gpu: 1210 },
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
        ttft_ms: 208, tpot_ms: 2.86, tokens_per_sec_per_gpu: 599 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 345, tpot_ms: 7.51, tokens_per_sec_per_gpu: 3324 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1317, tpot_ms: 33.78, tokens_per_sec_per_gpu: 3801 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2722, tpot_ms: 52.84, tokens_per_sec_per_gpu: 9773 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 89936, tpot_ms: 61.42, tokens_per_sec_per_gpu: 9336 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 238636, tpot_ms: 61.15, tokens_per_sec_per_gpu: 9432 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 258, tpot_ms: 4.2, tokens_per_sec_per_gpu: 243 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 460, tpot_ms: 10.97, tokens_per_sec_per_gpu: 1149 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1868, tpot_ms: 42.19, tokens_per_sec_per_gpu: 1336 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2917, tpot_ms: 99.32, tokens_per_sec_per_gpu: 2669 },
    ],
  },
  {
    // At conc 4096 the engine is saturated (running at its max batch), so extra requests
    // queue — the high TTFT is queue wait, not compute; throughput is at its ceiling here.
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 103678, tpot_ms: 43.99, tokens_per_sec_per_gpu: 4203 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 257656, tpot_ms: 42.13, tokens_per_sec_per_gpu: 4400 },
    ],
  },
  // ====================================================================
  // B300 + NVFP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 187, tpot_ms: 2.83, tokens_per_sec_per_gpu: 729 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 410, tpot_ms: 7.68, tokens_per_sec_per_gpu: 3407 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 206, tpot_ms: 4.14, tokens_per_sec_per_gpu: 251 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 425, tpot_ms: 10.51, tokens_per_sec_per_gpu: 1256 },
    ],
  },
  // ====================================================================
  // GB300 + FP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 434, tpot_ms: 3.72, tokens_per_sec_per_gpu: 513 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 735, tpot_ms: 9.95, tokens_per_sec_per_gpu: 2465 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1041, tpot_ms: 30.45, tokens_per_sec_per_gpu: 4022 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4291, tpot_ms: 85.9, tokens_per_sec_per_gpu: 6366 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 137866, tpot_ms: 93.14, tokens_per_sec_per_gpu: 6338 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 364274, tpot_ms: 93.27, tokens_per_sec_per_gpu: 6246 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 317, tpot_ms: 4.49, tokens_per_sec_per_gpu: 441 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 501, tpot_ms: 14.54, tokens_per_sec_per_gpu: 1934 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1088, tpot_ms: 50.17, tokens_per_sec_per_gpu: 2455 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 8122, tpot_ms: 156.18, tokens_per_sec_per_gpu: 3429 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 288182, tpot_ms: 185.19, tokens_per_sec_per_gpu: 2832 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 761128, tpot_ms: 188.23, tokens_per_sec_per_gpu: 2787 },
    ],
  },
  // ====================================================================
  // GB300 + NVFP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 430, tpot_ms: 3.51, tokens_per_sec_per_gpu: 537 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 734, tpot_ms: 10.59, tokens_per_sec_per_gpu: 2385 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 321, tpot_ms: 4.61, tokens_per_sec_per_gpu: 440 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 631, tpot_ms: 14.25, tokens_per_sec_per_gpu: 1921 },
    ],
  },
  // ====================================================================
  // H200 + FP8
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 183, tpot_ms: 3.26, tokens_per_sec_per_gpu: 632 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 655, tpot_ms: 10.11, tokens_per_sec_per_gpu: 2752 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 880, tpot_ms: 40.63, tokens_per_sec_per_gpu: 3156 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 46563, tpot_ms: 89.82, tokens_per_sec_per_gpu: 3226 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 217694, tpot_ms: 146.95, tokens_per_sec_per_gpu: 3975 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 576540, tpot_ms: 148.29, tokens_per_sec_per_gpu: 3920 },
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
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 242, tpot_ms: 3.37, tokens_per_sec_per_gpu: 603 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 498, tpot_ms: 10.19, tokens_per_sec_per_gpu: 2636 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 864, tpot_ms: 34.12, tokens_per_sec_per_gpu: 3072 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3222, tpot_ms: 116.3, tokens_per_sec_per_gpu: 3768 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 193812, tpot_ms: 126.31, tokens_per_sec_per_gpu: 4503 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 499528, tpot_ms: 125.07, tokens_per_sec_per_gpu: 4546 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 634, tpot_ms: 5.65, tokens_per_sec_per_gpu: 170 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1727, tpot_ms: 23.12, tokens_per_sec_per_gpu: 559 },
    ],
  },
  {
    // Capacity-bound on 8xH200 for the 1.6T model: KV fits only ~15 concurrent requests, so
    // tok/s/GPU is pinned (~535-572) from conc 64 through the ht conc-4096 cell and the excess
    // concurrency just queues — P50 TTFT climbs to ~46s here and minutes at higher conc. The
    // throughput numbers are real but reflect that ceiling, not linear scaling.
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 41506, tpot_ms: 26.14, tokens_per_sec_per_gpu: 589 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 209586, tpot_ms: 28.23, tokens_per_sec_per_gpu: 591 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 889185, tpot_ms: 66.39, tokens_per_sec_per_gpu: 594 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1833386, tpot_ms: 65.86, tokens_per_sec_per_gpu: 601 },
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
        ttft_ms: 205, tpot_ms: 3.19, tokens_per_sec_per_gpu: 319 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 469, tpot_ms: 8.48, tokens_per_sec_per_gpu: 1539 },
    ],
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 726, tpot_ms: 23.11, tokens_per_sec_per_gpu: 2306 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 35793, tpot_ms: 48.46, tokens_per_sec_per_gpu: 2416 },
    ],
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 209393, tpot_ms: 65.31, tokens_per_sec_per_gpu: 2252 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 476764, tpot_ms: 66.0, tokens_per_sec_per_gpu: 2248 },
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
