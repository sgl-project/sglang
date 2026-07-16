// DeepSeek-V4 per-cell benchmark numbers, keyed by the same `match` tuple as
// deepseek-v4.jsx cells. See _deployment.jsx for the speed/accuracy schema.
// Measured on sglang v0.5.12.post1.
// tokens_per_sec_per_gpu is total (input+output) tok/s/GPU: fp4/fp8 = measured
// output/GPU × (isl+osl)/osl; nvfp4 was measured as total already.
export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 87, tpot_ms: 3.68, tokens_per_sec_per_gpu: 585 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 290, tpot_ms: 6.21, tokens_per_sec_per_gpu: 4401 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4228, tpot_ms: 60.98, tokens_per_sec_per_gpu: 2025 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4628, tpot_ms: 88.25, tokens_per_sec_per_gpu: 5787 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 99949, tpot_ms: 67.46, tokens_per_sec_per_gpu: 8451 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 253310, tpot_ms: 66.11, tokens_per_sec_per_gpu: 8676 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2326, tpot_ms: 69.9, tokens_per_sec_per_gpu: 891 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 7242, tpot_ms: 152.09, tokens_per_sec_per_gpu: 1728 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // B300 + FP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 88, tpot_ms: 3.67, tokens_per_sec_per_gpu: 594 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 266, tpot_ms: 6.06, tokens_per_sec_per_gpu: 4455 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2363, tpot_ms: 34.4, tokens_per_sec_per_gpu: 3618 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2812, tpot_ms: 51.65, tokens_per_sec_per_gpu: 9828 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 97028, tpot_ms: 65.09, tokens_per_sec_per_gpu: 8694 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 243335, tpot_ms: 63.98, tokens_per_sec_per_gpu: 8982 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 261, tpot_ms: 5.01, tokens_per_sec_per_gpu: 207 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 364, tpot_ms: 11.37, tokens_per_sec_per_gpu: 1233 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1866, tpot_ms: 54.48, tokens_per_sec_per_gpu: 1251 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 6325, tpot_ms: 123.95, tokens_per_sec_per_gpu: 2133 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // GB200 + FP4
  // ====================================================================
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2560, tpot_ms: 39.71, tokens_per_sec_per_gpu: 3078 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3995, tpot_ms: 82.56, tokens_per_sec_per_gpu: 6462 },
    ],
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
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 463, tpot_ms: 4.19, tokens_per_sec_per_gpu: 315 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 436, tpot_ms: 8.93, tokens_per_sec_per_gpu: 3024 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2671, tpot_ms: 45.88, tokens_per_sec_per_gpu: 2691 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4823, tpot_ms: 94.04, tokens_per_sec_per_gpu: 5733 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 154868, tpot_ms: 104.84, tokens_per_sec_per_gpu: 5589 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 386489, tpot_ms: 103.37, tokens_per_sec_per_gpu: 5643 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // GB300 + NVFP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #25820",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 361.72 , tpot_ms: 3.62, tokens_per_sec_per_gpu: 480 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 422.96, tpot_ms: 8.19, tokens_per_sec_per_gpu: 3733 },
    ],
    accuracy: { gsm8k_pct: 96.44 },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #25820",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 364.33, tpot_ms: 5.33, tokens_per_sec_per_gpu: 381 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 694.58, tpot_ms: 16.03, tokens_per_sec_per_gpu: 1654 },
    ],
    accuracy: { gsm8k_pct: 96.36 },
  },
  // ====================================================================
  // H200 + FP8
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 204, tpot_ms: 3.38, tokens_per_sec_per_gpu: 612 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 538, tpot_ms: 11.42, tokens_per_sec_per_gpu: 2376 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 738, tpot_ms: 36.27, tokens_per_sec_per_gpu: 3465 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 39806, tpot_ms: 80.13, tokens_per_sec_per_gpu: 3537 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 195293, tpot_ms: 130.35, tokens_per_sec_per_gpu: 4437 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 502615, tpot_ms: 130.31, tokens_per_sec_per_gpu: 4410 },
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
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 193, tpot_ms: 3.38, tokens_per_sec_per_gpu: 603 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 598, tpot_ms: 10.46, tokens_per_sec_per_gpu: 2772 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // H100 + FP4
  // ====================================================================
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
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
