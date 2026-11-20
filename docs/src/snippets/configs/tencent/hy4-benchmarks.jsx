export const benchmarks = [
  {
    match: { hw: "h200", quant: "fp8", strategy: "low-latency" },
    sglang_version: "dd8756f1189c",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 847.39, tpot_ms: 5.775, tokens_per_sec_per_gpu: 175.74 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 66691.85, tpot_ms: 30.582, tokens_per_sec_per_gpu: 708.27 },
    ],
    accuracy: { gsm8k_pct: 95.53 },
  },
  {
    match: { hw: "h200", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "a3bfedf6656e",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 821.14, tpot_ms: 11.419, tokens_per_sec_per_gpu: 92.13 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 104232.10, tpot_ms: 38.962, tokens_per_sec_per_gpu: 535.53 },
    ],
    accuracy: { gsm8k_pct: 95.91 },
    notes: "Not verified: the run did not pass qualification. During the C64 cell, every rank logged a recovered CUDA allocator OOM and GPU0 free memory fell to 61 MiB (all 256 requests completed). 17 cold multimodal requests spent their 4,096-token budget on thinking and ended with finish_reason length. GSM8K stop rate 99.01%.",
  },
  {
    match: { hw: "gb300", quant: "fp8", strategy: "low-latency" },
    sglang_version: "dd8756f1189c",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 621.862, tpot_ms: 5.966, tokens_per_sec_per_gpu: 355.239 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 16671.795, tpot_ms: 37.873, tokens_per_sec_per_gpu: 2298.573 },
    ],
    accuracy: { gsm8k_pct: 95.60 },
  },
  {
    match: { hw: "gb300", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "dd8756f1189c",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 594.909, tpot_ms: 12.920, tokens_per_sec_per_gpu: 166.467 },
    ],
  },
  { match: { hw: "b300", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "b300", quant: "fp8", strategy: "high-throughput" } },
  { match: { hw: "b200", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "b200", quant: "fp8", strategy: "high-throughput" } },
];
