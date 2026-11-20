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
    sglang_version: "2dffce694e0b",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 821.07, tpot_ms: 11.427, tokens_per_sec_per_gpu: 92.07 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 124844.13, tpot_ms: 30.459, tokens_per_sec_per_gpu: 421.64 },
    ],
    accuracy: { gsm8k_pct: 95.53 },
    notes: "Automatic memory leaves a 130,944-token KV pool on 8× H200, about 14 concurrent 8k/1k requests; at C64 the remaining requests queue, which raises TTFT and lowers per-GPU throughput.",
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
    sglang_version: "2dffce694e0b",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 594.664, tpot_ms: 12.925, tokens_per_sec_per_gpu: 166.414 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 19209.296, tpot_ms: 58.691, tokens_per_sec_per_gpu: 1573.671 },
    ],
    accuracy: { gsm8k_pct: 95.91 },
  },
  { match: { hw: "b300", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "b300", quant: "fp8", strategy: "high-throughput" } },
  { match: { hw: "b200", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "b200", quant: "fp8", strategy: "high-throughput" } },
];
