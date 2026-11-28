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
    sglang_version: "f90693efd6cc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 820.75, tpot_ms: 11.426, tokens_per_sec_per_gpu: 92.07 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 104382.37, tpot_ms: 37.76, tokens_per_sec_per_gpu: 495.05 },
    ],
    accuracy: { gsm8k_pct: 95.68 },
    notes: "The 512-request cap leaves a 199,296-token KV pool on 8× H200 (130,944 at the default limit of 2,048), about 21 concurrent 8k/1k requests; the rest of a C64 load queues. C64 throughput is 17.6% higher than at the default limit, and median TPOT rises from 30.5 to 37.8 ms.",
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
    sglang_version: "f90693efd6cc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 595.863, tpot_ms: 12.887, tokens_per_sec_per_gpu: 167.07 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 19147.446, tpot_ms: 64.924, tokens_per_sec_per_gpu: 1724.268 },
    ],
    accuracy: { gsm8k_pct: 95.75 },
    notes: "The 512-request cap leaves a 590,848-token KV pool on 4× GB300 (522,048 at the default limit of 2,048). Across three runs the median C64 throughput is 1,725.00 tok/s/GPU, 9.6% higher than at the default limit, and median TPOT rises from about 59 to 65 ms.",
  },
  { match: { hw: "b300", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "b300", quant: "fp8", strategy: "high-throughput" } },
  { match: { hw: "b200", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "b200", quant: "fp8", strategy: "high-throughput" } },
];
