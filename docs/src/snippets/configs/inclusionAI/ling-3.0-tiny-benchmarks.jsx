// Measured on lmsysorg/sglang:dev-Ling-3.0-tiny, 1× H200. TTFT/TPOT are P50
// (median) from sglang.bench_serving (random ISL 8192 / OSL 1024, --flush-cache);
// tokens_per_sec_per_gpu = output tok/s × (isl+osl)/osl. Accuracy from sgl-eval
// full GSM8K (1319).
export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 69.95, tpot_ms: 2.87, tokens_per_sec_per_gpu: 3002 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 78.31, tpot_ms: 5.96, tokens_per_sec_per_gpu: 22446 },
    ],
    accuracy: { gsm8k_pct: 94.01 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 81.78, tpot_ms: 2.82, tokens_per_sec_per_gpu: 3072 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 79.50, tpot_ms: 5.57, tokens_per_sec_per_gpu: 23738 },
    ],
    accuracy: { gsm8k_pct: 94.69 },
  },
];
