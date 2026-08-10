// Measured on lmsysorg/sglang:dev-Ling-3.0-tiny, 1× H200 (tp 1), sgl-eval full
// GSM8K (1319). Speed uses the eval's output throughput; TTFT/TPOT need a
// sglang.bench_serving run and are left null (pending).
export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    speed: [
      { workload: { dataset: "gsm8k", isl: null, osl: null, max_concurrency: 64 },
        ttft_ms: null, tpot_ms: null, tokens_per_sec_per_gpu: 4221 },
    ],
    accuracy: { gsm8k_pct: 94.01 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    speed: [
      { workload: { dataset: "gsm8k", isl: null, osl: null, max_concurrency: 64 },
        ttft_ms: null, tpot_ms: null, tokens_per_sec_per_gpu: 5011 },
    ],
    accuracy: { gsm8k_pct: 94.69 },
  },
];
