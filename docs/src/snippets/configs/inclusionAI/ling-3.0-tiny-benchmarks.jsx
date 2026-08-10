// Accuracy measured on lmsysorg/sglang:dev-Ling-3.0-tiny, 1× H200,
// sgl-eval full GSM8K (1319).
// Speed (TTFT/TPOT/throughput) is NOT included yet: it needs a
// sglang.bench_serving run at a fixed random ISL/OSL, which has not been run —
// accuracy-eval throughput is not a comparable serving-speed number.
export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    accuracy: { gsm8k_pct: 94.01 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-Ling-3.0-tiny",
    accuracy: { gsm8k_pct: 94.69 },
  },
];
