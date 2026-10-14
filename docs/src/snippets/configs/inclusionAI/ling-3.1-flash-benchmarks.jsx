export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    accuracy: { gsm8k_pct: 97.19, stop_rate_pct: 99.92 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-cu13-ling-3.1-flash",
    accuracy: { gsm8k_pct: 97.42, stop_rate_pct: 99.92 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "hicache", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    accuracy: { gsm8k_pct: 97.27, stop_rate_pct: 99.92 },
  },
];
