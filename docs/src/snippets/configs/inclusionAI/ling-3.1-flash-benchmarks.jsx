export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    accuracy: { gsm8k_pct: 96.89, stop_rate_pct: 99.85 },
    notes: "NEXTN; checkpoint revision b426279f1833b8e097c7c3781d1efa08aba19309. Matched standard control: GSM8K 96.97%, stop rate 99.77%.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    notes: "{{GB300_NEXTN_SMOKE}}; {{GB300_NEXTN_BS1}}",
  },
];
