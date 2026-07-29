// LongCat-2.0 per-cell benchmark numbers, keyed by the same `match` tuple as longcat-2.0.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
export const benchmarks = [
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "SGLang nightly",
    accuracy: { gsm8k_pct: 95.8904109589041 },
    notes: "GSM8K was also spot-checked on 200 examples at 98.0%.",
  },
];
