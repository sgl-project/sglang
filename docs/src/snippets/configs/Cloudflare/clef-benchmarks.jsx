// Decision Index 0.2.1 GSM8K accuracy through /v1/systemone.
// All 1,319 test problems in both four-choice and ten-choice formats.
export const benchmarks = [
  {
    match: { hw: "h200", variant: "clef", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev-clef (bd2d73daa5af)",
    accuracy: { gsm8k_decision_pct: 80.71 },
    notes: "1,319 problems, each with 4 and 10 choices (2,638 requests).",
  },
  {
    match: { hw: "h200", variant: "flash", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev-clef (bd2d73daa5af)",
    accuracy: { gsm8k_decision_pct: 67.32 },
    notes: "1,319 problems, each with 4 and 10 choices (2,638 requests).",
  },
  {
    match: { hw: "b200", variant: "clef", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "nightly (37ae292e6f)",
    accuracy: { gsm8k_decision_pct: 80.59 },
    notes: "1,319 problems, each with 4 and 10 choices (2,638 requests).",
  },
  {
    match: { hw: "b200", variant: "flash", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "nightly (37ae292e6f)",
    accuracy: { gsm8k_decision_pct: 67.36 },
    notes: "1,319 problems, each with 4 and 10 choices (2,638 requests).",
  },
  {
    match: { hw: "b300", variant: "clef", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev-clef (bd2d73daa5af)",
    accuracy: { gsm8k_decision_pct: 80.59 },
    notes: "1,319 problems, each with 4 and 10 choices (2,638 requests).",
  },
  {
    match: { hw: "b300", variant: "flash", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev-clef (bd2d73daa5af)",
    accuracy: { gsm8k_decision_pct: 67.36 },
    notes: "1,319 problems, each with 4 and 10 choices (2,638 requests).",
  },
]
