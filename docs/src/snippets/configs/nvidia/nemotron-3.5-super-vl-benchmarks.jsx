// Text-only GSM8K scores from completed sgl-eval==0.1.2 runs.
// Accuracy verification applies to these NGC-weight source runs, not to the
// public Hugging Face checkpoint IDs or the planned cookbook Docker tag.
export const benchmarks = [
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 95.98 },
    notes: "1,266/1,319 correct; 20 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.33 },
    notes: "292/300 correct on a smaller cohort; three outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.13 },
    notes: "1,268/1,319 correct; 16 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.33 },
    notes: "292/300 correct on a smaller cohort; four outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 95.68 },
    notes: "1,262/1,319 correct; 25 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.67 },
    notes: "293/300 correct on a smaller cohort; five outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.44 },
    notes: "1,272/1,319 correct; 21 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.33 },
    notes: "292/300 correct on a smaller cohort; four outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.06 },
    notes: "1,267/1,319 correct; 21 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.67 },
    notes: "293/300 correct on a smaller cohort; four outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 95.83 },
    notes: "1,264/1,319 correct; 21 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.33 },
    notes: "292/300 correct on a smaller cohort; two outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "rtx-pro-6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.06 },
    notes: "1,267/1,319 correct; 21 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "rtx-pro-6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 97.0 },
    notes: "291/300 correct on a smaller cohort; five outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "rtx-pro-6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.44 },
    notes: "1,272/1,319 correct; 18 outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "rtx-pro-6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", mambaDtype: "float32" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.33 },
    notes: "289/300 correct on a smaller cohort; four outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 96.67 },
    notes: "290/300 correct on a smaller cohort; five outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", mambaDtype: "bfloat16" },
    sglang_version: "dev @ 2c7511f6f7cf",
    accuracy: { gsm8k_pct: 98.0 },
    notes: "294/300 correct on a smaller cohort; three outputs hit the 8,192-token cap; zero request errors. Text-only NGC checkpoint run.",
  },
];
