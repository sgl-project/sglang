// GSM8K: sgl-eval, full 1,319 questions, temperature 1.0 / top-p 0.95, on
// 4x GB300 with the PR #41993 head. Stop rate is 100% on every row except
// 1B + DSPARK (99.92%), and every recipe passed the image probes (16x / 4x / 16x-again / multi-image /
// max_slice_nums=1: 355 / 1291 / 355 / 556 / 93 prompt tokens, finish_reason=stop).

export const benchmarks = [
  {
    match: { hw: "gb300", variant: "1b", quant: "bf16", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 84.08 },
    notes: "1×GB300, TP=1. GSM8K stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "1b", quant: "bf16", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 83.24 },
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-1B-DSpark. GSM8K stop rate 99.92% (one of 1,319 requests did not emit EOS at temperature 1.0); accept length about 2.6.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "bf16", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.36 },
    notes: "2×GB300, TP=2. GSM8K stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "bf16", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    notes: "2×GB300, TP=2, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. GSM8K stop rate 100%; accept length about 4.1.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "fp8", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    notes: "2×GB300, TP=2. GSM8K stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "fp8", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    notes: "2×GB300, TP=2, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. GSM8K stop rate 100%; accept length about 4.0.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "int4", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    notes: "1×GB300, TP=1. GSM8K stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "int4", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.44 },
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. GSM8K stop rate 100%; accept length about 3.8.",
  },
];
