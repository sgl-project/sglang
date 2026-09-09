// Hy3 per-cell benchmark numbers, keyed by the same `match` tuple as hy3.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// H200 BF16 low-latency + balanced verified on 8×H200 (sgl-eval, single-shot, temp=0).
// FP8 cells not yet verified.
export const benchmarks = [
  {
    match: { hw: "a3", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "c5bd3d7dce",
    accuracy: { gsm8k_pct: 96.59 },
    latencyPercentile: "P50",
    speed: [{
      workload: { dataset: "ShareGPT", isl: 338.42, osl: 231.59, max_concurrency: 16, num_prompts: 128 },
      ttft_ms: 638.46,
      tpot_ms: 59.67,
      tokens_per_sec_per_gpu: 30.43,
      output_tokens_per_sec: 197.80,
    }],
    notes: "2026-09-09: Official image cann9.0.0-a3-v0.5.16, bundled SGLang c5bd3d7dce. GSM8K: 1274/1319 correct with EvalScope 1.11.1 and one fixed demonstration. ShareGPT: 128/128 requests succeeded in 149.87 s; mean input/output lengths are shown. BF16, TP16, EAGLE, no_think, temperature 0. ShareGPT uses ignore_eos.",
  },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" }, gsm8k_pct: 95.75 },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" }, gsm8k_pct: 95.83 },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
];
