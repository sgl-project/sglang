// Hy3 per-cell benchmark numbers, keyed by the same `match` tuple as hy3.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// H200 BF16 low-latency + balanced verified on 8×H200 (sgl-eval, single-shot, temp=0).
// BF16 high-throughput cell not yet verified — DeepEP num_max=1024 + DP-attention=8
// hits a HunyuanV3 rotary/attn reshape error on H200.
// FP8 cells not yet verified.
export const benchmarks = [
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" }, gsm8k_pct: 95.75 },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" }, gsm8k_pct: 95.83 },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "h200",  variant: "default",  quant: "fp8",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default",  quant: "fp8",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default",  quant: "fp8",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default",  quant: "fp8",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "default",  quant: "fp8",  strategy: "high-throughput", nodes: "single" } },
];
