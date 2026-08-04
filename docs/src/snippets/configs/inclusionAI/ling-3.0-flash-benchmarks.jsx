export const benchmarks = [
  { match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h800", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
];
