// One entry per cell `match` tuple (same 5 keys as config cells). Every entry is
// a bare match with no numbers, so the card shows "pending" — the launch recipes
// are verified but no measurement run has been published yet.

export const benchmarks = [
  { match: { hw: "h200",  variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200",  variant: "default", quant: "fp8",   strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "fp8",   strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "fp8",   strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi350x", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi350x", variant: "default", quant: "fp8",  strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8",  strategy: "balanced",        nodes: "single" } },
];
