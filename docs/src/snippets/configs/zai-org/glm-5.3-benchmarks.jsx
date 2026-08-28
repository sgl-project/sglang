// GLM-5.3 benchmark placeholders, keyed by the same `match` tuple as glm-5.3.jsx cells.
// Bare match stubs render as pending until speed and accuracy measurements are available.
export const benchmarks = [
  // NVIDIA FP8, single node.
  { match: { hw: "h200",  variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200",  variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "h200",  variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },

  // NVIDIA BF16.
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },

  // AMD ROCm, single node.
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
];
