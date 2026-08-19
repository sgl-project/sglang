// One entry per cell `match` tuple (same 5 keys as config cells). Every entry is
// a bare match with no numbers, so the card shows "pending".

export const benchmarks = [
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-4" } },
  { match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "multi-2" } },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "multi-4" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-4" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "multi-4" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "dspark", nodes: "multi-4" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "multi-4" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "dspark", nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "multi-8" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "dspark", nodes: "multi-8" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "multi-2" } },
  { match: { hw: "mi350x", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "mxfp4", strategy: "balanced", nodes: "single" } },
];
