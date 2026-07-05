// Hy3 per-cell benchmark numbers, keyed by the same `match` tuple as hy3.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// Day-0: numbers pending — verified cells have a GSM8K sanity only (§4.1 on the page).
export const benchmarks = [
  { match: { hw: "h200",  variant: "bf16", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200",  variant: "bf16", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "h200",  variant: "bf16", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200",  variant: "bf16", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "bf16", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "bf16", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300",  variant: "bf16", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300",  variant: "bf16", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b300",  variant: "bf16", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "bf16", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "bf16", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb300", variant: "bf16", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb200", variant: "bf16", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb200", variant: "bf16", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "gb200", variant: "bf16", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
];
