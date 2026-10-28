// One entry per cell `match`. Numbers pending — the card renders "pending"
// until speed/accuracy data is filled in from a measured run.
export const benchmarks = [
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "mtp",      nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dflash",   nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "dspark",   nodes: "single" } },

  { match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "mtp",      nodes: "single" } },
  { match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dflash",   nodes: "single" } },
  { match: { hw: "h100", variant: "default", quant: "nvfp4", strategy: "dspark",   nodes: "single" } },

  { match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "mtp",      nodes: "single" } },
  { match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dflash",   nodes: "single" } },
  { match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "dspark",   nodes: "single" } },
];
