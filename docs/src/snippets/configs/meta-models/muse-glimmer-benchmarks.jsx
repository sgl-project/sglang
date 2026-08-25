export const benchmarks = [
  { match: { hw: "rtx5090", variant: "default", quant: "gguf",  strategy: "standard", nodes: "single" } },
  { match: { hw: "rtx5090", variant: "default", quant: "gguf",  strategy: "dflash",   nodes: "single" } },
  { match: { hw: "rtx5090", variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" } },
  { match: { hw: "rtx5090", variant: "default", quant: "nvfp4", strategy: "dflash",   nodes: "single" } },

  { match: { hw: "rtx6000", variant: "default", quant: "bf16",  strategy: "standard", nodes: "single" } },
  { match: { hw: "rtx6000", variant: "default", quant: "bf16",  strategy: "dflash",   nodes: "single" } },
  { match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" } },
  { match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "dflash",   nodes: "single" } },

  { match: { hw: "h200",    variant: "default", quant: "bf16",  strategy: "standard", nodes: "single" } },
  { match: { hw: "h200",    variant: "default", quant: "bf16",  strategy: "dflash",   nodes: "single" } },

  { match: { hw: "b200",    variant: "default", quant: "bf16",  strategy: "standard", nodes: "single" } },
  { match: { hw: "b200",    variant: "default", quant: "bf16",  strategy: "dflash",   nodes: "single" } },
  { match: { hw: "b200",    variant: "default", quant: "nvfp4", strategy: "standard", nodes: "single" } },
  { match: { hw: "b200",    variant: "default", quant: "nvfp4", strategy: "dflash",   nodes: "single" } },

  { match: { hw: "mac",     variant: "default", quant: "mlx-q4",       strategy: "standard", nodes: "single" } },
  { match: { hw: "mac",     variant: "default", quant: "mlx-q4km",     strategy: "standard", nodes: "single" } },
  { match: { hw: "mac",     variant: "default", quant: "mlx-q4k-dyn",  strategy: "standard", nodes: "single" } },
];
