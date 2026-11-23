// One entry per cell `match` tuple. Values are not filled in yet; an entry
// with only `match` renders as pending. To add a measurement, give the entry
// `sglang_version` and `speed: [{ workload: { dataset, isl, osl, max_concurrency },
// ttft_ms, tpot_ms, tokens_per_sec_per_gpu }]`; for `accuracy`, also add
// config.accuracyLabels in inkling-2-small.jsx.

export const benchmarks = [
  { match: { hw: "vera-rubin", quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "vera-rubin", quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "tpu7x"     , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "tpu7x"     , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "gb300"  , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "gb200"  , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "b300"   , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "b200"   , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "h200"   , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "mi350x" , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "mi355x" , quant: "nvfp4", strategy: "balanced" } },
  { match: { hw: "gb300"  , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "gb200"  , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "b300"   , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "b200"   , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "h200"   , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "mi350x" , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "mi355x" , quant: "nvfp4", strategy: "dspark"   } },
  { match: { hw: "gb300"  , quant: "mxfp8", strategy: "balanced" } },
  { match: { hw: "gb200"  , quant: "mxfp8", strategy: "balanced" } },
  { match: { hw: "b300"   , quant: "mxfp8", strategy: "balanced" } },
  { match: { hw: "b200"   , quant: "mxfp8", strategy: "balanced" } },
  { match: { hw: "gb300"  , quant: "mxfp8", strategy: "dspark"   } },
  { match: { hw: "gb200"  , quant: "mxfp8", strategy: "dspark"   } },
  { match: { hw: "b300"   , quant: "mxfp8", strategy: "dspark"   } },
  { match: { hw: "b200"   , quant: "mxfp8", strategy: "dspark"   } },
  { match: { hw: "vera-rubin", quant: "mxfp8", strategy: "balanced" } },
  { match: { hw: "vera-rubin", quant: "mxfp8", strategy: "dspark"   } },
];
