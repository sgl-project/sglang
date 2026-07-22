// One entry per cell `match` tuple. Accuracy is per-cell measured (keyed to
// config.accuracyLabels), taken at reasoning effort max (0.99) on the balanced
// recipe for each platform. Speed is pending — fill tokens_per_sec_per_gpu /
// ttft_ms / tpot_ms once bench_serving has been run per cell.
//
// Accuracy provenance: BFCL v3 / MMAU / MMMU-Pro / AIME25 (pass@1, avg of 8) /
// NIAH single-needle / HLE (self-judge, text subset). NVIDIA cells ran on the
// lmsysorg/sglang:inkling-cu13 image (inkling-support branch); AMD on
// inkling-rocm700-mi35x. NIAH shows the two long-context buckets (512K / 1M) —
// all platforms score ~1.0 below ~220K. GB300 balanced HLE not yet run.

export const benchmarks = [
  { match: { hw: "b200"   , variant: "default" , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   },
    sglang_version: "inkling-support (inkling-cu13)",
    accuracy: { bfcl_pct: 77.9, mmau_pct: 78.3, mmmu_pro_pct: 74.0, aime25_pct: 94.6, niah_512k_pct: 93.9, niah_1m_pct: 75.8, hle_pct: 29.5 } },
  { match: { hw: "b300"   , variant: "default" , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   },
    sglang_version: "inkling-support (inkling-cu13)",
    accuracy: { bfcl_pct: 78.5, mmau_pct: 77.5, mmmu_pro_pct: 74.1, aime25_pct: 95.0, niah_512k_pct: 90.9, niah_1m_pct: 72.7, hle_pct: 29.3 } },
  { match: { hw: "gb200"  , variant: "default" , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "gb300"  , variant: "default" , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   },
    sglang_version: "inkling-support (inkling-cu13)",
    accuracy: { bfcl_pct: 77.9, mmau_pct: 78.4, mmmu_pro_pct: 74.0, aime25_pct: 96.3, niah_512k_pct: 93.9, niah_1m_pct: 81.8 } },
  { match: { hw: "h200"   , variant: "default" , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   },
    sglang_version: "inkling-support (inkling-cu13)",
    accuracy: { bfcl_pct: 77.0, mmau_pct: 77.7, mmmu_pro_pct: 74.3, aime25_pct: 96.7, niah_512k_pct: 90.9, niah_1m_pct: 75.8, hle_pct: 28.8 } },
  { match: { hw: "mi350x" , variant: "default" , quant: "bf16"  , strategy: "balanced"     , nodes: "single"   },
    sglang_version: "inkling-rocm700-mi35x",
    accuracy: { bfcl_pct: 77.8, mmau_pct: 76.3, mmmu_pro_pct: 74.4, aime25_pct: 95.0, hle_pct: 29.4 } },
  { match: { hw: "mi355x" , variant: "default" , quant: "bf16"  , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "b200"   , variant: "default" , quant: "nvfp4" , strategy: "mtp"          , nodes: "single"   } },
  { match: { hw: "b300"   , variant: "default" , quant: "nvfp4" , strategy: "mtp"          , nodes: "single"   } },
  { match: { hw: "gb200"  , variant: "default" , quant: "nvfp4" , strategy: "mtp"          , nodes: "single"   } },
  { match: { hw: "gb300"  , variant: "default" , quant: "nvfp4" , strategy: "mtp"          , nodes: "single"   } },
  { match: { hw: "h200"   , variant: "default" , quant: "nvfp4" , strategy: "mtp"          , nodes: "single"   } },
  { match: { hw: "b200"   , variant: "default" , quant: "nvfp4" , strategy: "long_context" , nodes: "single"   } },
  { match: { hw: "b300"   , variant: "default" , quant: "nvfp4" , strategy: "long_context" , nodes: "single"   } },
  { match: { hw: "gb200"  , variant: "default" , quant: "nvfp4" , strategy: "long_context" , nodes: "single"   } },
  { match: { hw: "gb300"  , variant: "default" , quant: "nvfp4" , strategy: "long_context" , nodes: "single"   } },
  { match: { hw: "gb300"  , variant: "default" , quant: "bf16"  , strategy: "balanced"     , nodes: "multi-2"  },
    sglang_version: "inkling-support (inkling-cu13)",
    accuracy: { bfcl_pct: 78.3, mmau_pct: 76.9, mmmu_pro_pct: 74.7, aime25_pct: 95.0, niah_512k_pct: 90.9, niah_1m_pct: 78.8 } },
  { match: { hw: "gb300"  , variant: "default" , quant: "bf16"  , strategy: "mtp"          , nodes: "multi-2"  } },
  { match: { hw: "b300"   , variant: "default" , quant: "bf16"  , strategy: "balanced"     , nodes: "single"   },
    sglang_version: "inkling-support (inkling-cu13)",
    accuracy: { bfcl_pct: 78.1, mmau_pct: 77.3, mmmu_pro_pct: 74.7, aime25_pct: 96.3, niah_512k_pct: 90.9, niah_1m_pct: 78.8, hle_pct: 29.7 } },
  { match: { hw: "b300"   , variant: "default" , quant: "bf16"  , strategy: "mtp"          , nodes: "single"   } },
  { match: { hw: "b200"   , variant: "default" , quant: "bf16"  , strategy: "balanced"     , nodes: "multi-2"  } },
  { match: { hw: "b200"   , variant: "default" , quant: "bf16"  , strategy: "mtp"          , nodes: "multi-2"  } },
  { match: { hw: "b200"   , variant: "lora"    , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "b300"   , variant: "lora"    , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "gb200"  , variant: "lora"    , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "gb300"  , variant: "lora"    , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "h200"   , variant: "lora"    , quant: "nvfp4" , strategy: "balanced"     , nodes: "single"   } },
  { match: { hw: "gb300"  , variant: "lora"    , quant: "bf16"  , strategy: "balanced"     , nodes: "multi-2"  } },
  { match: { hw: "h200"   , variant: "lora"    , quant: "bf16"  , strategy: "balanced"     , nodes: "single"   } },
];
