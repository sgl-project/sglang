// Hy4-Preview benchmark data — one entry per config cell `match` tuple.
//
// All entries are bare-match stubs (the card shows "pending"). When a cell's
// numbers land, fill the entry with the measured data and set
// `sglang_version` to the exact commit/tag they were measured on (a
// reproducible anchor — never a moving ref like "main").
//
// Speed shape when filling:
//   speed: [{ workload: {dataset, isl, osl, max_concurrency}, ttft_ms,
//             tpot_ms, tokens_per_sec_per_gpu }, ...]   // P50 latencies
// Accuracy: per-cell `accuracy: { gsm8k_pct: <pct> }` (harness settings in
// the config's benchmarkCommands.accuracy).

export const benchmarks = [
  // H200 (BF16 only — tested: SM90 cannot serve the MXFP8 checkpoint)
  { match: { hw: "h200",  variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "multi-2" } },
  // B200
  { match: { hw: "b200",  variant: "default", quant: "mxfp8", strategy: "low-latency",     nodes: "single"  } },
  { match: { hw: "b200",  variant: "default", quant: "mxfp8", strategy: "high-throughput", nodes: "single"  } },
  { match: { hw: "b200",  variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "multi-2" } },
  // B300
  { match: { hw: "b300",  variant: "default", quant: "mxfp8", strategy: "low-latency",     nodes: "single"  } },
  { match: { hw: "b300",  variant: "default", quant: "mxfp8", strategy: "high-throughput", nodes: "single"  } },
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single"  } },
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single"  } },
  // GB300
  { match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "low-latency",     nodes: "single"  } },
  { match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "high-throughput", nodes: "single"  } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "multi-2" } },
];
