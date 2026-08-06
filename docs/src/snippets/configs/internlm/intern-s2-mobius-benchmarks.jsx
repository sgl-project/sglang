// One benchmarks entry per cell `match` from intern-s2-mobius.jsx. Numbers are
// p50 latency (default `latencyPercentile: "P50"` from the engine). Speed workload
// = random 8K in / 1K out, --flush-cache, warmup=8, varied seed per run.
export const benchmarks = [
  // ==== H200, low-latency (MTP NEXTN 3-1-4) ====
  // Spec ON doubles-to-triples single-stream decode speed and lifts conc=16
  // total throughput from 9257 → 16061 tok/s (the better conc=16 cell).
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ e0828ee3",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 165.84, tpot_ms: 3.13, tokens_per_sec_per_gpu: 1285.6 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1414.39, tpot_ms: 7.35, tokens_per_sec_per_gpu: 8030.5 },
    ],
    accuracy: { gsm8k_pct: 96.66, gpqa_pct: 79.23 },
  },

  // ==== H200, high-throughput (spec off) ====
  // At conc≫16 the draft+verify overhead outweighs spec's latency gain — the
  // spec-off recipe reaches 20969 tok/s at conc=64 where MTP would only add compute.
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ e0828ee3",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4121.16, tpot_ms: 23.45, tokens_per_sec_per_gpu: 10484.4 },
    ],
    accuracy: { gsm8k_pct: 96.82 },
  },

  // ==== B200, 2-GPU low-latency — recipe inferred from H200; benchmarks pending. ====
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },

  // ==== B200, 1-GPU high-throughput — recipe inferred from H200; benchmarks pending. ====
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
];
