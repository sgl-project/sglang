// Intern-S2-Mobius per-cell benchmark numbers, keyed by the same `match` tuple as
// intern-s2-mobius.jsx cells. All H200 numbers measured in this work on 2xH200
// (TP=2, sglang main @ e0828ee3 + PR #33691 head — model landed in main 2026-08-08, so lmsysorg/sglang:dev is equivalent now). Speed workload = random 8K-in /
// 1K-out, --random-range-ratio 1.0, --flush-cache, warmup 8-16 prompts, varied seed.
// ttft_ms / tpot_ms are P50. tokens_per_sec_per_gpu = total (in+out)/GPU.
export const benchmarks = [
  // ==== H200, low-latency (MTP NEXTN 3-1-4) ====
  // EAGLE MTP doubles-to-triples single-stream decode and lifts mid-concurrency
  // total throughput; at conc=64 the spec recipe still wins on total throughput
  // (26033 vs 21395 tok/s) but the no-spec recipe at conc=256 wins overall
  // (34786 — its saturation point). Spec accept_length ~3.9/4.
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ e0828ee3",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 177.91, tpot_ms: 3.13, tokens_per_sec_per_gpu: 1283.2 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1139.88, tpot_ms: 6.84, tokens_per_sec_per_gpu: 9014.6 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4440.38, tpot_ms: 12.17, tokens_per_sec_per_gpu: 13016.5 },
    ],
    accuracy: { gsm8k_pct: 96.66, gpqa_pct: 79.23 },
  },

  // ==== H200, high-throughput (no speculative decoding) ====
  // Saturates at conc=1024 submitted (server clamps max_running to ~730 — the
  // real concurrency observed). Spec-off at the largest batch sizes reaches
  // ~35k total tok/s, >1.3× the spec-on peak at conc=64, which is why the
  // high-throughput recipe stays spec-free.
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ e0828ee3",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1166.47, tpot_ms: 14.26, tokens_per_sec_per_gpu: 4679.2 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4112.65, tpot_ms: 22.91, tokens_per_sec_per_gpu: 10697.7 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 16290.63, tpot_ms: 50.31, tokens_per_sec_per_gpu: 17393.4 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 182448.22, tpot_ms: 123.82, tokens_per_sec_per_gpu: 11875.5 },
    ],
    accuracy: { gsm8k_pct: 96.82 },
  },

  // ==== B200 recipes are inferred from the H200 ones — benchmarks pending. ====
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
];
