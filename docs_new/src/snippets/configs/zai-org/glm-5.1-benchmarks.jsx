// GLM-5.1 measured benchmarks — faithfully migrated from the legacy page's §5.
//
// Only ONE result survives the reproducibility rule:
//   • Speed (§5.1): H200 ×8, GLM-5.1-FP8, tp=8 — pinned to `commit 947927bdb`
//     (a reproducible commit-hash anchor). The measured deploy ran EAGLE
//     speculative decoding (Accept length ~3.5) and no DP-Attention, matching
//     the h200/fp8/low-latency cell. The bench deploy did NOT set parsers, so
//     the cell's parsers-OFF flags equal the measured command.
//
// DROPPED (not migrated):
//   • Accuracy GSM8K 0.955 / MMLU 0.877 (§5.2): the page's own <Note> states
//     these are SHARED WITH GLM-5 (cross-model) and "GLM-5.1 was not
//     independently benchmarked" — cross-model numbers are never inherited.
//   • AMD GSM8K 0.970 (§5.3): anchored to "AMD nightly CI" (a moving ref) and
//     PR #18911, which is titled "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test"
//     — a GLM-5 (cross-model) day-0 test on a nightly build. Dropped on both
//     the cross-model and the non-reproducible-anchor grounds.
//
// tokens_per_sec_per_gpu = output tok/s ÷ (tp × nnodes) = output tok/s ÷ 8.

export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "commit 947927bdb",
    speed: [
      // §5.1.1 Latency: isl=1000, osl=1000, concurrency=1 (num-prompts 10).
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 290.88, tpot_ms: 7.54, tokens_per_sec_per_gpu: 14.745 },
      // §5.1.2 Throughput: isl=1000, osl=1000, concurrency=100 (num-prompts 1000).
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 20613.80, tpot_ms: 38.73, tokens_per_sec_per_gpu: 151.871 },
    ],
  },
];
