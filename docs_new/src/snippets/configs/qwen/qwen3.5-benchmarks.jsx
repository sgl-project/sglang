// Measured accuracy for the Qwen3.5 cookbook — transcribed verbatim from the
// legacy page's §5 Benchmark section (H200 ×8, TP=8, NEXTN speculative
// decoding; the measured run also had the reasoning and tool-call parsers
// enabled, which the matching cell omits — parser flags are a Playground
// feature, never part of Deployment commands).
//
// The legacy page's speed numbers are NOT migrated: they were measured on a
// drifting "main branch" build, which is no version anchor — speed data is
// only meaningful against an exact, reproducible build (re-measure via the
// "⚡ Reproduce" commands and submit with a pinned release). Accuracy is far
// less build-sensitive and is kept. Cells without an entry render "pending".

export const benchmarks = [
  {
    match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    accuracy: { gsm8k_pct: 97.5, mmmu_pct: 97.8 },
    notes: "GSM8K via benchmark/gsm8k/bench_sglang.py (200 questions); MMMU via benchmark/mmmu/bench_sglang.py (91-sample val subset).",
  },

  // MI300X Qwen3.5-4B (BF16, TP=1) — measured on tw023 (8×MI300X).
  // Server: sglang v0.5.13.post1, --attention-backend triton.
  // Low-latency cell uses EAGLE speculative decoding (built-in head).
  // Workload: random ISL=1024, OSL=1024, --warmup-requests 64.
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      {
        workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 108,
        tpot_ms: 2.36,
        tokens_per_sec_per_gpu: 380,
      },
      {
        workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16 },
        ttft_ms: 178,
        tpot_ms: 5.61,
        tokens_per_sec_per_gpu: 2578,
      },
    ],
  },
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      {
        workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 166965,
        tpot_ms: 10.55,
        tokens_per_sec_per_gpu: 2314,
      },
      {
        workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 445790,
        tpot_ms: 10.13,
        tokens_per_sec_per_gpu: 2405,
      },
    ],
  },
];
