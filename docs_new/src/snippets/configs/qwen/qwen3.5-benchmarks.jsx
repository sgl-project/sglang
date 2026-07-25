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

  // MI300X Qwen3.5-4B (BF16, TP=1) — measured on 8×MI300X (single GPU used).
  // Server: sglang v0.5.16, --attention-backend aiter, SGLANG_USE_AITER=1.
  // Low-latency cell uses EAGLE speculative decoding (built-in head).
  // Workload: random ISL=8192, OSL=1024.
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    accuracy: { gsm8k_pct: 78.85 },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 138,
        tpot_ms: 3.87,
        tokens_per_sec_per_gpu: 243,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 597,
        tpot_ms: 7.03,
        tokens_per_sec_per_gpu: 1622,
      },
    ],
  },
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 315,
        tpot_ms: 23.93,
        tokens_per_sec_per_gpu: 2584,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4554,
        tpot_ms: 84.91,
        tokens_per_sec_per_gpu: 2978,
      },
    ],
  },
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 109282,
        tpot_ms: 86.18,
        tokens_per_sec_per_gpu: 2894,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 361004,
        tpot_ms: 82.25,
        tokens_per_sec_per_gpu: 2742,
      },
    ],
  },
];
