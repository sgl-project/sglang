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
];
