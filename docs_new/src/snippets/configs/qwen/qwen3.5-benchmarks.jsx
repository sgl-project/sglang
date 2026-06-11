// Measured numbers for the Qwen3.5 cookbook — transcribed verbatim from the
// legacy page's §5 Benchmark section (H200 ×8, TP=8, sglang "main branch",
// NEXTN speculative decoding; the measured run also had the reasoning and
// tool-call parsers enabled, which the matching cell omits — parser flags are
// a Playground feature, never part of Deployment commands).
// tokens_per_sec_per_gpu = output token throughput ÷ 8 GPUs. Cells without an
// entry render as "pending".

export const benchmarks = [
  {
    match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "main branch",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 190.4, tpot_ms: 3.96, tokens_per_sec_per_gpu: 27.9 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 14247.21, tpot_ms: 26.16, tokens_per_sec_per_gpu: 220.9 },
    ],
    accuracy: { gsm8k_pct: 97.5, mmmu_pct: 97.8 },
    notes: "GSM8K via benchmark/gsm8k/bench_sglang.py (200 questions); MMMU via benchmark/mmmu/bench_sglang.py (91-sample val subset).",
  },
];
