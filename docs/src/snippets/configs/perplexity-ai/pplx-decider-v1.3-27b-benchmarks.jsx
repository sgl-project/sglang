// PPLX-Decider-v1.3-27B on one NVIDIA H20 (96 GB), BF16.
//
// The model is prefill-only: the answer is read at the final <|fim_suffix|>
// marker, so TTFT is the whole request latency, TPOT is null, and throughput is
// prompt tokens/s/GPU. Each cell uses a closed-loop client, eight warmup
// requests, a cache flush, and three measured repeats. TTFT is the P50.
//
// SGLang code: PR #53432 at 9deba5ff7d. The server auto-selected FA3 for text
// and vision, Triton for GDN, and the Rust tree core. It also auto-disabled the
// radix cache and chunked prefill. Command adds --mem-fraction-static 0.85.
export const benchmarks = [
  {
    match: { hw: "h20", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #53432 @ 9deba5ff7d",
    speed: [
      { workload: { dataset: "random", isl: 382, osl: 0, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 158.6, tpot_ms: null, tokens_per_sec_per_gpu: 2408 },
      { workload: { dataset: "random", isl: 382, osl: 0, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 8966.1, tpot_ms: null, tokens_per_sec_per_gpu: 2711 },
      { workload: { dataset: "random", isl: 8224, osl: 0, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 3230.7, tpot_ms: null, tokens_per_sec_per_gpu: 2542 },
      { workload: { dataset: "random", isl: 8224, osl: 0, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 51298.2, tpot_ms: null, tokens_per_sec_per_gpu: 2565 },
    ],
    accuracy: { belebele_pct: 97.56, winogrande_pct: 94.95 },
    notes: "Prefill-only model: TTFT is the full decision latency and throughput is prompt tokens/s/GPU. H20 BF16 peak is substantially lower than H200/GB300, so compare within the same hardware.",
  },
]
