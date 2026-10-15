export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 512, max_concurrency: 1 },
        ttft_ms: 230.5, tpot_ms: 3.29, tokens_per_sec_per_gpu: 166.1 },
    ],
    accuracy: { gsm8k_pct: 97.12, stop_rate_pct: 99.92 },
    notes: "TP-only (EP=1) with NEXTN; full 1319-question GSM8K. bs=1 output 265.7 tok/s, acceptance length 2.58-2.60.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 290.71, tpot_ms: 3.74, tokens_per_sec_per_gpu: 279.1 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 331.71, tpot_ms: 12.44, tokens_per_sec_per_gpu: 1357.1 },
    ],
    accuracy: { gsm8k_pct: 96.89, stop_rate_pct: 99.77 },
    notes: "EP8 with NEXTN. bs=1 output 248.1 tok/s (this workload); accept length 2.51-2.61. Matched standard-decoding control: 96.97% / 99.77%.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 512, max_concurrency: 1 },
        ttft_ms: 256.95, tpot_ms: 5.14, tokens_per_sec_per_gpu: 224.9 },
    ],
    accuracy: { gsm8k_pct: 97.19, stop_rate_pct: 99.70 },
    notes: "NEXTN; full 1319-question GSM8K on this exact recipe (1282/1319). Acceptance length 2.58 (full run) / 2.59 (bench).",
  },
  {
    match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "rocm724-mi30x nightly 20261011 + ROCm gated-MLA fix",
    accuracy: { gsm8k_pct: 97.27, stop_rate_pct: 99.92 },
    notes: "Measured by AMD on 8x MI325X with a pre-release Ling-3.1-flash checkpoint, not rerun by SGLang on the public one. Full 1319-question GSM8K: 1283 correct, 1318 stops, one 8192-token truncation, zero request errors. AITER e7d2453f.",
  },
];
