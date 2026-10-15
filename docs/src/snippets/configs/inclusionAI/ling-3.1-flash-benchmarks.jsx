export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 512, max_concurrency: 1 },
        ttft_ms: 217.91, tpot_ms: 3.63, tokens_per_sec_per_gpu: 156.7 },
      { workload: { dataset: "random", isl: 4096, osl: 512, max_concurrency: 32 },
        ttft_ms: 680.80, tpot_ms: 19.06, tokens_per_sec_per_gpu: 1640.8 },
    ],
    accuracy: { gsm8k_pct: 96.89, stop_rate_pct: 99.85 },
    notes: "NEXTN. Matched standard-decoding control: GSM8K 96.97%, stop rate 99.77%. bs=1 output 250.7 tok/s, acceptance length 2.61 (random) / 2.13 (real text).",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
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
