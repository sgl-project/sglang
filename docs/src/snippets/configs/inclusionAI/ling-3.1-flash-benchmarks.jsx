export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 512, max_concurrency: 1 },
        ttft_ms: 228.0, tpot_ms: 3.35, tokens_per_sec_per_gpu: 168.3 },
    ],
    accuracy: { gsm8k_pct: 97.12, stop_rate_pct: 99.77 },
    notes: "TP-only (EP=1) with NEXTN; full 1319-question GSM8K. bs=1 output 269.2 tok/s, acceptance length 2.59-2.61.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 48 },
        ttft_ms: 600.8, tpot_ms: 27.60, tokens_per_sec_per_gpu: 1819.4 },
    ],
    accuracy: { gsm8k_pct: 96.82, stop_rate_pct: 99.92 },
    notes: "EP8 with NEXTN at full saturation (max_running_requests=48); pushing concurrency to 1024 only adds queue time (14.6k total tok/s). bs=1 output 239.5 tok/s, acceptance length 2.52-2.65.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 512, max_concurrency: 1 },
        ttft_ms: 296.0, tpot_ms: 5.22, tokens_per_sec_per_gpu: 219.1 },
    ],
    accuracy: { gsm8k_pct: 97.19, stop_rate_pct: 99.85 },
    notes: "TP-only (EP=1) with NEXTN; full 1319-question GSM8K (1282/1319). bs=1 output 175.3 tok/s vs 160.0 on EP4 (+9.6%). Requires the TP-requant loader plus the Triton shared-expert K-tail route on SM103.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 377.4, tpot_ms: 15.72, tokens_per_sec_per_gpu: 2142.3 },
    ],
    accuracy: { gsm8k_pct: 97.42, stop_rate_pct: 99.92 },
    notes: "NEXTN EP4; full 1319-question GSM8K on this exact recipe (1285/1319). bs=1 output 160.0 tok/s (TPOT 5.12 ms). Acceptance length 2.56-2.59.",
  },
  {
    match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "rocm724-mi30x nightly 20261011 + ROCm gated-MLA fix",
    accuracy: { gsm8k_pct: 97.27, stop_rate_pct: 99.92 },
    notes: "Measured by AMD on 8x MI325X with a pre-release Ling-3.1-flash checkpoint, not rerun by SGLang on the public one. Full 1319-question GSM8K: 1283 correct, 1318 stops, one 8192-token truncation, zero request errors. AITER e7d2453f.",
  },
];
