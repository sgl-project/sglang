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
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 48 },
        ttft_ms: 550, tpot_ms: 24.60, tokens_per_sec_per_gpu: 2012.2 },
    ],
    accuracy: { gsm8k_pct: 96.89, stop_rate_pct: 99.77 },
    notes: "EP8 with NEXTN at full saturation (max_running_requests=48); pushing concurrency to 1024 only adds queue time, throughput unchanged (16.1k total tok/s). bs=1 output 250.7 tok/s, accept length 2.51-2.61.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 512, max_concurrency: 1 },
        ttft_ms: 302.0, tpot_ms: 5.26, tokens_per_sec_per_gpu: 215.1 },
    ],
    accuracy: { gsm8k_pct: 97.04, stop_rate_pct: 99.92 },
    notes: "TP-only (EP=1) with NEXTN; full 1319-question GSM8K (1280/1319). bs=1 output 172.1 tok/s vs 162.5 on EP4 (+5.9%). Requires the TP-requant loader plus the Triton shared-expert K-tail route on SM103.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-ling-3.1-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 405.7, tpot_ms: 15.42, tokens_per_sec_per_gpu: 2220.8 },
    ],
    accuracy: { gsm8k_pct: 96.97, stop_rate_pct: 99.85 },
    notes: "NEXTN EP4; full 1319-question GSM8K on this exact recipe (1279/1319). bs=1 output 162.5 tok/s (TPOT 5.62 ms). Acceptance length 2.54-2.59.",
  },
  {
    match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "rocm724-mi30x nightly 20261011 + ROCm gated-MLA fix",
    accuracy: { gsm8k_pct: 97.27, stop_rate_pct: 99.92 },
    notes: "Measured by AMD on 8x MI325X with a pre-release Ling-3.1-flash checkpoint, not rerun by SGLang on the public one. Full 1319-question GSM8K: 1283 correct, 1318 stops, one 8192-token truncation, zero request errors. AITER e7d2453f.",
  },
];
