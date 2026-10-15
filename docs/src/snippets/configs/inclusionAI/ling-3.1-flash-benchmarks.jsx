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
    accuracy: { gsm8k_pct: 97.0, stop_rate_pct: 100.0 },
    notes: "NEXTN; acceptance length 2.59 (bench) / 2.40. Accuracy from the 100-question GSM8K smoke on this exact recipe (97/100, all stop).",
  },
];
