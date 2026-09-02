// Direct native SGLang measurements on NVIDIA H200 GPUs.
// Runtime source: 70997c0fc6429162bffd4476436c398951eb1647.
// Container image build: 20621aa1.
//
// Every displayed value is the mean of two independent server launches. Each
// server used BF16 runtime dtype, FA3 attention, and --reasoning-parser k2_v3.
// Tensor parallelism was TP1 for 0.9B/3.7B/7B, TP2 for 32B/36B, and TP8 for
// 375B.
//
// Speed workload: sglang.benchmark.serving with random token IDs, fixed
// 8192-token input and 1024-token output, random-range-ratio 1.0, request-rate
// inf, 64 warmup requests, cache flush, temperature 0, top-p 1, and seed
// 20260901. The C1 point uses 32 prompts. The high-concurrency point uses 256
// prompts at concurrency 64 (TP1), 32 (TP2), or 8 (TP8). TTFT and TPOT are the
// arithmetic mean of the two native median values reported by the benchmark.
// Throughput/GPU is the mean of total input-plus-output tok/s divided by TP.
//
// Accuracy workload: full GSM8K test split (1,319 examples), sgl-eval, 32
// threads, max_tokens 32768, temperature 0, top-p 0.95, seed 0, and reasoning
// effort high. Both launches had a 0% request error rate. Display values are
// rounded to two decimal places.
//
// Per-launch raw files are retained in campaign
// k2-horizon-direct-benchmark-70997c0fc-d501b3e7e-20260901-private-v1 under
// results/<job>/performance/{c1,high_concurrency}/benchmark.raw.jsonl and the
// corresponding results/<job>/accuracy directory. Job pairs:
//   0.9B: k2-horizon-0.9b-r0-2290615, k2-horizon-0.9b-r1-2290638
//   3.7B: k2-horizon-3.7b-r0-2290616, k2-horizon-3.7b-r1-2290639
//   7B:   k2-horizon-7b-r0-2290617,   k2-horizon-7b-r1-2290641
//   32B:  k2-horizon-32b-r0-2290618,  k2-horizon-32b-r1-2290640
//   36B:  k2-horizon-36b-r0-2290619,  k2-horizon-36b-r1-2290642
//   375B: k2-horizon-375b-r0-2290643, k2-horizon-375b-r1-2290644

export const benchmarks = [
  {
    match: { hw: "h200", variant: "0.9b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "source 70997c0fc",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 47.55,
        tpot_ms: 1.74,
        tokens_per_sec_per_gpu: 5025.24,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 1418.67,
        tpot_ms: 11.18,
        tokens_per_sec_per_gpu: 45773.21,
      },
    ],
    accuracy: { gsm8k_pct: 85.25 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 6.90% in both launches.",
  },
  {
    match: { hw: "h200", variant: "3.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "source 70997c0fc",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 158.71,
        tpot_ms: 5.10,
        tokens_per_sec_per_gpu: 1712.15,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 5174.71,
        tpot_ms: 28.83,
        tokens_per_sec_per_gpu: 16998.94,
      },
    ],
    accuracy: { gsm8k_pct: 92.00 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 4.09% and 4.70%.",
  },
  {
    match: { hw: "h200", variant: "7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "source 70997c0fc",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 249.31,
        tpot_ms: 6.74,
        tokens_per_sec_per_gpu: 1289.51,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 8175.68,
        tpot_ms: 33.81,
        tokens_per_sec_per_gpu: 13784.03,
      },
    ],
    accuracy: { gsm8k_pct: 94.88 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.91% and 1.14%.",
  },
  {
    match: { hw: "h200", variant: "32b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "source 70997c0fc",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 590.81,
        tpot_ms: 13.18,
        tokens_per_sec_per_gpu: 327.51,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 256 },
        ttft_ms: 10370.53,
        tpot_ms: 30.79,
        tokens_per_sec_per_gpu: 3503.00,
      },
    ],
    accuracy: { gsm8k_pct: 95.98 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.45% and 0.53%.",
  },
  {
    match: { hw: "h200", variant: "36b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "source 70997c0fc",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 218.54,
        tpot_ms: 10.77,
        tokens_per_sec_per_gpu: 410.17,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 256 },
        ttft_ms: 3639.21,
        tpot_ms: 27.67,
        tokens_per_sec_per_gpu: 4613.54,
      },
    ],
    accuracy: { gsm8k_pct: 95.15 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.83% and 0.99%.",
  },
  {
    match: { hw: "h200", variant: "375b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "source 70997c0fc",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 299.65,
        tpot_ms: 9.65,
        tokens_per_sec_per_gpu: 113.25,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 8, num_prompts: 256 },
        ttft_ms: 1408.10,
        tpot_ms: 15.68,
        tokens_per_sec_per_gpu: 524.00,
      },
    ],
    accuracy: { gsm8k_pct: 95.56 },
    notes: "Speed and GSM8K are the mean of two independent server launches. GSM8K truncation was 0.45% and 0.38%.",
  },
];
