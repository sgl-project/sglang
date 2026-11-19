// Hy4-flash benchmark data — one entry per config cell `match` tuple.
//
// Bare-match stubs show "pending" on the card. When a cell's numbers land,
// fill the entry with the measured data and set `sglang_version` to the exact
// commit/tag they were measured on.
// tokens_per_sec_per_gpu = total (input + output) tok/s ÷ GPUs.
//
// Speed shape when filling (8k1k sweep at C = 1 and 64):
//   speed: [{ workload: {dataset: "random", isl: 8192, osl: 1024, max_concurrency},
//             ttft_ms, tpot_ms, tokens_per_sec_per_gpu }, ...]   // P50 latencies
// Accuracy: per-cell `accuracy: { gsm8k_pct: <pct> }` (harness command in the
// config's benchmarkCommands.accuracy).

export const benchmarks = [
  // Low-Latency (DFly, block size 6). B300 measured 2026-11-18, TP4: GSM8K
  // only (sgl-eval 0.1.3, 1319 examples, --num-threads 32, default sampling);
  // mean accept len ≈ 3.2 (FP8) / 3.1 (BF16). 8k1k not measured: random
  // prompts make DFly acceptance meaningless. Other platforms pending.
  {
    match: { hw: "b300", variant: "hy4-flash", quant: "fp8", strategy: "low-latency" },
    sglang_version: "lmsysorg/sglang:hy4-flash",
    accuracy: { gsm8k_pct: 96.44 },
  },
  {
    match: { hw: "b300", variant: "hy4-flash", quant: "bf16", strategy: "low-latency" },
    sglang_version: "lmsysorg/sglang:hy4-flash",
    accuracy: { gsm8k_pct: 97.04 },
  },
  { match: { hw: "gb300", variant: "hy4-flash", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "gb300", variant: "hy4-flash", quant: "bf16", strategy: "low-latency" } },
  { match: { hw: "h200", variant: "hy4-flash", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "h200", variant: "hy4-flash", quant: "bf16", strategy: "low-latency" } },
  // B300 Balanced — measured 2026-11-18, TP4, vision tower loaded.
  // GSM8K: sgl-eval 0.1.3, 1319 examples, --num-threads 32, default sampling.
  {
    match: { hw: "b300", variant: "hy4-flash", quant: "fp8", strategy: "balanced" },
    sglang_version: "lmsysorg/sglang:hy4-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 196.6, tpot_ms: 6.20, tokens_per_sec_per_gpu: 352.3 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5404.6, tpot_ms: 19.08, tokens_per_sec_per_gpu: 5906.0 },
    ],
    accuracy: { gsm8k_pct: 96.89 },
  },
  {
    match: { hw: "b300", variant: "hy4-flash", quant: "bf16", strategy: "balanced" },
    sglang_version: "lmsysorg/sglang:hy4-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 187.5, tpot_ms: 5.72, tokens_per_sec_per_gpu: 381.4 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5585.6, tpot_ms: 21.36, tokens_per_sec_per_gpu: 5375.7 },
    ],
    accuracy: { gsm8k_pct: 96.74 },
  },
  // GB300
  { match: { hw: "gb300", variant: "hy4-flash", quant: "fp8",  strategy: "balanced" } },
  {
    match: { hw: "gb300", variant: "hy4-flash", quant: "bf16", strategy: "balanced" },
    // Measured 2026-11-17 before vision support landed
    // (run with --language-model-only), TP4. GSM8K: sgl-eval 0.1.3,
    // 1319 examples, --num-threads 32, sgl-eval default sampling.
    sglang_version: "lmsysorg/sglang:hy4-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 191.5, tpot_ms: 5.50, tokens_per_sec_per_gpu: 395.0 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5367.3, tpot_ms: 20.42, tokens_per_sec_per_gpu: 5621.4 },
    ],
    accuracy: { gsm8k_pct: 96.89 },
  },
  // H200
  // H200 FP8 Balanced — measured 2026-11-18, TP4,
  // vision tower loaded. Needs sglang-kernel>=0.4.11 and sgl-fa3 (ViT FA3 on
  // Hopper). GSM8K: sgl-eval 0.1.3, 1319 examples, --num-threads 32.
  {
    match: { hw: "h200", variant: "hy4-flash", quant: "fp8", strategy: "balanced" },
    sglang_version: "lmsysorg/sglang:hy4-flash",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 294.4, tpot_ms: 6.82, tokens_per_sec_per_gpu: 316.6 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 9576.3, tpot_ms: 27.76, tokens_per_sec_per_gpu: 3873.2 },
    ],
    accuracy: { gsm8k_pct: 96.06 },
  },
  { match: { hw: "h200",  variant: "hy4-flash", quant: "bf16", strategy: "balanced" } },
];
