export const benchmarks = [

  // ── H200 (8×H200, tp 8, sglang 0.5.15.post1, random ISL=8192/OSL=1024) ──
  // tokens_per_sec_per_gpu = (input+output) tok/s/GPU = output_tok_s / 8 * (8192+1024)/1024
  // TTFT/TPOT are mean values from bench_serving.

  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 23742, tpot_ms: 256.8, tokens_per_sec_per_gpu: 5264 },
    ],
    // BF16 reasons ~2× longer than FP8/INT4 (median 34.8k vs 16.9k tokens);
    // truncation at max_tokens=64000 invalidates the result. Needs max_tokens ≥ 131072.
    accuracy: { gsm8k_pct: 93.18, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 102.3, tpot_ms: 3.48, tokens_per_sec_per_gpu: 305 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 97.7, tpot_ms: 6.45, tokens_per_sec_per_gpu: 2094 },
    ],
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 16199, tpot_ms: 293.2, tokens_per_sec_per_gpu: 5175 },
    ],
    accuracy: { gsm8k_pct: 94.24, aime25_pct: 67.29 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 97.7, tpot_ms: 4.43, tokens_per_sec_per_gpu: 242 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 99.4, tpot_ms: 7.16, tokens_per_sec_per_gpu: 1956 },
    ],
    accuracy: { gsm8k_pct: 94.47, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 19886, tpot_ms: 318.5, tokens_per_sec_per_gpu: 5055 },
    ],
    accuracy: { gsm8k_pct: 95.00, aime25_pct: 67.71 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 113.1, tpot_ms: 3.83, tokens_per_sec_per_gpu: 276 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 94.1, tpot_ms: 6.40, tokens_per_sec_per_gpu: 2125 },
    ],
    accuracy: { gsm8k_pct: 94.24, aime25_pct: null },
  },

  // ── B300 (8×B300, tp 8, sglang 0.5.15.post1) ──

  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.71, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.63, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.39, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 95.30, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.62, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },

  // ── GB300 (4×GB300, tp 4, sglang dev build 2026-07-20) ──
  // AIME25 null: sgl-eval --thinking sends the generic `thinking` key; Laguna's template
  // requires `enable_thinking`. Scores would reflect thinking-off behavior — omitted.

  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 93.86, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 94.16, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15",
    accuracy: { gsm8k_pct: 95.15, aime25_pct: null },
  },
];
