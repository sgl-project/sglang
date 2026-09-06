export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.73, aime26_pct: 97.92 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.57, aime26_pct: 99.17 },
  },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.65, aime26_pct: 98.33, mmmu_pro_pct: 77.51 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.50, aime26_pct: 98.33, mmmu_pro_pct: 77.57 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.57, aime26_pct: 98.33, mmmu_pro_pct: 76.94 },
  },
  { match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.12, aime26_pct: 97.50, mmmu_pro_pct: 77.17 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.65, aime26_pct: 97.92, mmmu_pro_pct: 77.92 },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.65, aime26_pct: 99.58, mmmu_pro_pct: 77.34 },
  },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.42, aime26_pct: 98.33, mmmu_pro_pct: 77.34 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.65, aime26_pct: 98.33, mmmu_pro_pct: 77.69 },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main @ e17062a1d",
    accuracy: { gsm8k_pct: 97.50, aime26_pct: 99.17, mmmu_pro_pct: 76.42 },
  },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  // 2x DGX Spark, TP=2, lmsysorg/sglang:qwen38flashnext (SGLang 593134d17a),
  // 2026-09-04. GSM8K here is the chat-API protocol with thinking off, n=200
  // (not the full 1,319-question set the datacenter rows use); AIME26 and
  // MMMU-Pro not run.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    accuracy: { gsm8k_pct: 97.5 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 457.14, tpot_ms: 19.94, tokens_per_sec_per_gpu: 116 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 2027.14, tpot_ms: 79.16, tokens_per_sec_per_gpu: 379 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 24 },
        ttft_ms: 2301.03, tpot_ms: 101.95, tokens_per_sec_per_gpu: 464 },
    ],
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "multi-2" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    accuracy: { gsm8k_pct: 97.0 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 416.42, tpot_ms: 39.33, tokens_per_sec_per_gpu: 60 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 4035.18, tpot_ms: 90.87, tokens_per_sec_per_gpu: 372 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 96 },
        ttft_ms: 19531.95, tpot_ms: 301.92, tokens_per_sec_per_gpu: 603 },
    ],
  },
  // nvidia/Qwen3.8-Flash-Next-NVFP4 (ModelOpt MIXED_PRECISION) on the same
  // Spark pair, measured on the qwen4-main-squashed tip 9b2aee2283 (which
  // includes sgl-project/sglang#38121) — the shipped image cannot load this
  // export yet. Same GSM8K
  // protocol (chat API, thinking off, n=200) and bench workload as above.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    accuracy: { gsm8k_pct: 97.5 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 447.13, tpot_ms: 18.70, tokens_per_sec_per_gpu: 119 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 2161.72, tpot_ms: 73.20, tokens_per_sec_per_gpu: 415 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 24 },
        ttft_ms: 1608.84, tpot_ms: 97.80, tokens_per_sec_per_gpu: 496 },
    ],
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "multi-2" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    accuracy: { gsm8k_pct: 97.5 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 398.49, tpot_ms: 38.62, tokens_per_sec_per_gpu: 61 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 4949.99, tpot_ms: 90.58, tokens_per_sec_per_gpu: 364 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 96 },
        ttft_ms: 17962.94, tpot_ms: 289.52, tokens_per_sec_per_gpu: 632 },
    ],
  },
  // 1x RTX PRO 6000 Blackwell (96 GB), TP=1, lmsysorg/sglang:qwen38flashnext
  // (SGLang 593134d17a), 2026-09-05. Same chat-API GSM8K protocol as the DGX
  // Spark rows (thinking off, n=200); AIME26 and MMMU-Pro not run.
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    accuracy: { gsm8k_pct: 97.0 },
  },
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    accuracy: { gsm8k_pct: 98.0 },
  },
  // nvidia/Qwen3.8-Flash-Next-NVFP4 on the same card, measured 2026-09-06 on the
  // qwen4-main-squashed tip 9b2aee2283 (#38121 merged; the shipped image cannot
  // load this export yet). Same GSM8K protocol and bench workload; TP=1, so
  // tokens_per_sec_per_gpu is the server's output throughput.
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    accuracy: { gsm8k_pct: 96.5 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 204.45, tpot_ms: 6.13, tokens_per_sec_per_gpu: 145 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 851.64, tpot_ms: 19.10, tokens_per_sec_per_gpu: 628 },
    ],
  },
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    accuracy: { gsm8k_pct: 97.0 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 164.59, tpot_ms: 11.53, tokens_per_sec_per_gpu: 82 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 1409.63, tpot_ms: 25.40, tokens_per_sec_per_gpu: 518 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64 },
        ttft_ms: 3138.67, tpot_ms: 55.82, tokens_per_sec_per_gpu: 879 },
    ],
  },
  { match: { hw: "mi350x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi350x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
];
