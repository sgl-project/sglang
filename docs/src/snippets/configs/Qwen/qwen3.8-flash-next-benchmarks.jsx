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
  // 2026-09-04. GSM8K is the full 1,319-question set via the chat API (thinking
  // off, greedy, 8192 max tokens) on lmsysorg/sglang:dev-qwen38-next-local
  // (9b2aee2283), 2026-09-06; rows without it are still being run. AIME26 and
  // MMMU-Pro not run.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
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
  // 1x DGX Spark, TP=1, N-gram table file-backed on NVMe (PLE Offload = On
  // (NVMe file)), qwen4-main-squashed @ 9b2aee2283. Same bench workload as
  // the 2-node rows; single GPU, so per-GPU = total tok/s. Accuracy pending.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 648.92, tpot_ms: 33.59, tokens_per_sec_per_gpu: 137 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 4 },
        ttft_ms: 1089.20, tpot_ms: 60.10, tokens_per_sec_per_gpu: 288 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 8 },
        ttft_ms: 1341.05, tpot_ms: 94.64, tokens_per_sec_per_gpu: 358 },
    ],
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 580.18, tpot_ms: 61.63, tokens_per_sec_per_gpu: 80 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 7281.52, tpot_ms: 181.30, tokens_per_sec_per_gpu: 386 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 24 },
        ttft_ms: 8336.78, tpot_ms: 241.20, tokens_per_sec_per_gpu: 415 },
    ],
  },
  // nvidia/Qwen3.8-Flash-Next-NVFP4 (ModelOpt MIXED_PRECISION) on the same
  // Spark pair, measured on the qwen4-main-squashed tip 9b2aee2283 (which
  // includes sgl-project/sglang#38121) — the shipped image cannot load this
  // export yet. Same bench workload as above; GSM8K as in the RDXA rows (full
  // set, dev-qwen38-next-local image, 2026-09-06).
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "qwen4-main-squashed @ 9b2aee2283",
    accuracy: { gsm8k_pct: 96.8 },
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
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 398.49, tpot_ms: 38.62, tokens_per_sec_per_gpu: 61 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 4949.99, tpot_ms: 90.58, tokens_per_sec_per_gpu: 364 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 96 },
        ttft_ms: 17962.94, tpot_ms: 289.52, tokens_per_sec_per_gpu: 632 },
    ],
  },
  // 1x DGX Spark, TP=1, nvidia export with the N-gram table file-backed on NVMe,
  // lmsysorg/sglang:dev-qwen38-next-local (9b2aee2283), 2026-09-06. Same
  // 1024/256 bench workload; accuracy pending. The in-checkpoint MTP head is
  // used at TP=1.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 9b2aee2283",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 650.03, tpot_ms: 33.26, tokens_per_sec_per_gpu: 136 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 4 },
        ttft_ms: 1546.18, tpot_ms: 61.22, tokens_per_sec_per_gpu: 266 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 8 },
        ttft_ms: 2167.14, tpot_ms: 95.63, tokens_per_sec_per_gpu: 342 },
    ],
  },
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 9b2aee2283",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 604.33, tpot_ms: 63.54, tokens_per_sec_per_gpu: 78 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 6272.56, tpot_ms: 179.25, tokens_per_sec_per_gpu: 397 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 24 },
        ttft_ms: 7151.52, tpot_ms: 246.79, tokens_per_sec_per_gpu: 443 },
    ],
  },
  // 1x RTX PRO 6000 Blackwell (96 GB), TP=1, lmsysorg/sglang:dev-qwen38-next-local
  // (qwen4-main-squashed 9b2aee2283), 2026-09-06: all four cells run as the
  // command generator emits them. GSM8K is the full 1,319-question set with the
  // same chat protocol as the DGX Spark rows (chat completions API, thinking off,
  // greedy, answer parsed from a final "The answer is N" line); the run_eval
  // 5-shot / thinking-on figures are in the notes. Same 1024/256 bench workload;
  // tokens_per_sec_per_gpu is (input + output) tok/s on the one GPU; output alone
  // is one fifth of it (1024 in / 256 out, range ratio 1, ignore_eos).
  // AIME26 and MMMU-Pro not run.
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 9b2aee2283",
    accuracy: { gsm8k_pct: 95.8 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 235.23, tpot_ms: 6.21, tokens_per_sec_per_gpu: 705 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 1007.94, tpot_ms: 19.54, tokens_per_sec_per_gpu: 3065 },
    ],
  },
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 9b2aee2283",
    accuracy: { gsm8k_pct: 95.8 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 162.25, tpot_ms: 11.49, tokens_per_sec_per_gpu: 415 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 1327.21, tpot_ms: 25.11, tokens_per_sec_per_gpu: 2645 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64 },
        ttft_ms: 3212.36, tpot_ms: 54.93, tokens_per_sec_per_gpu: 4425 },
    ],
  },
  // nvidia/Qwen3.8-Flash-Next-NVFP4 on the same card and image.
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 9b2aee2283",
    accuracy: { gsm8k_pct: 95.9 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 225.43, tpot_ms: 5.86, tokens_per_sec_per_gpu: 745 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 1032.11, tpot_ms: 18.74, tokens_per_sec_per_gpu: 3170 },
    ],
  },
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 9b2aee2283",
    accuracy: { gsm8k_pct: 96.4 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 158.37, tpot_ms: 11.54, tokens_per_sec_per_gpu: 415 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 1184.19, tpot_ms: 25.12, tokens_per_sec_per_gpu: 2695 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64 },
        ttft_ms: 3314.12, tpot_ms: 55.67, tokens_per_sec_per_gpu: 4355 },
    ],
  },
  { match: { hw: "mi350x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi350x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
];
