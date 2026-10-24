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
  // 1x B200, TP=1, RadixArk/Qwen3.8-Flash-Next-NVFP4 @ 7b71922 on
  // lmsysorg/sglang:qwen38flashnext (593134d17a), Modal (8 CPU cores, 128 GiB
  // host RAM), 2026-09-30, from sgl-project/sglang#41919. The cell's flags as
  // they stood then: BF16 KV cache and QSA indexer, before #45245 pinned both to
  // FP8 (which needs v0.5.22). Two warmups, then the median of three runs; TTFT
  // and TPOT are per-run means. Only PLE Offload = Auto was run, so the entries
  // key on it. The GSM8K run (deprecated run_eval, zero-shot, legacy numeric
  // scorer) is not comparable with the other rows and is left out.
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", pleOffload: "auto" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    latencyPercentile: "Mean",
    notes: "Measured with a BF16 KV cache and QSA indexer (no --kv-cache-dtype / --qsa-indexer-dtype pins) on the qwen38flashnext image, before the FP8 defaults. From sgl-project/sglang#41919.",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 79.77, tpot_ms: 2.31, tokens_per_sec_per_gpu: 1911 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 181.76, tpot_ms: 8.5, tokens_per_sec_per_gpu: 8436 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 267.11, tpot_ms: 12.61, tokens_per_sec_per_gpu: 11284 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 214.6, tpot_ms: 2.25, tokens_per_sec_per_gpu: 3667 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 663.41, tpot_ms: 8.19, tokens_per_sec_per_gpu: 15468 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 1090.2, tpot_ms: 12.73, tokens_per_sec_per_gpu: 20075 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", pleOffload: "auto" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    latencyPercentile: "Mean",
    notes: "Measured with a BF16 KV cache and QSA indexer (no --kv-cache-dtype / --qsa-indexer-dtype pins) on the qwen38flashnext image, before the FP8 defaults. From sgl-project/sglang#41919.",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 69.59, tpot_ms: 4.78, tokens_per_sec_per_gpu: 992 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 369.51, tpot_ms: 8.49, tokens_per_sec_per_gpu: 8062 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 1097.89, tpot_ms: 16.28, tokens_per_sec_per_gpu: 15534 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 206.12, tpot_ms: 4.88, tokens_per_sec_per_gpu: 1774 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1983.33, tpot_ms: 9.68, tokens_per_sec_per_gpu: 12396 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 6591.61, tpot_ms: 21.07, tokens_per_sec_per_gpu: 20939 },
    ],
  },
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
  // 1x B300, TP=1, same checkpoint, image and protocol as the B200 NVFP4 rows
  // (Modal, 8 CPU cores, 128 GiB host RAM), 2026-09-30 to 2026-10-01, from
  // sgl-project/sglang#41937: BF16 KV cache and QSA indexer, before #45245.
  // Run with PLE Offload = Auto and = On (--ple-offload-embedding), so each
  // strategy has one entry per setting. GSM8K left out as for B200.
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", pleOffload: "auto" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    latencyPercentile: "Mean",
    notes: "Measured with a BF16 KV cache and QSA indexer (no --kv-cache-dtype / --qsa-indexer-dtype pins) on the qwen38flashnext image, before the FP8 defaults. From sgl-project/sglang#41937.",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 169.9, tpot_ms: 2.29, tokens_per_sec_per_gpu: 1696 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 298.05, tpot_ms: 11.99, tokens_per_sec_per_gpu: 5944 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 409.31, tpot_ms: 18.79, tokens_per_sec_per_gpu: 7673 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 229.27, tpot_ms: 2.16, tokens_per_sec_per_gpu: 3768 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 701.15, tpot_ms: 8.05, tokens_per_sec_per_gpu: 15826 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 1151.0, tpot_ms: 12.7, tokens_per_sec_per_gpu: 19926 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", pleOffload: "auto" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    latencyPercentile: "Mean",
    notes: "Measured with a BF16 KV cache and QSA indexer (no --kv-cache-dtype / --qsa-indexer-dtype pins) on the qwen38flashnext image, before the FP8 defaults. From sgl-project/sglang#41937.",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 149.66, tpot_ms: 4.72, tokens_per_sec_per_gpu: 944 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 476.31, tpot_ms: 8.48, tokens_per_sec_per_gpu: 7750 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 1171.11, tpot_ms: 16.38, tokens_per_sec_per_gpu: 15242 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 220.14, tpot_ms: 4.81, tokens_per_sec_per_gpu: 1792 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1937.25, tpot_ms: 9.62, tokens_per_sec_per_gpu: 12510 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 6286.6, tpot_ms: 20.9, tokens_per_sec_per_gpu: 21310 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single", pleOffload: "on" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    latencyPercentile: "Mean",
    notes: "Measured with a BF16 KV cache and QSA indexer (no --kv-cache-dtype / --qsa-indexer-dtype pins) on the qwen38flashnext image, before the FP8 defaults. From sgl-project/sglang#41937.",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 155.42, tpot_ms: 2.28, tokens_per_sec_per_gpu: 1732 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 287.21, tpot_ms: 11.49, tokens_per_sec_per_gpu: 6219 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 389.93, tpot_ms: 18.67, tokens_per_sec_per_gpu: 7736 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 230.31, tpot_ms: 2.25, tokens_per_sec_per_gpu: 3643 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 716.5, tpot_ms: 8.16, tokens_per_sec_per_gpu: 15636 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 1073.24, tpot_ms: 12.55, tokens_per_sec_per_gpu: 20261 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single", pleOffload: "on" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    latencyPercentile: "Mean",
    notes: "Measured with a BF16 KV cache and QSA indexer (no --kv-cache-dtype / --qsa-indexer-dtype pins) on the qwen38flashnext image, before the FP8 defaults. From sgl-project/sglang#41937.",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 144.25, tpot_ms: 4.7, tokens_per_sec_per_gpu: 952 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 447.02, tpot_ms: 8.47, tokens_per_sec_per_gpu: 7827 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 1174.15, tpot_ms: 16.39, tokens_per_sec_per_gpu: 15249 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 215.53, tpot_ms: 4.81, tokens_per_sec_per_gpu: 1794 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1920.87, tpot_ms: 9.64, tokens_per_sec_per_gpu: 12504 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 6318.35, tpot_ms: 20.92, tokens_per_sec_per_gpu: 21271 },
    ],
  },
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
  // (qwen4-main-squashed 4ccff141db). AIME26 and MMMU-Pro not run.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "qwen38flashnext image @ 593134d17a",
    accuracy: { gsm8k_pct: 97.0 },
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
  // (NVMe file)), qwen4-main-squashed @ 4ccff141db. Same bench workload as
  // the 2-node rows; single GPU, so per-GPU = total tok/s. GSM8K as in the
  // 2-node rows (full set, dev-qwen38-next-local image).
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "qwen4-main-squashed @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.1 },
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
    sglang_version: "qwen4-main-squashed @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.3 },
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
  // Spark pair, measured on qwen4-main-squashed 4ccff141db (which includes
  // sgl-project/sglang#38121); the qwen38flashnext image cannot load this
  // export. Same bench workload as above; GSM8K as in the RDXA rows (full
  // set, dev-qwen38-next-local image).
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "qwen4-main-squashed @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.0 },
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
    sglang_version: "qwen4-main-squashed @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.0 },
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
  // lmsysorg/sglang:dev-qwen38-next-local (4ccff141db). Same 1024/256 bench
  // workload; GSM8K as in the 2-node rows (full set).
  // The in-checkpoint MTP head is used at TP=1.
  {
    match: { hw: "dgx-spark", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.1 },
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
    sglang_version: "dev-qwen38-next-local image @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.0 },
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
  // (qwen4-main-squashed 4ccff141db, which carries the #36811 and #38290 router fixes), 2026-09-07:
  // all four cells run as the
  // command generator emits them. GSM8K is the full 1,319-question set with the
  // same chat protocol as the DGX Spark rows (chat completions API, thinking off,
  // greedy, 8,192-token budget, answer parsed from a final "The answer is N"
  // line); the run_eval
  // 5-shot / thinking-on figures are in the notes. Same 1024/256 bench workload;
  // tokens_per_sec_per_gpu is (input + output) tok/s on the one GPU; output alone
  // is one fifth of it (1024 in / 256 out, range ratio 1, ignore_eos).
  // AIME26 and MMMU-Pro not run.
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 4ccff141db",
    accuracy: { gsm8k_pct: 96.9 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 113.93, tpot_ms: 6.0, tokens_per_sec_per_gpu: 778 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 623.3, tpot_ms: 19.14, tokens_per_sec_per_gpu: 3426 },
    ],
  },
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 4ccff141db",
    accuracy: { gsm8k_pct: 96.9 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 115.53, tpot_ms: 11.44, tokens_per_sec_per_gpu: 422 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 919.64, tpot_ms: 24.95, tokens_per_sec_per_gpu: 2807 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64 },
        ttft_ms: 2977.69, tpot_ms: 54.09, tokens_per_sec_per_gpu: 4546 },
    ],
  },
  // nvidia/Qwen3.8-Flash-Next-NVFP4 on the same card and image.
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.3 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 113.49, tpot_ms: 6.02, tokens_per_sec_per_gpu: 776 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 611.69, tpot_ms: 18.62, tokens_per_sec_per_gpu: 3376 },
    ],
  },
  {
    match: { hw: "rtx6000", variant: "default", quant: "nvfp4-nvda", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-qwen38-next-local image @ 4ccff141db",
    accuracy: { gsm8k_pct: 97.0 },
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1 },
        ttft_ms: 116.27, tpot_ms: 11.47, tokens_per_sec_per_gpu: 420 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16 },
        ttft_ms: 927.28, tpot_ms: 24.97, tokens_per_sec_per_gpu: 2803 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64 },
        ttft_ms: 2958.76, tpot_ms: 54.4, tokens_per_sec_per_gpu: 4530 },
    ],
  },
  { match: { hw: "mi350x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi350x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
];
