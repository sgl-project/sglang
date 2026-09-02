// GLM-5.3 benchmark placeholders, keyed by the same `match` tuple as glm-5.3.jsx cells.
// Bare match stubs render as pending until speed and accuracy measurements are available.
export const benchmarks = [
  // NVIDIA FP8, single node.
  {
    match: { hw: "h200",  variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.42 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 780, tpot_ms: 3.71, tokens_per_sec_per_gpu: 252 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5499, tpot_ms: 14.20, tokens_per_sec_per_gpu: 920 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=3.5 with the EAGLE 5/1/6 draft (measured accept length 3.496 at concurrency 1, 3.476 at concurrency 16). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "h200",  variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.12, aime25_pct: 91.88 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7790, tpot_ms: 21.90, tokens_per_sec_per_gpu: 2440 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 57607, tpot_ms: 33.39, tokens_per_sec_per_gpu: 2558 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=2 with the EAGLE 1/1/2 draft (measured accept length 2.000 exactly at both concurrencies, saturating the 2-token draft). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "b200",  variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.12 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 400, tpot_ms: 3.11, tokens_per_sec_per_gpu: 321 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3044, tpot_ms: 7.84, tokens_per_sec_per_gpu: 1662 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=3.5 with the EAGLE 5/1/6 draft (measured accept length 3.507 at concurrency 1, 3.503 at concurrency 16). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "b200",  variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.42, aime25_pct: 90.83 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5908, tpot_ms: 16.53, tokens_per_sec_per_gpu: 3220 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 18916, tpot_ms: 31.88, tokens_per_sec_per_gpu: 5025 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=2 with the EAGLE 1/1/2 draft (measured accept length 2.000 exactly at both concurrencies, saturating the 2-token draft). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.73 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 372, tpot_ms: 3.68, tokens_per_sec_per_gpu: 556 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3305, tpot_ms: 10.02, tokens_per_sec_per_gpu: 2718 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=3.5 with the EAGLE 5/1/6 draft (measured accept length 3.489 at concurrency 1, 3.509 at concurrency 16). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.19, aime25_pct: 90.62 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7775, tpot_ms: 22.77, tokens_per_sec_per_gpu: 4743 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 25775, tpot_ms: 45.87, tokens_per_sec_per_gpu: 7501 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=2 with the EAGLE 1/1/2 draft (measured accept length 2.000 exactly at both concurrencies, saturating the 2-token draft). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "b300",  variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.12 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 342, tpot_ms: 3.10, tokens_per_sec_per_gpu: 328 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2851, tpot_ms: 7.77, tokens_per_sec_per_gpu: 1705 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=3.5 with the EAGLE 5/1/6 draft (measured accept length 3.485 at concurrency 1, 3.487 at concurrency 16). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "b300",  variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.42, aime25_pct: 92.08 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5586, tpot_ms: 16.37, tokens_per_sec_per_gpu: 3296 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 17848, tpot_ms: 32.48, tokens_per_sec_per_gpu: 5259 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=2 with the EAGLE 1/1/2 draft (measured accept length 2.000 exactly at both concurrencies, saturating the 2-token draft). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },

  // NVIDIA NVFP4, single node (RadixArk/GLM-5.3-NVFP4).
  { match: { hw: "b200",  variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "nvfp4", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "b200",  variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ 26fd7fdaa273",
    accuracy: { gsm8k_pct: 97.42, aime26_pct: 94.17 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 281, tpot_ms: 1.48, tokens_per_sec_per_gpu: 640 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1988, tpot_ms: 5.11, tokens_per_sec_per_gpu: 2306 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 26fd7fdaa273",
    accuracy: { gsm8k_pct: 97.42, aime26_pct: 94.17 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5227, tpot_ms: 12.40, tokens_per_sec_per_gpu: 3748 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 17513, tpot_ms: 29.97, tokens_per_sec_per_gpu: 5416 },
    ],
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ 26fd7fdaa273",
    accuracy: { gsm8k_pct: 97.42, aime26_pct: 94.17 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 307, tpot_ms: 1.71, tokens_per_sec_per_gpu: 1113 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2132, tpot_ms: 6.18, tokens_per_sec_per_gpu: 3979 },
    ],
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 26fd7fdaa273",
    accuracy: { gsm8k_pct: 97.42, aime26_pct: 94.17 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5828, tpot_ms: 16.69, tokens_per_sec_per_gpu: 6108 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 22330, tpot_ms: 39.63, tokens_per_sec_per_gpu: 8537 },
    ],
  },

  // NVIDIA BF16.
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.12 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 344, tpot_ms: 2.96, tokens_per_sec_per_gpu: 341 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2772, tpot_ms: 9.13, tokens_per_sec_per_gpu: 1521 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=3.5 with the EAGLE 5/1/6 draft (measured accept length 3.531 at concurrency 1, 3.506 at concurrency 16). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.27, aime25_pct: 93.75 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 9512, tpot_ms: 22.30, tokens_per_sec_per_gpu: 2279 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 75250, tpot_ms: 25.84, tokens_per_sec_per_gpu: 2333 },
    ],
    notes:
      "Speed measured under SGLANG_SIMULATE_ACC_LEN=2 with the EAGLE 1/1/2 draft (measured accept length 2.000 exactly at both concurrencies, saturating the 2-token draft). The pinned accept length makes these throughput-mechanism numbers only, never correctness evidence; the accuracy runs carry no such env.",
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 20a491d1d311",
    accuracy: { gsm8k_pct: 97.12 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 349227, tpot_ms: 65.45, tokens_per_sec_per_gpu: 2216 },
    ],
  },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },

  // AMD ROCm, single node.
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi355x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi325x", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced",        nodes: "single" } },
  { match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
];
