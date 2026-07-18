// GLM-5.2 per-cell benchmark numbers, keyed by the same `match` tuple as glm-5.2.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// Single-node NVIDIA cells re-measured on sglang v0.5.15 with the EXACT cookbook serve flags
// (env=[], no SGLANG_SIMULATE_ACC_LEN) and the exact benchmarkCommands (--random-range-ratio 1.0,
// --flush-cache, warmup 64) — so spec-decode reflects real acceptance, not a fixed simulate.
export const benchmarks = [
  // ---- H200 + FP8 ----  (serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
      {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 651, tpot_ms: 2.78, tokens_per_sec_per_gpu: 327 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5712, tpot_ms: 11.83, tokens_per_sec_per_gpu: 1033 },
    ],
  },
      {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7438, tpot_ms: 23.61, tokens_per_sec_per_gpu: 2315 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 85245, tpot_ms: 31.46, tokens_per_sec_per_gpu: 2225 },
    ],
  },
      {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 530003, tpot_ms: 58.93, tokens_per_sec_per_gpu: 1766 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2372600, tpot_ms: 66.17, tokens_per_sec_per_gpu: 1615 },
    ],
  },
  // ---- B200 + FP8 ----  (8-GPU single node, TP8; real weights, --random-range-ratio 1.0, flush-cache every run)
      {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 326, tpot_ms: 2.06, tokens_per_sec_per_gpu: 467 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3085, tpot_ms: 6.14, tokens_per_sec_per_gpu: 1895 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5344, tpot_ms: 17.55, tokens_per_sec_per_gpu: 3140 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 17796, tpot_ms: 33.42, tokens_per_sec_per_gpu: 4997 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 209794, tpot_ms: 46.77, tokens_per_sec_per_gpu: 4171 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1031213, tpot_ms: 47.9, tokens_per_sec_per_gpu: 4216 },
    ],
  },
  // ---- GB300 + FP8 ----  (4-GPU single node, TP4; real weights, --random-range-ratio 1.0, flush-cache every run)
      {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 381, tpot_ms: 2.36, tokens_per_sec_per_gpu: 816 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2992, tpot_ms: 8.0, tokens_per_sec_per_gpu: 3263 },
    ],
  },
      {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7316, tpot_ms: 25.0, tokens_per_sec_per_gpu: 4447 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 26107, tpot_ms: 48.9, tokens_per_sec_per_gpu: 6812 },
    ],
  },
  // GB300 HT: drop-flags (mfs/cgbs/mrr dropped) + env SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512.
  // DeepEP low_latency asserts x.size(0) <= num_max_dispatch_tokens_per_rank (deep_ep.cpp:1262,
  // default 128). Decode cuda-graph capture builds a dummy batch of `bs` tokens per rank (not
  // DP-split), so the 256/512 capture buckets trip the assert at default 128; GB300 (DP4) also
  // hits bs/4 = 256 > 128 at c1024 runtime. Raising the buffer to 512 fixes both and lets HT
  // drop --max-running-requests. Verified on main: with env=512 capture + serve pass; without
  // env the assert trips at the bs=512 capture bucket. No spec, so no SIMULATE_ACC_LEN.
      {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 260663, tpot_ms: 79.72, tokens_per_sec_per_gpu: 6120 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1370949, tpot_ms: 81.67, tokens_per_sec_per_gpu: 6033 },
    ],
  },
  // ---- B300 + FP8 ----  (8-GPU single node, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
      {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 315, tpot_ms: 2.0, tokens_per_sec_per_gpu: 480 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2978, tpot_ms: 6.13, tokens_per_sec_per_gpu: 1964 },
    ],
  },
      {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4978, tpot_ms: 17.22, tokens_per_sec_per_gpu: 3223 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 16569, tpot_ms: 32.14, tokens_per_sec_per_gpu: 5209 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 201346, tpot_ms: 45.51, tokens_per_sec_per_gpu: 4320 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 990502, tpot_ms: 46.48, tokens_per_sec_per_gpu: 4376 },
    ],
  },
  // ---- B300 + BF16 ----  (unquantized zai-org/GLM-5.2, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
    {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 311, tpot_ms: 2.1, tokens_per_sec_per_gpu: 464 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2738, tpot_ms: 7.12, tokens_per_sec_per_gpu: 1840 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 9079, tpot_ms: 24.57, tokens_per_sec_per_gpu: 2155 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 92206, tpot_ms: 28.86, tokens_per_sec_per_gpu: 2176 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 453045, tpot_ms: 50.85, tokens_per_sec_per_gpu: 2129 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2087134, tpot_ms: 54.13, tokens_per_sec_per_gpu: 2126 },
    ],
  },
  // ---- BF16 multi-node (inferred) ----  benchmarks pending
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "h200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "b200",  variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced",        nodes: "multi-2" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "multi-2" } },
  // ---- B200 + NVFP4 ----  (8-GPU single node, TP8; nvidia/GLM-5.2-NVFP4 via --quantization modelopt_fp4,
  // flush-cache every run.
  // ttft_ms/tpot_ms are P50; tokens_per_sec_per_gpu = total (in+out) tok/s/GPU (output/GPU × (isl+osl)/osl).
  // balanced & high-throughput add DP-Attention (dp8); low-latency uses MTP 5-1-6, balanced MTP 2-1-3.)
      {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 279, tpot_ms: 1.67, tokens_per_sec_per_gpu: 575 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2028, tpot_ms: 5.64, tokens_per_sec_per_gpu: 2154 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5359, tpot_ms: 12.8, tokens_per_sec_per_gpu: 3956 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 11188, tpot_ms: 28.04, tokens_per_sec_per_gpu: 5994 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 127628, tpot_ms: 64.71, tokens_per_sec_per_gpu: 5432 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 747146, tpot_ms: 64.49, tokens_per_sec_per_gpu: 5519 },
    ],
  },
  // ---- B300 + NVFP4 ----  (8-GPU single node, TP8; nvidia/GLM-5.2-NVFP4 via --quantization modelopt_fp4,
  // flush-cache every run.
  // tokens_per_sec_per_gpu = total (in+out) tok/s/GPU (measured output/GPU 51/224/153/205/430 × (isl+osl)/osl).
  // aime25 overrides the variant default (87.7 → 89.58, measured on this NVFP4 build); gsm8k inherits the default.)
    {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 278, tpot_ms: 1.65, tokens_per_sec_per_gpu: 584 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2308, tpot_ms: 5.04, tokens_per_sec_per_gpu: 2259 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5880, tpot_ms: 15.7, tokens_per_sec_per_gpu: 3237 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 14924, tpot_ms: 30.29, tokens_per_sec_per_gpu: 4922 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 85876, tpot_ms: 128.33, tokens_per_sec_per_gpu: 5312 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 698429, tpot_ms: 131.84, tokens_per_sec_per_gpu: 5281 },
    ],
  },
  // ---- MI355X + FP8 ----  gfx950, TP8, DSA tilelang, NO MTP (disabled on AMD).
  // Measured on lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260618, flush-cache every run.
  // No spec-decoding, so not directly comparable to the NVIDIA low-latency cells (EAGLE MTP).
  {
    match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 634, tpot_ms: 13.56, tokens_per_sec_per_gpu: 81 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5411, tpot_ms: 23.60, tokens_per_sec_per_gpu: 621 },
    ],
  },
  {
    match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 19526, tpot_ms: 46.50, tokens_per_sec_per_gpu: 1098 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 117866, tpot_ms: 56.12, tokens_per_sec_per_gpu: 1044 },
    ],
  },
  {
    match: { hw: "mi355x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 432058, tpot_ms: 106.44, tokens_per_sec_per_gpu: 1269 },
    ],
  },
  // ---- GB300 + NVFP4 (0.5.15 rebench: exact cookbook config + bench, real spec acceptance) ----
    {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 8086, tpot_ms: 17.45, tokens_per_sec_per_gpu: 5474 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 17684, tpot_ms: 43.43, tokens_per_sec_per_gpu: 7941 },
    ],
  },
    {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 186446, tpot_ms: 92.05, tokens_per_sec_per_gpu: 7617 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1101528, tpot_ms: 92.49, tokens_per_sec_per_gpu: 7541 },
    ],
  },
    {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 314, tpot_ms: 1.83, tokens_per_sec_per_gpu: 1046 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2429, tpot_ms: 6.11, tokens_per_sec_per_gpu: 3960 },
    ],
  },
];
