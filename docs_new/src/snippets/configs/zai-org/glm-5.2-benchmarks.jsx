// GLM-5.2 per-cell benchmark numbers, keyed by the same `match` tuple as glm-5.2.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// Single-node NVIDIA cells re-measured on sglang v0.5.15 with the EXACT cookbook serve flags
// (env=[], no SGLANG_SIMULATE_ACC_LEN) and the exact benchmarkCommands (--random-range-ratio 1.0,
// --flush-cache, warmup 64) — so spec-decode reflects real acceptance, not a fixed simulate.
export const benchmarks = [
  // ---- H200 + FP8 ----  (serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
      {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 654, tpot_ms: 2.78, tokens_per_sec_per_gpu: 318 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5983, tpot_ms: 11.73, tokens_per_sec_per_gpu: 1015 },
    ],
  },
      {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7420, tpot_ms: 23.67, tokens_per_sec_per_gpu: 2315 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 84626, tpot_ms: 30.52, tokens_per_sec_per_gpu: 2266 },
    ],
  },
      {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 530991, tpot_ms: 62.11, tokens_per_sec_per_gpu: 1711 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2379560, tpot_ms: 66.4, tokens_per_sec_per_gpu: 1599 },
    ],
  },
  // ---- B200 + FP8 ----  (8-GPU single node, TP8; real weights, --random-range-ratio 1.0, flush-cache every run)
      {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 322, tpot_ms: 2.05, tokens_per_sec_per_gpu: 471 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2721, tpot_ms: 6.24, tokens_per_sec_per_gpu: 2003 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5248, tpot_ms: 17.55, tokens_per_sec_per_gpu: 3160 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 17498, tpot_ms: 33.99, tokens_per_sec_per_gpu: 5003 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 133322, tpot_ms: 88.48, tokens_per_sec_per_gpu: 5020 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 746614, tpot_ms: 86.46, tokens_per_sec_per_gpu: 5097 },
    ],
  },
  // ---- GB300 + FP8 ----  (4-GPU single node, TP4; real weights, --random-range-ratio 1.0, flush-cache every run)
      {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 372, tpot_ms: 2.36, tokens_per_sec_per_gpu: 813 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3454, tpot_ms: 7.98, tokens_per_sec_per_gpu: 3103 },
    ],
  },
      {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7288, tpot_ms: 25.05, tokens_per_sec_per_gpu: 4430 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 26070, tpot_ms: 49.37, tokens_per_sec_per_gpu: 6739 },
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
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 259814, tpot_ms: 79.93, tokens_per_sec_per_gpu: 6119 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1367301, tpot_ms: 81.46, tokens_per_sec_per_gpu: 6054 },
    ],
  },
  // ---- B300 + FP8 ----  (8-GPU single node, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
      {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 313, tpot_ms: 2.0, tokens_per_sec_per_gpu: 480 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2944, tpot_ms: 6.14, tokens_per_sec_per_gpu: 1965 },
    ],
  },
      {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4992, tpot_ms: 17.22, tokens_per_sec_per_gpu: 3220 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 16720, tpot_ms: 31.89, tokens_per_sec_per_gpu: 5223 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 67532, tpot_ms: 116.39, tokens_per_sec_per_gpu: 6301 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 381190, tpot_ms: 235.26, tokens_per_sec_per_gpu: 6634 },
    ],
  },
  // ---- B300 + BF16 ----  (unquantized zai-org/GLM-5.2, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
    {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 309, tpot_ms: 2.09, tokens_per_sec_per_gpu: 466 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2495, tpot_ms: 7.26, tokens_per_sec_per_gpu: 1835 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 9287, tpot_ms: 24.29, tokens_per_sec_per_gpu: 2158 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 91939, tpot_ms: 29.09, tokens_per_sec_per_gpu: 2179 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 454915, tpot_ms: 50.99, tokens_per_sec_per_gpu: 2120 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2089630, tpot_ms: 55.61, tokens_per_sec_per_gpu: 2122 },
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
        ttft_ms: 279, tpot_ms: 1.69, tokens_per_sec_per_gpu: 572 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1963, tpot_ms: 5.53, tokens_per_sec_per_gpu: 2346 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 6046, tpot_ms: 12.37, tokens_per_sec_per_gpu: 3890 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 11467, tpot_ms: 28.0, tokens_per_sec_per_gpu: 5969 },
    ],
  },
      {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 72542, tpot_ms: 117.73, tokens_per_sec_per_gpu: 6094 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 551123, tpot_ms: 152.78, tokens_per_sec_per_gpu: 5971 },
    ],
  },
  // ---- B300 + NVFP4 ----  (8-GPU single node, TP8; nvidia/GLM-5.2-NVFP4 via --quantization modelopt_fp4,
  // flush-cache every run. Re-benched on v0.5.15 — speed only; accuracy inherits the variant default.)
    {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 276, tpot_ms: 1.63, tokens_per_sec_per_gpu: 587 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2177, tpot_ms: 5.02, tokens_per_sec_per_gpu: 2322 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 6359, tpot_ms: 13.46, tokens_per_sec_per_gpu: 3428 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 13694, tpot_ms: 30.4, tokens_per_sec_per_gpu: 4946 },
    ],
  },
    {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 80118, tpot_ms: 130.46, tokens_per_sec_per_gpu: 5501 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 310125, tpot_ms: 371.61, tokens_per_sec_per_gpu: 5920 },
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
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 8236, tpot_ms: 17.05, tokens_per_sec_per_gpu: 5724 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 17277, tpot_ms: 42.84, tokens_per_sec_per_gpu: 8181 },
    ],
  },
    {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 111104, tpot_ms: 182.37, tokens_per_sec_per_gpu: 7904 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 871704, tpot_ms: 229.03, tokens_per_sec_per_gpu: 7686 },
    ],
  },
    {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 298, tpot_ms: 1.82, tokens_per_sec_per_gpu: 1057 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2129, tpot_ms: 6.47, tokens_per_sec_per_gpu: 3964 },
    ],
  },
];
