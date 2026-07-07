// GLM-5.2 per-cell benchmark numbers, keyed by the same `match` tuple as glm-5.2.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// Numbers pending: each entry is a bare `match` stub (renders "pending") until measured
// end-to-end on the corresponding hardware, then filled with sglang_version + speed/accuracy.
export const benchmarks = [
  // ---- H200 + FP8 ----  (serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  // ---- B200 + FP8 ----  (8-GPU single node, TP8; real weights, --random-range-ratio 1.0, flush-cache every run)
  {
    // EAGLE MTP 5-1-6, mfs 0.8, no cuda-graph-max-bs. env SGLANG_SIMULATE_ACC_LEN=3.5
    // (match-expected: 50% accept 3 / 50% accept 4) fixes the acceptance length.
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 757, tpot_ms: 3.22, tokens_per_sec_per_gpu: 288 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3188, tpot_ms: 9.12, tokens_per_sec_per_gpu: 1476 },
    ],
  },
  {
    // Balanced: DP8 + deepep + mfs 0.85 + chunked-prefill 32768 + max-running 256, 1-1-2 EAGLE.
    // env SGLANG_SIMULATE_ACC_LEN=2 (match-expected: accept 2 of 2 draft tokens).
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5742, tpot_ms: 17.65, tokens_per_sec_per_gpu: 3078 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 18744, tpot_ms: 32.61, tokens_per_sec_per_gpu: 5022 },
    ],
  },
  {
    // HT: DP8 + deepep + mfs 0.85 + max-running 256. B200 (178GB) keeps --max-running-requests 256
    // (clamps the decode capture list to <=32 < the default 128 DeepEP buffer); no env buffer bump.
    // No spec, so no SIMULATE_ACC_LEN.
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 177620, tpot_ms: 47.99, tokens_per_sec_per_gpu: 4059 },
    ],
  },
  // ---- GB300 + FP8 ----  (4-GPU single node, TP4; real weights, --random-range-ratio 1.0, flush-cache every run)
  {
    // EAGLE MTP 5-1-6, mfs 0.85, no cuda-graph-max-bs; mrr auto-capped 48. env
    // SGLANG_SIMULATE_ACC_LEN=3.5 (match-expected: 50% accept 3 / 50% accept 4) fixes the
    // acceptance length so the spec numbers are comparable across runs.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 374, tpot_ms: 4.55, tokens_per_sec_per_gpu: 459 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3719, tpot_ms: 11.5, tokens_per_sec_per_gpu: 2376 },
    ],
  },
  {
    // Balanced: DP4 + deepep + mfs 0.85 + chunked-prefill 32768 (÷dp4 = 8192) + max-running 256,
    // 1-1-2 EAGLE. env SGLANG_SIMULATE_ACC_LEN=2 (match-expected: accept 2 of 2 draft tokens).
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7429, tpot_ms: 25.21, tokens_per_sec_per_gpu: 4437 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 27488, tpot_ms: 48.43, tokens_per_sec_per_gpu: 6804 },
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
    sglang_version: "main @ 09ca4fc",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 231101, tpot_ms: 86.01, tokens_per_sec_per_gpu: 6039 },
    ],
  },
  // ---- B300 + FP8 ----  (8-GPU single node, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  // ---- B300 + BF16 ----  (unquantized zai-org/GLM-5.2, TP8; serve recipe in glm-5.2.jsx; benchmark pending re-measurement)
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
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
  // measured on the lmsysorg/sglang:dev-glm52-nvfp4 preview image, flush-cache every run.
  // ttft_ms/tpot_ms are P50; tokens_per_sec_per_gpu = total (in+out) tok/s/GPU (output/GPU × (isl+osl)/osl).
  // balanced & high-throughput add DP-Attention (dp8); low-latency uses MTP 5-1-6, balanced MTP 2-1-3.)
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-glm52-nvfp4",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 295, tpot_ms: 1.85, tokens_per_sec_per_gpu: 527 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2491, tpot_ms: 5.43, tokens_per_sec_per_gpu: 2289 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "dev-glm52-nvfp4",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5837, tpot_ms: 12.70, tokens_per_sec_per_gpu: 3770 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 16736, tpot_ms: 30.00, tokens_per_sec_per_gpu: 5343 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-glm52-nvfp4",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 130174, tpot_ms: 67.12, tokens_per_sec_per_gpu: 5305 },
    ],
  },
  // ---- B300 + NVFP4 ----  (8-GPU single node, TP8; nvidia/GLM-5.2-NVFP4 via --quantization modelopt_fp4,
  // measured on the lmsysorg/sglang:dev-glm52-nvfp4 preview image, flush-cache every run.
  // tokens_per_sec_per_gpu = total (in+out) tok/s/GPU (measured output/GPU 51/224/153/205/430 × (isl+osl)/osl).
  // aime25 overrides the variant default (87.7 → 89.58, measured on this NVFP4 build); gsm8k inherits the default.)
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "dev-glm52-nvfp4",
    accuracy: { aime25_pct: 89.58 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 196, tpot_ms: 1.86, tokens_per_sec_per_gpu: 459 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 274, tpot_ms: 6.95, tokens_per_sec_per_gpu: 2016 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "dev-glm52-nvfp4",
    accuracy: { aime25_pct: 89.58 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 680, tpot_ms: 48.9, tokens_per_sec_per_gpu: 1377 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3010, tpot_ms: 149, tokens_per_sec_per_gpu: 1845 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "dev-glm52-nvfp4",
    accuracy: { aime25_pct: 89.58 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 6370, tpot_ms: 280, tokens_per_sec_per_gpu: 3870 },
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
];
