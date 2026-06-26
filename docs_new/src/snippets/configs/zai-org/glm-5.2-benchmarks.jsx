// GLM-5.2 per-cell benchmark numbers, keyed by the same `match` tuple as glm-5.2.jsx cells.
// See _deployment.jsx for the speed/accuracy schema.
// Numbers pending: each entry is a bare `match` stub (renders "pending") until measured
// end-to-end on the corresponding hardware, then filled with sglang_version + speed/accuracy.
export const benchmarks = [
  // ---- H200 + FP8 ----  (measured on the v0.5.13.post1 release image, flush-cache on every run)
  {
    // EAGLE MTP 5-1-6 (was 3-1-4): accept ~5.96/6 → +31%/+15% throughput, -25%/-11% TPOT vs 3-1-4.
    // KV stays bf16 (Hopper auto-default). fp8 KV measured worse on H200 (slower flashmla_kv prefill
    // + lower decode throughput): conc=1 31 gpu / TTFT 838, conc=16 96 gpu / TTFT 6650.
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 662, tpot_ms: 3.03, tokens_per_sec_per_gpu: 34 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5080, tpot_ms: 12.44, tokens_per_sec_per_gpu: 113 },
    ],
  },
  {
    // Tuned prefill (chunked-prefill 32768 + max-running 80): +44%/+78% throughput and
    // -59%/-49% TTFT vs the untuned default-chunked (2048) baseline (152/133 gpu) on post1.
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 8013, tpot_ms: 25.57, tokens_per_sec_per_gpu: 219 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 77790, tpot_ms: 29.08, tokens_per_sec_per_gpu: 236 },
    ],
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 450276, tpot_ms: 86.71, tokens_per_sec_per_gpu: 184 },
    ],
  },
  // ---- B200 + FP8 ----  (measured on the v0.5.13.post1 release image, flush-cache on every run)
  {
    // EAGLE MTP 5-1-6 (was 3-1-4): accept length ~5.98/6 → +33%/+22% throughput, -26%/-15% TPOT vs 3-1-4.
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 334, tpot_ms: 2.30, tokens_per_sec_per_gpu: 48 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2777, tpot_ms: 6.84, tokens_per_sec_per_gpu: 209 },
    ],
  },
  {
    // Re-measured on v0.5.13.post1 with tuned prefill (chunked-prefill 32768 + max-running 80):
    // +34%/+44% throughput and -55%/-39% TTFT vs the untuned default-chunked (2048) baseline.
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 5680, tpot_ms: 18.76, tokens_per_sec_per_gpu: 285 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 60665, tpot_ms: 23.91, tokens_per_sec_per_gpu: 297 },
    ],
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 178249, tpot_ms: 48.28, tokens_per_sec_per_gpu: 449 },
    ],
  },
  // ---- GB300 + FP8 ----  (4-GPU single node, TP4; measured on the v0.5.13.post1 release image, flush-cache on every run)
  {
    // EAGLE MTP 5-1-6 (was 3-1-4): accept length ~5.98/6 → +34%/+24% throughput, -28%/-18% TPOT vs 3-1-4.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 393, tpot_ms: 2.78, tokens_per_sec_per_gpu: 79 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3201, tpot_ms: 8.53, tokens_per_sec_per_gpu: 341 },
    ],
  },
  {
    // Balanced uses the tuned prefill (chunked-prefill 32768 + max-running 80), same lever as H200/B200.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 7678, tpot_ms: 27.77, tokens_per_sec_per_gpu: 411 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 76359, tpot_ms: 31.08, tokens_per_sec_per_gpu: 483 },
    ],
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 250727, tpot_ms: 68.55, tokens_per_sec_per_gpu: 641 },
    ],
  },
  // ---- B300 + FP8 ----  (8-GPU single node, TP8; measured on v0.5.13.post1, flush-cache every run.
  // B300 (sm103) trails B200 (sm100) per-GPU here — the deep_gemm/DSA kernels are tuned for sm100 and
  // fall to a slower path on sm103; the gap should close as sm103 gets first-class kernels.)
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 503, tpot_ms: 3.24, tokens_per_sec_per_gpu: 34 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 4731, tpot_ms: 9.56, tokens_per_sec_per_gpu: 140 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 6465, tpot_ms: 23.36, tokens_per_sec_per_gpu: 245 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 67814, tpot_ms: 26.19, tokens_per_sec_per_gpu: 265 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 206246, tpot_ms: 56.11, tokens_per_sec_per_gpu: 388 },
    ],
  },
  // ---- B300 + BF16 ----  (unquantized zai-org/GLM-5.2, TP8; measured on v0.5.13.post1, flush-cache every run.
  // balanced/HT run plain TP8 (no DP-Attention/DeepEP), so they trail the FP8 dp-attention recipe at high concurrency.)
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 470, tpot_ms: 2.93, tokens_per_sec_per_gpu: 37 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3474, tpot_ms: 10.33, tokens_per_sec_per_gpu: 146 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 14123, tpot_ms: 35.47, tokens_per_sec_per_gpu: 157 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 116633, tpot_ms: 40.65, tokens_per_sec_per_gpu: 167 },
    ],
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 525108, tpot_ms: 82.52, tokens_per_sec_per_gpu: 168 },
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
  // ---- NVFP4 (Blackwell Ultra) ----  benchmarks pending
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "balanced",    nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced",    nodes: "single" } },
];
