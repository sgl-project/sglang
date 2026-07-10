// GLM-5.1 measured benchmarks — faithfully migrated from the legacy page's §5.
//
// Only ONE result survives the reproducibility rule:
//   • Speed (§5.1): H200 ×8, GLM-5.1-FP8, tp=8 — pinned to `commit 947927bdb`
//     (a reproducible commit-hash anchor). The measured deploy ran EAGLE
//     speculative decoding (Accept length ~3.5) and no DP-Attention, matching
//     the h200/fp8/low-latency cell. The bench deploy did NOT set parsers, so
//     the cell's parsers-OFF flags equal the measured command.
//
// DROPPED (not migrated):
//   • Accuracy GSM8K 0.955 / MMLU 0.877 (§5.2): the page's own <Note> states
//     these are SHARED WITH GLM-5 (cross-model) and "GLM-5.1 was not
//     independently benchmarked" — cross-model numbers are never inherited.
//   • AMD GSM8K 0.970 (§5.3): anchored to "AMD nightly CI" (a moving ref) and
//     PR #18911, which is titled "[AMD] [GLM-5 Day 0] Add GLM-5 nightly test"
//     — a GLM-5 (cross-model) day-0 test on a nightly build. Dropped on both
//     the cross-model and the non-reproducible-anchor grounds.
//
// tokens_per_sec_per_gpu = output tok/s ÷ (tp × nnodes) = output tok/s ÷ 8.

export const benchmarks = [
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "commit 947927bdb",
    speed: [
      // §5.1.1 Latency: isl=1000, osl=1000, concurrency=1 (num-prompts 10).
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 290.88, tpot_ms: 7.54, tokens_per_sec_per_gpu: 14.745 },
      // §5.1.2 Throughput: isl=1000, osl=1000, concurrency=100 (num-prompts 1000).
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 100, num_prompts: 1000 },
        ttft_ms: 20613.80, tpot_ms: 38.73, tokens_per_sec_per_gpu: 151.871 },
    ],
  },
  {
    // MI300X ×8 / GLM-5.1-FP8 / BF16 KV cache / tp=8 / low-latency (no DP-Attention).
    // Measured on AMD Instinct MI300X (8×192 GB), sglang 0.5.13.post1,
    // docker lmsysorg/sglang:v0.5.13.post1-rocm720-mi30x.
    // Server flags: --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
    //   --chunked-prefill-size 131072 --watchdog-timeout 1200 --mem-fraction-static 0.8.
    // KV cache auto-set to bfloat16 (DSA on SM9 device).
    // Benchmark: sglang.bench_serving --dataset-name random --random-input-len 1000
    //   --random-output-len 1000 --warmup-requests 64 --request-rate inf.
    match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 585, tpot_ms: 28.38, tokens_per_sec_per_gpu: 4 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 16, num_prompts: 256 },
        ttft_ms: 338, tpot_ms: 51.63, tokens_per_sec_per_gpu: 37 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 64, num_prompts: 512 },
        ttft_ms: 634, tpot_ms: 87.27, tokens_per_sec_per_gpu: 85 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 256, num_prompts: 1024 },
        ttft_ms: 2017, tpot_ms: 169.96, tokens_per_sec_per_gpu: 169 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 104260, tpot_ms: 437.72, tokens_per_sec_per_gpu: 179 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 4096, num_prompts: 4096 },
        ttft_ms: 570017, tpot_ms: 437.19, tokens_per_sec_per_gpu: 188 },
    ],
    accuracy: [
      // GSM8K via sgl-eval, --no-thinking --max-tokens 8192 --num-threads 4, temperature=0.
      // 1.06% truncation rate at 8192 max-tokens (thinking model generates long CoT).
      { benchmark: "gsm8k", score: 0.9621, details: "1319 examples, 1.06% truncated, --no-thinking" },
    ],
  },
];
