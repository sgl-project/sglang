// Gemma 4 per-cell benchmark numbers, keyed by the same `match` tuple as
// gemma4.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// The original migration PR dropped all benchmarks because the legacy page's
// version string ("gemma4 branch") was a non-reproducible moving ref. These
// MI300X measurements are fresh, collected on sglang 0.5.13.post1 (a release
// anchor → reproducible).
//
// SPEED — sglang.bench_serving, random ISL/OSL 1000/1000, --warmup-requests 64,
//   --request-rate inf. Gemma 4 31B (dense, BF16) auto-selects tp=1 on MI300X
//   (59 GB fits in a single 192 GB GPU).
//
// ACCURACY — sgl-eval GSM8K, --max-tokens 8192, --num-threads 4, tp=1.
export const benchmarks = [
  {
    // MI300X ×1 / Gemma 4 31B / BF16 / tp=1 / balanced.
    // Measured on AMD Instinct MI300X (8×192 GB), sglang 0.5.13.post1,
    // docker lmsysorg/sglang:v0.5.13.post1-rocm720-mi30x.
    // Server flags: --model-path google/gemma-4-31B-it --mem-fraction-static 0.8.
    match: { hw: "mi300x", variant: "31b", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 130, tpot_ms: 21.49, tokens_per_sec_per_gpu: 46 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 16, num_prompts: 256 },
        ttft_ms: 151, tpot_ms: 31.45, tokens_per_sec_per_gpu: 492 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 64, num_prompts: 512 },
        ttft_ms: 588, tpot_ms: 62.93, tokens_per_sec_per_gpu: 951 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 256, num_prompts: 1024 },
        ttft_ms: 39891, tpot_ms: 192.91, tokens_per_sec_per_gpu: 1047 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 218436, tpot_ms: 681.22, tokens_per_sec_per_gpu: 1045 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 4096, num_prompts: 4096 },
        ttft_ms: 640807, tpot_ms: 1524.16, tokens_per_sec_per_gpu: 1044 },
    ],
    // sgl-eval GSM8K, --max-tokens 8192, --num-threads 4, tp=1.
    // 1319 examples, 97.04% correct, 0% truncated, 0% errors.
    accuracy: { gsm8k_pct: 97.04 },
  },
];
