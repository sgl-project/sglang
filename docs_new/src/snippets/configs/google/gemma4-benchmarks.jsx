// Gemma 4 per-cell benchmark numbers, keyed by the same `match` tuple as
// gemma4.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// The original migration PR dropped all benchmarks because the legacy page's
// version string ("gemma4 branch") was a non-reproducible moving ref. These
// MI300X measurements are fresh, collected on sglang 0.5.16 (a release
// anchor → reproducible).
//
// SPEED — sglang.bench_serving, random ISL/OSL 8192/1024, --flush-cache,
//   --request-rate inf. Gemma 4 31B (dense, BF16) auto-selects tp=1 on MI300X
//   (59 GB fits in a single 192 GB GPU).
//
// ACCURACY — sgl-eval GSM8K, --max-tokens 8192, --num-threads 4, tp=1.
export const benchmarks = [
  {
    // MI300X ×1 / Gemma 4 31B / BF16 / tp=1 / balanced.
    // Measured on AMD Instinct MI300X (8×192 GB), sglang 0.5.16,
    // docker lmsysorg/sglang:v0.5.16-rocm700-mi30x.
    // Server flags: --model-path google/gemma-4-31B-it --mem-fraction-static 0.8.
    match: { hw: "mi300x", variant: "31b", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 32 },
        ttft_ms: 801, tpot_ms: 19.80, tokens_per_sec_per_gpu: 47 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 32 },
        ttft_ms: 4334, tpot_ms: 38.68, tokens_per_sec_per_gpu: 288 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 36305, tpot_ms: 68.07, tokens_per_sec_per_gpu: 377 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256, num_prompts: 512 },
        ttft_ms: 229552, tpot_ms: 72.53, tokens_per_sec_per_gpu: 397 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 976716, tpot_ms: 71.51, tokens_per_sec_per_gpu: 398 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096, num_prompts: 4096 },
        ttft_ms: 2645151, tpot_ms: 71.59, tokens_per_sec_per_gpu: 401 },
    ],
    // sgl-eval GSM8K, --max-tokens 8192, --num-threads 4, tp=1.
    // 1319 examples, 97.04% correct, 0% truncated, 0% errors.
    accuracy: { gsm8k_pct: 97.04 },
  },
];
