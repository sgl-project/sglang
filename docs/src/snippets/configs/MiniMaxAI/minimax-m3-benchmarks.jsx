// MiniMax-M3 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m3.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Speed: bench_serving --flush-cache, P50, qwen traffic shape (random isl 8192 /
// osl 1024, --random-range-ratio 1.0, --warmup-requests 64), balanced tier @ conc
// 64 & 256 (num_prompts 128 / 512). h200 (tp8, bf16) / b300 / gb300 (tp4, MXFP8)
// measured on sglang 0.5.16. b200, gb200, and the AMD cells are pending — not yet
// measured on 0.5.16.
//
// Accuracy: sgl-eval (github.com/sgl-project/sgl-eval) run gsm8k (full 1319) /
// run gpqa (GPQA Diamond 198, n-repeats 4), chat endpoint with --thinking and M3's
// recommended sampling (temp 1.0 / top_p 0.95). This is the config's Reproduce command.
export const benchmarks = [
  {
    // B200 (tp8, MXFP8).
    match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    // Speed pending — not yet measured on 0.5.16. env PYTORCH_CUDA_ALLOC_CONF required.
    accuracy: { gpqa_pct: 89.1, gsm8k_pct: 96.5, mmmu_pro_pct: 72.7 }, // sgl-eval --thinking, recommended sampling (temp 1.0/top_p 0.95), tp8. GSM8K full 1319 = 96.51% (greedy 96.89%). GPQA Diamond 198, n-repeats 4 = pass@1[avg-of-4] 89.14% +/-1.73% (pass@4 95.45%, majority@4 93.52%). MMMU-Pro sgl-eval "standard (10 options)" test split, full 1730, single-shot 72.66% (thinking, temp 1.0/top_p 0.95).
  },
  {
    // H200: bf16 build (MXFP8 is Blackwell-only), tp8, built-in Triton sparse path.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 114134, tpot_ms: 11.5, tokens_per_sec_per_gpu: 550 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256, num_prompts: 512 },
        ttft_ms: 484256, tpot_ms: 11.6, tokens_per_sec_per_gpu: 552 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // sgl-eval --thinking, full 1319, recommended sampling (temp 1.0/top_p 0.95/top_k 40); 97.04% across 3 runs (std 0.0)
  },
  {
    match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 6795, tpot_ms: 28.2, tokens_per_sec_per_gpu: 4140 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256, num_prompts: 512 },
        ttft_ms: 26480, tpot_ms: 61.5, tokens_per_sec_per_gpu: 6043 },
    ],
    accuracy: { gsm8k_pct: null }, // pending; legacy few_shot 200: 87.5
  },
  // GB200: inferred-supported, not directly benchmarked.
  { match: { hw: "gb200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 7664, tpot_ms: 28.3, tokens_per_sec_per_gpu: 4020 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256, num_prompts: 512 },
        ttft_ms: 29568, tpot_ms: 64.0, tokens_per_sec_per_gpu: 6203 },
    ],
    accuracy: { gsm8k_pct: null }, // pending; legacy few_shot 200: 87.5
  },
  // MI355X (gfx950): native MXFP8. bench_serving 1024/1024 @ conc 64, tp8 →
  // ~420 tokens/sec/GPU (total, in+out). No TTFT/TPOT for this run.
  {
    match: { hw: "mi355x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: null, tpot_ms: null, tokens_per_sec_per_gpu: 420 },
    ],
    accuracy: { gsm8k_pct: null }, // pending; legacy run_eval 1319: 92.2
  },
  // MI350X (gfx950): inferred-supported from MI355X, not separately benchmarked.
  { match: { hw: "mi350x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  // MI300X (gfx942): MXFP8 -> block-fp8 [128,128].
  {
    match: { hw: "mi300x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    accuracy: { gsm8k_pct: null }, // pending; legacy run_eval 1319: 92.0
  },
  // MI325X (gfx942): inferred-supported from MI300X, not separately benchmarked.
  { match: { hw: "mi325x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
];
