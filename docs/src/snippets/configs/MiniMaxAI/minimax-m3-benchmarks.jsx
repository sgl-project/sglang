// MiniMax-M3 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m3.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// NVIDIA rows (B200 / B300 / GB300 / H200) re-measured 2026-11-04 on
// lmsysorg/sglang:v0.5.23-cu130 with each cell's exact launch flags. The sparse step
// runs on the built-in Triton path (the image does not bundle the MSA kernel).
//
// SPEED: sglang.benchmark.serving --dataset-name random --flush-cache (default
// --random-range-ratio), 1 warm-up run discarded, mean of 3 measured runs.
// tokens_per_sec_per_gpu = total (input + output) tok/s / GPU count. TTFT/TPOT are
// means. GB200 is a bare-match stub (inferred-supported, not benchmarked). AMD:
// MI355X at 8-GPU tp8 (native MXFP8) carries a bench_serving speed row; MI300X
// (MXFP8 -> block-fp8) was accuracy-only. MI350X / MI325X inherit their same-arch
// sibling's recipe (stubs).
//
// ACCURACY: sgl-eval 0.1.3 (the python/pyproject test-extra pin), `run gsm8k` (full
// 1319) with --thinking and M3's recommended sampling (temp 1.0 / top_p 0.95), one
// run, 75 min wall-time cap. A few runs had 1-2 runaway generations still running
// at the cap; those are scored on the completed questions (noted per row). Single
// runs at temp 1.0 vary by roughly +/-1 point. Rows still pending show
// `gsm8k_pct: null` (no GSM8K row rendered).
export const benchmarks = [
  {
    match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "v0.5.23",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 413, tpot_ms: 25.6, tokens_per_sec_per_gpu: 2261 },
    ],
    accuracy: { gpqa_pct: 89.1, gsm8k_pct: 96.7, mmmu_pro_pct: 72.7 }, // GSM8K 96.74% (1319)
    notes: "GPQA Diamond (89.1, n-repeats 4) and MMMU-Pro (72.7) are from the earlier PR #27944 run (2026-06) and were not re-run on v0.5.23.",
  },
  {
    // Hopper H200: bf16 build (MXFP8 is Blackwell-only) at tp8; fa3 + page 128,
    // Triton sparse path. Host driver 570.211.01 (the CUDA 13 image runs on it).
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "v0.5.23",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 888, tpot_ms: 41.8, tokens_per_sec_per_gpu: 1381 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // 97.04% (1319)
  },
  {
    match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "v0.5.23",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 446, tpot_ms: 31.5, tokens_per_sec_per_gpu: 3703 },
    ],
    accuracy: { gsm8k_pct: 97.1 }, // 97.12% on 1318/1319 (1 runaway generation at the 75 min cap)
  },
  // GB200: inferred-supported, not directly benchmarked.
  { match: { hw: "gb200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  {
    // Measured with the cell's --decode-attention-backend trtllm_mha split. fa4-only
    // on the same image: 2048/256 c64 TTFT 440 / TPOT 31.5 / 3725 tok/s/GPU;
    // 8192/256 c24 TTFT 1442 / TPOT 19.8 / 5230 tok/s/GPU; GSM8K 97.0% (1319).
    // Split + MSA (MiniMax-AI/MSA#27): TPOT 54.3 / 23.4 ms, GSM8K 95.4% (1317/1319).
    match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "v0.5.23",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 435, tpot_ms: 31.5, tokens_per_sec_per_gpu: 3715 },
      { workload: { dataset: "random", isl: 8192, osl: 256, max_concurrency: 24, num_prompts: 24 },
        ttft_ms: 1422, tpot_ms: 20.0, tokens_per_sec_per_gpu: 5121 },
    ],
    accuracy: { gsm8k_pct: 95.3 }, // 95.29% on 1317/1319 (2 runaway generations at the 75 min cap)
  },
  // MI355X (gfx950): native MXFP8. Speed: bench_serving 1024/1024 @ conc 64, tp8
  // -> 1678 output tok/s (3355 total incl. input); 3355 / 8 = ~420 tokens/sec/GPU (total, in+out).
  // No TTFT/TPOT reported for this run.
  {
    match: { hw: "mi355x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: null, tpot_ms: null, tokens_per_sec_per_gpu: 420 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on MI355X (legacy run_eval 1319: 92.2)
  },
  // MI350X (gfx950): inferred-supported from MI355X, not separately benchmarked.
  { match: { hw: "mi350x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  // MI300X (gfx942): MXFP8 -> block-fp8 [128,128].
  {
    match: { hw: "mi300x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on MI300X (legacy run_eval 1319: 92.0, triton 0.917-0.929 / aiter ~0.929)
  },
  // MI325X (gfx942): inferred-supported from MI300X, not separately benchmarked.
  { match: { hw: "mi325x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
];
