// MiniMax-M3 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m3.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// 2026-08 RE-BENCH (sglang 0.5.16, pinned release): b200 / b300 / gb300 speed
// re-measured cache-cold (--flush-cache), P50, standardized to isl 2048 / osl 256
// at conc 24 & 64. b200 required PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
// (shipped --mem-fraction-static 0.65 OOMs at cuda-graph capture on 0.5.16).
// STILL PENDING a 0.5.16 re-bench (numbers below are the original "PR #27944"
// measurement): h200 (no 8x-h200 whole-node currently placeable), gb200 (no box),
// and the AMD cells mi300x/mi325x/mi350x/mi355x (no AMD devbox in the fleet).
// The provenance notes below describe those original measurements.
//
// SPEED — bench_serving --flush-cache, random isl2048/osl256, max_concurrency 64,
// CUDA graph on. B200 (tp8, MXFP8, MSA fmha_sm100 path; re-measured 2026-06-15
// with piecewise CUDA graph default-on) and H200 (tp8, bf16, built-in Triton
// sparse) are measured on PR #27944 — warm steady-state from a 3-run sweep (the
// B200 3-run is identical; the H200 cold-start first run, ~2x slower, is
// excluded). B300 / GB300
// rows are the earlier 2026-06-11 tp4 MSA numbers (pre-piecewise),
// pending a #27944 re-measure on their own boxes. GB200 is a bare-match
// stub (inferred-supported, not benchmarked). AMD: MI355X at 8-GPU tp8 (native
// MXFP8) carries a bench_serving speed row; MI300X (MXFP8 -> block-fp8) was
// accuracy-only. MI350X / MI325X inherit their same-arch sibling's recipe
// (stubs). (sgl-eval does NOT measure serving throughput — TTFT/TPOT/tok-s come
// from sglang.bench_serving.)
//
// GSM8K / GPQA — unified on a SINGLE harness: sgl-eval (github.com/sgl-project/sgl-eval)
// `run gsm8k` (full 1319) / `run gpqa` (GPQA Diamond 198, n-repeats 4), chat
// endpoint with --thinking (M3's reasoning path) + M3's recommended sampling
// (temp 1.0 / top_p 0.95), symbolic grading. This is the config's Reproduce command.
// H200 is stable at GSM8K 97.04% (std 0.0). B200 was re-measured 2026-06-15 on
// minimax-m3-upstream (piecewise + MSA decode fix): GSM8K 96.51% recommended /
// 96.89% greedy (stable single-run), GPQA pass@1[avg-of-4] 89.14% — the merged
// MSA decode fix resolves the earlier fresh-server-94.4%-then-drift under-load issue.
// Per-platform re-measurement under sgl-eval is in progress; rows still pending
// show `gsm8k_pct: null` (no GSM8K row rendered) with the legacy-harness number
// kept in a comment. Legacy harnesses were NOT comparable across platforms
// (NVIDIA: few_shot_gsm8k --num-questions 200; AMD: run_eval gsm8k 1319 examples) —
// which is exactly why we re-measure on one harness.
export const benchmarks = [
  {
    // B200 re-measured 2026-06-15 at tp8 on minimax-m3-upstream (piecewise CUDA
    // graph default-on + AR-fusion revert/off + MSA decode fix). The earlier
    // #27944 tp4 speed + GSM8K drift were pre-fix; the merged MSA decode fix
    // resolves the drift (stable single-run greedy 96.89% / recommended 96.51%).
    match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      // Re-bench on 0.5.16 (cache-cold, --flush-cache, tp8; P50). NOTE: required
      // PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — the shipped
      // --mem-fraction-static 0.65 OOMs at cuda-graph capture on 0.5.16 without it.
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 24, num_prompts: 24 },
        ttft_ms: 784, tpot_ms: 14.4, tokens_per_sec_per_gpu: 1549 },
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1459, tpot_ms: 21.3, tokens_per_sec_per_gpu: 2662 },
    ],
    accuracy: { gpqa_pct: 89.1, gsm8k_pct: 96.5, mmmu_pro_pct: 72.7 }, // 2026-06-15, sgl-eval --thinking, recommended sampling (temp 1.0/top_p 0.95), tp8. GSM8K full 1319 = 96.51% (greedy 96.89%). GPQA Diamond 198, n-repeats 4 = pass@1[avg-of-4] 89.14% +/-1.73% (pass@4 95.45%, majority@4 93.52%). MMMU-Pro 2026-06-18, sgl-eval "standard (10 options)" test split, full 1730, single-shot 72.66% (thinking, temp 1.0/top_p 0.95).
  },
  {
    // Hopper H200: bf16 build (MXFP8 is Blackwell-only) at tp8, built-in Triton
    // sparse path (MSA is Blackwell-only). GSM8K re-measured on #27944.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      // bench_serving --flush-cache, bf16 Triton path; warm steady-state (3-run, cold-start run-1 excluded).
      // NOTE: these TTFT/TPOT are the original MEAN measurement (config latencyPercentile
      // is P50 for the 0.5.16-re-benched cells) — h200 pending an 8x-node re-bench to P50.
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1054, tpot_ms: 70.8, tokens_per_sec_per_gpu: 1044 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // #27944, sgl-eval --thinking, full 1319, recommended sampling (temp 1.0/top_p 0.95/top_k 40); stable 97.04% across all 3 runs (std 0.0)
  },
  {
    match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      // Re-bench on 0.5.16 (cache-cold, --flush-cache, tp4; P50).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 24, num_prompts: 24 },
        ttft_ms: 916, tpot_ms: 17.4, tokens_per_sec_per_gpu: 2578 },
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1764, tpot_ms: 26.2, tokens_per_sec_per_gpu: 4331 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on B300 (legacy few_shot 200: 87.5)
  },
  // GB200: inferred-supported, not directly benchmarked.
  { match: { hw: "gb200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      // Re-bench on 0.5.16 (cache-cold, --flush-cache, tp4; P50). Standardized to
      // isl 2048 / osl 256 at conc 24 & 64 (dropped the old isl-8192 conc-24 row).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 24, num_prompts: 24 },
        ttft_ms: 902, tpot_ms: 16.9, tokens_per_sec_per_gpu: 2638 },
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1545, tpot_ms: 25.2, tokens_per_sec_per_gpu: 4583 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on GB300 (legacy few_shot 200: 87.5)
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
