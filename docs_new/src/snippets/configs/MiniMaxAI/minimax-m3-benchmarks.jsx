// MiniMax-M3 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m3.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// SPEED — bench_serving --flush-cache, random isl2048/osl256, max_concurrency 64,
// CUDA graph on. B200 (tp8, MXFP8, MSA fmha_sm100 path; re-measured 2026-06-15
// with piecewise CUDA graph default-on) and H200 (tp8, bf16, built-in Triton
// sparse) are measured on PR #27944 — warm steady-state from a 3-run sweep (the
// B200 3-run is identical; the H200 cold-start first run, ~2x slower, is
// excluded). B300 / GB300
// rows are the earlier 2026-06-11 tp4 MSA numbers (pre-piecewise),
// pending a #27944 re-measure on their own boxes. GB200 is a bare-match
// stub (inferred-supported, not benchmarked). AMD: MI350X / MI355X (native
// MXFP8) and MI300X (MXFP8 -> block-fp8) carry bench_serving speed rows.
// (sgl-eval does NOT measure serving throughput — TTFT/TPOT/tok-s come from
// sglang.bench_serving.)
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
    sglang_version: "PR #27944",
    speed: [
      // bench_serving --flush-cache, MSA path, tp8; warm steady-state (3-run, identical).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1580, tpot_ms: 24.1, tokens_per_sec_per_gpu: 265 },
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
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1054, tpot_ms: 70.8, tokens_per_sec_per_gpu: 116 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // #27944, sgl-eval --thinking, full 1319, recommended sampling (temp 1.0/top_p 0.95/top_k 40); stable 97.04% across all 3 runs (std 0.0)
  },
  {
    match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64 },
        ttft_ms: null, tpot_ms: 32.8, tokens_per_sec_per_gpu: 365 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on B300 (legacy few_shot 200: 87.5)
  },
  // GB200: inferred-supported, not directly benchmarked.
  { match: { hw: "gb200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "gb300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64 },
        ttft_ms: 4746, tpot_ms: 39.3, tokens_per_sec_per_gpu: 277 },
      { workload: { dataset: "random", isl: 8192, osl: 256, max_concurrency: 24 },
        ttft_ms: 3324, tpot_ms: 32.9, tokens_per_sec_per_gpu: 131 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on GB300 (legacy few_shot 200: 87.5)
  },
  // MI355X (gfx950): native MXFP8. Measured on 8xMI355X with the public
  // MiniMaxAI/MiniMax-M3-MXFP8 model id and
  // aigmkt/minimax-m3-sglang-rocm720-mi35x:latest. Speed uses the config's
  // Reproduce command with fixed random-range-ratio=1.0:
  // 2016.75 output tok/s (4033.50 total incl. input); 2016.75 / 8 = 252.09
  // tokens/sec/GPU. GSM8K uses sgl-eval with the config's Reproduce command.
  {
    match: { hw: "mi355x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: 1616.84, tpot_ms: 30.17, tokens_per_sec_per_gpu: 252.09 },
    ],
    accuracy: { gsm8k_pct: 97.2 }, // sgl-eval: 97.19%, 1319 examples, stop_rate=100%, truncated_rate=0%.
  },
  // MI350X (gfx950): native MXFP8. Measured on 8xMI350X with the public
  // MiniMaxAI/MiniMax-M3-MXFP8 model id and
  // aigmkt/minimax-m3-sglang-rocm720-mi35x:latest. Speed uses the config's
  // Reproduce command with fixed random-range-ratio=1.0:
  // 2012.54 output tok/s (4025.09 total incl. input); 2012.54 / 8 = 251.57
  // tokens/sec/GPU. GSM8K uses sgl-eval.
  {
    match: { hw: "mi350x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: 902.30, tpot_ms: 30.94, tokens_per_sec_per_gpu: 251.57 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // sgl-eval: 97.04%, 1319 examples, stop_rate=100%, truncated_rate=0%.
  },
  // MI300X (gfx942): MXFP8 -> block-fp8 [128,128]. Fresh run on 8xMI300X with
  // aigmkt/minimax-m3-sglang-rocm700-mi30x:latest: bench_serving 1024/1024
  // @ conc 64, tp8 -> 1431.04 output tok/s (2862.08 total incl. input);
  // 1431.04 / 8 = 178.88 tokens/sec/GPU. GSM8K uses sgl-eval with the
  // config's Reproduce command.
  {
    match: { hw: "mi300x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: 2714.17, tpot_ms: 42.08, tokens_per_sec_per_gpu: 178.88 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // sgl-eval: 97.04%, 1319 examples, stop_rate=100%, truncated_rate=0%.
  },
  // MI325X (gfx942): same-arch sibling of MI300X; use the MI300X measured row.
  {
    match: { hw: "mi325x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: 2714.17, tpot_ms: 42.08, tokens_per_sec_per_gpu: 178.88 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // Same values as the MI300X row.
  },
];
