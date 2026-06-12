// MiniMax-M3 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m3.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// SPEED — bench_serving --flush-cache, random isl2048/osl256, max_concurrency 64,
// CUDA graph on. B200 (tp4, MXFP8, MSA fmha_sm100 path) and H200 (tp8, bf16,
// built-in Triton sparse) are measured on PR #27944 (run-1; a 3-run mean is
// pending). B300 / GB300 rows are the earlier sglang main (2026-06-11) tp4 MSA
// numbers, pending a #27944 re-measure on their own boxes. GB200 is a bare-match
// stub (inferred-supported, not benchmarked). AMD: MI355X at 8-GPU tp8 (native
// MXFP8) carries a bench_serving speed row; MI300X (MXFP8 -> block-fp8) was
// accuracy-only. MI350X / MI325X inherit their same-arch sibling's recipe
// (stubs). (sgl-eval does NOT measure serving throughput — TTFT/TPOT/tok-s come
// from sglang.bench_serving.)
//
// GSM8K — unified on a SINGLE harness: sgl-eval (github.com/sgl-project/sgl-eval)
// `run gsm8k`, full 1319-question test split, chat endpoint with --thinking
// (M3's reasoning path) + M3's recommended sampling (temp 1.0 / top_p 0.95 /
// top_k 40), symbolic grading. This is the config's Reproduce command. B200
// (MSA path) and H200 (bf16, built-in Triton sparse) are measured on PR #27944;
// a 3-run mean±std and the matching #27944 bench_serving speed are in progress.
// Per-platform re-measurement under sgl-eval is in progress; rows still pending
// show `gsm8k_pct: null` (no GSM8K row rendered) with the legacy-harness number
// kept in a comment. Legacy harnesses were NOT comparable across platforms
// (NVIDIA: few_shot_gsm8k --num-questions 200; AMD: run_eval gsm8k 1319 examples) —
// which is exactly why we re-measure on one harness.
export const benchmarks = [
  {
    match: { hw: "b200", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      // bench_serving --flush-cache, MSA path; run-1 (3-run mean pending).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 2410, tpot_ms: 148.4, tokens_per_sec_per_gpu: 124 },
    ],
    accuracy: { gsm8k_pct: 94.4 }, // #27944, sgl-eval --thinking, full 1319, recommended sampling (temp 1.0/top_p 0.95/top_k 40), MSA path; run-1 94.39% (greedy 94.16%); --no-thinking 88.6%
  },
  {
    // Hopper H200: bf16 build (MXFP8 is Blackwell-only) at tp8, built-in Triton
    // sparse path (MSA is Blackwell-only). GSM8K re-measured on #27944.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "PR #27944",
    speed: [
      // bench_serving --flush-cache, bf16 Triton path; run-1 (3-run mean pending).
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 1068, tpot_ms: 78.0, tokens_per_sec_per_gpu: 105 },
    ],
    accuracy: { gsm8k_pct: 97.0 }, // #27944, sgl-eval --thinking, full 1319, recommended sampling (temp 1.0/top_p 0.95/top_k 40); run-1 97.04%
  },
  {
    match: { hw: "b300", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main (2026-06-11)",
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
    sglang_version: "main (2026-06-11)",
    speed: [
      { workload: { dataset: "random", isl: 2048, osl: 256, max_concurrency: 64 },
        ttft_ms: 4746, tpot_ms: 39.3, tokens_per_sec_per_gpu: 277 },
      { workload: { dataset: "random", isl: 8192, osl: 256, max_concurrency: 24 },
        ttft_ms: 3324, tpot_ms: 32.9, tokens_per_sec_per_gpu: 131 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on GB300 (legacy few_shot 200: 87.5)
  },
  // MI355X (gfx950): native MXFP8. Speed: bench_serving 1024/1024 @ conc 64, tp8
  // -> 1678 output tok/s (3355 total incl. input); 1678 / 8 = ~210 tokens/sec/GPU.
  // No TTFT/TPOT reported for this run.
  {
    match: { hw: "mi355x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main (2026-06-11)",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 640 },
        ttft_ms: null, tpot_ms: null, tokens_per_sec_per_gpu: 210 },
    ],
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on MI355X (legacy run_eval 1319: 92.2)
  },
  // MI350X (gfx950): inferred-supported from MI355X, not separately benchmarked.
  { match: { hw: "mi350x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
  // MI300X (gfx942): MXFP8 -> block-fp8 [128,128].
  {
    match: { hw: "mi300x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" },
    sglang_version: "main (2026-06-11)",
    accuracy: { gsm8k_pct: null }, // TODO: pending sgl-eval re-measure on MI300X (legacy run_eval 1319: 92.0, triton 0.917-0.929 / aiter ~0.929)
  },
  // MI325X (gfx942): inferred-supported from MI300X, not separately benchmarked.
  { match: { hw: "mi325x", variant: "default", quant: "mxfp8", strategy: "balanced", nodes: "single" } },
];
