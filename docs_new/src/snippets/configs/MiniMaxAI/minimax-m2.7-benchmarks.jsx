// MiniMax-M2.7 per-cell benchmark numbers, keyed by the same `match` tuple as
// minimax-m2.7.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// All numbers are transcribed verbatim from the legacy page's §5 measured
// blocks — Test Environment: 2x NVIDIA GB300 (275GB/die), FP8, TP=2, docker
// lmsysorg/sglang:v0.5.10.post1-cu130, SGLang version 0.5.10.post1 (a release
// anchor → reproducible, so the result migrates). The cell this attaches to is
// GB300 / FP8 / low-latency (TP=2), which matches that environment.
//
// SPEED — sglang.bench_serving, random isl/osl 1000/1000 (legacy §5.2):
//   low concurrency  (max_concurrency 1,   num_prompts 10):  Mean TTFT 50.28 ms,
//     Mean TPOT 8.02 ms, output 122.92 tok/s -> 122.92 / (tp 2 x 1 node) ≈ 61.
//   high concurrency (max_concurrency 100, num_prompts 500): Mean TTFT 247.94 ms,
//     Mean TPOT 35.75 ms, output 2521.66 tok/s -> 2521.66 / 2 ≈ 1261.
//
// ACCURACY — the legacy §5.1 NVIDIA NeMo-Skills (`ns eval`) runs, all with
// thinking ON (--reasoning-parser minimax-append-think + ++parse_reasoning=True);
// the speed bench above ran WITHOUT parsers. GPQA Diamond / AIME 2025 are
// pass@1 (avg-of-8); MMLU-Pro is pass@1 (greedy); GSM8K is 8-shot CoT (full 1319,
// 1218 correct). NeMo-Skills sampling: temp 0.6 / top_p 0.95 (MMLU-Pro greedy
// temp 0.0). The MMLU-Pro 32K token cap left an 18.75% no-answer rate (legacy
// §5.1.3 note: a 120K rerun is expected to improve it) — kept verbatim.
export const benchmarks = [
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.10.post1",
    speed: [
      // Low concurrency: random 1000/1000, max_concurrency 1, num_prompts 10.
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 10 },
        ttft_ms: 50.28, tpot_ms: 8.02, tokens_per_sec_per_gpu: 61 },
      // High concurrency: random 1000/1000, max_concurrency 100, num_prompts 500.
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 100, num_prompts: 500 },
        ttft_ms: 247.94, tpot_ms: 35.75, tokens_per_sec_per_gpu: 1261 },
    ],
    // NeMo-Skills, thinking ON. GPQA pass@1[avg-of-8] (majority@8 88.89%, pass@8
    // 96.46%); AIME25 pass@1[avg-of-8] ±5.56% (majority@8 97.08%, pass@8 100%);
    // MMLU-Pro pass@1 greedy (18.75% no-answer at the 32K cap); GSM8K 8-shot CoT.
    accuracy: { gpqa_pct: 84.91, aime25_pct: 92.50, mmlu_pro_pct: 69.41, gsm8k_pct: 92.34 },
  },
  {
    // MI300X ×2 / MiniMax-M2.7 / FP8 KV cache (fp8_e4m3) / tp=2 ep=2 / low-latency.
    // Measured on AMD Instinct MI300X (8×192 GB), sglang 0.5.13.post1,
    // docker lmsysorg/sglang:v0.5.13.post1-rocm720-mi30x.
    // Server flags: --attention-backend triton --mem-fraction-static 0.85
    //   --kv-cache-dtype fp8_e4m3.
    // Benchmark: sglang.bench_serving --dataset-name random --random-input-len 1000
    //   --random-output-len 1000 --warmup-requests 64 --request-rate inf.
    match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 210, tpot_ms: 14.94, tokens_per_sec_per_gpu: 33 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 16, num_prompts: 256 },
        ttft_ms: 167, tpot_ms: 31.80, tokens_per_sec_per_gpu: 243 },
    ],
  },
  {
    // MI300X ×4 / MiniMax-M2.7 / FP8 KV cache (fp8_e4m3) / tp=4 ep=4 / balanced.
    match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 64, num_prompts: 512 },
        ttft_ms: 419, tpot_ms: 42.48, tokens_per_sec_per_gpu: 351 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 256, num_prompts: 1024 },
        ttft_ms: 337, tpot_ms: 71.72, tokens_per_sec_per_gpu: 809 },
    ],
  },
  {
    // MI300X ×8 / MiniMax-M2.7 / FP8 KV cache (fp8_e4m3) / tp=8 ep=8 / high-throughput.
    match: { hw: "mi300x", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.13.post1",
    speed: [
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 3547, tpot_ms: 239.18, tokens_per_sec_per_gpu: 538 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 4096, num_prompts: 4096 },
        ttft_ms: 18251, tpot_ms: 431.16, tokens_per_sec_per_gpu: 1043 },
    ],
  },
];
