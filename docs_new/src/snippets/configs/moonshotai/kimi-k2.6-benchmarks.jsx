// Kimi-K2.6 per-cell benchmark numbers, keyed by the same `match` tuple as
// kimi-k2.6.jsx cells. See _deployment.jsx for the speed/accuracy schema.
// Transcribed verbatim from the legacy page's measured blocks (§5).
//
// VERSION ANCHOR: sglang 0.5.9 (a real pinned release), so the speed numbers
// migrate (hard rule 2). The deploy/bench commands are in the config's
// benchmarkCommands (⚡ Reproduce).
//
// PARSERS NOTE (hard rule 3): the measured runs had the kimi_k2 reasoning +
// tool-call parsers ON (the accuracy suites depend on them; the AMD §5.3
// deploy command shows them explicitly). The Deployment cells still ship
// WITHOUT the parser flags — they are Playground-only — so the verified-cell
// flag-equality rule is moot here anyway (no cell is verified: the maintainer
// has not attested any K2.6 deploy command).
//
// SPEED — sglang.bench_serving, random dataset, --request-rate inf.
//   * H200 (8x, tp8, INT4): §5.2. IMPORTANT — the legacy page states these
//     speed numbers were measured on K2.5 and carried over as a reference
//     (K2.6 shares K2.5's architecture); see the `notes` field. They attach to
//     the bare H200/INT4 recipe (no DP-attention) = low-latency cell.
//   * MI350X (4x, tp4, INT4, --kv-cache-dtype fp8_e4m3): §5.3, measured on
//     Kimi-K2.6 directly. ROCm 7.0, docker lmsysorg/sglang:v0.5.9-rocm700-mi35x.
//   tokens_per_sec_per_gpu = output tok/s / (tp x nnodes). TTFT/TPOT = the
//   Mean rows. `workload.num_prompts` carries each block's --num-prompts.
//
// ACCURACY — §5.1, measured on 8xH200 / INT4 / sglang 0.5.9, parsers on. Each
// suite is its own external harness (see benchmarkCommands.accuracy):
//   toolcall_valid_pct = K2-Vendor-Verifier Tool Call Valid (869/970).
//   aime25_pct         = AIME 2025 majority@32 (pass@1 avg-of-32 was 98.9%).
//   gpqa_pct           = GPQA Diamond pass@1 avg across 4 epochs (553/792).
//   ocrbench_pct       = OCRBench pass@1 (1,000 image questions).
//   mmmu_pro_pct       = MMMU Pro Vision pass@1 (1,481/1,730 completed).
export const benchmarks = [
  {
    // H200 / INT4 / low-latency (bare recipe, DP-attention off) — carries the
    // §5.1 accuracy suite AND the §5.2 speed table.
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.9",
    speed: [
      // Scenario 1 — Chat (1K/1K)
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1,   num_prompts: 10 },
        ttft_ms: 177,  tpot_ms: 9.22,   tokens_per_sec_per_gpu: 13 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 16,  num_prompts: 80 },
        ttft_ms: 374,  tpot_ms: 53.25,  tokens_per_sec_per_gpu: 32 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 100, num_prompts: 500 },
        ttft_ms: 1290, tpot_ms: 364.70, tokens_per_sec_per_gpu: 32 },
      // Scenario 2 — Reasoning (1K/8K)
      { workload: { dataset: "random", isl: 1000, osl: 8000, max_concurrency: 1,   num_prompts: 10 },
        ttft_ms: 206,  tpot_ms: 14.36,  tokens_per_sec_per_gpu: 8 },
      { workload: { dataset: "random", isl: 1000, osl: 8000, max_concurrency: 16,  num_prompts: 80 },
        ttft_ms: 359,  tpot_ms: 111.18, tokens_per_sec_per_gpu: 16 },
      // Scenario 3 — Summarization (8K/1K)
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 1,   num_prompts: 10 },
        ttft_ms: 1626, tpot_ms: 24.95,  tokens_per_sec_per_gpu: 4 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 16,  num_prompts: 80 },
        ttft_ms: 2460, tpot_ms: 140.57, tokens_per_sec_per_gpu: 13 },
      { workload: { dataset: "random", isl: 8000, osl: 1000, max_concurrency: 64,  num_prompts: 320 },
        ttft_ms: 2710, tpot_ms: 443.84, tokens_per_sec_per_gpu: 17 },
    ],
    accuracy: {
      toolcall_valid_pct: 89.6,
      aime25_pct: 100.0,
      gpqa_pct: 96.9,
      ocrbench_pct: 90.8,
      mmmu_pro_pct: 82.2,
    },
    notes: "Accuracy measured on 8xH200 / INT4 / sglang 0.5.9 with the kimi_k2 reasoning + tool-call parsers on. AIME 2025: majority@32 100% (pass@1 avg-of-32 98.9%, pass@32 100%). GPQA Diamond: pass@1 avg across 4 epochs (553/792 samples). MMMU Pro Vision: pass@1 over 1,481/1,730 completed; use max-tokens >=32768 (a reasoning model exhausts a small budget on thinking). Speed numbers (§5.2) were measured on Kimi-K2.5, which shares K2.6's architecture, and carried over as an equivalent-performance reference.",
  },
  {
    // MI350X / INT4 / low-latency — §5.3 AMD speed (measured on K2.6 directly).
    match: { hw: "mi350x", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.9",
    speed: [
      // Latency (1K/1K), AITER + fp8 KV cache, tp4.
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 1,  num_prompts: 10 },
        ttft_ms: 564, tpot_ms: 35.61,  tokens_per_sec_per_gpu: 7 },
      { workload: { dataset: "random", isl: 1000, osl: 1000, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 989, tpot_ms: 191.04, tokens_per_sec_per_gpu: 19 },
    ],
    notes: "AMD Instinct MI350X (4x), ROCm 7.0, docker lmsysorg/sglang:v0.5.9-rocm700-mi35x. Launched with SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0, --mem-fraction-static 0.8, --kv-cache-dtype fp8_e4m3, and the kimi_k2 parsers on. TP must be <=4 on AMD (Kimi-K2.6 has 64 attention heads; the AITER MLA kernel needs heads_per_gpu % 16 == 0).",
  },
  {
    // NVFP4 (B300) — no SGLang-measured serving numbers on the legacy page.
    // NVIDIA's published NVFP4-vs-INT4 accuracy reference is recorded in notes
    // (a different suite set than the §5.1 labels, so not structured rows).
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    notes: "NVIDIA-reported accuracy for nvidia/Kimi-K2.6-NVFP4 vs the native INT4 baseline (temperature 1.0, top_p 0.95, max tokens 128,000) — GPQA Diamond 90.4 vs 90.9, SciCode 54.4 vs 52.6, t2-Bench Telecom 98.0 vs 98.2, MMMU Pro 76.5 vs 75.6, AA-LCR 71.8 vs 71.0, IFBench 73.9 vs 73.9. These are NVIDIA's published references, not measured on this page.",
  },
];
