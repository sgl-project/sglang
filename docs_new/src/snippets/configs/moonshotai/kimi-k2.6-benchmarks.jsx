// Kimi-K2.6 per-cell benchmark numbers, keyed by the same `match` tuple as
// kimi-k2.6.jsx cells. See _deployment.jsx for the speed/accuracy schema.
// Transcribed verbatim from the legacy page's measured blocks (§5).
//
// VERSION ANCHOR: sglang 0.5.9 (a real pinned release). Only same-model,
// version-pinned, reproducible results are migrated:
//   * DROPPED — the §5.2 H200 speed table: the legacy page states those numbers
//     were measured on Kimi-K2.5 and carried over as a reference. Cross-model
//     numbers are not this model's measurements, so they are not migrated.
//   * DROPPED — the NVFP4/B300 block: NVIDIA's published NVFP4-vs-INT4 reference,
//     not measured on this page and with no pinned sglang version → not
//     reproducible, not migrated.
//   * KEPT — §5.1 H200 accuracy (measured on K2.6 @ 0.5.9) and §5.3 MI350X speed
//     (measured on K2.6 @ 0.5.9).
//
// PARSERS NOTE (hard rule 3): the measured runs had the kimi_k2 reasoning +
// tool-call parsers ON (the accuracy suites depend on them). The Deployment
// cells still ship WITHOUT the parser flags — they are Playground-only — and
// no cell is verified (the maintainer has not attested any K2.6 deploy command).
//
// ACCURACY — §5.1, measured on 8xH200 / INT4 / sglang 0.5.9, parsers on. Each
// suite is its own external harness (see benchmarkCommands.accuracy):
//   toolcall_valid_pct = K2-Vendor-Verifier Tool Call Valid (869/970).
//   aime25_pct         = AIME 2025 majority@32 (pass@1 avg-of-32 was 98.9%).
//   gpqa_pct           = GPQA Diamond pass@1 avg across 4 epochs (553/792).
//   ocrbench_pct       = OCRBench pass@1 (1,000 image questions).
//   mmmu_pro_pct       = MMMU Pro Vision pass@1 (1,481/1,730 completed).
//
// SPEED — sglang.bench_serving, random dataset, --request-rate inf.
//   * MI350X (4x, tp4, INT4, --kv-cache-dtype fp8_e4m3): §5.3, measured on
//     Kimi-K2.6 directly. ROCm 7.0, docker lmsysorg/sglang:v0.5.9-rocm700-mi35x.
//   tokens_per_sec_per_gpu = output tok/s / (tp x nnodes). TTFT/TPOT = the
//   Mean rows. `workload.num_prompts` carries each block's --num-prompts.
export const benchmarks = [
  {
    // H200 / INT4 / low-latency — §5.1 accuracy suite (measured on K2.6 @ 0.5.9).
    // The §5.2 speed table is NOT here: it was measured on K2.5 (see header).
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.9",
    accuracy: {
      toolcall_valid_pct: 89.6,
      aime25_pct: 100.0,
      gpqa_pct: 96.9,
      ocrbench_pct: 90.8,
      mmmu_pro_pct: 82.2,
    },
    notes: "Accuracy measured on 8xH200 / INT4 / sglang 0.5.9 with the kimi_k2 reasoning + tool-call parsers on. AIME 2025: majority@32 100% (pass@1 avg-of-32 98.9%, pass@32 100%). GPQA Diamond: pass@1 avg across 4 epochs (553/792 samples). MMMU Pro Vision: pass@1 over 1,481/1,730 completed; use max-tokens >=32768 (a reasoning model exhausts a small budget on thinking).",
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
];
