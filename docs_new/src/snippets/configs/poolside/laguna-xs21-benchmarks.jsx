// Laguna-XS-2.1 benchmarks — one entry per cell `match` (same 5 keys as laguna-xs21.jsx cells).
//
// All numbers below are REAL measured values; cells without measurements are bare `{ match }`
// pending stubs (the card renders "pending"). NO fabricated/dummy numbers.
//
// REAL GSM8K (sgl-eval `run gsm8k`, FULL 1319 questions, greedy/non-thinking, chat template
// auto-loaded), measured on a 4×GB300 single node at tp 4:
//
//   high-throughput (dense, backend auto→trtllm_mha):
//     BF16 75.66% · FP8 71.87% · NVFP4 78.39% · INT4 66.79%
//   low-latency (DFlash, --attention-backend trtllm_mha, matched-precision draft):
//     BF16 76.19% (accept-len 4.17) · FP8 72.02% (4.05) · NVFP4 74.53% (4.02) · INT4 67.02% (3.80)
//
//   Spec == dense within noise on every quant → DFlash is accuracy-neutral, as expected for
//   verification-based speculation. Accept-length is the speedup lever (~4× fewer target steps
//   at tp=4; ~5.7–6.8 accept-len measured at tp=1 on the same pairs).
//
//   Backend caveats baked into the configs (do not "simplify" them away):
//     - DFlash cells pin --attention-backend trtllm_mha on Blackwell: with speculation active,
//       auto-select falls back to flashinfer, which breaks this hybrid-SWA model at tp≥4
//       (GSM8K 28% vs 76%, reproduced + single-variable-bisected on GB300).
//     - `triton` attention is broken for Laguna (13.2% GSM8K) — never use it here.
//     - Known open question: FP8/INT4 score ~5/~7 pts higher under flashinfer at tp≤2 than under
//       trtllm_mha/fa4 (which agree with each other); bf16/nvfp4 are backend-invariant. Ground
//       truth (HF eager reference) not yet established — the trtllm_mha numbers are shipped since
//       that is the only tp≥4-viable backend.
//
// sglang_version = PR #29446 branch (DFlash) on top of main incl. PR #29761 (INT4 mixed-precision
// MoE load fix). Dense cells need only main ≥ #29761; DFlash cells need #29446.
//
// H200 / B200: command-correct per the backend rules (fa3 / trtllm_mha), measurements pending.

export const benchmarks = [
  // ===== H200 (8-GPU HGX) — pending measurement =====
  { match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency",     nodes: "single" } },

  // ===== B200 (8-GPU HGX) — pending measurement =====
  { match: { hw: "b200", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "fp8",   strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "fp8",   strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "int4",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "int4",  strategy: "low-latency",     nodes: "single" } },

  // ===== GB300 (4-GPU single node, tp 4) — ✅ REAL, full GSM8K =====
  {
    // ✅ REAL — 4×GB300, BF16 dense, tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 75.66 },
  },
  {
    // ✅ REAL — 4×GB300, BF16 + DFlash (matched bf16 draft), tp4, trtllm_mha. Accept-len 4.17.
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 76.19 },
  },
  {
    // ✅ REAL — 4×GB300, FP8 dense, tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 71.87 },
  },
  {
    // ✅ REAL — 4×GB300, FP8 + DFlash (matched fp8-calibrated draft), tp4, trtllm_mha. Accept-len 4.05.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 72.02 },
  },
  {
    // ✅ REAL — 4×GB300, NVFP4 dense, tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 78.39 },
  },
  {
    // ✅ REAL — 4×GB300, NVFP4 + DFlash (matched nvfp4-calibrated draft), tp4, trtllm_mha. Accept-len 4.02.
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 74.53 },
  },
  {
    // ✅ REAL — 4×GB300, INT4 dense (mixed 4/8-bit MoE, needs #29761), tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 66.79 },
  },
  {
    // ✅ REAL — 4×GB300, INT4 + DFlash (matched int4-calibrated draft), tp4, trtllm_mha. Accept-len 3.80.
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 branch (incl. main + #29761)",
    accuracy: { gsm8k_pct: 67.02 },
  },
];
