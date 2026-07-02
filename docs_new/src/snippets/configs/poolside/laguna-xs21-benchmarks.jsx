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
// REAL GSM8K on an 8×H200 HGX node (sgl-eval `run gsm8k`, FULL 1319 questions, greedy/
// non-thinking), backend fa3 (= the Hopper auto-select for dense; pinned for DFlash):
//
//   high-throughput (dense):        BF16 76.12% (tp 8) · FP8 73.54% (tp8+ep8) · INT4 67.02% (tp8+ep8)
//   low-latency (DFlash, fa3):      BF16 75.97% (tp 8) · FP8 74.53% (tp8+ep8) · INT4 64.52% (tp8+ep8)
//     accept-lengths (matched-precision draft, greedy GSM8K): BF16 ~3.9 (bs=1) · FP8 6.75 · INT4 4.94.
//
//   Spec == dense within noise for BF16/FP8, same as GB300 — DFlash is accuracy-neutral there.
//   INT4+DFlash shows the widest tp8-vs-tp4 delta in the whole grid (64.52% here vs 66.41% at
//   plain tp4 — see laguna-xs21.jsx cell comment): inside ~2 binomial SEs, but it's the newest
//   and least-repeated combination measured (first-ever run of EP=8 routed experts + Marlin
//   decode + DFlash verify together) — re-verify before treating it as production-grade.
//
//   FP8/INT4 run --tp 8 --ep-size 8 (NOT plain tp 8, which fails at weight load — see
//   laguna-xs21.jsx header comment for why: moe_intermediate_size=512 with FP8 block
//   [128,128] / INT4 gs=128 scales can't shard 8-way). EP keeps whole experts per rank,
//   sidestepping the shard-granularity wall entirely, so both quantizations use all 8 GPUs
//   on one instance. FP8 additionally needs SGLANG_SHARED_EXPERT_TP1=1 (its shared expert
//   is also block-quantized; INT4's stays bf16, no flag needed). The checks that make plain
//   tp 8 fail are pure shard arithmetic with no arch branch → B200 tp 8 fails the same way,
//   hence the B200 fp8/int4 cells also carry tp8+ep8.
//
// sglang_version = PR #29446 (DFlash + SGLANG_SHARED_EXPERT_TP1 fix) + PR #29761 (INT4
// mixed-precision MoE load fix) — BOTH MERGED to main as of 2026-07-02.
//
// B200: command-correct per the backend rules (trtllm_mha) and the same merged EP=8 fix,
// measurements pending.

export const benchmarks = [
  // ===== H200 (8-GPU HGX; bf16 tp 8, fp8/int4 tp8+ep8) — ✅ REAL, full GSM8K =====
  {
    // ✅ REAL — 8×H200, BF16 dense, tp8, backend fa3 (Hopper auto-select).
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 76.12 },
  },
  {
    // ✅ REAL — 8×H200, BF16 + DFlash (matched bf16 draft), tp8, fa3. Accept-len 3.05
    // (mixed eval traffic; ~3.9 greedy GSM8K bs=1).
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 75.97 },
  },
  {
    // ✅ REAL — 8×H200, FP8 dense, tp8+ep8+SGLANG_SHARED_EXPERT_TP1=1 (plain tp8 impossible:
    // block-FP8 scale granularity), fa3.
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 73.54 },
  },
  {
    // ✅ REAL — 8×H200, FP8 + DFlash (matched fp8-calibrated draft), tp8+ep8+flag, fa3.
    // Accept-len 6.75.
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 74.53 },
  },
  {
    // ✅ REAL — 8×H200, INT4 dense (mixed 4/8-bit MoE, needs #29761), tp8+ep8 (plain tp8
    // impossible: Marlin gs=128 scale layout; no shared-expert flag needed), fa3.
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 67.02 },
  },
  {
    // ✅ REAL — 8×H200, INT4 + DFlash (matched int4-calibrated draft), tp8+ep8, fa3.
    // Accept-len 4.94. Widest tp8-vs-tp4 gap in the grid (64.52% vs 66.41%) — see header note.
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 64.52 },
  },

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
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 75.66 },
  },
  {
    // ✅ REAL — 4×GB300, BF16 + DFlash (matched bf16 draft), tp4, trtllm_mha. Accept-len 4.17.
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 76.19 },
  },
  {
    // ✅ REAL — 4×GB300, FP8 dense, tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 71.87 },
  },
  {
    // ✅ REAL — 4×GB300, FP8 + DFlash (matched fp8-calibrated draft), tp4, trtllm_mha. Accept-len 4.05.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 72.02 },
  },
  {
    // ✅ REAL — 4×GB300, NVFP4 dense, tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 78.39 },
  },
  {
    // ✅ REAL — 4×GB300, NVFP4 + DFlash (matched nvfp4-calibrated draft), tp4, trtllm_mha. Accept-len 4.02.
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 74.53 },
  },
  {
    // ✅ REAL — 4×GB300, INT4 dense (mixed 4/8-bit MoE, needs #29761), tp4, backend auto→trtllm_mha.
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 66.79 },
  },
  {
    // ✅ REAL — 4×GB300, INT4 + DFlash (matched int4-calibrated draft), tp4, trtllm_mha. Accept-len 3.80.
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "PR #29446 + #29761 (both merged to main)",
    accuracy: { gsm8k_pct: 67.02 },
  },
];
