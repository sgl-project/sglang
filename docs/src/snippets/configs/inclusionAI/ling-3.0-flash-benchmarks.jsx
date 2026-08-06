// Ling-3.0-flash per-cell benchmark numbers, keyed by the same `match` tuple as
// ling-3.0-flash.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Accuracy harness (one harness for the whole GSM8K column, per
// config.benchmarkCommands.accuracy): sgl-eval run gsm8k, full 1319 questions,
// --num-threads 32. Every filled entry below also recorded 100% stop /
// 0% truncated / 0% error.
//
// Speed numbers are not measured yet — entries carry accuracy only.
//
// Cells with no entry (H20-3e / H800 / H100, both quantizations) had no matching
// allocation and were never gated.
export const benchmarks = [
  // ====================================================================
  // H200 + BF16 (TP4)
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ c5071ded",
    accuracy: { gsm8k_pct: 96.59 },
  },
  {
    // Rejected by the full GSM8K gate at request 1319: a no-EOS runaway generated
    // >33k tokens. Recipe stays `verified: false` in ling-3.0-flash.jsx.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { gsm8k_pct: null },
    notes: "Full GSM8K gate did not complete: one request ran away without emitting EOS (>33k generated tokens).",
  },

  // ====================================================================
  // H200 + FP8 (TP4 + EP4)
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 95.83 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 96.51 },
  },

  // ====================================================================
  // H200 + HiCache (Mooncake tiered cache)
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "hicache", nodes: "single" },
    sglang_version: "PR #33561 @ 51bcd89c",
    accuracy: { gsm8k_pct: 96.44 },
  },

  // ====================================================================
  // B200 + BF16 (TP4)
  // ====================================================================
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ c5071ded",
    accuracy: { gsm8k_pct: 96.44 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ c5071ded",
    accuracy: { gsm8k_pct: 96.51 },
  },

  // ====================================================================
  // B200 + FP8 (TP4 + EP4)
  // ====================================================================
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 96.59 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 97.04 },
  },

  // ====================================================================
  // GB300 + BF16 (TP4)
  // ====================================================================
  // TODO: both cells are `verified: true` (gated on the final head) but the GSM8K
  // percentages were not recorded in the PR body or in the verifying commits —
  // fill from the run logs.
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },

  // ====================================================================
  // GB300 + FP8 (TP4 + EP4)
  // ====================================================================
  // TODO: both cells are `verified: true` on the final head; the 96.66% / 96.44%
  // pair in the PR body predates the TP+EP change — fill with the re-measured values.
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
];
