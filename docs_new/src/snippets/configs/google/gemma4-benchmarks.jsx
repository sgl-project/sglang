// Measured ACCURACY benchmarks for the Gemma 4 cookbook.
//
// Source: the legacy Gemma4 page §5.2 (Accuracy Benchmark), measured on H200
// with the base (MTP-off) launch command `sglang serve --model-path
// google/gemma-4-<variant>-it` — no parser flags, no speculative decoding — so
// every entry is keyed strategy:"high-throughput" (MTP off) / quant:"bf16".
//
// SPEED numbers from the legacy §5.1 are intentionally NOT migrated: the page
// pinned only the moving "gemma4 branch" ref, which is no reproducible build
// anchor (hard rule 2 — no version anchor ⇒ drop speed AND `sglang_version`,
// keep accuracy). Accuracy is far less build-sensitive; `benchmarkCommands`
// (config) still drives ⚡ Reproduce for re-measurement against a pinned release.
//
// Values are the legacy page's measured figures verbatim (0–1 fractions):
//   MMLU  = §5.2 "MMLU" table, Overall column.
//   MMMU  = §5.2 "MMMU" table, Overall column (900-sample val split; see the
//           per-domain JSON "Overall": {"num": 900}).
//   GSM8K = only gemma-4-12B-it, and only the chat-template run_eval score
//           (0.950), which is command-consistent with
//           config.benchmarkCommands.accuracy.gsm8k (sglang.test.run_eval
//           --eval-name gsm8k). The §5.2 few-shot-completion GSM8K table
//           (sglang.test.few_shot_gsm8k) and the sgl-eval CLI numbers use
//           different harnesses and are NOT migrated.

export const benchmarks = [
  {
    match: { hw: "h200", variant: "e2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.720, mmmu: 0.307 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
  {
    match: { hw: "h200", variant: "e4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.810, mmmu: 0.396 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
  {
    match: { hw: "h200", variant: "12b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.859, mmmu: 0.683, gsm8k: 0.950 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split. GSM8K 0.950 via the chat-template run_eval harness on the 1319-question test set (the raw few-shot harness under-elicits this reasoning-oriented variant).",
  },
  {
    match: { hw: "h200", variant: "31b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.896, mmmu: 0.589 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
  {
    match: { hw: "h200", variant: "26b-a4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.891, mmmu: 0.549 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
];
