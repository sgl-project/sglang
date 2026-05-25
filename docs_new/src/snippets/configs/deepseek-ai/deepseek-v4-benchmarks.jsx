// DeepSeek-V4 cookbook benchmarks — paired with deepseek-v4.jsx + _deployment.jsx.
//
// One entry per cell in `config.cells`, keyed by the same `match` tuple
// (hw × variant × quant × strategy × nodes). When the user picks a
// combination in the Deploy panel, the engine looks up the entry whose
// `match` equals the current selection and renders it as a benchmark
// sub-card under the command box. Cells without any measured numbers
// render as the empty state ("Benchmark data pending …").
//
// Schema (everything except `match` is optional — present fields render,
// absent ones collapse out):
//
//   {
//     match:           { hw, variant, quant, strategy, nodes },     // REQUIRED
//     sglang_version:  string,                                       // card header
//     speed: [
//       {
//         workload: { dataset, isl, osl, max_concurrency },          // structured
//         ttft_ms, tpot_ms, tokens_per_sec_per_gpu,                  // measured
//       },                                                           // one entry per
//       ...                                                          //   workload (typically
//                                                                    //   varying max-concurrency
//                                                                    //   for a Pareto sweep)
//     ],
//     accuracy: { gsm8k_pct, ... },                                  // extensible — add
//                                                                    //   more keys when new
//                                                                    //   accuracy benches land
//     notes:    string,                                              // optional caveat
//   }
//
// The engine renders `speed` as a metric × workload table:
//   - rows: TTFT, TPOT, tokens/sec/GPU, interactivity (1000/TPOT_ms)
//   - columns: one per workload entry
//   - shared workload parts (dataset, in/out) lift to an italic line above
//   - per-column header shows the differing parts (typically `c=N`)
//
// `interactivity` is derived from `tpot_ms` and never stored. Any
// measured field set to null (or absent) renders as "—" in its cell —
// so partial sweeps (only TTFT/TPOT at c=1; only tokens/sec at c=100)
// degrade gracefully.
//
// `accuracy` renders as a single muted row ABOVE the speed table
// (semantic priority — model quality leads serving speed). The engine
// concatenates whichever accuracy fields are present, so adding a new
// benchmark (e.g. `math_pct`) only requires updating ACCURACY_LABELS
// in _deployment.jsx and adding the key here.
//
// `speed` also accepts a single object (e.g. `speed: { workload: ..., ttft_ms: ... }`)
// for cookbooks that only measure one workload per cell; the engine
// wraps it to `[speed]` internally.
//
// Editing policy: this file turns over independently of the cell
// catalog (new sglang version → new numbers, but no flag changes).
// Keep the entry count and ordering in sync with deepseek-v4.jsx's
// `cells: [...]` so the two files diff cleanly side by side.

// All numbers below were measured on **sglang v0.5.12**. When that version
// is updated, refresh the `sglang_version` field on every entry (or sweep
// with a single find-and-replace).
//
// Historical note on the migration: cells originally carried separate
// `latency` (concurrency=1) and `throughput` (concurrency=100) blocks.
// These have been unified into the `speed` array — low-latency-only
// cells now have one entry at c=1, max-throughput-only cells have one
// entry at c=100, and balanced cells have both. The "—" cells in the
// rendered table mark metrics not measured at that particular workload
// and are candidates for the next benchmark sweep.
export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 86, tpot_ms: 3.42 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 3464, tpot_ms: 5.63 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 677 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 1215 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 171, tpot_ms: 5.12 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 183, tpot_ms: 8.49 },
      // c=100 column shows up empty so the planned throughput slot is
      // visible — refill numbers once the throughput bench re-runs.
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 } },
    ],
    notes: "Throughput bench pending re-validation on this cell.",
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 372 },
    ],
  },

  // ====================================================================
  // B300 + FP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 84, tpot_ms: 3.33 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 157, tpot_ms: 5.59 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 1036 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 1213 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 115, tpot_ms: 5.04 },
    ],
    accuracy: { gsm8k_pct: 96.5 },
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 1507, tpot_ms: 8.43 },
      // c=100 placeholder — fill in when the throughput bench re-runs.
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 } },
    ],
    notes: "Throughput bench pending re-validation on this cell.",
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 359 },
    ],
  },

  // ====================================================================
  // GB200 + FP4
  // ====================================================================
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 194, tpot_ms: 3.53 },
    ],
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 335, tpot_ms: 6.18 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 614 },
    ],
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 718 },
    ],
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 263, tpot_ms: 6.91 },
    ],
  },
  // GB200 Pro balanced: no cookbook recipe completed both lat+thp on 2
  // nodes — §3.1 cell carries the yellow Auto-Estimated badge, no benchmark
  // numbers written. Re-run once the deepep / megamoe-w4a8 bench finishes.
  { match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" } },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "multi-2" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 226 },
    ],
  },

  // ====================================================================
  // GB300 + FP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 226, tpot_ms: 4.30 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 5131, tpot_ms: 6.43 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 373 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 682 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 286, tpot_ms: 6.83 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 410, tpot_ms: 10.30 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 342 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 378 },
    ],
  },

  // ====================================================================
  // H200 + FP8
  // ====================================================================
  // FP8 sweep on H200 has not been benchmarked yet — all FP8 cells render
  // the pending-state empty card. Fill these in once the H200 FP8 deploy
  // (sgl-project repackaging on Hopper) is re-verified.
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "max-throughput", nodes: "multi-2" } },

  // ====================================================================
  // H200 + FP4
  //   Flash low-latency / max-throughput → marlin (cookbook)
  //   Flash balanced                     → flashinfer_mxfp4 (playground, +26% thp)
  //   Pro all 3 strategies               → flashinfer_mxfp4 (playground TP=4 sweep)
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 132, tpot_ms: 3.48 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 139, tpot_ms: 4.71 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 746 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 644 },
    ],
  },
  // H200 Pro on FP4: numbers transcribed from the dsv4-benchmark playground
  // sweep (TP=4, flashinfer-mxfp4 runner). The cookbook command in §3.1
  // mirrors the same backend; cookbook bench re-run on TP=8 is the next
  // round of work.
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 176, tpot_ms: 5.28 },
    ],
    accuracy: { gsm8k_pct: 97.5 },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 190, tpot_ms: 7.47 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 270 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 100 },
        tokens_per_sec_per_gpu: 288 },
    ],
  },

  // ====================================================================
  // H100 + FP4 (all cells pending — H100 was not part of the v0.5.12 sweep)
  // ====================================================================
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "multi-2" } },
];
