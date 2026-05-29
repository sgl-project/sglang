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

// All numbers below were measured on **sglang v0.5.12.post1**. When that version
// is updated, refresh the `sglang_version` field on every entry (or sweep
// with a single find-and-replace).
//
// Historical note on the migration: cells originally carried separate
// `latency` (concurrency=1) and `throughput` (concurrency=100) blocks.
// These have been unified into the `speed` array — low-latency-only
// cells now have one entry at c=1, high-throughput-only cells have one
// entry at c=100, and balanced cells have both. The "—" cells in the
// rendered table mark metrics not measured at that particular workload
// and are candidates for the next benchmark sweep.
export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 428, tpot_ms: 3.53, tokens_per_sec_per_gpu: 44 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3111, tpot_ms: 23.82, tokens_per_sec_per_gpu: 121 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 4228, tpot_ms: 60.98, tokens_per_sec_per_gpu: 225 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4628, tpot_ms: 88.25, tokens_per_sec_per_gpu: 643 },
    ],
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 105918, tpot_ms: 70.73, tokens_per_sec_per_gpu: 881 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 273356, tpot_ms: 71.61, tokens_per_sec_per_gpu: 889 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2326, tpot_ms: 69.9, tokens_per_sec_per_gpu: 99 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 7242, tpot_ms: 152.09, tokens_per_sec_per_gpu: 192 },
    ],
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // B300 + FP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 205, tpot_ms: 3.43, tokens_per_sec_per_gpu: 54 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1856, tpot_ms: 14.82, tokens_per_sec_per_gpu: 205 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2363, tpot_ms: 34.4, tokens_per_sec_per_gpu: 402 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 2812, tpot_ms: 51.65, tokens_per_sec_per_gpu: 1092 },
    ],
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 82556, tpot_ms: 55.37, tokens_per_sec_per_gpu: 1130 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 207987, tpot_ms: 54.05, tokens_per_sec_per_gpu: 1171 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 239, tpot_ms: 5.04, tokens_per_sec_per_gpu: 24 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 830, tpot_ms: 15.55, tokens_per_sec_per_gpu: 101 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1866, tpot_ms: 54.48, tokens_per_sec_per_gpu: 139 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 6325, tpot_ms: 123.95, tokens_per_sec_per_gpu: 237 },
    ],
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 99139, tpot_ms: 44.37, tokens_per_sec_per_gpu: 476 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 241544, tpot_ms: 43.51, tokens_per_sec_per_gpu: 492 },
    ],
  },
  // ====================================================================
  // GB200 + FP4
  // ====================================================================
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 335, tpot_ms: 3.67, tokens_per_sec_per_gpu: 47 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2440, tpot_ms: 15.95, tokens_per_sec_per_gpu: 163 },
    ],
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2560, tpot_ms: 39.71, tokens_per_sec_per_gpu: 342 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 3995, tpot_ms: 82.56, tokens_per_sec_per_gpu: 718 },
    ],
    accuracy: { gsm8k_pct: 99 },
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 128397, tpot_ms: 84.95, tokens_per_sec_per_gpu: 757 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 330479, tpot_ms: 86.7, tokens_per_sec_per_gpu: 741 },
    ],
    accuracy: { gsm8k_pct: 98 },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 343, tpot_ms: 6.47, tokens_per_sec_per_gpu: 18 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1345, tpot_ms: 23.85, tokens_per_sec_per_gpu: 65 },
    ],
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
  },
  // ====================================================================
  // GB300 + FP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 380, tpot_ms: 4.4, tokens_per_sec_per_gpu: 38 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2960, tpot_ms: 21.26, tokens_per_sec_per_gpu: 125 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 2671, tpot_ms: 45.88, tokens_per_sec_per_gpu: 299 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4823, tpot_ms: 94.04, tokens_per_sec_per_gpu: 637 },
    ],
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 146954, tpot_ms: 97.24, tokens_per_sec_per_gpu: 662 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 368557, tpot_ms: 99.33, tokens_per_sec_per_gpu: 651 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 363, tpot_ms: 6.53, tokens_per_sec_per_gpu: 36 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1275, tpot_ms: 20.75, tokens_per_sec_per_gpu: 152 },
    ],
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // H200 + FP8
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 204, tpot_ms: 3.38, tokens_per_sec_per_gpu: 68 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 538, tpot_ms: 11.42, tokens_per_sec_per_gpu: 264 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 738, tpot_ms: 36.27, tokens_per_sec_per_gpu: 385 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 39806, tpot_ms: 80.13, tokens_per_sec_per_gpu: 393 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 195293, tpot_ms: 130.35, tokens_per_sec_per_gpu: 493 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 502615, tpot_ms: 130.31, tokens_per_sec_per_gpu: 490 },
    ],
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "low-latency", nodes: "multi-2" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "balanced", nodes: "multi-2" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
  },
  // ====================================================================
  // H200 + FP4
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 193, tpot_ms: 3.38, tokens_per_sec_per_gpu: 67 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 598, tpot_ms: 10.46, tokens_per_sec_per_gpu: 308 },
    ],
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  // ====================================================================
  // H100 + FP4
  // ====================================================================
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "high-throughput", nodes: "single" },
  },
  {
    match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
  },
  {
    match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" },
  },
  {
    match: { hw: "h100", variant: "pro", quant: "fp4", strategy: "high-throughput", nodes: "multi-2" },
  },
];
