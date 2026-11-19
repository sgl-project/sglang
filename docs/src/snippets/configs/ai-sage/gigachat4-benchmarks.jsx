// H100: two nodes, TP16/EP1, FP8 target, measured on 2026-11-18.
// Native streaming /generate, random ISL 8192 / OSL 1024, greedy, four warmup
// requests followed by cache flush. Throughput includes input and output tokens.
// Off retains three completed repeats (median); EAGLE and DFlash2 use one run.
// H200 entries remain pending. Accuracy will be added after the final harness run.

export const benchmarks = [
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "multi-2" },
    sglang_version: "0c0cfff57a (PR #50613, 2026-11-18)",
    notes: "Median of three measured runs. Speculative decoding off.",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 844.81, tpot_ms: 23.68, tokens_per_sec_per_gpu: 22.97 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 8023.83, tpot_ms: 42.07, tokens_per_sec_per_gpu: 180.84 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 28907.87, tpot_ms: 70.13, tokens_per_sec_per_gpu: 310.12 },
    ] },
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "0c0cfff57a (PR #50613, 2026-11-18)",
    notes: "One measured run per concurrency. EAGLE 3/1/4 with bundled MTP.",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 889.19, tpot_ms: 7.37, tokens_per_sec_per_gpu: 65.99 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2645.11, tpot_ms: 24.78, tokens_per_sec_per_gpu: 298.98 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 13970.14, tpot_ms: 66.18, tokens_per_sec_per_gpu: 409.27 },
    ] },
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "dflash2", nodes: "multi-2" },
    sglang_version: "0c0cfff57a (PR #50613, 2026-11-18)",
    notes: "One measured run per concurrency. DFlash2 block 8 with a BF16 draft; effective running-request limit 52, so concurrency 64 includes queueing.",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 879.15, tpot_ms: 4.26, tokens_per_sec_per_gpu: 99.35 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 4830.51, tpot_ms: 15.70, tokens_per_sec_per_gpu: 347.44 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 34228.15, tpot_ms: 37.26, tokens_per_sec_per_gpu: 426.74 },
    ] },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "dflash2", nodes: "single" } },
];
