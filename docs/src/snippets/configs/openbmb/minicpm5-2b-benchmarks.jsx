// One entry per cell `match` tuple (same keys as the config cells).
//
// Only the RTX 5090 row carries numbers. They were measured on a single
// RTX 5090 32GB (`zijiexia-5090-1gpu`, `lmsysorg/sglang:dev` @ 30705c004c) with
// `python3 -m sglang.bench_serving`, random 1024/1024, against the exact cell
// command on this page — including `--mem-fraction-static 0.75
// --cuda-graph-max-bs 128`, which is what keeps concurrency 64 and 128
// graph-backed (see the model page's Configuration Tips).
//
// `tokens_per_sec_per_gpu` is total (input+output) per GPU = the measured
// output throughput x 2 at 1024/1024 on one GPU.
//
// Two operating points only: concurrency 1 (single-user latency) and 128 (the
// saturated end of the sweep). The intermediate 8 / 32 / 64 rows were measured
// but are not published here.
//
// The source runs did not restate a percentile, so the values are recorded as
// Mean (bench_serving's headline lines) rather than claimed as P50 — same
// convention as the LFM2.5 page. TTFT was not captured at concurrency 128, so
// that cell carries `null` rather than a back-filled guess.
//
// No accuracy rows: the only GSM8K numbers taken so far came from a custom
// chat-template harness, not the `sgl-eval run gsm8k` command the Reproduce
// modal would show, and the two are not comparable.

export const benchmarks = [
  {
    match: { hw: "rtx5090", variant: "default", quant: "bf16", nodes: "single" },
    sglang_version: "dev @ 30705c004c",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 34, tpot_ms: 4.0, tokens_per_sec_per_gpu: 496 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 128 },
        ttft_ms: null, tpot_ms: 11.8, tokens_per_sec_per_gpu: 19280 },
    ],
  },
  {
    // Single DGX Spark (GB10), same image and commit as the RTX 5090 row.
    // TTFT is Mean; at concurrency 64 the Median is 246 ms, because the first
    // wave of requests all queue behind one prefill.
    match: { hw: "dgx-spark", variant: "default", quant: "bf16", nodes: "single" },
    sglang_version: "dev @ 30705c004c",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1 },
        ttft_ms: 85, tpot_ms: 27.7, tokens_per_sec_per_gpu: 72 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64 },
        ttft_ms: 1373, tpot_ms: 42.9, tokens_per_sec_per_gpu: 2892 },
    ],
  },
  // Pending — no numbers taken on these platforms yet.
  { match: { hw: "h200",    variant: "default", quant: "bf16", nodes: "single" } },
  { match: { hw: "rtx6000", variant: "default", quant: "bf16", nodes: "single" } },
];
