// Measured benchmarks for the Gemma-4 cookbook.
// ACCURACY: from the legacy page's measured blocks (H200 bf16) — MMLU + MMMU-val
// (+ GSM8K chat-template for 12B); kept per skill rule 2 (accuracy is build-robust).
// SPEED: fresh cache-cold re-bench on the pinned release recorded per cell
// (sglang_version), corrected workload: low-latency @ conc 1 & 16, high-throughput
// @ conc 1024 & 4096, --random-range-ratio 1.0 --warmup-requests 64 --flush-cache.
// tokens_per_sec_per_gpu = output tok/s / tp * (isl+osl)/osl (actual served tp).
// Cells without a speed row are re-bench-in-progress; mi300x has no box.

export const benchmarks = [
  {
    match: { hw: "h200", variant: "e2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.720, mmmu: 0.307 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
  {
    match: { hw: "h200", variant: "e4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    accuracy: { mmlu: 0.810, mmmu: 0.396 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 365323, tpot_ms: 41.38, tokens_per_sec_per_gpu: 21623 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1599142, tpot_ms: 46.98, tokens_per_sec_per_gpu: 21569 },
    ],
  },
  {
    match: { hw: "h200", variant: "12b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.859, mmmu: 0.683, gsm8k: 0.950 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split. GSM8K 0.950 via the chat-template run_eval harness on the 1319-question test set (the raw few-shot harness under-elicits this reasoning-oriented variant).",
  },
  {
    match: { hw: "h200", variant: "26b-a4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.891, mmmu: 0.549 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
  {
    match: { hw: "h200", variant: "31b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { mmlu: 0.896, mmmu: 0.589 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
  },
  {
    match: { hw: "b200", variant: "e2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 40094, tpot_ms: 30.43, tokens_per_sec_per_gpu: 122973 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 247521, tpot_ms: 33.14, tokens_per_sec_per_gpu: 121481 },
    ],
  },
  {
    match: { hw: "b200", variant: "e4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 115508, tpot_ms: 20.7, tokens_per_sec_per_gpu: 64440 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 529945, tpot_ms: 22.15, tokens_per_sec_per_gpu: 64570 },
    ],
  },
  {
    match: { hw: "b200", variant: "12b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 316010, tpot_ms: 15.86, tokens_per_sec_per_gpu: 26556 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1340393, tpot_ms: 17.32, tokens_per_sec_per_gpu: 26498 },
    ],
  },
  {
    match: { hw: "b300", variant: "e2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 65736, tpot_ms: 92.46, tokens_per_sec_per_gpu: 47510 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 672088, tpot_ms: 120.11, tokens_per_sec_per_gpu: 44315 },
    ],
  },
  {
    match: { hw: "b300", variant: "e4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 202738, tpot_ms: 61.37, tokens_per_sec_per_gpu: 32300 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1082482, tpot_ms: 66.91, tokens_per_sec_per_gpu: 31208 },
    ],
  },
];
