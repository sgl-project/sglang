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
    sglang_version: "0.5.16",
    accuracy: { mmlu: 0.720, mmmu: 0.307 },
    notes: "H200, base command (no MTP/parsers). MMLU overall (Humanities/Social Sciences/STEM/Other); MMMU overall on the 900-sample val split.",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 197298, tpot_ms: 63.07, tokens_per_sec_per_gpu: 33775 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1025363, tpot_ms: 68.28, tokens_per_sec_per_gpu: 32720 },
    ],
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
    match: { hw: "b200", variant: "31b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1358488, tpot_ms: 19.89, tokens_per_sec_per_gpu: 6394 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 5496429, tpot_ms: 19.73, tokens_per_sec_per_gpu: 6386 },
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
    match: { hw: "b300", variant: "e2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 114, tpot_ms: 2.94, tokens_per_sec_per_gpu: 2890 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 119, tpot_ms: 4.29, tokens_per_sec_per_gpu: 27799 },
    ],
  },
  {
    match: { hw: "b300", variant: "e2b", quant: "qat", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 65459, tpot_ms: 88.43, tokens_per_sec_per_gpu: 48051 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 658565, tpot_ms: 106.9, tokens_per_sec_per_gpu: 45530 },
    ],
  },
  {
    match: { hw: "b300", variant: "e2b", quant: "qat", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 113, tpot_ms: 2.94, tokens_per_sec_per_gpu: 2886 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 117, tpot_ms: 4.31, tokens_per_sec_per_gpu: 24637 },
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
  {
    match: { hw: "b300", variant: "e4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 146, tpot_ms: 3.04, tokens_per_sec_per_gpu: 2762 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 272, tpot_ms: 4.73, tokens_per_sec_per_gpu: 23768 },
    ],
  },
  {
    match: { hw: "b300", variant: "e4b", quant: "qat", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 202798, tpot_ms: 61.24, tokens_per_sec_per_gpu: 32308 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1080602, tpot_ms: 66.63, tokens_per_sec_per_gpu: 31268 },
    ],
  },
  {
    match: { hw: "b300", variant: "e4b", quant: "qat", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 146, tpot_ms: 3.14, tokens_per_sec_per_gpu: 2620 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 273, tpot_ms: 4.77, tokens_per_sec_per_gpu: 22137 },
    ],
  },
  {
    match: { hw: "b300", variant: "12b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 516652, tpot_ms: 39.87, tokens_per_sec_per_gpu: 15814 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2141638, tpot_ms: 45.55, tokens_per_sec_per_gpu: 16213 },
    ],
  },
  {
    match: { hw: "b300", variant: "12b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 325, tpot_ms: 4.51, tokens_per_sec_per_gpu: 1178 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 339, tpot_ms: 12.69, tokens_per_sec_per_gpu: 7325 },
    ],
  },
  {
    match: { hw: "b300", variant: "26b-a4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "rx: read: read tcp 192.168.1.66:63259->52.42.240.243:443: read: connection reset by peer",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 397855, tpot_ms: 41.32, tokens_per_sec_per_gpu: 19830 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1673928, tpot_ms: 46.52, tokens_per_sec_per_gpu: 20762 },
    ],
  },
  {
    match: { hw: "b300", variant: "26b-a4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 137, tpot_ms: 3.06, tokens_per_sec_per_gpu: 1222 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 142, tpot_ms: 4.91, tokens_per_sec_per_gpu: 11883 },
    ],
  },
  {
    match: { hw: "b300", variant: "26b-a4b", quant: "qat", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 398903, tpot_ms: 41.42, tokens_per_sec_per_gpu: 19769 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1678558, tpot_ms: 46.65, tokens_per_sec_per_gpu: 20704 },
    ],
  },
  {
    match: { hw: "b300", variant: "26b-a4b", quant: "qat", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 138, tpot_ms: 3.02, tokens_per_sec_per_gpu: 1237 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 165, tpot_ms: 5.06, tokens_per_sec_per_gpu: 11636 },
    ],
  },
  {
    match: { hw: "b300", variant: "31b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1815517, tpot_ms: 41.63, tokens_per_sec_per_gpu: 4784 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 7238899, tpot_ms: 44.89, tokens_per_sec_per_gpu: 4949 },
    ],
  },
  {
    match: { hw: "b300", variant: "31b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 802, tpot_ms: 5.64, tokens_per_sec_per_gpu: 1415 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 5258, tpot_ms: 22.74, tokens_per_sec_per_gpu: 4924 },
    ],
  },
  {
    match: { hw: "b300", variant: "31b", quant: "qat", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1812272, tpot_ms: 41.49, tokens_per_sec_per_gpu: 4792 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 7213738, tpot_ms: 45.0, tokens_per_sec_per_gpu: 4962 },
    ],
  },
  {
    match: { hw: "b300", variant: "31b", quant: "qat", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 799, tpot_ms: 5.64, tokens_per_sec_per_gpu: 1339 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3310, tpot_ms: 23.41, tokens_per_sec_per_gpu: 4969 },
    ],
  },
];
