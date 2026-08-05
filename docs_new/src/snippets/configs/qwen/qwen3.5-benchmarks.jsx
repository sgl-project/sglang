// Measured speed + accuracy benchmarks for the Qwen3.5 cookbook.
// Speed RE-BENCHED with the corrected workload: low-latency @ conc 1 & 16, high-throughput
// @ conc 1024 & 4096 (was conc 1 & 100), --random-range-ratio 1.0 --warmup-requests 64
// --flush-cache. tokens_per_sec_per_gpu = output tok/s / (tp*nnodes) * (isl+osl)/osl, using
// the ACTUAL served tp (cells that omit --tp serve tp=1). Accuracy (P50) preserved from the
// prior fill. Cells without an entry render "pending" (in-progress re-bench, or non-NVIDIA
// hardware we can't re-measure: xeon/AMD — their old conc-100 speed was dropped).

export const benchmarks = [
  {
    match: { hw: "h100", variant: "0.8b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 69122, tpot_ms: 24.06, tokens_per_sec_per_gpu: 89684 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 373400, tpot_ms: 26.57, tokens_per_sec_per_gpu: 89045 },
    ],
  },
  {
    match: { hw: "h100", variant: "0.8b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 53, tpot_ms: 0.82, tokens_per_sec_per_gpu: 10322 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 525, tpot_ms: 1.52, tokens_per_sec_per_gpu: 70370 },
    ],
  },
  {
    match: { hw: "h100", variant: "2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 87296, tpot_ms: 27.24, tokens_per_sec_per_gpu: 71717 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 472778, tpot_ms: 29.33, tokens_per_sec_per_gpu: 71058 },
    ],
  },
  {
    match: { hw: "h100", variant: "2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 76, tpot_ms: 1.03, tokens_per_sec_per_gpu: 8149 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 530, tpot_ms: 2.05, tokens_per_sec_per_gpu: 53951 },
    ],
  },
  {
    match: { hw: "h100", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 287640, tpot_ms: 26.62, tokens_per_sec_per_gpu: 27675 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1307071, tpot_ms: 27.72, tokens_per_sec_per_gpu: 27507 },
    ],
  },
  {
    match: { hw: "h100", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 154, tpot_ms: 1.64, tokens_per_sec_per_gpu: 4632 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 650, tpot_ms: 4.31, tokens_per_sec_per_gpu: 26143 },
    ],
  },
  {
    match: { hw: "h100", variant: "9b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 403316, tpot_ms: 29.02, tokens_per_sec_per_gpu: 20379 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1771338, tpot_ms: 31.24, tokens_per_sec_per_gpu: 20284 },
    ],
  },
  {
    match: { hw: "h100", variant: "9b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 240, tpot_ms: 2.51, tokens_per_sec_per_gpu: 3187 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 704, tpot_ms: 6.4, tokens_per_sec_per_gpu: 18546 },
    ],
  },
  {
    match: { hw: "h100", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 446, tpot_ms: 4.76, tokens_per_sec_per_gpu: 919 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 894, tpot_ms: 10.8, tokens_per_sec_per_gpu: 5255 },
    ],
  },
  {
    match: { hw: "h100", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 521, tpot_ms: 4.9, tokens_per_sec_per_gpu: 1636 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 7448, tpot_ms: 10.29, tokens_per_sec_per_gpu: 7409 },
    ],
  },
  {
    match: { hw: "h100", variant: "35b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 136, tpot_ms: 1.66, tokens_per_sec_per_gpu: 2368 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 195, tpot_ms: 5.64, tokens_per_sec_per_gpu: 10515 },
    ],
  },
  {
    match: { hw: "h100", variant: "35b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 149, tpot_ms: 2.14, tokens_per_sec_per_gpu: 3948 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 183, tpot_ms: 6.5, tokens_per_sec_per_gpu: 18803 },
    ],
  },
  {
    match: { hw: "h100", variant: "122b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 210, tpot_ms: 2.12, tokens_per_sec_per_gpu: 923 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2698, tpot_ms: 6.82, tokens_per_sec_per_gpu: 3315 },
    ],
  },
  {
    match: { hw: "h200", variant: "0.8b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 31920, tpot_ms: 39.01, tokens_per_sec_per_gpu: 103327 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 268655, tpot_ms: 44.92, tokens_per_sec_per_gpu: 103300 },
    ],
  },
  {
    match: { hw: "h200", variant: "0.8b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 53, tpot_ms: 1.07, tokens_per_sec_per_gpu: 8003 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 454, tpot_ms: 1.47, tokens_per_sec_per_gpu: 74681 },
    ],
  },
  {
    match: { hw: "h200", variant: "2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 56520, tpot_ms: 50.01, tokens_per_sec_per_gpu: 82164 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 356073, tpot_ms: 52.59, tokens_per_sec_per_gpu: 82825 },
    ],
  },
  {
    match: { hw: "h200", variant: "2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 75, tpot_ms: 1.09, tokens_per_sec_per_gpu: 7700 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 496, tpot_ms: 1.73, tokens_per_sec_per_gpu: 61120 },
    ],
  },
  {
    match: { hw: "h200", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 214067, tpot_ms: 37.52, tokens_per_sec_per_gpu: 34910 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1001438, tpot_ms: 39.77, tokens_per_sec_per_gpu: 34740 },
    ],
  },
  {
    match: { hw: "h200", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 147, tpot_ms: 1.35, tokens_per_sec_per_gpu: 5433 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 688, tpot_ms: 3.55, tokens_per_sec_per_gpu: 30634 },
    ],
  },
  {
    match: { hw: "h200", variant: "9b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 290575, tpot_ms: 45.39, tokens_per_sec_per_gpu: 25256 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1391792, tpot_ms: 48.46, tokens_per_sec_per_gpu: 25199 },
    ],
  },
  {
    match: { hw: "h200", variant: "9b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 229, tpot_ms: 1.92, tokens_per_sec_per_gpu: 3891 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 900, tpot_ms: 5.13, tokens_per_sec_per_gpu: 21671 },
    ],
  },
  {
    match: { hw: "h200", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1400465, tpot_ms: 39.1, tokens_per_sec_per_gpu: 6380 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 5850046, tpot_ms: 39.13, tokens_per_sec_per_gpu: 6367 },
    ],
  },
  {
    match: { hw: "h200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 735, tpot_ms: 5.89, tokens_per_sec_per_gpu: 1430 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 807, tpot_ms: 15.27, tokens_per_sec_per_gpu: 7185 },
    ],
  },
  {
    match: { hw: "h200", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 915087, tpot_ms: 35.15, tokens_per_sec_per_gpu: 9712 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 3832689, tpot_ms: 35.34, tokens_per_sec_per_gpu: 9701 },
    ],
  },
  {
    match: { hw: "h200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 520, tpot_ms: 4.61, tokens_per_sec_per_gpu: 1778 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1051, tpot_ms: 12.16, tokens_per_sec_per_gpu: 9438 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 453328, tpot_ms: 27.62, tokens_per_sec_per_gpu: 18996 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1943739, tpot_ms: 27.76, tokens_per_sec_per_gpu: 18906 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 182, tpot_ms: 2.16, tokens_per_sec_per_gpu: 3974 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 222, tpot_ms: 7.29, tokens_per_sec_per_gpu: 16706 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 195793, tpot_ms: 42.93, tokens_per_sec_per_gpu: 35770 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 981448, tpot_ms: 44.79, tokens_per_sec_per_gpu: 35481 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 153, tpot_ms: 2.13, tokens_per_sec_per_gpu: 4035 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 188, tpot_ms: 5.78, tokens_per_sec_per_gpu: 20771 },
    ],
  },
  {
    match: { hw: "h200", variant: "122b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 295006, tpot_ms: 39.51, tokens_per_sec_per_gpu: 6352 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1420693, tpot_ms: 40.0, tokens_per_sec_per_gpu: 6354 },
    ],
  },
  {
    match: { hw: "h200", variant: "122b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 193, tpot_ms: 1.91, tokens_per_sec_per_gpu: 1033 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 740, tpot_ms: 6.5, tokens_per_sec_per_gpu: 4586 },
    ],
  },
  {
    match: { hw: "h200", variant: "122b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 350514, tpot_ms: 41.81, tokens_per_sec_per_gpu: 11673 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1538417, tpot_ms: 41.88, tokens_per_sec_per_gpu: 11630 },
    ],
  },
  {
    match: { hw: "h200", variant: "122b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 227, tpot_ms: 2.2, tokens_per_sec_per_gpu: 1706 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1191, tpot_ms: 6.86, tokens_per_sec_per_gpu: 7885 },
    ],
  },
  {
    match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 540099, tpot_ms: 33.5, tokens_per_sec_per_gpu: 1975 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2287294, tpot_ms: 34.78, tokens_per_sec_per_gpu: 1975 },
    ],
  },
  {
    match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    accuracy: { gsm8k_pct: 97.5, mmmu_pct: 97.8 },
    notes: "GSM8K via benchmark/gsm8k/bench_sglang.py (200 questions); MMMU via benchmark/mmmu/bench_sglang.py (91-sample val subset).",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 220, tpot_ms: 2.24, tokens_per_sec_per_gpu: 439 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 452, tpot_ms: 8.31, tokens_per_sec_per_gpu: 1764 },
    ],
  },
  {
    match: { hw: "h200", variant: "397b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 270545, tpot_ms: 62.74, tokens_per_sec_per_gpu: 3340 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1274582, tpot_ms: 72.74, tokens_per_sec_per_gpu: 3322 },
    ],
  },
  {
    match: { hw: "h200", variant: "397b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 219, tpot_ms: 2.73, tokens_per_sec_per_gpu: 374 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 452, tpot_ms: 7.35, tokens_per_sec_per_gpu: 2003 },
    ],
  },
  {
    match: { hw: "b200", variant: "0.8b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 8825, tpot_ms: 32.4, tokens_per_sec_per_gpu: 164828 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 166434, tpot_ms: 36.17, tokens_per_sec_per_gpu: 165961 },
    ],
  },
  {
    match: { hw: "b200", variant: "0.8b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 40, tpot_ms: 0.56, tokens_per_sec_per_gpu: 14875 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 321, tpot_ms: 0.97, tokens_per_sec_per_gpu: 109107 },
    ],
  },
  {
    match: { hw: "b200", variant: "2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 15323, tpot_ms: 38.16, tokens_per_sec_per_gpu: 131259 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 186143, tpot_ms: 45.68, tokens_per_sec_per_gpu: 137597 },
    ],
  },
  {
    match: { hw: "b200", variant: "2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 48, tpot_ms: 0.73, tokens_per_sec_per_gpu: 11452 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 342, tpot_ms: 1.25, tokens_per_sec_per_gpu: 87543 },
    ],
  },
  {
    match: { hw: "b200", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 121155, tpot_ms: 29.49, tokens_per_sec_per_gpu: 59014 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 588650, tpot_ms: 29.21, tokens_per_sec_per_gpu: 59379 },
    ],
  },
  {
    match: { hw: "b200", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 88, tpot_ms: 1.06, tokens_per_sec_per_gpu: 7306 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 344, tpot_ms: 2.45, tokens_per_sec_per_gpu: 45972 },
    ],
  },
  {
    match: { hw: "b200", variant: "9b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 156800, tpot_ms: 33.34, tokens_per_sec_per_gpu: 45150 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 775556, tpot_ms: 34.1, tokens_per_sec_per_gpu: 45132 },
    ],
  },
  {
    match: { hw: "b200", variant: "9b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 128, tpot_ms: 1.5, tokens_per_sec_per_gpu: 4971 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 581, tpot_ms: 3.24, tokens_per_sec_per_gpu: 34333 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 714034, tpot_ms: 30.91, tokens_per_sec_per_gpu: 12335 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 3007462, tpot_ms: 31.02, tokens_per_sec_per_gpu: 12334 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 404, tpot_ms: 3.51, tokens_per_sec_per_gpu: 2077 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 425, tpot_ms: 9.67, tokens_per_sec_per_gpu: 11772 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 525882, tpot_ms: 28.91, tokens_per_sec_per_gpu: 16623 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2219201, tpot_ms: 28.92, tokens_per_sec_per_gpu: 16668 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 293, tpot_ms: 3.09, tokens_per_sec_per_gpu: 2628 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 553, tpot_ms: 7.71, tokens_per_sec_per_gpu: 15276 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 200042, tpot_ms: 22.08, tokens_per_sec_per_gpu: 39275 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 917826, tpot_ms: 22.14, tokens_per_sec_per_gpu: 39527 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 98, tpot_ms: 1.61, tokens_per_sec_per_gpu: 5393 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 114, tpot_ms: 4.72, tokens_per_sec_per_gpu: 26318 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 163938, tpot_ms: 24.49, tokens_per_sec_per_gpu: 47846 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 746740, tpot_ms: 24.44, tokens_per_sec_per_gpu: 48093 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 104, tpot_ms: 1.66, tokens_per_sec_per_gpu: 5346 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 109, tpot_ms: 4.09, tokens_per_sec_per_gpu: 29762 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 558311, tpot_ms: 16.33, tokens_per_sec_per_gpu: 8116 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2298264, tpot_ms: 16.36, tokens_per_sec_per_gpu: 8110 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 150, tpot_ms: 1.84, tokens_per_sec_per_gpu: 2215 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 568, tpot_ms: 6.21, tokens_per_sec_per_gpu: 9722 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1019354, tpot_ms: 12.92, tokens_per_sec_per_gpu: 9060 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 4145598, tpot_ms: 12.93, tokens_per_sec_per_gpu: 9056 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 231, tpot_ms: 2.24, tokens_per_sec_per_gpu: 3656 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 7687, tpot_ms: 5.19, tokens_per_sec_per_gpu: 10712 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 219990, tpot_ms: 23.23, tokens_per_sec_per_gpu: 9339 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 969702, tpot_ms: 23.22, tokens_per_sec_per_gpu: 9363 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 142, tpot_ms: 1.81, tokens_per_sec_per_gpu: 1116 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 316, tpot_ms: 4.91, tokens_per_sec_per_gpu: 6166 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 327151, tpot_ms: 27.35, tokens_per_sec_per_gpu: 6371 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1439502, tpot_ms: 27.6, tokens_per_sec_per_gpu: 6349 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 174, tpot_ms: 2.04, tokens_per_sec_per_gpu: 1005 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 332, tpot_ms: 6.3, tokens_per_sec_per_gpu: 4826 },
    ],
  },
  {
    match: { hw: "b300", variant: "0.8b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 10030, tpot_ms: 45.63, tokens_per_sec_per_gpu: 134890 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 137605, tpot_ms: 57.2, tokens_per_sec_per_gpu: 153486 },
    ],
  },
  {
    match: { hw: "b300", variant: "0.8b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 42, tpot_ms: 0.7, tokens_per_sec_per_gpu: 12088 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 303, tpot_ms: 1.04, tokens_per_sec_per_gpu: 104609 },
    ],
  },
  {
    match: { hw: "b300", variant: "2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 17043, tpot_ms: 60.15, tokens_per_sec_per_gpu: 116740 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 145386, tpot_ms: 76.15, tokens_per_sec_per_gpu: 135474 },
    ],
  },
  {
    match: { hw: "b300", variant: "2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 53, tpot_ms: 0.72, tokens_per_sec_per_gpu: 11634 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 327, tpot_ms: 1.24, tokens_per_sec_per_gpu: 87590 },
    ],
  },
  {
    match: { hw: "b300", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 81967, tpot_ms: 43.91, tokens_per_sec_per_gpu: 61188 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 538374, tpot_ms: 45.23, tokens_per_sec_per_gpu: 61316 },
    ],
  },
  {
    match: { hw: "b300", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 90, tpot_ms: 1.06, tokens_per_sec_per_gpu: 7213 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 378, tpot_ms: 2.42, tokens_per_sec_per_gpu: 46068 },
    ],
  },
  {
    match: { hw: "b300", variant: "9b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 113592, tpot_ms: 50.2, tokens_per_sec_per_gpu: 47147 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 692689, tpot_ms: 51.72, tokens_per_sec_per_gpu: 47319 },
    ],
  },
  {
    match: { hw: "b300", variant: "9b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 127, tpot_ms: 1.48, tokens_per_sec_per_gpu: 5239 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 680, tpot_ms: 3.14, tokens_per_sec_per_gpu: 34577 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 584145, tpot_ms: 46.76, tokens_per_sec_per_gpu: 14053 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2590354, tpot_ms: 46.99, tokens_per_sec_per_gpu: 14086 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 397, tpot_ms: 3.49, tokens_per_sec_per_gpu: 2088 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 417, tpot_ms: 9.54, tokens_per_sec_per_gpu: 11934 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 445654, tpot_ms: 42.79, tokens_per_sec_per_gpu: 18131 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2006619, tpot_ms: 42.71, tokens_per_sec_per_gpu: 18213 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 291, tpot_ms: 3.07, tokens_per_sec_per_gpu: 2642 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 421, tpot_ms: 7.46, tokens_per_sec_per_gpu: 15372 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 141478, tpot_ms: 33.38, tokens_per_sec_per_gpu: 48837 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 715475, tpot_ms: 33.89, tokens_per_sec_per_gpu: 48448 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 100, tpot_ms: 1.63, tokens_per_sec_per_gpu: 5372 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 188, tpot_ms: 4.69, tokens_per_sec_per_gpu: 26394 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 113104, tpot_ms: 34.69, tokens_per_sec_per_gpu: 54011 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 633449, tpot_ms: 35.52, tokens_per_sec_per_gpu: 54051 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 104, tpot_ms: 1.62, tokens_per_sec_per_gpu: 5438 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 141, tpot_ms: 4.05, tokens_per_sec_per_gpu: 29812 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 262945, tpot_ms: 33.21, tokens_per_sec_per_gpu: 15032 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1202204, tpot_ms: 33.55, tokens_per_sec_per_gpu: 15077 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 142, tpot_ms: 1.83, tokens_per_sec_per_gpu: 2230 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 568, tpot_ms: 6.07, tokens_per_sec_per_gpu: 10041 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 547822, tpot_ms: 28.65, tokens_per_sec_per_gpu: 15961 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2323042, tpot_ms: 28.72, tokens_per_sec_per_gpu: 15956 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 224, tpot_ms: 2.22, tokens_per_sec_per_gpu: 3694 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1021, tpot_ms: 7.7, tokens_per_sec_per_gpu: 14780 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 168, tpot_ms: 2.1, tokens_per_sec_per_gpu: 982 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 304, tpot_ms: 7.65, tokens_per_sec_per_gpu: 4001 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 332198, tpot_ms: 30.81, tokens_per_sec_per_gpu: 12096 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1494254, tpot_ms: 30.84, tokens_per_sec_per_gpu: 12150 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 175, tpot_ms: 2.26, tokens_per_sec_per_gpu: 1853 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 198, tpot_ms: 7.02, tokens_per_sec_per_gpu: 8805 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 888809, tpot_ms: 19.61, tokens_per_sec_per_gpu: 5138 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 3628300, tpot_ms: 19.67, tokens_per_sec_per_gpu: 5136 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 232, tpot_ms: 2.5, tokens_per_sec_per_gpu: 1623 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2874, tpot_ms: 8.04, tokens_per_sec_per_gpu: 5672 },
    ],
  },
];
