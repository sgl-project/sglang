// Measured speed + accuracy benchmarks for the Qwen3.5 cookbook.
// Speed RE-BENCHED (cache-cold run1) with the corrected workload: low-latency @ conc
// 1 & 16, high-throughput @ conc 1024 (conc 4096 dropped — past saturation);
// --random-range-ratio 1.0 --warmup-requests 64 --flush-cache. TTFT/TPOT are P50
// (median). tokens_per_sec_per_gpu = output tok/s / (tp*nnodes) * (isl+osl)/osl, using
// the ACTUAL served tp. Accuracy preserved from the prior fill (build-robust).
// NOTE: four high-throughput cells are left PENDING because their throughput sits at or
// below the low-latency peak — b200/122b (bf16 & fp8), b300/397b (fp8), h200/27b (bf16).
// These memory-bound decodes saturate by ~conc 16-256, so the conc-1024 HT point only
// queues (huge TTFT) with no throughput gain; the throughput-optimal point is nearer the
// low-latency concurrency. (Cells with no entry render pending: xeon/AMD have no box.)

export const benchmarks = [
  {
    match: { hw: "h100", variant: "0.8b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 69122, tpot_ms: 24.06, tokens_per_sec_per_gpu: 89684 },
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
    ],
  },
  {
    match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    accuracy: { gsm8k_pct: 97.5, mmmu_pct: 97.8 },
    notes: "Accuracy (GSM8K/MMMU) is from the prior fill on an unpinned main-branch build, NOT 0.5.16; only the speed rows are 0.5.16.",
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
    match: { hw: "b300", variant: "397b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 232, tpot_ms: 2.5, tokens_per_sec_per_gpu: 1623 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2874, tpot_ms: 8.04, tokens_per_sec_per_gpu: 5672 },
    ],
  },

  // MI300X Qwen3.5-4B (BF16, TP=1) — measured on 8×MI300X (single GPU used).
  // Server: sglang v0.5.16, --attention-backend aiter, SGLANG_USE_AITER=1.
  // Low-latency cell uses EAGLE speculative decoding (built-in head).
  // Workload: random ISL=8192, OSL=1024.
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    accuracy: { gsm8k_pct: 78.85 },
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 138,
        tpot_ms: 3.87,
        tokens_per_sec_per_gpu: 243,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 597,
        tpot_ms: 7.03,
        tokens_per_sec_per_gpu: 1622,
      },
    ],
  },
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 315,
        tpot_ms: 23.93,
        tokens_per_sec_per_gpu: 2584,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 256 },
        ttft_ms: 4554,
        tpot_ms: 84.91,
        tokens_per_sec_per_gpu: 2978,
      },
    ],
  },
  {
    match: { hw: "mi300x", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 109282,
        tpot_ms: 86.18,
        tokens_per_sec_per_gpu: 2894,
      },
      {
        workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 361004,
        tpot_ms: 82.25,
        tokens_per_sec_per_gpu: 2742,
      },
    ],
  },
];
