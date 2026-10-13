// Speed: `python3 -m sglang.bench_serving --flush-cache` against the exact cell
// command on one GPU, PR #41993 head on top of nightly main, so sglang_version
// names the PR rather than a release. Base rows use random 8192/1024; DSPARK
// rows use sharegpt, because random tokens collapse draft acceptance.
// tokens_per_sec_per_gpu is bench_serving's total (input+output) throughput;
// sharegpt has no fixed in/out ratio, so it is read from the logs, not derived.
//
// GSM8K (GB300 only): sgl-eval, full 1,319 questions, temperature 1.0 /
// top-p 0.95. The accuracy runs predate the `auto` parser flags. Every recipe
// on GB300 and H200 passed the image probes (16x / 4x / 16x-again /
// multi-image / max_slice_nums=1: 355 / 1291 / 355 / 556 / 93 prompt tokens,
// finish_reason=stop).

export const benchmarks = [
  {
    match: { hw: "gb300", variant: "1b", quant: "bf16", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 84.08 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 69.25, tpot_ms: 1.85, tokens_per_sec_per_gpu: 4703 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 470.91, tpot_ms: 2.24, tokens_per_sec_per_gpu: 53270 },
    ],
    notes: "1×GB300, TP=1. Speed: bench_serving random 8192/1024, --flush-cache. GSM8K stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "1b", quant: "bf16", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 83.24 },
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 56.8, tpot_ms: 2.02, tokens_per_sec_per_gpu: 1003 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 85.2, tpot_ms: 5.73, tokens_per_sec_per_gpu: 6745 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-1B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K stop rate 99.92% (one of 1,319 requests did not emit EOS at temperature 1.0); accept length about 2.6.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "bf16", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.36 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 153.3, tpot_ms: 4.36, tokens_per_sec_per_gpu: 1996 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1242.16, tpot_ms: 9.01, tokens_per_sec_per_gpu: 14094 },
    ],
    notes: "1×GB300, TP=1. Speed: bench_serving random 8192/1024, --flush-cache. GSM8K (measured at TP=2) stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "bf16", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 136.35, tpot_ms: 2.8, tokens_per_sec_per_gpu: 633 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 199.66, tpot_ms: 13.69, tokens_per_sec_per_gpu: 2901 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K (measured at TP=2) stop rate 100%; accept length about 4.1.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "fp8", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 137.07, tpot_ms: 4.44, tokens_per_sec_per_gpu: 1967 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1057.92, tpot_ms: 7.94, tokens_per_sec_per_gpu: 16059 },
    ],
    notes: "1×GB300, TP=1. Speed: bench_serving random 8192/1024, --flush-cache. GSM8K (measured at TP=2) stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "fp8", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 131.4, tpot_ms: 2.76, tokens_per_sec_per_gpu: 638 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 210.89, tpot_ms: 12.6, tokens_per_sec_per_gpu: 3120 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K (measured at TP=2) stop rate 100%; accept length about 4.0.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "int4", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.97 },
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 683.09, tpot_ms: 8.56, tokens_per_sec_per_gpu: 976 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 6068.66, tpot_ms: 51.34, tokens_per_sec_per_gpu: 2516 },
    ],
    notes: "1×GB300, TP=1. Speed: bench_serving random 8192/1024, --flush-cache. GSM8K stop rate 100%.",
  },
  {
    match: { hw: "gb300", variant: "35b-a3b", quant: "int4", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    accuracy: { gsm8k_pct: 96.44 },
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 155.88, tpot_ms: 11.05, tokens_per_sec_per_gpu: 183 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 425.46, tpot_ms: 54.8, tokens_per_sec_per_gpu: 722 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K stop rate 100%; accept length about 3.8.",
  },
  {
    match: { hw: "h200", variant: "1b", quant: "bf16", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 81.49, tpot_ms: 1.55, tokens_per_sec_per_gpu: 5523 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 438.99, tpot_ms: 2.14, tokens_per_sec_per_gpu: 55898 },
    ],
    notes: "1×H200, TP=1. Speed: bench_serving random 8192/1024, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "1b", quant: "bf16", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 51.55, tpot_ms: 1.65, tokens_per_sec_per_gpu: 1189 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 228.59, tpot_ms: 6.27, tokens_per_sec_per_gpu: 5778 },
    ],
    notes: "1×H200, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-1B-DSpark. Speed: bench_serving sharegpt, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "bf16", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 208.83, tpot_ms: 3.69, tokens_per_sec_per_gpu: 2311 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1347.42, tpot_ms: 10.59, tokens_per_sec_per_gpu: 12103 },
    ],
    notes: "1×H200, TP=1. Speed: bench_serving random 8192/1024, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "bf16", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 155.49, tpot_ms: 3.02, tokens_per_sec_per_gpu: 593 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 216.21, tpot_ms: 11.92, tokens_per_sec_per_gpu: 3111 },
    ],
    notes: "1×H200, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "fp8", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 183.85, tpot_ms: 3.55, tokens_per_sec_per_gpu: 2413 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1080.17, tpot_ms: 8.37, tokens_per_sec_per_gpu: 15280 },
    ],
    notes: "1×H200, TP=1. Speed: bench_serving random 8192/1024, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "fp8", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 129.71, tpot_ms: 2.61, tokens_per_sec_per_gpu: 694 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 178.4, tpot_ms: 12.4, tokens_per_sec_per_gpu: 3196 },
    ],
    notes: "1×H200, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "int4", nodes: "single", spec: "none" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 228.26, tpot_ms: 3.47, tokens_per_sec_per_gpu: 2439 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1519.9, tpot_ms: 7.63, tokens_per_sec_per_gpu: 15808 },
    ],
    notes: "1×H200, TP=1. Speed: bench_serving random 8192/1024, --flush-cache.",
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "int4", nodes: "single", spec: "dspark" },
    sglang_version: "PR #41993",
    speed: [
      { workload: { dataset: "sharegpt", max_concurrency: 1 },
        ttft_ms: 92.01, tpot_ms: 2.38, tokens_per_sec_per_gpu: 781 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 177.82, tpot_ms: 11.2, tokens_per_sec_per_gpu: 3457 },
    ],
    notes: "1×H200, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache.",
  },
  { match: { hw: "h100", variant: "1b",      quant: "bf16", nodes: "single" } },
  { match: { hw: "h100", variant: "35b-a3b", quant: "fp8",  nodes: "single" } },
  { match: { hw: "h100", variant: "35b-a3b", quant: "int4", nodes: "single" } },
];
