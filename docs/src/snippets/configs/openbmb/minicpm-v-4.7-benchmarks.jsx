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
        ttft_ms: 57.97, tpot_ms: 1.92, tokens_per_sec_per_gpu: 1020 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 248.32, tpot_ms: 7.48, tokens_per_sec_per_gpu: 5452 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-1B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K stop rate 99.92% (one of 1,319 requests did not emit EOS at temperature 1.0); accept length about 2.5.",
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
        ttft_ms: 198.91, tpot_ms: 2.56, tokens_per_sec_per_gpu: 645 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 842.83, tpot_ms: 16.38, tokens_per_sec_per_gpu: 2267 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K (measured at TP=2) stop rate 100%; accept length about 3.3.",
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
        ttft_ms: 131.54, tpot_ms: 2.46, tokens_per_sec_per_gpu: 718 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 458.23, tpot_ms: 14.94, tokens_per_sec_per_gpu: 2546 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K (measured at TP=2) stop rate 100%; accept length about 3.3.",
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
        ttft_ms: 160.53, tpot_ms: 10.25, tokens_per_sec_per_gpu: 197 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 643.95, tpot_ms: 51.04, tokens_per_sec_per_gpu: 747 },
    ],
    notes: "1×GB300, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache. GSM8K stop rate 100%; accept length about 3.3.",
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
        ttft_ms: 44.09, tpot_ms: 1.51, tokens_per_sec_per_gpu: 1357 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 233.95, tpot_ms: 6.63, tokens_per_sec_per_gpu: 6240 },
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
        ttft_ms: 99.15, tpot_ms: 2.71, tokens_per_sec_per_gpu: 699 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 230.7, tpot_ms: 11.33, tokens_per_sec_per_gpu: 3314 },
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
        ttft_ms: 97.82, tpot_ms: 2.38, tokens_per_sec_per_gpu: 783 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 170.36, tpot_ms: 11.12, tokens_per_sec_per_gpu: 3506 },
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
        ttft_ms: 90.02, tpot_ms: 2.15, tokens_per_sec_per_gpu: 867 },
      { workload: { dataset: "sharegpt", max_concurrency: 16 },
        ttft_ms: 169.74, tpot_ms: 10.92, tokens_per_sec_per_gpu: 3601 },
    ],
    notes: "1×H200, TP=1, DSPARK with openbmb/MiniCPM-V-4.7-35B-A3B-DSpark. Speed: bench_serving sharegpt, --flush-cache.",
  },
  { match: { hw: "h100", variant: "1b",      quant: "bf16", nodes: "single" } },
  { match: { hw: "h100", variant: "35b-a3b", quant: "fp8",  nodes: "single" } },
  { match: { hw: "h100", variant: "35b-a3b", quant: "int4", nodes: "single" } },
];
