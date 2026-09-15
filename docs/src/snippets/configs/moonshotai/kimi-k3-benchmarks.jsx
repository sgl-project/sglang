export const benchmarks = [
  { match: { hw: "b300",  pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "b300",  pdMode: "unified", strategy: "low-latency" } },
  { match: { hw: "b300",  pdMode: "unified", strategy: "high-throughput"    } },
  { match: { hw: "b200",  pdMode: "unified", strategy: "low-latency" } },
  { match: { hw: "b200",  pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "b200",  pdMode: "unified", strategy: "high-throughput"    } },
  { match: { hw: "mi350x", pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "mi355x", pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "h100",  pdMode: "unified", strategy: "low-latency" } },
  { match: { hw: "h100",  pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "h100",  pdMode: "unified", strategy: "high-throughput"    } },
  { match: { hw: "h200",  pdMode: "unified", strategy: "low-latency" } },
  { match: { hw: "h200",  pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "h200",  pdMode: "unified", strategy: "high-throughput"    } },
  { match: { hw: "gb300", pdMode: "unified", strategy: "low-latency" } },
  { match: { hw: "gb300", pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "gb300", pdMode: "unified", strategy: "high-throughput"    } },
  { match: { hw: "gb200", pdMode: "unified", strategy: "low-latency" } },
  { match: { hw: "gb200", pdMode: "unified", strategy: "balanced"    } },
  { match: { hw: "gb200", pdMode: "unified", strategy: "high-throughput"    } },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "low-latency", quant: "mxfp4", spec: "none" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 378, tpot_ms: 8.51, tokens_per_sec_per_gpu: 127 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3539, tpot_ms: 19.47, tokens_per_sec_per_gpu: 785 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "low-latency", quant: "mxfp4", spec: "dspark" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 389, tpot_ms: 2.84, tokens_per_sec_per_gpu: 351 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3942, tpot_ms: 9.88, tokens_per_sec_per_gpu: 1319 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "low-latency", quant: "nvfp4", spec: "none" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 369, tpot_ms: 10.12, tokens_per_sec_per_gpu: 107 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3387, tpot_ms: 20.94, tokens_per_sec_per_gpu: 742 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "low-latency", quant: "nvfp4", spec: "dspark" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 380, tpot_ms: 3.24, tokens_per_sec_per_gpu: 313 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 3765, tpot_ms: 9.77, tokens_per_sec_per_gpu: 1345 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "balanced", quant: "mxfp4", spec: "none" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 11635, tpot_ms: 40.19, tokens_per_sec_per_gpu: 1395 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "balanced", quant: "mxfp4", spec: "dspark" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 12038, tpot_ms: 24.47, tokens_per_sec_per_gpu: 1987 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "balanced", quant: "nvfp4", spec: "none" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 11063, tpot_ms: 41.55, tokens_per_sec_per_gpu: 1373 },
    ],
  },
  {
    match: { hw: "b300", pdMode: "unified", strategy: "balanced", quant: "nvfp4", spec: "dspark" },
    sglang_version: "v0.5.18 @ 71de97b2",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 11664, tpot_ms: 22.17, tokens_per_sec_per_gpu: 1946 },
    ],
  },
  {
    // DSPARK acceptance pinned to 5 of 8 draft tokens (SGLANG_SIMULATE_ACC_LEN=5
    // SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_RAGGED_VERIFY_MODE=static), so the
    // rows are independent of the benchmark's random prompts; measured accept length 5.00.
    match: { hw: "mi350x", pdMode: "unified", strategy: "balanced", quant: "mxfp4", spec: "dspark" },
    sglang_version: "v0.5.19 @ 12771786",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 604, tpot_ms: 5.39, tokens_per_sec_per_gpu: 186 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 6035, tpot_ms: 15.32, tokens_per_sec_per_gpu: 864 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 23720, tpot_ms: 31.45, tokens_per_sec_per_gpu: 913 },
    ],
  },
  {
    match: { hw: "mi350x", pdMode: "unified", strategy: "balanced", quant: "mxfp4", spec: "none" },
    sglang_version: "v0.5.19 @ 12771786",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 590, tpot_ms: 18.55, tokens_per_sec_per_gpu: 58 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 6331, tpot_ms: 32.80, tokens_per_sec_per_gpu: 462 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64 },
        ttft_ms: 18665, tpot_ms: 70.39, tokens_per_sec_per_gpu: 813 },
    ],
  },
];
