// Measured speed benchmarks for the Qwen3.6 cookbook. Cache-cold, pinned release:
// each cell launched from its exact Deployment config (config.cells), benched with
// benchmarkCommands.speed (random isl 8192 / osl 1024, --random-range-ratio 1.0,
// --warmup-requests 64, --flush-cache), run1 landed. tokens_per_sec_per_gpu =
// output_throughput / tp * (isl+osl)/osl (tp=1). LL @ conc 1+16, HT @ conc 1024+4096.
//
// Pending cells (H100 HT sweeps, 35B-A3B bf16 LL, Xeon) are not yet measured.
//
// 35B-A3B NVFP4 (MoE) requires --moe-runner-backend flashinfer_cutlass on Blackwell
// (baked into the config); 27B NVFP4 is dense and unaffected.
// NOTE: the nvidia/Qwen3.6-*-NVFP4 checkpoints quantize the MoE experts as W4A16 (4-bit
// weights, 16-bit activations) — memory-oriented (~half the weight VRAM), NOT a throughput
// win: the expert matmuls run in 16-bit and don't hit the NVFP4 (W4A4) tensor-core path.
// So NVFP4 speed ~ FP8, with a smaller footprint. See the Qwen3.6.mdx §3.2 NVFP4 note.

export const benchmarks = [
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 182, tpot_ms: 2.23, tokens_per_sec_per_gpu: 3751 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 223, tpot_ms: 8.13, tokens_per_sec_per_gpu: 15394 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 390479, tpot_ms: 32.67, tokens_per_sec_per_gpu: 21029 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1738129, tpot_ms: 32.93, tokens_per_sec_per_gpu: 21090 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 263654, tpot_ms: 27.9, tokens_per_sec_per_gpu: 30201 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1206050, tpot_ms: 28.22, tokens_per_sec_per_gpu: 30217 },
    ],
  },
  {
    match: { hw: "h200", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 138, tpot_ms: 1.98, tokens_per_sec_per_gpu: 4331 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 189, tpot_ms: 6.2, tokens_per_sec_per_gpu: 20434 },
    ],
  },
  {
    match: { hw: "h200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 734, tpot_ms: 5.61, tokens_per_sec_per_gpu: 1433 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1198, tpot_ms: 17.28, tokens_per_sec_per_gpu: 7083 },
    ],
  },
  {
    match: { hw: "h200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 487, tpot_ms: 4.3, tokens_per_sec_per_gpu: 1884 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 734, tpot_ms: 12.58, tokens_per_sec_per_gpu: 9895 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 206078, tpot_ms: 22.45, tokens_per_sec_per_gpu: 38134 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 934685, tpot_ms: 22.47, tokens_per_sec_per_gpu: 38729 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 98, tpot_ms: 1.81, tokens_per_sec_per_gpu: 4921 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 126, tpot_ms: 5.27, tokens_per_sec_per_gpu: 24712 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 166960, tpot_ms: 24.3, tokens_per_sec_per_gpu: 46963 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 757978, tpot_ms: 24.73, tokens_per_sec_per_gpu: 47242 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 116, tpot_ms: 1.81, tokens_per_sec_per_gpu: 4755 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 221, tpot_ms: 4.62, tokens_per_sec_per_gpu: 27810 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b-a3b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 161551, tpot_ms: 27.86, tokens_per_sec_per_gpu: 45714 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 770396, tpot_ms: 27.8, tokens_per_sec_per_gpu: 45626 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b-a3b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 119, tpot_ms: 1.93, tokens_per_sec_per_gpu: 4583 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 140, tpot_ms: 5.26, tokens_per_sec_per_gpu: 25245 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 724659, tpot_ms: 31.56, tokens_per_sec_per_gpu: 12154 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 3048776, tpot_ms: 31.61, tokens_per_sec_per_gpu: 12164 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 402, tpot_ms: 4.29, tokens_per_sec_per_gpu: 1966 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 605, tpot_ms: 10.84, tokens_per_sec_per_gpu: 11136 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 538675, tpot_ms: 29.35, tokens_per_sec_per_gpu: 16229 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2271693, tpot_ms: 29.51, tokens_per_sec_per_gpu: 16284 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 292, tpot_ms: 3.4, tokens_per_sec_per_gpu: 2532 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 321, tpot_ms: 8.59, tokens_per_sec_per_gpu: 14362 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1238608, tpot_ms: 61.09, tokens_per_sec_per_gpu: 7048 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 5241929, tpot_ms: 60.77, tokens_per_sec_per_gpu: 7051 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 966, tpot_ms: 3.56, tokens_per_sec_per_gpu: 2070 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1458, tpot_ms: 17.26, tokens_per_sec_per_gpu: 6707 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b-a3b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 140285, tpot_ms: 33.17, tokens_per_sec_per_gpu: 49190 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 707855, tpot_ms: 33.54, tokens_per_sec_per_gpu: 48944 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b-a3b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 95, tpot_ms: 1.7, tokens_per_sec_per_gpu: 5235 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 153, tpot_ms: 5.17, tokens_per_sec_per_gpu: 24971 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b-a3b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 112976, tpot_ms: 34.69, tokens_per_sec_per_gpu: 54080 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 628702, tpot_ms: 34.83, tokens_per_sec_per_gpu: 54488 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 101, tpot_ms: 1.71, tokens_per_sec_per_gpu: 5108 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 161, tpot_ms: 4.47, tokens_per_sec_per_gpu: 28910 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b-a3b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 123937, tpot_ms: 39.39, tokens_per_sec_per_gpu: 51550 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 645007, tpot_ms: 39.84, tokens_per_sec_per_gpu: 52068 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b-a3b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 115, tpot_ms: 1.88, tokens_per_sec_per_gpu: 4705 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 128, tpot_ms: 5.2, tokens_per_sec_per_gpu: 25907 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 575160, tpot_ms: 45.8, tokens_per_sec_per_gpu: 14272 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 2551444, tpot_ms: 46.23, tokens_per_sec_per_gpu: 14298 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 389, tpot_ms: 4.04, tokens_per_sec_per_gpu: 2037 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 410, tpot_ms: 11.03, tokens_per_sec_per_gpu: 11740 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 440199, tpot_ms: 42.42, tokens_per_sec_per_gpu: 18327 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 1985294, tpot_ms: 42.22, tokens_per_sec_per_gpu: 18403 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 284, tpot_ms: 3.31, tokens_per_sec_per_gpu: 2576 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 420, tpot_ms: 8.28, tokens_per_sec_per_gpu: 14977 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024 },
        ttft_ms: 1057860, tpot_ms: 91.17, tokens_per_sec_per_gpu: 7666 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096 },
        ttft_ms: 4765554, tpot_ms: 93.51, tokens_per_sec_per_gpu: 7667 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 937, tpot_ms: 3.35, tokens_per_sec_per_gpu: 2227 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 955, tpot_ms: 16.65, tokens_per_sec_per_gpu: 7212 },
    ],
  },
  {
    match: { hw: "h100", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 770, tpot_ms: 7.41, tokens_per_sec_per_gpu: 1082 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 67026, tpot_ms: 8.76, tokens_per_sec_per_gpu: 1898 },
    ],
  },
  {
    match: { hw: "h100", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 522, tpot_ms: 5.27, tokens_per_sec_per_gpu: 1562 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 8322, tpot_ms: 11.27, tokens_per_sec_per_gpu: 7001 },
    ],
  },
  {
    match: { hw: "h100", variant: "35b-a3b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.16",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 147, tpot_ms: 2.21, tokens_per_sec_per_gpu: 3982 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 181, tpot_ms: 7.17, tokens_per_sec_per_gpu: 17730 },
    ],
  },
];
