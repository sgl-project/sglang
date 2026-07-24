// Measured accuracy for the Qwen3.5 cookbook — transcribed verbatim from the
// legacy page's §5 Benchmark section (H200 ×8, TP=8, NEXTN speculative
// decoding; the measured run also had the reasoning and tool-call parsers
// enabled, which the matching cell omits — parser flags are a Playground
// feature, never part of Deployment commands).
//
// The legacy page's speed numbers are NOT migrated: they were measured on a
// drifting "main branch" build, which is no version anchor — speed data is
// only meaningful against an exact, reproducible build (re-measure via the
// "⚡ Reproduce" commands and submit with a pinned release). Accuracy is far
// less build-sensitive and is kept. Cells without an entry render "pending".

export const benchmarks = [
{
    match: { hw: "h200", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    accuracy: { gsm8k_pct: 97.5, mmmu_pct: 97.8 },
    notes: "GSM8K via benchmark/gsm8k/bench_sglang.py (200 questions); MMMU via benchmark/mmmu/bench_sglang.py (91-sample val subset).",
  },
  {
    match: { hw: "b200", variant: "2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 40, tpot_ms: 1.58, tokens_per_sec_per_gpu: 5158 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 142, tpot_ms: 12.62, tokens_per_sec_per_gpu: 67581 },
    ],
  },
  {
    match: { hw: "b200", variant: "2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 109, tpot_ms: 0.73, tokens_per_sec_per_gpu: 8064 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 5177, tpot_ms: 8.65, tokens_per_sec_per_gpu: 47706 },
    ],
  },
  {
    match: { hw: "b200", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 60, tpot_ms: 2.58, tokens_per_sec_per_gpu: 3282 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 105, tpot_ms: 19.69, tokens_per_sec_per_gpu: 44316 },
    ],
  },
  {
    match: { hw: "b200", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 63, tpot_ms: 1.09, tokens_per_sec_per_gpu: 6386 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 4944, tpot_ms: 8.14, tokens_per_sec_per_gpu: 49171 },
    ],
  },
  {
    match: { hw: "b200", variant: "9b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 84, tpot_ms: 3.91, tokens_per_sec_per_gpu: 2192 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 123, tpot_ms: 23.44, tokens_per_sec_per_gpu: 37044 },
    ],
  },
  {
    match: { hw: "b200", variant: "9b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 86, tpot_ms: 1.54, tokens_per_sec_per_gpu: 4865 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 6076, tpot_ms: 10.03, tokens_per_sec_per_gpu: 39995 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 258, tpot_ms: 10.45, tokens_per_sec_per_gpu: 814 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 16131, tpot_ms: 46.25, tokens_per_sec_per_gpu: 11354 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 264, tpot_ms: 4.25, tokens_per_sec_per_gpu: 1824 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 23571, tpot_ms: 19.46, tokens_per_sec_per_gpu: 13498 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 185, tpot_ms: 7.54, tokens_per_sec_per_gpu: 1126 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 8093, tpot_ms: 43.51, tokens_per_sec_per_gpu: 14862 },
    ],
  },
  {
    match: { hw: "b200", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 254, tpot_ms: 3.39, tokens_per_sec_per_gpu: 2226 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 19173, tpot_ms: 22.6, tokens_per_sec_per_gpu: 14775 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 88, tpot_ms: 3.0, tokens_per_sec_per_gpu: 2783 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 143, tpot_ms: 31.6, tokens_per_sec_per_gpu: 27951 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 90, tpot_ms: 1.69, tokens_per_sec_per_gpu: 4684 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 7174, tpot_ms: 12.24, tokens_per_sec_per_gpu: 33369 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 104, tpot_ms: 2.91, tokens_per_sec_per_gpu: 2817 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 147, tpot_ms: 31.53, tokens_per_sec_per_gpu: 28180 },
    ],
  },
  {
    match: { hw: "b200", variant: "35b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 114, tpot_ms: 1.68, tokens_per_sec_per_gpu: 4429 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 7621, tpot_ms: 12.79, tokens_per_sec_per_gpu: 31750 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 119, tpot_ms: 4.1, tokens_per_sec_per_gpu: 1018 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 22221, tpot_ms: 22.88, tokens_per_sec_per_gpu: 6727 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 193, tpot_ms: 1.85, tokens_per_sec_per_gpu: 1159 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 28826, tpot_ms: 9.92, tokens_per_sec_per_gpu: 6685 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 150, tpot_ms: 4.75, tokens_per_sec_per_gpu: 1743 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 45672, tpot_ms: 16.75, tokens_per_sec_per_gpu: 8384 },
    ],
  },
  {
    match: { hw: "b200", variant: "122b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 154, tpot_ms: 2.11, tokens_per_sec_per_gpu: 3241 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 45206, tpot_ms: 6.02, tokens_per_sec_per_gpu: 9433 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 145, tpot_ms: 4.6, tokens_per_sec_per_gpu: 226 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 236, tpot_ms: 45.76, tokens_per_sec_per_gpu: 2422 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 218, tpot_ms: 1.48, tokens_per_sec_per_gpu: 543 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 10718, tpot_ms: 18.5, tokens_per_sec_per_gpu: 2828 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 153, tpot_ms: 4.67, tokens_per_sec_per_gpu: 444 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 181, tpot_ms: 42.0, tokens_per_sec_per_gpu: 5259 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 156, tpot_ms: 1.78, tokens_per_sec_per_gpu: 952 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 366, tpot_ms: 28.92, tokens_per_sec_per_gpu: 7131 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 181, tpot_ms: 5.31, tokens_per_sec_per_gpu: 389 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 252, tpot_ms: 54.25, tokens_per_sec_per_gpu: 4103 },
    ],
  },
  {
    match: { hw: "b200", variant: "397b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 240, tpot_ms: 2.12, tokens_per_sec_per_gpu: 800 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 14513, tpot_ms: 23.83, tokens_per_sec_per_gpu: 4215 },
    ],
  },
  {
    match: { hw: "b300", variant: "2b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 37, tpot_ms: 1.51, tokens_per_sec_per_gpu: 5493 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 115, tpot_ms: 13.56, tokens_per_sec_per_gpu: 64518 },
    ],
  },
  {
    match: { hw: "b300", variant: "2b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 95, tpot_ms: 0.72, tokens_per_sec_per_gpu: 8615 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 207, tpot_ms: 13.22, tokens_per_sec_per_gpu: 66333 },
    ],
  },
  {
    match: { hw: "b300", variant: "4b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 55, tpot_ms: 2.55, tokens_per_sec_per_gpu: 3356 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 85, tpot_ms: 17.49, tokens_per_sec_per_gpu: 49610 },
    ],
  },
  {
    match: { hw: "b300", variant: "4b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 56, tpot_ms: 1.06, tokens_per_sec_per_gpu: 6654 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 4558, tpot_ms: 7.58, tokens_per_sec_per_gpu: 52557 },
    ],
  },
  {
    match: { hw: "b300", variant: "9b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 76, tpot_ms: 3.76, tokens_per_sec_per_gpu: 2287 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 114, tpot_ms: 21.94, tokens_per_sec_per_gpu: 39499 },
    ],
  },
  {
    match: { hw: "b300", variant: "9b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 81, tpot_ms: 1.49, tokens_per_sec_per_gpu: 5041 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 5516, tpot_ms: 9.13, tokens_per_sec_per_gpu: 43877 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 241, tpot_ms: 10.2, tokens_per_sec_per_gpu: 838 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 335, tpot_ms: 65.08, tokens_per_sec_per_gpu: 13327 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 250, tpot_ms: 4.18, tokens_per_sec_per_gpu: 1877 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 16045, tpot_ms: 27.14, tokens_per_sec_per_gpu: 15160 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 178, tpot_ms: 7.46, tokens_per_sec_per_gpu: 1140 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 255, tpot_ms: 52.41, tokens_per_sec_per_gpu: 16630 },
    ],
  },
  {
    match: { hw: "b300", variant: "27b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 243, tpot_ms: 3.33, tokens_per_sec_per_gpu: 2284 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 14595, tpot_ms: 24.68, tokens_per_sec_per_gpu: 16723 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 69, tpot_ms: 2.87, tokens_per_sec_per_gpu: 2952 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 103, tpot_ms: 27.56, tokens_per_sec_per_gpu: 31328 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 66, tpot_ms: 1.57, tokens_per_sec_per_gpu: 4973 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 6224, tpot_ms: 10.64, tokens_per_sec_per_gpu: 38280 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 70, tpot_ms: 2.82, tokens_per_sec_per_gpu: 2983 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 118, tpot_ms: 26.09, tokens_per_sec_per_gpu: 33375 },
    ],
  },
  {
    match: { hw: "b300", variant: "35b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 77, tpot_ms: 1.64, tokens_per_sec_per_gpu: 4732 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 6034, tpot_ms: 10.27, tokens_per_sec_per_gpu: 39460 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 101, tpot_ms: 4.04, tokens_per_sec_per_gpu: 1048 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 175, tpot_ms: 43.14, tokens_per_sec_per_gpu: 10053 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 163, tpot_ms: 1.81, tokens_per_sec_per_gpu: 1293 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 11434, tpot_ms: 18.8, tokens_per_sec_per_gpu: 10612 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 144, tpot_ms: 4.67, tokens_per_sec_per_gpu: 1792 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 11841, tpot_ms: 40.52, tokens_per_sec_per_gpu: 13847 },
    ],
  },
  {
    match: { hw: "b300", variant: "122b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 145, tpot_ms: 2.08, tokens_per_sec_per_gpu: 3366 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 17251, tpot_ms: 15.45, tokens_per_sec_per_gpu: 17775 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 118, tpot_ms: 4.93, tokens_per_sec_per_gpu: 430 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 13697, tpot_ms: 34.88, tokens_per_sec_per_gpu: 3589 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 154, tpot_ms: 1.6, tokens_per_sec_per_gpu: 1118 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 15017, tpot_ms: 10.35, tokens_per_sec_per_gpu: 5377 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 121, tpot_ms: 5.42, tokens_per_sec_per_gpu: 785 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 193, tpot_ms: 45.83, tokens_per_sec_per_gpu: 9471 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 123, tpot_ms: 2.2, tokens_per_sec_per_gpu: 1713 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 8963, tpot_ms: 19.43, tokens_per_sec_per_gpu: 11654 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 153, tpot_ms: 5.79, tokens_per_sec_per_gpu: 729 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 35317, tpot_ms: 24.97, tokens_per_sec_per_gpu: 4730 },
    ],
  },
  {
    match: { hw: "b300", variant: "397b", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.15.post1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 157, tpot_ms: 2.61, tokens_per_sec_per_gpu: 1452 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 100 },
        ttft_ms: 37580, tpot_ms: 9.17, tokens_per_sec_per_gpu: 5358 },
    ],
  },
];
