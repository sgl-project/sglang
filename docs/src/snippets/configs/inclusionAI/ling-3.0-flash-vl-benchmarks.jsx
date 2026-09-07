// Ling-3.0-flash-VL per-cell benchmark numbers, keyed by the same `match` tuple as
// ling-3.0-flash-vl.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Speed: bench_serving `image` dataset, one random 720p JPEG per request (883 vision
// tokens on top of the text ISL), --random-range-ratio 1, --flush-cache, temperature 0.
// TTFT/TPOT are P50; tokens_per_sec_per_gpu is total (text + vision + output) tok/s/GPU.
// Accuracy: sgl-eval MMMU-Pro, full 1730 examples, single-shot, thinking on (template
// default), temperature 0 / top_p 0.95; stop rate 99.65%, truncated 0.35%, error 0%.
export const benchmarks = [
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev @ 3fcf9ddcdf",
    speed: [
      { workload: { dataset: "image", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 8 },
        ttft_ms: 339.27, tpot_ms: 4.41, tokens_per_sec_per_gpu: 306 },
      { workload: { dataset: "image", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 32 },
        ttft_ms: 1486.45, tpot_ms: 8.75, tokens_per_sec_per_gpu: 2278 },
      { workload: { dataset: "image", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 128 },
        ttft_ms: 4005.60, tpot_ms: 15.36, tokens_per_sec_per_gpu: 4811 },
    ],
    accuracy: { mmmu_pro_pct: 77.86 },
    notes: "2×GB300, TP=2. Speed: one 720p image per request adds 883 vision tokens to the listed text ISL; throughput counts text, vision, and output tokens. Short-QA workload (image, in/out=256/128): max-concurrency=16: TTFT 1480.46 ms, TPOT 7.91 ms, 4105 tok/s/GPU; max-concurrency=64: TTFT 3198.87 ms, TPOT 16.95 ms, 7484 tok/s/GPU. Accuracy: MMMU-Pro (sgl-eval, 1730 examples, single-shot, thinking on, temperature 0 / top-p 0.95), stop rate 99.65%.",
  },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev @ fec770ad42",
    accuracy: { mmmu_pro_pct: 76.71 },
    notes: "4×H200, TP=4. Accuracy: MMMU-Pro (sgl-eval, 1730 examples, single-shot, thinking on), stop rate 99.19%, truncated 0.81%, error 0%. Also validated on this cell: GSM8K 97.35% (stop rate 100%), eager mode (--disable-cuda-graph), streaming reasoning split, structured output, auto parser resolution (reasoning + tool-call → ling3). Speed numbers pending.",
  },
  { match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
];
