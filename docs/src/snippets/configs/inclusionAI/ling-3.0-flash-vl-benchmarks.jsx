// Ling-3.0-flash-VL per-cell benchmark numbers, keyed by the same `match` tuple as
// ling-3.0-flash-vl.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Speed: bench_serving --flush-cache --random-range-ratio 1, temperature 0. Speed cards
// use the `random` dataset (text-only, isl 8192 / osl 1024) across LL (conc 1/16) and HT
// (conc 1024/4096); per-cell notes carry the separate `image` workload (one random 720p
// JPEG per request, +883 vision tokens, isl/osl 1024/1024, conc 1/16/64/128). TTFT/TPOT
// are P50; tokens_per_sec_per_gpu = total (input + output) token throughput ÷ GPU count.
// HT columns are queue-dominated because the KDA state cache caps concurrent requests
// (GB300 TP=4: 935; H200 TP=4: 314).
// Accuracy: sgl-eval MMMU-Pro, full 1730 examples, single-shot, thinking on (template
// default), temperature 0 / top_p 0.95.
export const benchmarks = [
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev @ 1242867bcb",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 8 },
        ttft_ms: 190.51, tpot_ms: 3.37, tokens_per_sec_per_gpu: 612 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 1443.44, tpot_ms: 6.92, tokens_per_sec_per_gpu: 4335 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 41335.64, tpot_ms: 124.74, tokens_per_sec_per_gpu: 12586 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096, num_prompts: 8192 },
        ttft_ms: 555688.12, tpot_ms: 119.86, tokens_per_sec_per_gpu: 12296 },
    ],
    accuracy: { mmmu_pro_pct: 77.86 },
    notes: "4×GB300, TP=4. Speed: bench_serving --flush-cache, temperature 0; tok/s/GPU = total (input + output) token throughput ÷ 4. HT columns are queue-dominated (KDA state cache caps concurrent requests at 935 on this cell) — judge HT by TPOT/throughput, not TTFT. Image workload (one 720p JPEG per request, +883 vision tokens, in/out=1024/1024): conc 1: TTFT 320.99 ms, TPOT 3.91 ms, 173 tok/s/GPU; conc 16: TTFT 1490.73 ms, TPOT 6.53 ms, 1381 tok/s/GPU; conc 64: TTFT 3279.24 ms, TPOT 12.67 ms, 2996 tok/s/GPU; conc 128: TTFT 6887.76 ms, TPOT 15.56 ms, 4226 tok/s/GPU. Accuracy: MMMU-Pro (sgl-eval, 1730 examples, single-shot, thinking on, temperature 0 / top-p 0.95) measured at 2×GB300 TP=2, stop rate 99.65%.",
  },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev @ 1242867bcb",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 8 },
        ttft_ms: 232.57, tpot_ms: 3.32, tokens_per_sec_per_gpu: 634 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 1216.30, tpot_ms: 8.03, tokens_per_sec_per_gpu: 3898 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 153954.62, tpot_ms: 42.85, tokens_per_sec_per_gpu: 10405 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 4096, num_prompts: 8192 },
        ttft_ms: 800898.01, tpot_ms: 46.01, tokens_per_sec_per_gpu: 10467 },
    ],
    accuracy: { mmmu_pro_pct: 76.71 },
    notes: "4×H200, TP=4. Speed: bench_serving --flush-cache, temperature 0; tok/s/GPU = total (input + output) token throughput ÷ 4. HT columns are queue-dominated (KDA state cache caps concurrent requests at 314 on this cell) — judge HT by TPOT/throughput, not TTFT. Image workload (one 720p JPEG per request, +883 vision tokens, in/out=1024/1024): conc 1: TTFT 221.40 ms, TPOT 3.69 ms, 187 tok/s/GPU; conc 16: TTFT 1228.40 ms, TPOT 7.23 ms, 1378 tok/s/GPU; conc 64: TTFT 3932.72 ms, TPOT 13.02 ms, 2766 tok/s/GPU; conc 128: TTFT 6722.44 ms, TPOT 18.24 ms, 3737 tok/s/GPU. Accuracy: MMMU-Pro (sgl-eval, 1730 examples, single-shot, thinking on), stop rate 99.19%; also GSM8K 97.35% (stop rate 100%), eager, streaming, structured output, and auto parser resolution validated on this cell.",
  },
  { match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
];
