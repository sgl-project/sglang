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
    sglang_version: "dev @ bf254483a1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 8 },
        ttft_ms: 190.51, tpot_ms: 3.37, tokens_per_sec_per_gpu: 612 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 1443.44, tpot_ms: 6.92, tokens_per_sec_per_gpu: 4335 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 41335.64, tpot_ms: 124.74, tokens_per_sec_per_gpu: 12586 },
    ],
    accuracy: { mmmu_pro_pct: 75.78 },
    notes: "4×GB300, TP=4. Speed: bench_serving --flush-cache, temperature 0; tok/s/GPU = total (input + output) token throughput ÷ 4. max-concurrency=4096 (8192 prompts): TTFT 555688.12 ms, TPOT 119.86 ms, 12296 tok/s/GPU. HT columns are queue-dominated (KDA state cache caps concurrent requests at 935 on this cell) — judge HT by TPOT/throughput, not TTFT. Image workload (one 720p JPEG per request, +883 vision tokens, in/out=1024/1024): conc 1: TTFT 320.99 ms, TPOT 3.91 ms, 173 tok/s/GPU; conc 16: TTFT 1490.73 ms, TPOT 6.53 ms, 1381 tok/s/GPU; conc 64: TTFT 3279.24 ms, TPOT 12.67 ms, 2996 tok/s/GPU; conc 128: TTFT 6887.76 ms, TPOT 15.56 ms, 4226 tok/s/GPU. Accuracy: MMMU-Pro (sgl-eval, 1730 examples, single-shot, thinking on, temperature 0 / top-p 0.95) measured on this recipe, stop rate 99.08%. The same checkpoint measured 77.86% at 2×GB300 TP=2 and 76.71% at 4×H200 TP=4.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "dev @ bf254483a1",
    accuracy: { mmmu_pro_pct: 76.01, gsm8k_pct: 97.19 },
    notes: "4×GB300, TP=4. Measured with online dynamic FP8 (--quantization fp8 on the BF16 checkpoint), the same serving path the FP8 variant uses. Accuracy vs BF16 on the same box: MMMU-Pro 76.01% vs 77.86% (stop 99.36%), GSM8K 97.19% vs 97.35% (stop 100%). Speed (LL points, same protocol as the BF16 card): text 8192/1024 conc 1: TTFT 174.80 ms, TPOT 3.95 ms, 546 tok/s/GPU; conc 16: TTFT 981.25 ms, TPOT 7.49 ms, 4273 tok/s/GPU. Image 1024/1024 conc 1: TTFT 272.09 ms, TPOT 4.51 ms, 153 tok/s/GPU; conc 16: TTFT 1389.27 ms, TPOT 7.15 ms, 1369 tok/s/GPU. FP8 prefill (TTFT) is consistently faster than BF16 while TPOT is ~10% slower.",
  },
  { match: { hw: "b300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "dev @ bf254483a1",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 8 },
        ttft_ms: 232.57, tpot_ms: 3.32, tokens_per_sec_per_gpu: 634 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 1216.30, tpot_ms: 8.03, tokens_per_sec_per_gpu: 3898 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1024, num_prompts: 2048 },
        ttft_ms: 153954.62, tpot_ms: 42.85, tokens_per_sec_per_gpu: 10405 },
    ],
    accuracy: { mmmu_pro_pct: 76.71 },
    notes: "4×H200, TP=4. Speed: bench_serving --flush-cache, temperature 0; tok/s/GPU = total (input + output) token throughput ÷ 4. max-concurrency=4096 (8192 prompts): TTFT 800898.01 ms, TPOT 46.01 ms, 10467 tok/s/GPU. HT columns are queue-dominated (KDA state cache caps concurrent requests at 314 on this cell) — judge HT by TPOT/throughput, not TTFT. Image workload (one 720p JPEG per request, +883 vision tokens, in/out=1024/1024): conc 1: TTFT 221.40 ms, TPOT 3.69 ms, 187 tok/s/GPU; conc 16: TTFT 1228.40 ms, TPOT 7.23 ms, 1378 tok/s/GPU; conc 64: TTFT 3932.72 ms, TPOT 13.02 ms, 2766 tok/s/GPU; conc 128: TTFT 6722.44 ms, TPOT 18.24 ms, 3737 tok/s/GPU. Accuracy: MMMU-Pro (sgl-eval, 1730 examples, single-shot, thinking on), stop rate 99.19%; also GSM8K 97.35% (stop rate 100%), eager, streaming, structured output, and auto parser resolution validated on this cell.",
  },
  { match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
  { match: { hw: "h100", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" } },
];
