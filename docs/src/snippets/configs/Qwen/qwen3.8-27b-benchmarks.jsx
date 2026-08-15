// Qwen3.8-27B per-cell benchmark numbers, keyed by the same `match` tuple as
// qwen3.8-27b.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// All six rows are ONE-BATCH measurements (sglang.bench_serving --flush-cache,
// random dataset, ISL=1024 / OSL=1024, --random-range-ratio 1, request-rate inf,
// max_concurrency 1/16/64, n=64/64/256 respectively) on a single GB300 GPU
// (Blackwell Ultra SM103, 288GB HBM) on 2026-08-14. SGLang binary:
// lmsysorg/sglang:dev resolving to commit c4271c3fe1262fc2adbd162c33b25de5255251c5.
// With no --attention-backend pin, that commit on GB300 resolves attention to
// triton; a newer sglang (c7c03ec+) resolves trtllm_mha. Same binary and
// protocol for all six cells so the numbers are apples-to-apples. MTP rows
// show mean speculative accept_length across all decode batches.
//
// Accuracy (sgl-eval run gsm8k, full 1319, stop_rate 1.0, truncated_rate 0.0):
// only the W4A4-NVFP4 checkpoint has been GSM8K'd on a GB300 single-GPU box —
// via a sibling experiment pinned to sglang c7c03ec53b1e664c2d415db4f02e43f86661f31d
// with an explicit `--kv-cache-dtype` A/B. The bf16-KV row (96.82%) is the better
// baseline to cite; the default fp8-KV the NVFP4 checkpoint auto-enables scores
// 96.44%. FP8 / BF16 GB300 GSM8K has not been measured and stays null below
// (see journal 2026-08-14-1048-claude-jrn_f6c1265be8cbdf86c44fe36c).
export const benchmarks = [
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    sglang_version: "lmsysorg/sglang:dev @ c4271c3fe",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 79, tpot_ms: 6.4, tokens_per_sec_per_gpu: 155 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 715, tpot_ms: 8.5, tokens_per_sec_per_gpu: 1742 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 1216, tpot_ms: 13.6, tokens_per_sec_per_gpu: 4316 },
    ],
    accuracy: { gsm8k_pct: 96.44 },
    notes: "NVFP4 = RadixArk W4A4-0811 (private). KV auto fp8_e4m3 from ckpt-declared kv_cache_quant_algo. GSM8K fp8-KV = 96.44 / bf16-KV = 96.82 (sgl-eval, `c7c03ec`, sibling experiment on `/scratch/qwen38-w4a4-kv-ab-0814` on `qwen38-27b-nvfp4-convert-0812`); both stop_rate 100% / truncated 0% on full 1319.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "lmsysorg/sglang:dev @ c4271c3fe",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 73, tpot_ms: 2.3, tokens_per_sec_per_gpu: 415 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 216, tpot_ms: 4.4, tokens_per_sec_per_gpu: 3256 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 3085, tpot_ms: 8.6, tokens_per_sec_per_gpu: 5155 },
    ],
    accuracy: { gsm8k_pct: 96.44 },
    notes: "MTP accept_length mean ~3.31 across conc. Suspected cap is `--speculative-num-draft-tokens 4` (max 4 accepted/reject step).",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    sglang_version: "lmsysorg/sglang:dev @ c4271c3fe",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 73, tpot_ms: 8.2, tokens_per_sec_per_gpu: 121 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 586, tpot_ms: 11.2, tokens_per_sec_per_gpu: 1355 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 1539, tpot_ms: 19.7, tokens_per_sec_per_gpu: 3012 },
    ],
    accuracy: { gsm8k_pct: null },
    notes: "FP8 = Qwen/Qwen3.8-27B-FP8 revision 032ba94c96507998d543d5a43f7b4ebcfa6b4fa9 (private). GSM8K not yet measured on GB300 for FP8; TP4 B300 sibling experiment scored it 1276/1319 (96.74%) under thinking sampling.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "lmsysorg/sglang:dev @ c4271c3fe",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 74, tpot_ms: 3.2, tokens_per_sec_per_gpu: 302 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 174, tpot_ms: 5.3, tokens_per_sec_per_gpu: 2771 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 3376, tpot_ms: 9.5, tokens_per_sec_per_gpu: 4599 },
    ],
    accuracy: { gsm8k_pct: null },
    notes: "MTP accept_length mean ~3.16.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "lmsysorg/sglang:dev @ c4271c3fe",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 72, tpot_ms: 10.3, tokens_per_sec_per_gpu: 97 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 490, tpot_ms: 12.5, tokens_per_sec_per_gpu: 1235 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 1622, tpot_ms: 18.4, tokens_per_sec_per_gpu: 3208 },
    ],
    accuracy: { gsm8k_pct: null },
    notes: "BF16 = Qwen/Qwen3.8-27B revision 08cb37595cfe9ce298e6f172c6b50d814ef42413 (private). GSM8K not yet measured on GB300 for BF16.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "lmsysorg/sglang:dev @ c4271c3fe",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 1, num_prompts: 64 },
        ttft_ms: 76, tpot_ms: 4.0, tokens_per_sec_per_gpu: 245 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 16, num_prompts: 64 },
        ttft_ms: 211, tpot_ms: 6.3, tokens_per_sec_per_gpu: 2306 },
      { workload: { dataset: "random", isl: 1024, osl: 1024, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 3894, tpot_ms: 10.9, tokens_per_sec_per_gpu: 3995 },
    ],
    accuracy: { gsm8k_pct: null },
    notes: "MTP accept_length mean ~3.22.",
  },
];
