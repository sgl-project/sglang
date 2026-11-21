// Qwen3.8-27B per-cell benchmark numbers. See _deployment.jsx for the
// speed/accuracy schema.
//
// STRUCTURALLY UNMATCHED since the strategy->overlay migration: the six legacy
// GB300 entries key on the removed `strategy` match dim, and the three gb300
// `high-throughput` cells they were measured against were deleted (MTP is now
// the Speculative Decoding OVERLAY row, which never participates in cell
// matching). No current cell tuple matches these rows, so the UI shows no
// benchmark card. The numbers are kept as measurement provenance ONLY — do
// not re-key them onto the new cells without re-measuring: the overlay-era
// commands differ (explicit --kv-cache-dtype fp8_e4m3, strategy flag emitted
// by the tier row) and the accept lengths are protocol-specific.
//
// All six rows are ONE-BATCH measurements (sglang.benchmark.serving --flush-cache,
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
  // H200 FP8 / bf16 SSM-state measurements and raw runs reported in
  // https://github.com/sgl-project/sglang/issues/42024 by dhruvrnaik.
  {
    match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single", spec: "eagle", tier: "high-throughput", ssmDtype: "bfloat16" },
    sglang_version: "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 528.48, tpot_ms: 14.17, tokens_per_sec_per_gpu: 9416.88 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 542.77, tpot_ms: 4.74, tokens_per_sec_per_gpu: 1707.48 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1999.6, tpot_ms: 13.52, tokens_per_sec_per_gpu: 8972.17 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 3298.16, tpot_ms: 23.6, tokens_per_sec_per_gpu: 10412.2 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 78.89, tpot_ms: 4.73, tokens_per_sec_per_gpu: 994.92 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 300.25, tpot_ms: 9.16, tokens_per_sec_per_gpu: 7382.3 },
    ],
    accuracy: { gsm8k_pct: 90.67 },
    notes: "Reported by dhruvrnaik in #42024, 2026-09-30 to 2026-10-01: H200 TP1, FP8 weights, bf16 SSM state. Mean latency, not P50; median of three retained runs after two warmups. Total input+output throughput per GPU. Mamba ratio 3.74 (average request length 5120, target concurrency 64). GSM8K: all 1319 questions, zero-shot, greedy, thinking off; not comparable with the thinking-mode scores on this page.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single", spec: "none", tier: "high-throughput", ssmDtype: "bfloat16" },
    sglang_version: "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 2729.66, tpot_ms: 17.86, tokens_per_sec_per_gpu: 11212.39 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 525.83, tpot_ms: 10.22, tokens_per_sec_per_gpu: 839.05 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 5465.65, tpot_ms: 15.32, tokens_per_sec_per_gpu: 6973.43 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 18683.15, tpot_ms: 33.63, tokens_per_sec_per_gpu: 11104.9 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 75.25, tpot_ms: 10.1, tokens_per_sec_per_gpu: 482.74 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 884.36, tpot_ms: 11.47, tokens_per_sec_per_gpu: 5363.16 },
    ],
    accuracy: { gsm8k_pct: 90.37 },
    notes: "Reported by dhruvrnaik in #42024, 2026-09-30 to 2026-10-01: H200 TP1, FP8 weights, bf16 SSM state. Mean latency, not P50; median of three retained runs after two warmups. Total input+output throughput per GPU. Mamba ratio 1.87 (average request length 5120, target concurrency 64). GSM8K: all 1319 questions, zero-shot, greedy, thinking off; not comparable with the thinking-mode scores on this page.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single", spec: "eagle", tier: "low-latency", ssmDtype: "bfloat16" },
    sglang_version: "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 547.13, tpot_ms: 13.88, tokens_per_sec_per_gpu: 9564.27 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 545.59, tpot_ms: 4.7, tokens_per_sec_per_gpu: 1722.41 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 1953.29, tpot_ms: 13.53, tokens_per_sec_per_gpu: 8985.12 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 32, num_prompts: 160 },
        ttft_ms: 3267.16, tpot_ms: 23.55, tokens_per_sec_per_gpu: 10461.12 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 83.49, tpot_ms: 4.75, tokens_per_sec_per_gpu: 987.25 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 308.8, tpot_ms: 9.04, tokens_per_sec_per_gpu: 7391.09 },
    ],
    accuracy: { gsm8k_pct: 90.37 },
    notes: "Reported by dhruvrnaik in #42024, 2026-09-30 to 2026-10-01: H200 TP1, FP8 weights, bf16 SSM state. Mean latency, not P50; median of three retained runs after two warmups. Total input+output throughput per GPU. Mamba ratio 4.21 (average request length 5120, target concurrency 64). GSM8K: all 1319 questions, zero-shot, greedy, thinking off; not comparable with the thinking-mode scores on this page.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", nodes: "single", spec: "none", tier: "low-latency", ssmDtype: "bfloat16" },
    sglang_version: "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
    latencyPercentile: "Mean",
    speed: [
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 2772.09, tpot_ms: 17.86, tokens_per_sec_per_gpu: 11162.95 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 532.95, tpot_ms: 10.18, tokens_per_sec_per_gpu: 841.27 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 5525.68, tpot_ms: 15.34, tokens_per_sec_per_gpu: 6945.62 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 64, num_prompts: 320 },
        ttft_ms: 18904.39, tpot_ms: 33.82, tokens_per_sec_per_gpu: 11016.86 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 1, num_prompts: 16 },
        ttft_ms: 75.17, tpot_ms: 10.07, tokens_per_sec_per_gpu: 483.93 },
      { workload: { dataset: "random", isl: 1024, osl: 256, max_concurrency: 16, num_prompts: 80 },
        ttft_ms: 874.09, tpot_ms: 11.46, tokens_per_sec_per_gpu: 5380.37 },
    ],
    accuracy: { gsm8k_pct: 90.3 },
    notes: "Reported by dhruvrnaik in #42024, 2026-09-30 to 2026-10-01: H200 TP1, FP8 weights, bf16 SSM state. Mean latency, not P50; median of three retained runs after two warmups. Total input+output throughput per GPU. Mamba ratio 2.34 (average request length 5120, target concurrency 64). GSM8K: all 1319 questions, zero-shot, greedy, thinking off; not comparable with the thinking-mode scores on this page.",
  },
  {
    match: { hw: "gb300", variant: "default", quant: "nvfp4-fp4-head", strategy: "balanced", nodes: "single" },
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
    match: { hw: "gb300", variant: "default", quant: "nvfp4-fp4-head", strategy: "high-throughput", nodes: "single" },
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
