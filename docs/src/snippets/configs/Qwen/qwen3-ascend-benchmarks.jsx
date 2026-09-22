// Measured on 2026-09-12; original Qwen3 checkpoints, PD co-located.
// Source: sgl-project/sglang c389dd086340878934bf96bc209534f1fbb8440c.
// CANN 9.0.0, torch/torch-npu 2.10.0, sgl-kernel-npu 2026.9.0,
// triton-ascend 3.2.1.dev20260530, transformers 5.8.1. This is the measured
// platform environment, not a claim that every current-main dependency is pinned.
// The legacy tokens_per_sec_per_gpu field is divided by the logical NPU device
// count; this config labels that denominator "NPU device" in the shared UI.
// ShareGPT corpus SHA-256:
// 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4.
// Fixed input/output lengths, seed 1, range ratio 1, flush-cache, one measured run.
// GPQA/AIME: EvalScope 1.11.1, datasets 4.8.4, PyArrow 24.0.0.
// SHA-256 of loaded rows serialized as UTF-8 JSONL in original row order,
// sorted keys, compact separators, ensure_ascii=False, LF after every row:
// AI-ModelScope/gpqa_diamond (198 rows):
// 5bbd882a04a7ed72380b42064fbf1179cd2c0e569c8fa291d25c1296eb7dfde6.
// evalscope/aime25 (30 rows):
// 93226328a3bc794c9b7cf8ccd90beb65491bafa0ac58b132f3c11fe9976131b9.
// GSM8K: upstream sglang.test.run_eval, five examples excluded from 1319 rows.
// Original test.jsonl SHA-256:
// 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14.
export const benchmarks = [
  {
    match: { hw: "a3", variant: "8b", quant: "w8a8", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ c389dd0863",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: { dataset: "random", isl: 3500, osl: 1500, max_concurrency: 1, num_prompts: 4 },
        ttft_ms: 920.20,
        tpot_ms: 4.11,
        tokens_per_sec_per_gpu: 352.69,
      },
    ],
    accuracy: { gsm8k_pct: 88.05 },
    notes: "1 physical A3 card, 2 logical NPU devices (TP2). 4/4 performance requests succeeded; output throughput 211.62 tok/s for the service. Median TTFT 172.17 ms and median TPOT 4.14 ms. GSM8K: 1157/1314 correct, 1314/1314 successful requests, no empty responses or invalid answer extractions. One full 5-shot completion run at temperature 0, maximum output 512 tokens, client concurrency 1; the five examples are excluded from scoring. Main checkpoint: vllm-ascend/Qwen3-8B-w8a8; draft: Zjcxy-SmartAI/Eagle3-Qwen3-8B-zh.",
  },
  {
    match: { hw: "a3", variant: "8b", quant: "w8a8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ c389dd0863",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: { dataset: "random", isl: 3500, osl: 1500, max_concurrency: 64, num_prompts: 256 },
        ttft_ms: 4578.23,
        tpot_ms: 29.94,
        tokens_per_sec_per_gpu: 5927.99,
      },
    ],
    accuracy: { gpqa_diamond_pct: 61.11 },
    notes: "1 physical A3 card allocated, 1 logical NPU device used (TP1). 256/256 performance requests succeeded; output throughput 1778.40 tok/s for the service. Median TTFT 1840.44 ms and median TPOT 28.18 ms. GPQA Diamond: 198/198 questions completed without request errors, one full 0-shot run at temperature 1.0 with a 40000-token output budget. One length-limit stop remains included in the score. Main checkpoint: vllm-ascend/Qwen3-8B-w8a8; draft: Zjcxy-SmartAI/Eagle3-Qwen3-8B-zh.",
  },
  {
    match: { hw: "a3", variant: "30b-a3b", quant: "w8a8", strategy: "low-latency", nodes: "single" },
    sglang_version: "main @ c389dd0863",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: { dataset: "random", isl: 3500, osl: 1500, max_concurrency: 1, num_prompts: 1 },
        ttft_ms: 169.15,
        tpot_ms: 5.57,
        tokens_per_sec_per_gpu: 292.81,
      },
    ],
    accuracy: { aime25_pct: 76.67 },
    notes: "1 physical A3 card, 2 logical NPU devices (TP2). 1/1 request succeeded, matching the reference workload's request count; output throughput 175.69 tok/s for the service. AIME 2025: 23/30 correct, 30/30 requests completed without errors, one full 0-shot run at temperature 1.0 with a 32768-token output budget. Three output-budget stops remain included in the score. Both 30B workloads use the same server configuration and share this accuracy run. Main checkpoint: Eco-Tech/Qwen3-30B-A3B-w8a8; draft: vllm-ascend/Qwen3-a3B_eagle3.",
  },
  {
    match: { hw: "a3", variant: "30b-a3b", quant: "w8a8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "main @ c389dd0863",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: { dataset: "random", isl: 3500, osl: 1500, max_concurrency: 160, num_prompts: 640 },
        ttft_ms: 3619.48,
        tpot_ms: 46.05,
        tokens_per_sec_per_gpu: 5044.12,
      },
    ],
    accuracy: { aime25_pct: 76.67 },
    notes: "1 physical A3 card, 2 logical NPU devices (TP2). 640/640 requests succeeded; output throughput 3026.47 tok/s for the service. Median TTFT 669.44 ms and median TPOT 40.10 ms. AIME 2025: 23/30 correct, 30/30 requests completed without errors, one full 0-shot run at temperature 1.0 with a 32768-token output budget. Three output-budget stops remain included in the score. Both 30B workloads use the same server configuration and share this accuracy run. Main checkpoint: Eco-Tech/Qwen3-30B-A3B-w8a8; draft: vllm-ascend/Qwen3-a3B_eagle3.",
  },
];
