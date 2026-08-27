export const benchmarks = [
  {
    match: { hw: "gb300", strategy: "low-latency" },
    sglang_version: "d6ab04bdf1",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 598.41,
        tpot_ms: 6.43,
        tokens_per_sec_per_gpu: 2280.83,
      },
    ],
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) on the current release-image tree (d6ab04bdf1), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept length confirmed 3.00 in the bench summary and server log): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,824.66 aggregate output tok/s after two discarded warmups. Simulated accept length makes this a throughput-mechanism number. Accuracy is from the shared non-simulated full GSM8K gate: 97.50% with a 100% stop rate over all 1,319 problems.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "fp8-trtllm" },
    sglang_version: "d6ab04bdf1",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 609.9,
        tpot_ms: 6.25,
        tokens_per_sec_per_gpu: 2317.35,
      },
    ],
    notes:
      "The Low Latency recipe with FP8 KV + TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) on the current release-image tree (d6ab04bdf1), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept 3.00): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,853.88 aggregate output tok/s — 1.6% above the BF16 + TileLang Low Latency row, with mean TPOT 6.25 ms vs 6.43 ms. Draft and target full-graph capture succeeded for this combination. The speed rows were measured with the NEXTN spelling and --disable-shared-experts-fusion, which resolve to the same runtime path as the published command on this tree.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput" },
    sglang_version: "d6ab04bdf1",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 586.33,
        tpot_ms: 11.51,
        tokens_per_sec_per_gpu: 1451.53,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 1461.23,
        tpot_ms: 18.36,
        tokens_per_sec_per_gpu: 3325.3,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 3912.74,
        tpot_ms: 37.69,
        tokens_per_sec_per_gpu: 6035.41,
      },
    ],
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) on the current release-image tree (d6ab04bdf1), speculative decoding off, after two discarded warmups per row: 1,161.22 / 2,660.24 / 4,828.33 aggregate output tok/s at concurrency 16 / 64 / 256 (80 / 320 / 1,280 random requests at 1,024 input / 256 output tokens). The server ran exactly the published cell command. Throughput at 256 is still scaling but sublinear (prefill queueing). Accuracy is from the shared non-simulated full GSM8K gate: 97.50% with a 100% stop rate over all 1,319 problems. With HiCache L1+L2 (32 GB host tier, 16k prefill chunks) the same protocol measured 1,202.07 / 2,696.20 / 4,634.47 tok/s — within 4% of the non-HiCache rows; the random dataset has no prefix reuse, so L2 benefit was not exercised.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", kvDsaPair: "fp8-trtllm" },
    sglang_version: "d6ab04bdf1",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 581.52,
        tpot_ms: 10.79,
        tokens_per_sec_per_gpu: 1533.84,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 1459.62,
        tpot_ms: 17.68,
        tokens_per_sec_per_gpu: 3423.26,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 3908.3,
        tpot_ms: 36.12,
        tokens_per_sec_per_gpu: 6221.28,
      },
    ],
    accuracy: { gsm8k_pct: 97.35 },
    notes:
      "FP8 KV cache with TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) on the current release-image tree (d6ab04bdf1), same protocol as the BF16 rows: 1,227.07 / 2,738.61 / 4,977.02 aggregate output tok/s at concurrency 16 / 64 / 256 — 2.9–5.7% above BF16 + TileLang across the curve, and the FP8 pool holds 12.6M tokens per rank vs 7.0M at BF16 (1.8x capacity at identical pool bytes). Accuracy is the full GSM8K gate on this variant: 97.35% vs 97.50% on BF16 KV, a 0.15-point gap inside sampling noise, with a 100% stop rate over all 1,319 problems. With HiCache L1+L2 (32 GB host tier, 16k prefill chunks) the same protocol measured 1,263.85 / 2,763.31 / 4,773.95 tok/s — within 5% of the non-HiCache rows; the random dataset has no prefix reuse, so L2 benefit was not exercised.",
  },
  { match: { hw: "h100", strategy: "low-latency" } },
  { match: { hw: "h100", strategy: "high-throughput" } },
  { match: { hw: "h200", strategy: "low-latency" } },
  { match: { hw: "h200", strategy: "high-throughput" } },
  { match: { hw: "b200", strategy: "low-latency" } },
  { match: { hw: "b200", strategy: "high-throughput" } },
  { match: { hw: "b300", strategy: "low-latency" } },
  { match: { hw: "b300", strategy: "high-throughput" } },
  { match: { hw: "gb200", strategy: "low-latency" } },
  { match: { hw: "gb200", strategy: "high-throughput" } },
];
