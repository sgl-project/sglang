export const benchmarks = [
  {
    match: { hw: "gb300", strategy: "low-latency" },
    sglang_version: "f13cb6f6a7",
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
        ttft_ms: 599.91,
        tpot_ms: 6.48,
        tokens_per_sec_per_gpu: 2266.08,
      },
    ],
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) at the rc2 cut (f13cb6f6a7), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept length confirmed 3.00 in the bench summary and server log): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,812.86 aggregate output tok/s after two discarded warmups. Simulated accept length makes this a throughput-mechanism number. Accuracy is from the shared non-simulated full GSM8K gate: 97.50% with a 100% stop rate over all 1,319 problems.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "fp8-trtllm" },
    sglang_version: "f13cb6f6a7",
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
        ttft_ms: 604.2,
        tpot_ms: 6.19,
        tokens_per_sec_per_gpu: 2338.53,
      },
    ],
    notes:
      "The Low Latency recipe with FP8 KV + TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) at rc2 (f13cb6f6a7), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept 3.00): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,870.82 aggregate output tok/s — 3.2% above the BF16 + TileLang Low Latency row, with mean TPOT 6.19 ms vs 6.48 ms. Draft and target full-graph capture succeeded for this combination.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput" },
    sglang_version: "f13cb6f6a7",
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
        ttft_ms: 684.63,
        tpot_ms: 11.53,
        tokens_per_sec_per_gpu: 1410.4,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 1691.64,
        tpot_ms: 19.88,
        tokens_per_sec_per_gpu: 3023.31,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 5192.23,
        tpot_ms: 43.35,
        tokens_per_sec_per_gpu: 4856.19,
      },
    ],
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) at the rc2 cut (f13cb6f6a7), speculative decoding off, after two discarded warmups per row: 1,128.32 / 2,418.65 / 3,884.95 aggregate output tok/s at concurrency 16 / 64 / 256 (80 / 320 / 1,280 random requests at 1,024 input / 256 output tokens). The measured server added --max-running-requests 256, --cuda-graph-backend-decode full, --cuda-graph-max-bs-decode 256, --chunked-prefill-size 8192, --max-prefill-tokens 8192, and --disable-shared-experts-fusion to the cell command — the concurrency 64/256 rows are not reproducible at the cell's automatic sizing. Throughput at 256 is still scaling but sublinear (prefill queueing). Accuracy is the full GSM8K gate on the same server: 97.50% with a 100% stop rate over all 1,319 problems. With HiCache L1+L2 (32 GB host tier) the same protocol measured 1,125.45 / 2,403.57 / 3,839.57 tok/s — a 0.3-1.2% overhead; the random dataset has no prefix reuse, so L2 benefit was not exercised.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", kvDsaPair: "fp8-trtllm" },
    sglang_version: "f13cb6f6a7",
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
        ttft_ms: 681.3,
        tpot_ms: 10.81,
        tokens_per_sec_per_gpu: 1487.45,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 1693.9,
        tpot_ms: 19.25,
        tokens_per_sec_per_gpu: 3096.09,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 5208.7,
        tpot_ms: 41.96,
        tokens_per_sec_per_gpu: 4965.65,
      },
    ],
    accuracy: { gsm8k_pct: 97.35 },
    notes:
      "FP8 KV cache with TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) at rc2 (f13cb6f6a7), same protocol as the BF16 rows: 1,189.96 / 2,476.87 / 3,972.52 aggregate output tok/s at concurrency 16 / 64 / 256 — 2.3–5.5% above BF16 + TileLang across the curve, and the FP8 pool holds 13.5M tokens per rank vs 7.5M at BF16 (1.8x capacity at identical pool bytes). Accuracy is the full GSM8K gate on this variant: 97.35% vs 97.50% on BF16 KV, a 0.15-point gap inside sampling noise, with a 99.92% stop rate (one truncated run of 1,319). With HiCache L1+L2 (32 GB host tier) the same protocol measured 1,187.13 / 2,464.89 / 3,941.37 tok/s — a 0.2-0.8% overhead; the random dataset has no prefix reuse, so L2 benefit was not exercised.",
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
