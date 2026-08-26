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
        ttft_ms: 589.2,
        tpot_ms: 6.48,
        tokens_per_sec_per_gpu: 2277.46,
      },
    ],
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) at the rc2 cut (f13cb6f6a7), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept length confirmed 3.00 in the bench summary and server log): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,821.97 aggregate output tok/s after two discarded warmups. Simulated accept length makes this a throughput-mechanism number. Accuracy is from the shared non-simulated full GSM8K gate: 97.50% with a 100% stop rate over all 1,319 problems.",
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
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) at the rc2 cut (f13cb6f6a7), speculative decoding off, after two discarded warmups per row: 1,128.32 / 2,418.65 / 3,884.95 aggregate output tok/s at concurrency 16 / 64 / 256 (80 / 320 / 1,280 random requests at 1,024 input / 256 output tokens; the high-concurrency rows with --max-running-requests 256 and decode graph batch 256). Throughput at 256 is still scaling but sublinear (prefill queueing). Accuracy is the full GSM8K gate on the same server: 97.50% with a 100% stop rate over all 1,319 problems.",
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
