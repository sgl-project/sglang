export const benchmarks = [
  {
    match: {
      hw: "a3",
      variant: "instruct",
      quant: "w8a8",
      deployment: "pd-colocated",
      nodes: "single",
    },
    sglang_version: "v0.5.16 Ascend NPU run (2026-09-08)",
    latencyPercentile: "Mean",
    accuracy: {
      gsm8k_pct: 97.0,
    },
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 3500,
          osl: 1500,
          max_concurrency: 1,
          num_prompts: 1,
        },
        ttft_ms: 563.17,
        tpot_ms: 11.84,
        // The run used TP=2, so this is total throughput divided by two dies.
        tokens_per_sec_per_gpu: 136.18,
      },
    ],
    notes: "One-request random run at 3.5k input / 1.5k output. Total throughput was 272.35 tok/s across two NPU dies; output throughput was 81.71 tok/s and mean speculative accept length was 3.93. The 11.84 ms mean TPOT is below the 20 ms best-practice target. Re-run on your target host before comparing results.",
  },
];
