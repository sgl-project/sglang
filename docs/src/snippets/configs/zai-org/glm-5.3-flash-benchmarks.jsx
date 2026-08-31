export const benchmarks = [
  {
    match: { hw: "gb300", quant: "fp8", strategy: "low-latency" },
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
    match: { hw: "gb300", quant: "fp8", strategy: "low-latency", kvDsaPair: "fp8-trtllm" },
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
    match: { hw: "gb300", quant: "fp8", strategy: "low-latency", kvDsaPair: "fp8-trtllm", dcp: "4" },
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
        ttft_ms: 534.1,
        tpot_ms: 7.31,
        tokens_per_sec_per_gpu: 2100.76,
      },
    ],
    notes:
      "The Low Latency recipe with FP8 KV + TRT-LLM DSA and DCP4 (--dcp-size 4 --dcp-comm-backend a2a --dcp-replicate-q-proj) on 4x GB300, final weights (c5b82b63e37b) on the d6ab04bdf1 tree, adaptive MTP 5/1/6 with full decode graph: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,680.61 aggregate output tok/s at a 3.937 accept length — about 10% below the non-DCP FP8 Low Latency row. TRT-LLM DSA DCP decode returns the LSE natively, so this arm needs no patch.",
  },
  {
    match: { hw: "gb300", quant: "fp8", strategy: "low-latency", kvDsaPair: "bf16-tilelang", dcp: "4" },
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
        ttft_ms: 510.0,
        tpot_ms: 8.0,
        tokens_per_sec_per_gpu: 1957.25,
      },
    ],
    notes:
      "The Low Latency recipe with BF16 KV + TileLang DSA and DCP4 on 4x GB300, final weights (c5b82b63e37b) on the d6ab04bdf1 tree, adaptive MTP 5/1/6 with full decode graph: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,565.8 aggregate output tok/s at a 3.90 accept length. TileLang DSA DCP decode needs the LSE fix that ships in the current release image.",
  },
  {
    match: { hw: "gb300", quant: "fp8", strategy: "high-throughput" },
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
    match: { hw: "gb300", quant: "fp8", strategy: "high-throughput", kvDsaPair: "fp8-trtllm" },
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
  {
    match: { hw: "h100", quant: "fp8", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H100 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27%. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "h100", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H100 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.50% for the recommended selection; 97.27-97.50% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "h200", quant: "fp8", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.04 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.04%. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "h200", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.35 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.35% for the recommended selection; 97.19-97.57% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "b200", quant: "fp8", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27% for the recommended selection; 97.12-97.27% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "b200", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27% for the recommended selection; 96.97-97.35% across all 8 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "b300", quant: "fp8", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 96.82 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B300 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 96.82% for the recommended selection; 96.82-97.27% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "b300", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 96.97 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B300 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 96.97% for the recommended selection; 96.97-97.04% across all 8 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  { match: { hw: "gb200", quant: "fp8", strategy: "low-latency" } },
  { match: { hw: "gb200", quant: "fp8", strategy: "high-throughput" } },
  {
    match: { hw: "mi300x", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "aa8c950a3d",
    accuracy: { gsm8k_pct: 97.04 },
    notes:
      "Accuracy-only validation on 8x MI300X (gfx942, TP8/EP8) with zai-org/GLM-5.3-Flash revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a and SGLang PR #36507 commit aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e. The exact Triton MoE/full-decode-graph command captured batch sizes 1 and 32 on all eight ranks. Full GSM8K scored 1,280/1,319 with thinking enabled, temperature 1.0, top-p 0.95, a 32,768-token limit, and 32 evaluator threads. All 1,319 IDs were unique and finished with stop; there were zero missing records, duplicate IDs, empty generations, evaluator errors, truncations, server faults, or graph-false decode entries, and 512 decode entries used captured graphs. No throughput or latency benchmark was run.",
  },
  {
    match: { hw: "mi325x", quant: "fp8", strategy: "high-throughput" },
    sglang_version: "aa8c950a3d",
    accuracy: { gsm8k_pct: 97.04 },
    notes:
      "Architecture-equivalent verification from the exact 8x MI300X gfx942 TP8/EP8 measurement: zai-org/GLM-5.3-Flash revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a on SGLang PR #36507 commit aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e scored 1,280/1,319 (97.04%) on full GSM8K with full decode graphs. MI325X uses the same gfx942 runtime path and has greater HBM capacity, so the recipe is carried as verified; this result was not measured in a separate MI325X run. No throughput or latency benchmark was run.",
  },
  { match: { hw: "mi350x", quant: "fp8", strategy: "high-throughput" } },
  { match: { hw: "mi355x", quant: "fp8", strategy: "high-throughput" } },
  {
    match: { hw: "mi350x", quant: "mxfp4", strategy: "mxfp4-tp4" },
    sglang_version: "654df43cbe",
    accuracy: { gsm8k_pct: 97.19 },
    notes:
      "Accuracy-only validation on 4x MI350X (gfx950, TP4) with amd/GLM-5.3-Flash-Quark-MXFP4 revision fc676278b68ba33f6b4724be286af9d7f6c814c2 and SGLang PR #36607 commit 654df43cbee108a81fa1736c34ba8c701f199285. Full GSM8K scored 1,282/1,319 with thinking enabled and a 100% stop rate. There were zero duplicate IDs, empty generations, evaluator errors, truncations, or server faults. No throughput or latency benchmark was run.",
  },
  {
    match: { hw: "mi350x", quant: "mxfp4", strategy: "mxfp4-tp8-ep8" },
    sglang_version: "654df43cbe",
    accuracy: { gsm8k_pct: 97.12 },
    notes:
      "Accuracy-only validation on 8x MI350X (gfx950, TP8/EP8) with amd/GLM-5.3-Flash-Quark-MXFP4 revision fc676278b68ba33f6b4724be286af9d7f6c814c2 and SGLang PR #36607 commit 654df43cbee108a81fa1736c34ba8c701f199285. Full GSM8K scored 1,281/1,319 with thinking enabled and a 100% stop rate. There were zero duplicate IDs, empty generations, evaluator errors, truncations, or server faults; 415 decode entries used captured graphs. No throughput or latency benchmark was run.",
  },
];
