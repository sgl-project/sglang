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
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 219.15,
        tpot_ms: 3.94,
        tokens_per_sec_per_gpu: 261.03,
      },
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
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) on the current release-image tree (d6ab04bdf1), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept length confirmed 3.00 in the bench summary and server log): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,824.66 aggregate output tok/s after two discarded warmups. Simulated accept length makes this a throughput-mechanism number. Re-verified on the current release image (tree fe236ea6c3) within 1.5% on 4x GB300; the concurrency-1 entry (8 requests) comes from that re-run. Accuracy is from the shared non-simulated full GSM8K gate: 97.50% with a 100% stop rate over all 1,319 problems.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "fp8-trtllm", quant: "fp8" },
    sglang_version: "d6ab04bdf1",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 211.25,
        tpot_ms: 3.65,
        tokens_per_sec_per_gpu: 279.53,
      },
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
      "The Low Latency recipe with FP8 KV + TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) on the current release-image tree (d6ab04bdf1), adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3 (accept 3.00): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,853.88 aggregate output tok/s — 1.6% above the BF16 + TileLang Low Latency row, with mean TPOT 6.25 ms vs 6.43 ms. Draft and target full-graph capture succeeded for this combination. The speed rows were measured with the NEXTN spelling and --disable-shared-experts-fusion, which resolve to the same runtime path as the published command on this tree. Re-verified on the current release image (tree fe236ea6c3) within 1.5% on 4x GB300; the concurrency-1 entry (8 requests) comes from that re-run.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "fp8-trtllm", dcp: "4", quant: "fp8" },
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
      "The Low Latency recipe with FP8 KV + TRT-LLM DSA and DCP4 (--dcp-size 4 --dcp-comm-backend a2a --dcp-replicate-q-proj) on 4x GB300, final weights (c5b82b63e37b) on the d6ab04bdf1 tree, adaptive MTP 5/1/6 with full decode graph: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,680.61 aggregate output tok/s at a 3.937 accept length — about 10% below the non-DCP FP8 Low Latency row. TRT-LLM DSA DCP decode returns the LSE natively, so this arm needs no patch. Re-verified on the current release image (tree fe236ea6c3) within 1.3% on 4x GB300.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "bf16-tilelang", dcp: "4", quant: "fp8" },
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
      "The Low Latency recipe with BF16 KV + TileLang DSA and DCP4 on 4x GB300, final weights (c5b82b63e37b) on the d6ab04bdf1 tree, adaptive MTP 5/1/6 with full decode graph: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,565.8 aggregate output tok/s at a 3.90 accept length. TileLang DSA DCP decode needs the LSE fix that ships in the current release image. Re-verified on the current release image (tree fe236ea6c3) within 1.3% on 4x GB300.",
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
      "Measured on 4x GB300 (TP4/EP4) with the final weights (zai-org/GLM-5.3-Flash, c5b82b63e37b) on the current release-image tree (d6ab04bdf1), speculative decoding off, after two discarded warmups per row: 1,161.22 / 2,660.24 / 4,828.33 aggregate output tok/s at concurrency 16 / 64 / 256 (80 / 320 / 1,280 random requests at 1,024 input / 256 output tokens). The server ran exactly the published cell command. Throughput at 256 is still scaling but sublinear (prefill queueing). Accuracy is from the shared non-simulated full GSM8K gate: 97.50% with a 100% stop rate over all 1,319 problems. With HiCache L1+L2 (32 GB host tier, 16k prefill chunks) the same protocol measured 1,202.07 / 2,696.20 / 4,634.47 tok/s — within 4% of the non-HiCache rows; the random dataset has no prefix reuse, so L2 benefit was not exercised. Re-verified on the current release image (tree fe236ea6c3) within 1.5% on 4x GB300.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", kvDsaPair: "fp8-trtllm", quant: "fp8" },
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
      "FP8 KV cache with TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) on the current release-image tree (d6ab04bdf1), same protocol as the BF16 rows: 1,227.07 / 2,738.61 / 4,977.02 aggregate output tok/s at concurrency 16 / 64 / 256 — 2.9–5.7% above BF16 + TileLang across the curve, and the FP8 pool holds 12.6M tokens per rank vs 7.0M at BF16 (1.8x capacity at identical pool bytes). Accuracy is the full GSM8K gate on this variant: 97.35% vs 97.50% on BF16 KV, a 0.15-point gap inside sampling noise, with a 100% stop rate over all 1,319 problems. With HiCache L1+L2 (32 GB host tier, 16k prefill chunks) the same protocol measured 1,263.85 / 2,763.31 / 4,773.95 tok/s — within 5% of the non-HiCache rows; the random dataset has no prefix reuse, so L2 benefit was not exercised. Re-verified on the current release image (tree fe236ea6c3) within 1.5% on 4x GB300.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4", kvDsaPair: "bf16-tilelang" },
    sglang_version: "fe236ea6c3",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 175.18,
        tpot_ms: 3.84,
        tokens_per_sec_per_gpu: 276.74,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 479.9,
        tpot_ms: 6.33,
        tokens_per_sec_per_gpu: 2437.28,
      },
    ],
    accuracy: { gsm8k_pct: 97.14, aime2026_pct: 92.45 },
    notes:
      "RadixArk/GLM-5.3-Flash-NVFP4 — NVFP4 W4A4 post-training quantization of zai-org/GLM-5.3-Flash-BF16 with NVIDIA Model Optimizer 0.46.0 (abs-max scaling, group size 16): routed and shared experts plus the dense MLPs are FP4, while all attention (KDA, DSA indexer, MLA), the router, norms, the vision tower, the MTP layer, embeddings, and the LM head stay BF16. Accuracy measured on 4x GB300 with the lmsysorg/sglang:glm-5.3-flash image (PR #36507 head 033446bb05), adaptive MTP 5/1/6, BF16 KV + TileLang DSA, and the flashinfer_cutlass MoE runner. GSM8K 97.14% over the full 1,319-example split x 4 seeds (per-seed range 96.89-97.42%, stop rate 99.85-100%) and AIME 2026 92.45% (30 problems x 16 repeats x 4 seeds = 1,920 generations, per-seed range 91.67-93.54%), both at temperature 1.0 / top_p 0.95. The accuracy runs used the NEXTN spelling of --speculative-algorithm, which resolves to the same runtime path as the published EAGLE command on this tree. Speed measured on 4x GB300 with the current release image (tree fe236ea6c3) with SGLANG_SIMULATE_ACC_LEN=3 pinning the accept length (3.00): 8 requests at concurrency 1 and 80 at concurrency 16 (1,024 input / 256 output tokens) produced 221.39 and 1,949.82 aggregate output tok/s after two discarded warmups — the c16 cell lands within 5.3% of the B300 measurement, while the c1 cell runs ~20% lower on the Grace host (a host-latency effect visible only at bs1). Simulated accept length makes these throughput-mechanism numbers.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4", kvDsaPair: "bf16-tilelang" },
    sglang_version: "fe236ea6c3",
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
        ttft_ms: 447.92,
        tpot_ms: 10.09,
        tokens_per_sec_per_gpu: 1691.08,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 1094.05,
        tpot_ms: 15.19,
        tokens_per_sec_per_gpu: 4114.2,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 2890.32,
        tpot_ms: 31.85,
        tokens_per_sec_per_gpu: 7399.36,
      },
    ],
    accuracy: { gsm8k_pct: 97.14, aime2026_pct: 92.45 },
    notes:
      "RadixArk/GLM-5.3-Flash-NVFP4 with speculative decoding off — same checkpoint, image, and 4x GB300 measurement stack as the NVFP4 Low Latency row (ModelOpt 0.46.0 NVFP4 W4A4, abs-max, group size 16; MoE and dense MLPs in FP4, attention/router/MTP/embeddings BF16). Accuracy is a checkpoint-level result carried from that arm: GSM8K 97.14% over the full 1,319-example split x 4 seeds (per-seed range 96.89-97.42%, stop rate 99.85-100%) and AIME 2026 92.45% (30 problems x 16 repeats x 4 seeds, per-seed range 91.67-93.54%). Those runs used the NEXTN spelling of --speculative-algorithm on the adaptive-MTP arm, which resolves to the same runtime path as the published EAGLE command. Speed measured on 4x GB300 with the current release image (tree fe236ea6c3): 80 / 320 / 1,280 random requests at concurrency 16 / 64 / 256 (1,024 input / 256 output tokens) produced 1,352.86 / 3,291.36 / 5,919.49 aggregate output tok/s after two discarded warmups — at or above B300 parity, and above the FP8 gb300 High Throughput row at c256 (5,919.49 vs 4,828.33), as expected for W4A4.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4", kvDsaPair: "fp8-trtllm" },
    sglang_version: "fe236ea6c3",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 176.24,
        tpot_ms: 3.6,
        tokens_per_sec_per_gpu: 291.64,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 475.37,
        tpot_ms: 6.09,
        tokens_per_sec_per_gpu: 2517.1,
      },
    ],
    notes:
      "FP8 KV + TRT-LLM DSA pairing of the NVFP4 recipe, TP4-only with the flashinfer_cutlass MoE runner and adaptive MTP 5/1/6. Speed measured on 4x GB300 with the current release image (tree fe236ea6c3) with SGLANG_SIMULATE_ACC_LEN=3 pinning the accept length (3.00): 8 requests at concurrency 1 and 80 at concurrency 16 (1,024 input / 256 output tokens) produced 233.31 and 2,013.68 aggregate output tok/s after two discarded warmups — the c16 cell lands within 3.4% of the B300 measurement, while the c1 cell runs ~20% lower on the Grace host (a host-latency effect visible only at bs1). Simulated accept length makes these throughput-mechanism numbers.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4", kvDsaPair: "fp8-trtllm" },
    sglang_version: "fe236ea6c3",
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
        ttft_ms: 447.34,
        tpot_ms: 9.38,
        tokens_per_sec_per_gpu: 1799.06,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 1090.95,
        tpot_ms: 14.4,
        tokens_per_sec_per_gpu: 4285.73,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 2891.78,
        tpot_ms: 30.2,
        tokens_per_sec_per_gpu: 7688.08,
      },
    ],
    notes:
      "FP8 KV + TRT-LLM DSA pairing of the NVFP4 recipe with speculative decoding off — same TP4-only flashinfer_cutlass stack as the NVFP4 fp8-trtllm Low Latency row. Speed measured on 4x GB300 with the current release image (tree fe236ea6c3): 80 / 320 / 1,280 random requests at concurrency 16 / 64 / 256 (1,024 input / 256 output tokens) produced 1,439.25 / 3,428.58 / 6,150.46 aggregate output tok/s after two discarded warmups — at or above B300 parity, and above the FP8 gb300 High Throughput row at c256 (6,150.46 vs 4,977.02), as expected for W4A4.",
  },
  {
    match: { hw: "h100", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 8192,
          osl: 1024,
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 345.66,
        tpot_ms: 4.37,
        tokens_per_sec_per_gpu: 239.19,
      },
      {
        workload: {
          dataset: "random",
          isl: 8192,
          osl: 1024,
          max_concurrency: 16,
          num_prompts: 32,
        },
        ttft_ms: 3442.21,
        tpot_ms: 10.24,
        tokens_per_sec_per_gpu: 1323.44,
      },
    ],
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H100 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27%. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Speed was measured by the Z.ai team on the published image lmsysorg/sglang:glm-5.3-flash (sha256:aa9210e3…, tree fe236ea6c3), on the published Low Latency cell recipe with the BF16 KV + TileLang DSA pairing and SGLANG_SIMULATE_ACC_LEN=3 in the environment (accept 3.00 at c1, 2.98 at c16): random 8,192-input / 1,024-output requests (range ratio 1, sglang-oai backend) produced 212.61 aggregate output tok/s at concurrency 1 (8 requests) and 1,176.39 at concurrency 16 (32 requests) after two discarded warmups — simulated accept length makes these throughput-mechanism numbers. With HiCache L1+L2 the same protocol measured 1,206.11 at c16 (TTFT 2,985.08 ms, TPOT 10.35 ms). The retokenized-token count in these runs is about a quarter of the generated tokens because reasoning_content is not retokenized (benign).",
  },
  {
    match: { hw: "h100", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.50 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H100 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.50% for the recommended selection; 97.27-97.50% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "h200", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.04 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.04%. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "h200", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.35 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.35% for the recommended selection; 97.19-97.57% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "b200", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 8192,
          osl: 1024,
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 352.45,
        tpot_ms: 3.75,
        tokens_per_sec_per_gpu: 274.8,
      },
      {
        workload: {
          dataset: "random",
          isl: 8192,
          osl: 1024,
          max_concurrency: 16,
          num_prompts: 32,
        },
        ttft_ms: 2456.16,
        tpot_ms: 6.43,
        tokens_per_sec_per_gpu: 2037.52,
      },
    ],
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27% for the recommended selection; 97.12-97.27% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Speed was measured by the Z.ai team on the published image lmsysorg/sglang:glm-5.3-flash (sha256:aa9210e3…, tree fe236ea6c3), on the published Low Latency cell recipe with the BF16 KV + TileLang DSA pairing and SGLANG_SIMULATE_ACC_LEN=3 in the environment (accept 3.00 at c1, 2.98 at c16): random 8,192-input / 1,024-output requests (range ratio 1, sglang-oai backend) produced 244.27 aggregate output tok/s at concurrency 1 (8 requests) and 1,811.13 at concurrency 16 (32 requests) after two discarded warmups — simulated accept length makes these throughput-mechanism numbers; the FP8 KV + TRT-LLM pairing recommended on Blackwell was not speed-measured on this platform. With HiCache L1+L2 the same protocol measured 258.05 at c1 (TTFT 245.25 ms) and 2,136.62 at c16 (TTFT 1,590.40 ms).",
  },
  {
    match: { hw: "b200", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27% for the recommended selection; 96.97-97.35% across all 8 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  {
    match: { hw: "b300", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    latencyPercentile: "Mean",
    speed: [
      {
        workload: {
          dataset: "random",
          isl: 8192,
          osl: 1024,
          max_concurrency: 1,
          num_prompts: 8,
        },
        ttft_ms: 230.34,
        tpot_ms: 3.37,
        tokens_per_sec_per_gpu: 313.02,
      },
      {
        workload: {
          dataset: "random",
          isl: 8192,
          osl: 1024,
          max_concurrency: 16,
          num_prompts: 32,
        },
        ttft_ms: 1908.02,
        tpot_ms: 5.83,
        tokens_per_sec_per_gpu: 2338.81,
      },
    ],
    accuracy: { gsm8k_pct: 96.82 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B300 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 96.82% for the recommended selection; 96.82-97.27% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Speed was measured by the Z.ai team on the published image lmsysorg/sglang:glm-5.3-flash (sha256:aa9210e3…, tree fe236ea6c3), on the published Low Latency cell recipe with the BF16 KV + TileLang DSA pairing and SGLANG_SIMULATE_ACC_LEN=3 in the environment (accept 3.00 at c1, 2.98 at c16): random 8,192-input / 1,024-output requests (range ratio 1, sglang-oai backend) produced 278.24 aggregate output tok/s at concurrency 1 (8 requests) and 2,078.94 at concurrency 16 (32 requests) after two discarded warmups — simulated accept length makes these throughput-mechanism numbers; the FP8 KV + TRT-LLM pairing recommended on Blackwell was not speed-measured on this platform. With HiCache L1+L2 the same protocol measured 278.01 at c1 (TTFT 214.12 ms) and 2,323.89 at c16 (TTFT 1,433.37 ms).",
  },
  {
    match: { hw: "b300", strategy: "high-throughput" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 96.97 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B300 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 96.97% for the recommended selection; 96.97-97.04% across all 8 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
  },
  { match: { hw: "gb200", strategy: "low-latency" } },
  { match: { hw: "gb200", strategy: "high-throughput" } },
  { match: { hw: "gb200", strategy: "low-latency", quant: "nvfp4" } },
  { match: { hw: "gb200", strategy: "high-throughput", quant: "nvfp4" } },
  { match: { hw: "b200", strategy: "low-latency", quant: "nvfp4" } },
  { match: { hw: "b200", strategy: "high-throughput", quant: "nvfp4" } },
  { match: { hw: "b300", strategy: "low-latency", quant: "nvfp4" } },
  { match: { hw: "b300", strategy: "high-throughput", quant: "nvfp4" } },
];
