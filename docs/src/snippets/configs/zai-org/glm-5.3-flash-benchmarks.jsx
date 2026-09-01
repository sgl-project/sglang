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
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "fp8-trtllm", quant: "fp8" },
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
      "The Low Latency recipe with FP8 KV + TRT-LLM DSA and DCP4 (--dcp-size 4 --dcp-comm-backend a2a --dcp-replicate-q-proj) on 4x GB300, final weights (c5b82b63e37b) on the d6ab04bdf1 tree, adaptive MTP 5/1/6 with full decode graph: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,680.61 aggregate output tok/s at a 3.937 accept length — about 10% below the non-DCP FP8 Low Latency row. TRT-LLM DSA DCP decode returns the LSE natively, so this arm needs no patch.",
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
      "The Low Latency recipe with BF16 KV + TileLang DSA and DCP4 on 4x GB300, final weights (c5b82b63e37b) on the d6ab04bdf1 tree, adaptive MTP 5/1/6 with full decode graph: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,565.8 aggregate output tok/s at a 3.90 accept length. TileLang DSA DCP decode needs the LSE fix that ships in the current release image.",
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
      "FP8 KV cache with TRT-LLM DSA on 4x GB300, final weights (c5b82b63e37b) on the current release-image tree (d6ab04bdf1), same protocol as the BF16 rows: 1,227.07 / 2,738.61 / 4,977.02 aggregate output tok/s at concurrency 16 / 64 / 256 — 2.9–5.7% above BF16 + TileLang across the curve, and the FP8 pool holds 12.6M tokens per rank vs 7.0M at BF16 (1.8x capacity at identical pool bytes). Accuracy is the full GSM8K gate on this variant: 97.35% vs 97.50% on BF16 KV, a 0.15-point gap inside sampling noise, with a 100% stop rate over all 1,319 problems. With HiCache L1+L2 (32 GB host tier, 16k prefill chunks) the same protocol measured 1,263.85 / 2,763.31 / 4,773.95 tok/s — within 5% of the non-HiCache rows; the random dataset has no prefix reuse, so L2 benefit was not exercised.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4", kvDsaPair: "bf16-tilelang" },
    sglang_version: "033446bb05",
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
        ttft_ms: 148.18,
        tpot_ms: 3.84,
        tokens_per_sec_per_gpu: 256.69,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 242.66,
        tpot_ms: 16.95,
        tokens_per_sec_per_gpu: 1022.61,
      },
    ],
    accuracy: { gsm8k_pct: 97.14, aime2026_pct: 92.45 },
    notes:
      "RadixArk/GLM-5.3-Flash-NVFP4 — NVFP4 W4A4 post-training quantization of zai-org/GLM-5.3-Flash-BF16 with NVIDIA Model Optimizer 0.46.0 (abs-max scaling, group size 16): routed and shared experts plus the dense MLPs are FP4, while all attention (KDA, DSA indexer, MLA), the router, norms, the vision tower, the MTP layer, embeddings, and the LM head stay BF16. Measured on 4x GB300 with the lmsysorg/sglang:glm-5.3-flash image (PR #36507 head 033446bb05), adaptive MTP 5/1/6, BF16 KV + TileLang DSA, and the flashinfer_cutlass MoE runner. GSM8K 97.14% over the full 1,319-example split x 4 seeds (per-seed range 96.89-97.42%, stop rate 99.85-100%) and AIME 2026 92.45% (30 problems x 16 repeats x 4 seeds = 1,920 generations, per-seed range 91.67-93.54%), both at temperature 1.0 / top_p 0.95. The accuracy runs used the NEXTN spelling of --speculative-algorithm, which resolves to the same runtime path as the published EAGLE command on this tree. The speed row was measured on the unpatched docker tree with SGLANG_SIMULATE_ACC_LEN=3 pinning the accept length (confirmed 2.98): 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 818.09 aggregate output tok/s after two discarded warmups. Simulated accept length makes this a throughput-mechanism number; a same-pod A/B against the SwiGLU-fusion-disabled tree landed within ±3%, so the unpatched numbers are the published ones. The companion GSM8K gate on this stack scored 97.27% with a 99.92% stop rate over all 1,319 problems. The gap to the fp8 rows (~45%) is recipe-level, not kernel-isolated: flashinfer_cutlass + BF16 KV + TileLang DSA + TP4 without EP vs the fp8 cells' deep_gemm + FP8 KV + TRT-LLM DSA + EP4. The concurrency-1 entry uses the same protocol (8 requests, simulated accept 3.00); the fp8-trtllm pairing is ~7% ahead on decode at c1, consistent with the c16 ranking.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4", kvDsaPair: "bf16-tilelang" },
    sglang_version: "033446bb05",
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
        ttft_ms: 188.6,
        tpot_ms: 22.01,
        tokens_per_sec_per_gpu: 786.79,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 255.67,
        tpot_ms: 56.88,
        tokens_per_sec_per_gpu: 1317.68,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 413.08,
        tpot_ms: 123.03,
        tokens_per_sec_per_gpu: 2518.95,
      },
    ],
    accuracy: { gsm8k_pct: 97.14, aime2026_pct: 92.45 },
    notes:
      "RadixArk/GLM-5.3-Flash-NVFP4 with speculative decoding off — same checkpoint, image, and 4x GB300 measurement stack as the NVFP4 Low Latency row (ModelOpt 0.46.0 NVFP4 W4A4, abs-max, group size 16; MoE and dense MLPs in FP4, attention/router/MTP/embeddings BF16). Accuracy is a checkpoint-level result carried from that arm: GSM8K 97.14% over the full 1,319-example split x 4 seeds (per-seed range 96.89-97.42%, stop rate 99.85-100%) and AIME 2026 92.45% (30 problems x 16 repeats x 4 seeds, per-seed range 91.67-93.54%). Those runs used the NEXTN spelling of --speculative-algorithm on the adaptive-MTP arm, which resolves to the same runtime path as the published EAGLE command. The speed rows were measured on the unpatched docker tree after two discarded warmups per row: 629.43 / 1,054.14 / 2,015.16 aggregate output tok/s at concurrency 16 / 64 / 256 (80 / 320 / 1,280 random requests at 1,024 input / 256 output tokens), with decode on CUDA graphs through bs256 — the published cell command carries no --cuda-graph-max-bs cap. The companion GSM8K gate on this stack scored 97.27% with a 99.92% stop rate over all 1,319 problems. The gap to the fp8 rows (~40-55%) is recipe-level, not kernel-isolated: flashinfer_cutlass + BF16 KV + TileLang DSA + TP4 without EP vs the fp8 cells' deep_gemm + FP8 KV + TRT-LLM DSA + EP4.",
  },
  {
    match: { hw: "gb300", strategy: "low-latency", quant: "nvfp4", kvDsaPair: "fp8-trtllm" },
    sglang_version: "033446bb05",
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
        ttft_ms: 143.16,
        tpot_ms: 3.55,
        tokens_per_sec_per_gpu: 274.63,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 16,
          num_prompts: 80,
        },
        ttft_ms: 143.81,
        tpot_ms: 12.42,
        tokens_per_sec_per_gpu: 1424.79,
      },
    ],
    notes:
      "FP8 KV + TRT-LLM DSA pairing of the NVFP4 recipe on 4x B300 (SXM6), stock unpatched lmsysorg/sglang:glm-5.3-flash image (tree 033446bb05), TP4-only with the flashinfer_cutlass MoE runner, adaptive MTP 5/1/6 with SGLANG_SIMULATE_ACC_LEN=3: 80 random requests at 1,024 input / 256 output tokens and concurrency 16 produced 1,139.83 aggregate output tok/s after two discarded warmups — 39% above the bf16-tilelang NVFP4 row on the otherwise identical recipe. Simulated accept length makes this a throughput-mechanism number (accept 2.98 confirms the simulation engaged; smoke output is N/A by design under simulated acceptance). EP is not available for this checkpoint on the stock image: --ep-size 4 crashes on the first forward pass (the shared-expert NVFP4 weight arrives 1-D under EP, ValueError in the modelopt_fp4 apply), so the NVFP4 cells are TP-only for now. The concurrency-1 entry uses the same protocol (8 requests, simulated accept 3.00); it stays ~7% ahead of bf16-tilelang on decode, consistent with the c16 ranking.",
  },
  {
    match: { hw: "gb300", strategy: "high-throughput", quant: "nvfp4", kvDsaPair: "fp8-trtllm" },
    sglang_version: "033446bb05",
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
        ttft_ms: 107.02,
        tpot_ms: 15.98,
        tokens_per_sec_per_gpu: 1079.78,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 64,
          num_prompts: 320,
        },
        ttft_ms: 150.56,
        tpot_ms: 36.94,
        tokens_per_sec_per_gpu: 1998.81,
      },
      {
        workload: {
          dataset: "random",
          isl: 1024,
          osl: 256,
          max_concurrency: 256,
          num_prompts: 1280,
        },
        ttft_ms: 279.77,
        tpot_ms: 81.04,
        tokens_per_sec_per_gpu: 3750.86,
      },
    ],
    notes:
      "FP8 KV + TRT-LLM DSA pairing of the NVFP4 recipe — same 4x B300 (SXM6), stock unpatched image (tree 033446bb05), TP4-only flashinfer_cutlass stack as the NVFP4 fp8-trtllm Low Latency row, speculative decoding off, after two discarded warmups per row: 863.82 / 1,599.05 / 3,000.69 aggregate output tok/s at concurrency 16 / 64 / 256 (80 / 320 / 1,280 random requests at 1,024 input / 256 output tokens), 37-52% above the bf16-tilelang NVFP4 rows on the otherwise identical recipe. EP is not available for this checkpoint on the stock image: --ep-size 4 crashes on the first forward pass (the shared-expert NVFP4 weight arrives 1-D under EP), so the NVFP4 cells are TP-only for now.",
  },
  {
    match: { hw: "h100", strategy: "low-latency" },
    sglang_version: "f040cc72e6",
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x H100 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27%. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
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
    accuracy: { gsm8k_pct: 97.27 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.27% for the recommended selection; 97.12-97.27% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
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
    accuracy: { gsm8k_pct: 96.82 },
    notes:
      "Full GSM8K (all 1,319 problems) on 8x B300 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 96.82% for the recommended selection; 96.82-97.27% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement.",
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
