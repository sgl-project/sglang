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
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "fp8-trtllm", dcp: "4" },
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
    match: { hw: "gb300", strategy: "low-latency", kvDsaPair: "bf16-tilelang", dcp: "4" },
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
      "Full GSM8K (all 1,319 problems) on 8x H100 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.50% for the recommended selection; 97.27-97.50% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. Accuracy only, no speed measurement. " +
      "DSA backend selection on this cell is NOT MEASURED ON H100. The `--dsa-*-backend tilelang` pin was dropped here on two grounds: (1) `_dsa_split_backend_resolution` in python/sglang/srt/arg_groups/overrides.py branches only on `torch.cuda.get_device_capability()[0]`, and H100 and H200 are both SM90, so auto-detection resolves to the identical flashmla_sparse + fa3 pair on either card; (2) the swap was measured on 8x H200 at +9.0% output tok/s / -12.7% TTFT p95 with GSM8K parity. Both cards are the same GH100 die and differ only in HBM, so the kernel path is the same, but no H100 node was available to confirm the delta or re-run the accuracy gate. Treat the H100 speed claim as inherited from H200, not verified. " +
      "Note also that 80 GB cards are materially more pool-constrained than the 141 GB H200: after ~38 GB/rank of weights and the fp32 KDA state pool, an 8x H100 node serves roughly 36-45 concurrent requests at ~30k context (derived, not measured), so `--mamba-full-memory-ratio` matters more here than on H200 -- see the note on the h200 cell.",
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
      "Full GSM8K (all 1,319 problems) on 8x H200 (TP8/EP8) with zai-org/GLM-5.3-Flash at f040cc72e6: 97.35% for the recommended selection; 97.19-97.57% across all 4 measured selections. Run with `sgl-eval run gsm8k --base-url http://localhost:30000/v1 --num-threads 32 --max-tokens 32768`; gsm8k's registered default leaves thinking off, so these are non-thinking numbers and are not directly comparable to the GB300 rows above. " +
      "DSA backend selection (measured on 8x H200 TP8/EP8, real weights, lmsysorg/sglang:glm-5.3-flash, warmed plateau with the first rep discarded, 24k shared-prefix 3-turn traffic at 16k chunked prefill): leaving DSA to auto-detection -- which resolves to flashmla_sparse prefill + fa3 decode on SM90 -- gave 589.9 output tok/s at concurrency 32 and 346.5 at 128, versus 541.1 and 299.8 with the previously published `--dsa-*-backend tilelang` pin: +9.0% and +15.6%, TTFT p95 20.98s vs 24.04s (-12.7%). Rep-to-rep spread 0.07-0.26%. " +
      "Accuracy parity for that swap was checked on the same box and weights with a 5-shot completion GSM8K over all 1,319 problems: 91.9% auto vs 92.2% pinned, a 0.3-point gap against a 0.75-point binomial standard error, zero invalid generations in either arm. That harness is completion-style and is NOT comparable to the 97.35% chat-style figure above; it is an A/B control only. " +
      "The gain is context-dependent: on random 1024/256 the same swap is worth only +1.4% to +5.1%, consistent with index_topk=2048 exceeding a 1k prompt entirely. " +
      "Measured and deliberately NOT changed: `--moe-runner-backend triton` is 6.3% (c=32) and 11.9% (c=128) slower than deep_gemm on this workload even after tuning a fused-MoE config for this model's E=36/N=2048 geometry, so deep_gemm remains the recommendation.",
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
  {
    match: { hw: "mi300x", strategy: "high-throughput" },
    sglang_version: "9e692c9216",
    accuracy: { gsm8k_pct: 97.35 },
    notes:
      "Accuracy-only validation on 8x MI300X (gfx942, TP8) with zai-org/GLM-5.3-Flash revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a, SGLang PR #36607 head 9e692c9216c3b5e5c443fecf6b995700eb68d2e4 (validated source manifest 2c240e0e01d5fdf04acc485ebfa25f8a1793ba45fb07f165eecedfba7ec1db80), and lmsysorg/sglang:v0.5.18-rocm720-mi30x with the PR source mounted over the image tree. Full GSM8K scored 1,284/1,319 with a 100% stop rate and zero request errors, empty generations, or truncations. No throughput or latency benchmark was run.",
  },
  { match: { hw: "mi325x", strategy: "high-throughput" } },
  {
    match: { hw: "mi355x", strategy: "high-throughput" },
    sglang_version: "9e692c9216",
    accuracy: { gsm8k_pct: 97.65 },
    notes:
      "Accuracy-only validation on 8x MI355X (gfx950, TP8) with zai-org/GLM-5.3-Flash revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a, SGLang PR #36607 head 9e692c9216c3b5e5c443fecf6b995700eb68d2e4 (validated source manifest 2c240e0e01d5fdf04acc485ebfa25f8a1793ba45fb07f165eecedfba7ec1db80), and lmsysorg/sglang:v0.5.18-rocm720-mi35x with the PR source mounted over the image tree. Full GSM8K scored 1,288/1,319 with a 100% stop rate and zero request errors, empty generations, or truncations. No throughput or latency benchmark was run.",
  },
];
