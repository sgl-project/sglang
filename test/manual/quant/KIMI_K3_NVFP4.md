# Kimi K3 NVFP4 experimental optimizations

This branch collects five independently controlled optimizations for independent testing on NVIDIA B300 (SM103), targeting `nvidia/Kimi-K3-NVFP4`. It is a draft integration, not a validated all-options combination. Install SGLang from the repository root with `pip install -e "python"`. The tested Cake adapter uses FlashInfer 0.7.2 private prepared-stage APIs and requires the Cake modules in that installation.

Routing/quant fusion and the aligned KV projection CUTLASS path are enabled by default. Cake, CuTe DCP prefill and BCG MHA remain disabled by default. Set the corresponding environment variable to `0` to disable a path. All runtime coverage guards still apply.

## Switches and coverage

| Environment variable (set to `1`) | Coverage | Fallback / interaction |
|---|---|---|
| `SGLANG_K3_NVFP4_ROUTE_QUANT` | TRTLLM-layout NVFP4 MoE, EP1, SM103, 1–64 activation rows, hidden size 3584; BF16 or FP32 input | Uncovered routing uses the existing quantization path. Disabled when Cake is enabled. FP8 KV cache is compatible. |
| `SGLANG_K3_NVFP4_CAKE` | TP8/EP1 NVFP4 SiTU MoE; activation row counts 1, 8, 16, 32, 64, 128; equal gate/up dequant scales | Other shapes use the original runner. Replaces the routing/quant candidate rather than combining with it. Warm supported shapes before CUDA graph capture. |
| `SGLANG_K3_KV_B_PROJ_CUTLASS` | SM103 block128 FP8 GEMM, BF16 input, N=3072, K=512, 1536≤M≤131072, M divisible by128, original backend TRTLLM | Selects scale layout before GEMM. Unaligned tails and small decode shapes keep the original runner. Applicable when prefill actually executes this projection, independently of KV-cache dtype or BCG. |
| `SGLANG_K3_DCP_CUTE_PREFILL` | BCG ordinary extend with DCP metadata, MLA dimensions 512+64 | Uses gathered KV with FlashInfer `cute-dsl-monolithic`. No effect for DCP1. This is an alternative MLA prefill path, not the MHA path below. |
| `SGLANG_K3_BCG_MHA` | Kimi K3, BCG ordinary extend, TP8 with BF16 KV/DCP8 or FP8 E4M3 KV/DCP1, BF16 activations, no attention CP/DP, chunked prefix cache enabled | Dynamic MHA/KV writes between graph segments. Other dtype/parallel configurations retain the existing attention selection. |

Row counts refer to actual GEMM/verify activation rows, not request batch size or accepted output length. All switches must be set before starting the server. Enabling an option does not prove it was used; confirm the kernel path in a separate profile.

## Independent comparisons

Keep the launch command, checkpoint, prompts, cache counts and other environment settings unchanged. Set all five switches to `0` for the baseline; unsetting a variable uses its default. For TP8/FP8-KV/DFlash block8 without DCP, run these candidates separately:

```bash
export SGLANG_K3_NVFP4_ROUTE_QUANT=1  # candidate A; other switches explicitly set to 0
# Run your unchanged launch command and benchmark.
```

```bash
export SGLANG_K3_NVFP4_ROUTE_QUANT=0
export SGLANG_K3_NVFP4_CAKE=1        # candidate B; requires Cake APIs
```

For prefill, set both MoE switches to `0` and separately enable `SGLANG_K3_KV_B_PROJ_CUTLASS=1`. Confirm that the original FP8 GEMM backend is TRTLLM and the measured request reaches the eligible KV projection. CuTe DCP prefill requires a separately labeled DCP configuration. The MHA candidate supports TP8 with either FP8 E4M3 KV/DCP1 or BF16 KV/DCP8 and requires explicitly enabled BCG (`--cuda-graph-backend-prefill breakable`) and `SGLANG_K3_BCG_MHA=1`. When comparing MHA and CuTe MLA, enable only the corresponding attention switch. The combined MHA plus KV-CUTLASS candidate is a separate run.

Use `bench_one_batch_server` for isolated prefill and steady decode, output length1024 and fixed acceptance4 for performance. Measure 8k/32k/128k, and prefill with cache0 and approximately95%, recording actual cache counts. Use natural acceptance for accuracy. A single-node isolated comparison does not establish PD transfer or multi-turn SLO performance; PD hardware is not required to test these kernel paths.

## Regression checks

```bash
python -m pytest -q test/manual/quant/test_kimi_k3_nvfp4_handoff.py
python -m pytest -q test/manual/quant/test_kimi_k3_nvfp4_route.py
python -m pytest -q test/manual/quant/test_kimi_k3_kv_projection.py
python -m pytest -q test/manual/quant/test_kimi_k3_bcg_mha_rows.py
```

The handoff test checks consume-once behavior, scale/layout identity and fallback. The GPU tests cover exact routing/FP4 bytes, changed-input CUDA graph replay, and KV projection dispatch/accuracy with nonuniform scales. Cake and the attention integrations still need full-model startup, replay and quality checks on this main-based branch. These tests are not substitutes for model accuracy checks.

Historical measurements came from earlier experimental branches: routing/quant improved 8k/B1 no-spec decode from101.89 to104.81 tok/s; Cake+CuTe improved fixed-acceptance decode by2.68–4.57% and one128k cached-prefill case from2221.9 to762.5ms. A later BF16-KV/DCP8 comparison improved128k cached-prefill from705.1 to557.6ms with MHA+BCG and then496.1ms with KV CUTLASS. These are separate experiments, not additive gains or results for this exact branch/FP8-KV/DCP1 configuration.

FP8-KV/DCP1 BCG MHA validation is in progress. Startup and padded graph replay succeeded after matching KV write locations to the real token count; the five row/metadata regression checks passed. Cached retrieval passed 6/6 cases at 8k and 6/6 at 32k with natural DFlash acceptance. Full quality and matched performance comparisons are pending; this option remains disabled by default.
