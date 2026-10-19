# Preserve Step5 BF16 experts

Step5 now honors the checkpoint's expert exclusions in `quantization_config.ignored_layers` instead of quantizing them to FP8 during loading. The current checkpoint excludes expert layers 88–90; selection follows metadata, not fixed layer numbers. Both expert matrices must have matching precision. Unexpected BF16-to-FP8 conversion fails with an actionable error.

`SGLANG_STEP5_QUANTIZE_BF16_MOE=1` explicitly restores load-time FP8 quantization. Its default is `0`. The same policy covers the expert layers a ModelOpt NVFP4 export left in BF16 (`quantization_config.exclude_modules`).

The variable only selects how checkpoint BF16 experts are served. Weight updates after the checkpoint load re-quantize BF16 weights sent for FP8 experts to 128x128 block FP8 (with scales) either way, and refuse FP8 weights sent for BF16 experts.

## MegaMoE with TRTLLM BF16 experts

Build the FlashInfer 0.7.1 overlay using the [existing builder](README.md#isolated-build). Patch 04 and its source hashes are included in the manifest. The main Dockerfiles do not apply this patch automatically; a deployment image must include the patched package and rebuilt module. Do not reuse an older AOT MoE binary.

Keep the existing launch arguments and select:

```bash
export SGLANG_STEP5_QUANTIZE_BF16_MOE=0
# sglang serve ... --moe-a2a-backend megamoe \
#   --moe-runner-backend flashinfer_trtllm
```

Retained BF16 experts use TRTLLM automatically; no BF16 backend environment variable is needed. Missing FlashInfer SwigluStep support fails startup. For the validated B300 TP4 configuration with attention DP=1:

- FP8 experts use `forward_mega_moe_local_experts`.
- Checkpoint BF16 experts bypass MegaMoE and use the routed TRTLLM BF16 runner, with StandardDispatcher and Step5's existing reduction of local expert contributions.
- MegaMoE may normalize the global runner to `deep_gemm` and EP to TP. The BF16 layer's explicit runner selection remains TRTLLM.

The BF16 implementation uses a BF16 FC1 result with logical `[up, gate]` layout, FP32 `min(silu(gate), limit) * clamp(up, -limit, limit)`, BF16 activation storage, and BF16 FC2. It does not introduce FP8 expert weights or activations. Precomputed routing weights stay FP32. This preserves the intended formula and cast boundaries, not bitwise equality across GEMM backends.

## Validation

The B300 tests cover six configurations: single token, unaligned rows, nonzero EP offset, clamp boundaries, 4096/1536 matrix dimensions, and PDL disabled. They compare against a PyTorch BF16 reference and require exact eager/graph replay agreement. Run against the built overlay:

```bash
python docker/patches/step5/test_bf16_swiglu_step.py
```

Full TP4 startup, CUDA graph capture, single/batched generation, and six `bench_one_batch_server` measurements passed with MegaMoE and BF16 tail layers. The synthetic performance runs used random IDs, seed 42 and simulated acceptance length 2.6; this does not establish real MTP acceptance or model accuracy.

| Input / output / batch | Earlier FP8 tail mean | BF16 tail mean | Change |
|---|---:|---:|---:|
| 128K / 1K / 1 | 7.188 s | 7.212 s | +0.33% |
| 8K / 1K / 64 | 19.767 s | 20.025 s | +1.31% |
| 160K / 3K / 16, 95% cache hit | 25.144 s | 25.165 s | +0.08% |

Each mean has two runs. Both performance variants also included separately developed router and Q/W packing optimizations, which are not part of this change. The FP8 baseline was measured earlier, not interleaved; small differences may be normal variation. Validation does not cover Hopper or attention DP>1.
