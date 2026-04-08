# ModelOpt FP8 PR Ready Notes

Use [PR_BODY.md](./PR_BODY.md) as the primary draft PR description.

## Recommended Scope

- Add diffusion-side ModelOpt FP8 checkpoint support in SGLang
- Add a converter for ModelOpt diffusers FP8 exports
- Add trajectory-latent / final-frame quant validation tooling
- Claim FLUX.2 end-to-end FP8 support
- Claim WAN2.2 A14B primary-transformer FP8 exact-nightly benchmark support

## Do Not Overclaim

- Do not claim LTX-2 ModelOpt FP8 support
- Do not collapse WAN2.2 primary-transformer nightly benchmarking into “full WAN2.2 dual-transformer exact-nightly FP8”
- If mentioning dual-transformer WAN2.2 FP8, label it as a validated smoke / recipe path rather than the benchmarked nightly path

## Artifact Roots

- FLUX.2: `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408`
- WAN2.2: `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22`
- Trajectory similarity JSONs on H100:
  - `/tmp/modelopt_accuracy/flux2_similarity.json`
  - `/tmp/modelopt_accuracy/wan22_similarity.json`

## Current Code Areas

| File | Summary |
| --- | --- |
| `python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py` | registers diffusion ModelOpt quantization methods |
| `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py` | adds ModelOpt FP8 config and linear handling |
| `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py` | resolves flat ModelOpt FP8/NVFP4 configs and packed-QKV markers |
| `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py` | disables incompatible offload modes for ModelOpt FP8 |
| `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py` | constrains FLUX.2 packed-QKV loading to packed NVFP4 checkpoints |
| `python/sglang/multimodal_gen/tools/convert_modelopt_fp8_checkpoint.py` | converts ModelOpt FP8 exports into SGLang-loadable checkpoints |
| `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py` | compares BF16 vs quantized trajectories and outputs |
| `python/sglang/multimodal_gen/test/unit/test_diffusion_modelopt_quant.py` | covers quant-config resolution and offload guards |
| `python/sglang/multimodal_gen/test/unit/test_convert_modelopt_fp8_checkpoint.py` | covers FP8 conversion and BF16 fallback behavior |
| `python/sglang/multimodal_gen/test/unit/test_compare_diffusion_trajectory_similarity.py` | covers similarity helper behavior |
