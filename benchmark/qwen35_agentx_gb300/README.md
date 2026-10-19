# Qwen3.5-397B-A17B NVFP4 AgentX on GB300

This branch is the SGLang tree for reproducing the AgentX submission of
Qwen3.5-397B-A17B NVFP4 on GB300: aggregated configurations served by SGLang,
and prefill/decode-disaggregated configurations served by SGLang with NVIDIA
Dynamo.

## Recipes

The serving recipes of the submitted points are part of the InferenceX AgentX
submission for this model. A link to them will be added here.

## Container image

The container image is built from this branch with the
`.github/workflows/release-docker-dev.yml` workflow, so the image contains this
tree as is.

## Switches

Every optimization on this branch is behind an `SGLANG_Q35MF_*` environment
switch that is off by default. Most switches are registered in
`python/sglang/srt/environ.py`; the rest are read where they are used.

## FP8 bmm tactic table

`python/sglang/srt/layers/quantization/configs/fp8_bmm_tactics_gb300_tp2.json`
pins the cuBLASLt algorithm of FlashInfer's per-tensor FP8 bmm for the TP2
shapes of this model. To use it, set `SGLANG_Q35MF_FP8_BMM_TACTICS` to the
file's path; in the image that is
`/sgl-workspace/sglang/python/sglang/srt/layers/quantization/configs/fp8_bmm_tactics_gb300_tp2.json`.
The algorithm indices refer to cuBLASLt's heuristic list for each shape, so
they assume the cuBLAS version of that image.
