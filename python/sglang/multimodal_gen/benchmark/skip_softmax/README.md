# Skip-Softmax (BLASST) calibration for Wan2.2-T2V-A14B

The `flashinfer_trtllm_skip_softmax` attention backend can be driven either by a
raw `fixed_scale_factor` or by a `target_sparsity` + a calibration file. This
directory holds the offline tool that produces that calibration file.

`fixed_scale_factor` is a raw FlashInfer scale (`threshold = scale_factor /
seq_len`), so the effective sparsity drifts with resolution and is not portable.
Prefer `target_sparsity` + `calibration_path`: the backend uses the calibrated
`scale_factor = a * exp(b * target_sparsity)` to hit the requested sparsity at the
actual sequence length.

## 1. Generate the calibration file

Requires `nvidia-modelopt` and a checkout of the Model Optimizer repo that ships
`examples/diffusers/sparsity/wan22_skip_softmax.py` (this tool reuses that
example's forward-loop / sparse-config builders).

```bash
python calibrate_wan22_skip_softmax.py \
  --modelopt-example /path/Model-Optimizer/examples/diffusers/sparsity/wan22_skip_softmax.py \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --width 1280 --height 720 --num-frames 81 \
  --target-sparsity 0.5 --calib-steps 8 --calib-size 2 \
  --out ./wan22_calib_14b_720p.json
```

Notes:
- **Calibrate at the resolution you serve.** The fitted `(a, b)` are
  resolution dependent and do not transfer across resolutions.
- Output JSON contains per-component `(a, b)` for `transformer` /
  `transformer_2` plus a flat top-level `(a, b)`.
- All flags also accept env vars (`WAN22_CALIB_W/H/F`, `WAN22_CALIB_STEPS`,
  `WAN22_CALIB_SIZE`, `WAN22_CALIB_TARGET`, `WAN22_CALIB_OUT`,
  `WAN22_MODEL_PATH`, `MODELOPT_WAN22_EXAMPLE`).

## 2. Serve with the calibration file

```bash
sglang generate \
  --backend sglang \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --attention-backend flashinfer_trtllm_skip_softmax \
  --attention-backend-config 'target_sparsity=0.5,calibration_path=./wan22_calib_14b_720p.json' \
  --prompt "..."
```

Confirm engagement in the logs (`Using FlashInfer trtllm-gen Skip-Softmax (BLASST)
Attention backend`). Target HW is SM 10.x (B200/B300).
