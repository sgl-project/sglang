# Loop 8 / task5 — GLM-5.1 channel-mask ARTIFACT provenance (generated on 8×H200, 2026-06-07)

The real calibration artifact was generated and runtime-validated this round (R3), closing the
hardware-gated remainder of AC-3. The CPU recipe/contract (R2) and this artifact together complete the
"calibration recipe + safetensors artifact" task5 deliverable.

## Artifact
- **Path:** `/models/glm51-fp8-channel-mask.safetensors` (1,278,520 bytes)
- **content_sha256:** `e7dbf4c9308f9cd9b2380244e2862eac9c5b70c52889759e4edd5bba33727e3a`
- **Tensor:** `channel_selection` int32 `[78, 64, 32]` (layers × heads × label_dim); `channel_weights`
  float32 same shape; channel indices ∈ `[0, 191]` ⊂ `[0, head_dim=192)`.

## Artifact contract (mask `__metadata__`)
```json
{
  "schema_version": "1",
  "head_dim": "192",          // GLM no-PE width (qk_nope_head_dim) — matches runtime verify
  "label_dim": "32",          // DEC-3 GLM-native (not V3.2's 16)
  "page_size": "64",
  "dtype": "fp8_e4m3",
  "num_samples": "32",        // reduced-sample bring-up artifact; recipe production = 256
  "block_size": "512",
  "seed": "42",
  "dataset_source": "file:runs/20260528_dsv32_mvp/calib_corpus_pileval.txt",
  "created_at": "2026-06-07T12:18:34Z",
  "calibration_source": "real",
  "content_sha256": "e7dbf4c9308f9cd9b2380244e2862eac9c5b70c52889759e4edd5bba33727e3a"
}
```

## Verification (passed)
`load_channel_mask` re-verified `content_sha256` and the `[0,192)` index bound; then:
- `verify_bind_shapes(model_nope_head_dim=192, num_local_heads=8, tp_size=8, num_hidden_layers=78,
  server_page_size=64, server_label_dim=32, server_kv_cache_dtype=fp8_e4m3)` → **PASS**.
- Negative: `verify_bind_shapes(model_nope_head_dim=128, …)` → **REJECTED** naming `head_dim`
  (a V3.2-shaped mask cannot bind to GLM).

## Repro
- Dry-run (hooks fired on all 78 layers, head_dim=192, float8_present=True):
  `runs/20260607_glm51_loop8/calib_dryrun3.log`.
- Full calibration: `runs/20260607_glm51_loop8/calib_full.log`. Command (validated):
  ```bash
  pip install accelerate
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model /cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db \
    --dtype fp8_e4m3 --tp 8 --output /models/glm51-fp8-channel-mask.safetensors \
    --label-dim 32 --page-size 64 \
    --dataset runs/20260528_dsv32_mvp/calib_corpus_pileval.txt \
    --num-samples 32 --block-size 512 -v
  ```

## Notes for the next hardware steps
- This is a **reduced-sample (32) bring-up artifact** — sufficient to bring up + smoke + gate DS on GLM;
  regenerate with `--num-samples 256` for the production mask before the final landing numbers.
- Serve it via the env-overrides in `task5_calibration_recipe.md`. At bind, the R0 `verify_bind_shapes`
  gate will hard-error if the runtime GLM dims ever disagree with this mask.
- Boot-chain facts established this round: `accelerate` required; `expandable_segments` required (fixes
  the FP8 fragmentation OOM); a per-GPU `max_memory` cap is WRONG (forces cpu-offload → FP8 quantizer
  rejects); `glm_moe_dsa` is HF-registered (no `deepseek_v3`-style remap, unlike `deepseek_v32`); the
  Triton finegrained-fp8 fallback engages via `_force_triton_fp8_for_calibration` (DeepGEMM skipped).
