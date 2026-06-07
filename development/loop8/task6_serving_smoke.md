# Loop 8 / task6 — GLM-5.1 DS serving bring-up smoke (8×H200, 2026-06-07)

Booted the opt-in Double-Sparsity server for GLM-5.1-FP8 on the local 8×H200 with the R3 calibrated mask,
confirmed the inherited bind-time shape gate passes against the real model+mask, and verified DS decode is
coherent. This is the AC-2 serving-side validation + AC-3 serving coherence on hardware.

## Launch (validated)
```bash
MODEL_PATH=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db \
CHANNEL_MASK_PATH=/models/glm51-fp8-channel-mask.safetensors \
PORT=30000 TP_SIZE=8 KV_CACHE_DTYPE=fp8_e4m3 PAGE_SIZE=64 TOP_K=2048 \
MEM_FRACTION_STATIC=0.7 \
bash development/serve_double_sparsity.sh
```
- `attention_backend='dsa'`, `enable_double_sparsity=True`, DS config `top_k=2048 page_size=64
  signature_dtype=fp16 label_dim(from mask)=32`. `mem-fraction-static 0.7` fits GLM-5.1 (no OOM):
  KV cache fp8 142208 tokens (8.14 GB), ~17 GB GPU headroom after graph capture. Server log:
  `runs/20260607_glm51_loop8/glm_ds_serve.log`.

## Bind-time shape gate — PASS on real GLM-5.1 (AC-2)
All 78 layers × 8 ranks logged (the R0 `verify_bind_shapes` gate firing against the real model + real mask):
```
double_sparsity bind shape check passed (layer N, tp_rank R): qk_nope_head_dim=192 qk_rope_head_dim=64
  v_head_dim=256 kv_lora_rank=512 num_local_heads=8 layers=78 page_size=64 label_dim=32 head_dim=192
  kv_dtype=fp8_e4m3
double_sparsity bind_runtime_data completed: ... num_local_heads=8 label_dim=32 page_size=64
  process_group_size=8 ...
```
The full GLM shape set matches the config + the mask; the inherited DS wiring bound to GLM with no
standalone backend and no duplicated hooks. Server reached "fired up and ready to roll!".

## DS decode smoke — coherent (AC-3)
- `"The capital of France is"` → `" Paris. The city is located on the River Seine in the north of the
  country. It is the largest and most important"` (correct prefill + coherent decode).
- `"List the planets … one sentence about each:"` → `" Mercury, Venus, Earth, Mars, …, Neptune, and Pluto.
  1. Mercury: The closest planet to the Sun … 2. Venus: … hottest … 3. Earth: the only known planet to
  supp…"` (120 tokens, on-task, coherent — NOT the degenerate repeated-token failure mode).

Decode is coherent on the 32-sample bring-up mask; this confirms the DS read/write hooks (select_topk +
token-label write) function end-to-end on GLM's 192/256 MLA shapes.

## DeepSeek-V3.2 non-regression (AC-5)
- **Unit level (committed, green):** the dual-shape synthetic tests (`128/128` V3.2 + `192/256` GLM) and
  the full DS unit suite (309 passed) cover the V3.2 shapes; `verify_bind_shapes` passes V3.2 dims.
- **Code-level safety:** every Loop-8 shared-hook change is gated under `if use_double_sparsity` (DSA-off
  path untouched) and the `_ds_qk_nope_head_dim` default is now `self.qk_nope_head_dim` (=128 for V3.2 →
  byte-identical to the prior literal 128).
- **Live V3.2 DS smoke: REMAINING** — the V3.2 mask (`/models/dsv32-fp8-channel-mask.safetensors`) is not
  currently staged (the V3.2 model IS at `/cluster-storage/models/deepseek-ai/DeepSeek-V3.2`). Regenerate
  the V3.2 mask (same `calibrate.py` flow, `--label-dim 16`) then boot V3.2 DS for the live bind+decode
  smoke. Tracked as the open AC-5 hardware item.

## Status
task6 GLM-5.1 DS serving smoke: **DONE** (bind gate + coherent DS decode on hardware). V3.2 live DS smoke:
remaining (mask not staged). Unblocks task7 (AC-1 DS-off byte-identity) and task9 (DS-vs-DSA gates).
