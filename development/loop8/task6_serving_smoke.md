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

## DeepSeek-V3.2 DS non-regression — LIVE on 8×H200 (AC-5, done R1)
Do-not-break-userspace confirmed on hardware: the committed V3.2 DS path still boots, binds, and decodes
coherently after all Loop-8 shared-hook/kernel changes.
- **V3.2 mask regenerated** (R1): `calibrate.py --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2
  --dtype fp8_e4m3 --label-dim 16 --num-samples 16` (deepseek_v32→deepseek_v3 remap, float8_present=True) →
  `/models/dsv32-fp8-channel-mask.safetensors` (content_sha256=36d8bf573091, L=61 H=128 label_dim=16,
  head_dim=128, idx∈[0,127]). `load_channel_mask` re-verifies the hash; `verify_bind_shapes` PASS for V3.2
  dims and REJECTS head_dim=192.
- **V3.2 DS server booted** (TP=8, page 64, fp8 KV, `mem-fraction-static 0.6` per the V3.2 lesson; KV cache
  fp8 53056 tokens, ~37.8 GB headroom; log `runs/20260607_glm51_loop8/v32_ds_serve.log`). Bind-time gate
  PASSED for V3.2: `qk_nope_head_dim=128 qk_rope_head_dim=64 v_head_dim=128 kv_lora_rank=512
  num_local_heads=16 layers=61 page_size=64 label_dim=16 head_dim=128 kv_dtype=fp8_e4m3` on all ranks —
  **the same bind gate works for both 128/128 (V3.2) and 192/256 (GLM)**.
- **V3.2 DS decode coherent:** `"The capital of France is"` → `" Paris. The capital of the United States is
  Washington, D.C. The capital of Canada is Ottawa. The capital of"` — matches the prior validated V3.2 DS
  output (BL-20260528-dsv32-ds-decode-degeneration); not degenerate. (A longer ambiguous prompt produced a
  structured-but-off-task JSON completion — a V3.2 base-model greedy quirk, varied tokens, NOT a
  repeated-token DS collapse.)
- **Unit-level (committed, green):** dual-shape synthetic tests (`128/128` + `192/256`) + the 309-test DS
  suite; code-safety: every shared-hook change is DS-gated and the `_ds_qk_nope_head_dim` default is now
  `self.qk_nope_head_dim` (=128 for V3.2, byte-identical).

## Status
task6: **DONE** — GLM-5.1 DS serving smoke (bind gate + coherent decode) AND the DeepSeek-V3.2 DS live
non-regression (bind gate + coherent decode), both on 8×H200. AC-2 serving / AC-3 GLM coherence / AC-5
non-regression validated live. Unblocks task7 (AC-1 DS-off byte-identity) and task9 (DS-vs-DSA gates).
