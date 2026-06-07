# Loop 8 / task5 (CPU portion) — GLM-5.1 channel-mask calibration recipe + artifact contract

**Routing:** `coding`. This is the **CPU-authorable** part of task5: the calibration recipe, the DEC-3
`label_dim` decision, the artifact-contract field list, and a CPU round-trip proof of the contract. The
**artifact itself** (the offline FP8 forward that collects `q_b_proj`/`kv_b_proj` activations) requires
8×H200 + the GLM-5.1 weights and remains deferred to a hardware round.

## DEC-3 — GLM-native `label_dim` = **32** (not the DeepSeek-V3.2 value 16)

Decision: **`label_dim = 32`**. Justification:
- **GLM-native, not 16** (satisfies DEC-3): chosen from GLM's wider MLA shapes, not reused from V3.2.
- **Free relative to 24 in the kernel.** The selection kernels launch with
  `LABEL_DIM_POW2 = _next_pow2(label_dim)` (`selection_kernel.py:336`). `_next_pow2(24) == _next_pow2(32)
  == 32`, so 24 and 32 cost the **same** Triton work — 24 just wastes 8 padded lanes. 32 uses them.
- **Proportionate to the wider no-PE head.** V3.2 used `16/128 = 1/8` of the no-PE width. GLM's no-PE width
  is `qk_nope_head_dim = 192`; `32/192 ≈ 1/6` — slightly richer than V3.2, sensible because (a) the head is
  wider and (b) DS on GLM is the long-context recall fallback, where more retained channels aid recall.
- **In range:** `32 ≤ 192` (`calibrate.py` rejects `--label-dim > head_dim`).
- **Memory note (for the hardware round):** the per-rank `TokenLabelTable` scales with `label_dim`, so 32
  is 2× the V3.2 label width (16). At GLM's 78 layers + KV-slot count this enlarges the table; size it into
  `--mem-fraction-static` on the serve step (V3.2 used 0.6; GLM weights differ — re-tune on hardware).

## Calibration HF-load readiness (CPU-verified — no code change needed)

Unlike `deepseek_v32` (which transformers does not register, forcing the `deepseek_v32 → deepseek_v3`
remap in `calibrate._resolve_calibration_config`), **`glm_moe_dsa` IS registered in transformers 5.8.1**:
- `CONFIG_MAPPING['glm_moe_dsa'] → GlmMoeDsaConfig`, so `AutoConfig.from_pretrained(..., trust_remote_code=
  True)` (the `calibrate._resolve_calibration_config` fall-through, `calibrate.py:317`) loads GLM-5.1
  directly — **no remap branch is required**.
- `GlmMoeDsaConfig` surfaces the MLA dims under the **exact names** `calibrate.py` reads
  (`calibrate.py:471–498`): `qk_nope_head_dim` (default **192**), `v_head_dim` (**256**),
  `qk_rope_head_dim` (**64**), `kv_lora_rank`, `q_lora_rank`, `num_attention_heads`, plus the indexer dims
  `index_topk`/`index_head_dim`/`index_n_heads`. So calibrate derives `k_head_dim = qk_nope_head_dim = 192`,
  `full_mla_k_width = H*(192+256)`, `full_mla_q_width = H*(192+64)` — correct for GLM with **no code
  change**. (`head_dim` is not a config field, which is fine: the MLA branch uses `qk_nope_head_dim`.)

**Hardware-round pre-run checklist** (verifiable only with the weights + GPUs):
1. The transformers `glm_moe_dsa` *modeling* exposes per-layer `self_attn.kv_b_proj` / `self_attn.q_b_proj`
   (MLA) so the calibrate hooks attach (they early-skip layers without them and raise if none fire).
2. The FP8 forward: GLM-5.1 is FP8 block-quant (128×128). If transformers routes it through
   `finegrained_fp8`, the existing model-agnostic `calibrate._force_triton_fp8_for_calibration` patch
   (forces the Triton fallback, avoids the deep-gemm hub-kernel 429/schema failure — BL-20260528) already
   covers it; if GLM uses a different FP8 path, add the equivalent there. Do NOT set `HF_HUB_OFFLINE=1`
   (publisher-trust check needs the online API). `pip install zstandard` for the Pile-val corpus.
3. The one-block `--dry-run-blocks 1` placement guard must show FP8 sharding (no bf16 upcast) and the
   K/Q hooks firing on all 78 layers at `H, head_dim=192` before the full run.

## Recipe — calibration command (VALIDATED on 8×H200, 2026-06-07)

```bash
SNAP=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db
# expandable_segments is REQUIRED: the large FP8 load otherwise OOMs mid-conversion
# on the fullest GPU (~12 GiB sits reserved-but-unallocated = fragmentation). Do NOT
# pass a per-GPU --max-memory cap: it makes accelerate spill to cpu/disk, which the
# on-the-fly finegrained-fp8 quantizer hard-rejects. device_map="auto" (the default)
# fits the model on the 8 GPUs; expandable_segments reclaims the fragmented reserve.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
  --model "$SNAP" \
  --dtype fp8_e4m3 --tp 8 \
  --output /models/glm51-fp8-channel-mask.safetensors \
  --label-dim 32 \
  --page-size 64 \
  --num-samples 256 --block-size 512 \
  --dataset runs/20260528_dsv32_mvp/calib_corpus_pileval.txt \
  -v
# --dtype fp8_e4m3 is REQUIRED (not optional). --tp 8 is metadata only — calibrate
# does a single-process forward and shards via HF device_map="auto" (needs the
# `accelerate` package: `pip install accelerate`). Do NOT pass --head-dim: calibrate
# auto-derives the no-PE width (192) from the config and writes it as the mask's
# head_dim metadata, exactly what runtime verify_bind_shapes(model_nope_head_dim=
# qk_nope_head_dim) checks. Run --dry-run-blocks 1 first.
```

### On-hardware dry-run evidence (2026-06-07, `runs/20260607_glm51_loop8/calib_dryrun3.log`)
`--dry-run-blocks 1` on the real GLM-5.1-FP8 across 8× H200 logged:
`DRY RUN complete: calibration hooks fired on all 78 layers (H=64, head_dim=192) over 1 block` with the
parameter report `dtype histogram={bfloat16:624, float8_e4m3fn:930, float32:930}, float8_present=True`
(no bf16 upcast of the FP8 weights) sharded `cuda:0..7`, via the Triton finegrained-fp8 fallback
(`_force_triton_fp8_for_calibration` — DeepGEMM skipped). This confirms the calibration path is correct
for GLM at head_dim=192 with NO calibrate.py code change. (Boot-chain hazards needed: `pip install
accelerate`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; do NOT set `HF_HUB_OFFLINE=1` — the
Triton kernel publisher-trust check needs the online API.)

## Recipe — serving the calibrated mask (hardware round)

`development/serve_double_sparsity.sh` is already `MODEL_PATH`/`CHANNEL_MASK_PATH`-parameterized, so no new
script is needed — serve GLM-5.1 DS via env overrides (DSA-native baseline = `serve_native_nsa.sh` with the
same `MODEL_PATH`, for the same-node DS-vs-DSA gate):

```bash
MODEL_PATH=/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/<hash> \
CHANNEL_MASK_PATH=/models/glm51-fp8-channel-mask.safetensors \
PAGE_SIZE=64 TP_SIZE=8 KV_CACHE_DTYPE=fp8_e4m3 TOP_K=2048 \
MEM_FRACTION_STATIC=<re-tune for GLM weight size> \
bash development/serve_double_sparsity.sh
```
At bind, the new `verify_bind_shapes` (Loop-8 R0) hard-errors if the mask's `head_dim`/`label_dim`/page/
layers/head-count don't match GLM — so a wrong mask fails fast with a named field rather than mis-selecting.

## Artifact contract (what the mask file must carry — AC-3)

`save_channel_mask` (called by `calibrate`) writes, and `load_channel_mask` re-verifies:
- `channel_selection` int32 `[layers=78, num_heads, label_dim=32]`, indices in `[0, head_dim=192)`
- `channel_weights` float32, same shape as `channel_selection`
- `dtype` = `fp8_e4m3`; `head_dim` = **192** (the no-PE width); `page_size` = **64**; `label_dim` = **32**
- `content_sha256` (re-verified on load); `schema_version`; `created_at`
- `extra_metadata`: `calibration_source` (real/synthetic), `dataset_source`, `seed`, `num_samples`,
  `block_size`
The full provenance set the result record cites (AC-3): `label_dim`, `page_size`, TP layout, `layers`,
`q_lora`, `kv_lora`, `qk_rope`, `qk_nope`, `v_head`, index dims, mask tensor shape, `content_sha256`,
output path, validation output.

## CPU verification done this round
`TestGlmArtifactContractRoundTrip` (`test_double_sparsity_unit.py`): a synthetic GLM-shape mask
(`head_dim=192`, `label_dim=32`, `layers=78`) is `save_channel_mask`-written, `load_channel_mask`-loaded
(content_sha256 re-verifies, indices in `[0,192)`), and **passes** `verify_bind_shapes` against GLM dims;
a `head_dim=128` (V3.2) mask **fails** `verify_bind_shapes` against GLM (names `head_dim`). This proves the
on-hardware artifact will be loadable and runtime-valid before any GPU time is spent.

## Deferred to the hardware round (8×H200)
The actual calibration run (offline FP8 forward to collect activations + write the real mask with
`content_sha256` over real channel importances), then the serving bring-up, byte-identity, and gates
(task6-GPU / task7 / task9).
