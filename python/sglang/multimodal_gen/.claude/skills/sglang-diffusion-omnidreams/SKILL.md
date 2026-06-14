---
name: sglang-diffusion-omnidreams
description: Use when working on the OmniDreams model port — implementing features, debugging, reviewing changes, running tests, or checking file organization. OmniDreams is NVIDIA's autoregressive video world model for autonomous driving simulation.
---

# OmniDreams (FlashDreams Port) — Source Guide

Use this skill for any OmniDreams task: adding features, debugging regressions,
reviewing PRs, running tests, or checking file organization.

## Key Files (Absolute Paths)

All paths are relative to `python/sglang/multimodal_gen/`.

### Pipeline & Stages

| File | Purpose |
|---|---|
| `runtime/pipelines/omnidreams_pipeline.py` | Pipeline loader (non-Diffusers flat `.pt`), VAE/text-encoder/scheduler resolution, stage wiring |
| `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` | `OmniDreamsBeforeDenoisingStage` (text encode, i2v image, HD-map, schedule) + `OmniDreamsDenoisingStage` (AR rollout loop) |

### DiT Model

| File | Purpose |
|---|---|
| `runtime/models/dits/omnidreams.py` | `OmniDreamsDiT` (2.06B, 28 blocks), `OmniDreamsBlock`, `OmniDreamsAttention` |
| `runtime/models/dits/omnidreams_kvcache.py` | `BlockKVCache` — windowed KV cache with sink + split-copy roll |
| `runtime/models/dits/omnidreams_rope.py` | `RotaryPositionEmbedding3D` (NeoX 44:42:42), `apply_rope_freqs`, `to_cos_sin_cache` |

### Text Encoding (Cosmos-Reason1-7B / Qwen2.5-VL)

| File | Purpose |
|---|---|
| `runtime/models/encoders/omnidreams_text.py` | `full_concat_embeddings` — 28-layer hidden-state mean-normalize → 100352-dim |

### Configs

| File | Purpose |
|---|---|
| `configs/models/dits/omnidreams.py` | `OmniDreamsDiTArchConfig` (2048-d, 28 blocks, 16 heads, HDMap 16ch) |
| `configs/pipeline_configs/omnidreams.py` | `OmniDreamsPipelineConfig` (I2V, bf16 DiT, fp32 VAE, 2-step warp sigma) |
| `configs/models/vaes/wanvae.py` | `WanVAEConfig` (shared z_dim=16 VAE config) |
| `configs/sample/omnidreams.py` | `OmniDreamsSamplingParams` (720p, 2-step, len_t=2, window=6) |

### Registry

| File | Purpose |
|---|---|
| `registry.py` | `register_configs(...)` entry with `model_detectors` for "omnidreams"/"omni-dreams" |

### Tests

| File | Purpose |
|---|---|
| `test/unit/test_omnidreams_scaffold.py` | Structural/fixture checks |
| `test/unit/test_omnidreams_components.py` | CPU-component tests: RoPE, KV-cache, flow-match scheduler, full-concat, tiny-DiT forward, AR rollout orchestration |
| `test/unit/test_omnidreams_regression.py` | End-to-end regression tests (needs checkpoint + GPU) |
| `test/unit/test_omnidreams_hdmap_validation.py` | HD-map input validation |

## Architecture Facts (Do NOT Re-derive)

- **Checkpoint:** `single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt` — flat bf16, 570 keys, DiT-only (no VAE/text-encoder weights)
- **DiT:** 2048 hidden, 16 heads x 128 head-dim, 28 blocks, MLP ratio 4 (8192), AdaLN-LoRA (rank 256), no bias except `crossattn_proj.0.bias`
- **Attention:** NeoX-style RoPE (44:42:42 T:H:W split, h/w extrap 3.0, t 1.0). Q/K RMSNorm (128-d) before RoPE. Self-attn has RoPE + KV-cache; cross-attn has neither. No GQA.
- **Final layer:** 2-chunk AdaLN (shift+scale only, no gate) → linear 2048→64 → channel shuffle fused in `post_load_weights`
- **Channels:** 72-in = (16 latent + 1 cond_mask + 1 pad_mask) × patch 1x2x2. HDMap: additional 16ch → 64 × patch = 64-in. Post-fusion: 70-in (pad-mask channel dropped at load time).
- **Text:** Cosmos-Reason1-7B (Qwen2.5-VL). 28 transformer layers x 3584 = 100352 full-concat. Padded to 512 tokens. NO attention mask — model trained on unmasked padded sequences.
- **Schedule:** 2-step flow-match, warp `s = shift*s/(1+(shift-1)*s)` with shift=5, sigma_min=0 → sigmas [1.0, 0.8036, 0.0]
- **AR rollout:** Per-chunk `shift_t(ar_idx)` RoPE, context-noise=128 re-forward, KV-cache window=6 sink=0
- **HD-map:** Per-frame rasters VAE-encoded as causal clip, sliced into per-chunk tokens via `additional_patch_embedding`
- **Cross-view:** Architecture ready, gated by `enable_cross_view_attn=False` (single-view checkpoint). Raises `NotImplementedError` when enabled.

## Known Design Decisions

1. **No attention mask in text encoding.** The DiT was trained with FlashDreams' encoder that runs the full padded sequence unmasked. Adding a mask changed padding-token hidden states by ~99 abs diff, causing severe blur (diagnosed in `omnidreams_blur_diagnosis_report.md`).

2. **Custom scheduler, not framework default.** `SelfForcingFlowMatchScheduler.set_timesteps()` does not match the OmniDreams warp schedule. Solution: `OmniDreamsFlowMatchScheduler` with explicit `denoising_timesteps=(1000, 450)` and `warp_flow_match_sigmas()`.

3. **Flat .pt loading bypasses Diffusers.** The checkpoint is a single `.pt` file, not a diffusers layout. Pipeline overrides `_load_config` (synthesizes model_index) and `load_modules` (loads DiT via `load_model_from_full_model_state_dict`, VAE from diffusers Wan, text encoder from HF).

4. **No CFG.** The distilled checkpoint disables classifier-free guidance (`guidance_scale=1.0`).

5. **TP: yes. SP: no.** Column/RowParallel linear layers for TP. SP (ulysses/ring) is explicitly rejected in the AR rollout stage.

## Performance Optimizations (Implemented)

The following optimizations are active in the current codebase:

| # | Optimization | File | What |
|---|---|---|---|
| T1 | AdaLN fusion | `omnidreams.py` (DiT) | `LayerNormScaleShift` replaces `nn.LayerNorm + manual scale/shift` — dispatches to CuTe DSL fused kernel on CUDA |
| T2 | RoPE kernel | `omnidreams_rope.py` + `omnidreams.py` (DiT) | `to_cos_sin_cache()` + `_apply_rotary_emb` dispatch (FlashInfer on CUDA, Triton fallback) |
| T3 | KV-cache split-copy | `omnidreams_kvcache.py` | Split-copy eliminates per-block `.clone()` allocation during window roll |
| T4 | Text encoder cache | `omnidreams.py` (stage) | `OrderedDict` LRU cache (max 32) keyed on prompt string, embeddings stored on CPU |
| T5 | Cross-attn KV precompute | `omnidreams.py` (DiT) | K/V projected once per prompt, reused across all AR chunks |
| T6 | `torch.compile` on blocks | `omnidreams.py` (DiT) | `_compile_conditions = [isinstance(m, OmniDreamsBlock)]` |

## How to Serve

```bash
sglang serve --model-path /path/to/omnidreams-checkpoint \
  --pipeline-class-name OmniDreamsPipeline \
  --enable-torch-compile --warmup \
  --text-encoder-cpu-offload --vae-cpu-offload
```

## How to Generate

```bash
sglang generate --model-path /path/to/omnidreams-checkpoint \
  --pipeline-class-name OmniDreamsPipeline \
  --prompt "A car driving down a city street" \
  --image-path first_frame.png \
  --hdmap-path hdmap.mp4 \
  --enable-torch-compile --warmup \
  --seed 42 --save-output
```

## How to Test

```bash
# CPU component tests (run anywhere, no GPU needed)
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_components.py -x -v
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_scaffold.py -x -v

# End-to-end regression (needs GPU + checkpoint)
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_regression.py -x -v

# HD-map validation tests
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_hdmap_validation.py -x -v
```

## Diagnostics

```bash
# Per-frame tensor stats (mean/std/min/max at VAE encode + AR concat)
SGLANG_OMNIDREAMS_DIAGNOSTICS=1 sglang generate ...

# Stage-level timing breakdown
SGLANG_DIFFUSION_STAGE_LOGGING=1 sglang generate ...
```

## Related Documents

- `omni.md` — Original implementation plan (Phases 0–6)
- `OmniDreamsOptimizationFindings.md` — Deep-dive report on all optimization targets
- `omnidreams_blur_diagnosis_report.md` — Root-cause analysis of the blur bug (attention_mask training-inference mismatch)
- `progress.md` — HD-map validation results and GPU memory observations
