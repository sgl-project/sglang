# Weight Loader v2 — design and migration plan

Tracking issue: [RFC #24703](https://github.com/sgl-project/sglang/issues/24703)
Demo PR: [#28671](https://github.com/sgl-project/sglang/pull/28671)

---

## Why this design

**Complete loading, no stragglers.** Every checkpoint tensor that enters the loader and is not
explicitly skipped must produce a `WeightLoadRecord`, or `verify_complete` raises
`IncompleteWeightLoadError`. That replaces silent `logger.warning` drops and ad-hoc expert-shard
checks scattered across individual model files.

**Clean, visible weight mapping.** Each load is recorded as
`(checkpoint_name → runtime_name, shard_id, expert_id)`. Stacked qkv/gate_up and MoE expert
shards are first-class — not buried inside 70-line per-model loops. Mapping tables live in
`StackedParamsDispatch`, `ExpertParamsDispatch`, and `RemapRegistry` instead of being
copy-pasted 116 times across `srt/models/`.

**Enables advanced weight flows (e.g. P2P / streaming updates).** A mergeable `WeightLoadResult`
with per-shard records lets callers assert that a partial update batch finished, accumulate
across multiple `load_weights` calls, and know exactly which runtime slot each HF tensor
touched. The load / `post_load_weights` split separates tensor placement from optional
GPU-specific derivation (MLA `w_kc`/`w_vc`, etc.), which advanced transports need to control
independently.

All migrations stay behind `SGLANG_ENABLE_WEIGHT_LOADER_V2` (opt-in, default off). Legacy
`load_weights` paths remain until explicitly removed per model.

---

## Migration overview (~180 `load_weights` files)

Cumulative auto-loader coverage by PR (unique model implementation files, ~180 total):

| PR | Adds (files) | Cumulative | Coverage |
|----|-------------:|-----------:|---------:|
| **PR1** | 2 (`qwen2.py`, `transformers.py`) | 2 | 1% |
| **PR2** | ~40 | ~42 | 23% |
| **PR3** | ~22 | ~64 | 36% |
| **PR4** | ~5 | ~69 | 38% |
| **PR5** | 1 | ~70 | 39% |
| **PR6** | 1 | ~71 | 39% |
| **PR7** | 1 | ~72 | 40% |
| **PR8** | ~25 | ~97 | 54% |
| **PR9** | ~35 wrappers | ~132 | 73% |

`transformers.py` uses `AutoWeightsLoader` today (no v2 gate); counted in PR1 as already on the
walker stack. PR9 wrapper files add no new loader logic — they inherit v2 from a migrated
parent — but are included in the cumulative file count once smoke-tested.

### Protocol extensions (introduced across the PR stack)

| Extension | PR | Purpose |
|-----------|-----|---------|
| `post_load_weights` split | PR1 / PR4 | Separate tensor load from GPU derivations |
| Shared-expert preload hook | PR4 | Rename `shared_experts` → routed expert slot before dispatch |
| MLA `RemapRegistry` entries | PR4 | `fused_qkv_a_proj_with_mqa`, scale remaps |
| `FusedExpertDispatch` | PR5 | One fused ckpt key → multiple expert shard records |
| Packed-shard fan-out | PR5 | One ckpt tensor → multiple column shard records |
| Hybrid mixer submodule loaders | PR6 | Mamba + MoE in one decoder block |
| Custom loader decomposition | PR7 | Large bespoke loops → submodule overrides |
| Vision / audio tower `load_weights` | PR8 | Walker delegates to per-tower loaders |

---

## PR1 — Infrastructure + first model (this PR)

**Addresses:** Core abstractions and the reference dense migration pattern.

**Coverage:** +2 files (`qwen2.py`, `transformers.py` on walker stack) (~1%).

**Work:**

| Area | Deliverable |
|------|-------------|
| `auto_loader.py` | `WeightLoadRecord`, `WeightLoadResult`, `StackedParamsDispatch`, `ExpertParamsDispatch`, `RemapRegistry`, `verify_complete` |
| `models/utils.py` | `AutoWeightsLoader` returns `WeightLoadResult` |
| Load protocol | `load_weights(..., run_post_load=True)` + default no-op `post_load_weights()` |
| `models/qwen2.py` | v2 gate; `Qwen2Attention` / `Qwen2MLP` submodule loaders |
| `models/transformers.py` | already on `AutoWeightsLoader`; document as covered (no v2 gate) |
| `environ.py` | `SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)` |
| Tests | `test_auto_loader.py`, `test_auto_loader_moe.py`, `test_weight_loader_v2_equiv.py` |

Quant `process_weights_after_loading` remains in `loader.py` (unchanged).

---

## PR2 — Standard dense models (~41 files)

**Addresses:** Models with `stacked_params_mapping` only (qkv + gate_up), no expert mapping,
no MLA mixin. Mechanical migration: env gate, submodule loaders, optional `RemapRegistry`.

**Coverage:** +~40 files (~23% cumulative).

**Work per file:** Rename loop → `_legacy_load_weights`; add `_load_weights_v2` using
`AutoWeightsLoader` + `StackedParamsDispatch` on attention/MLP submodules; register FP8/kv
remaps where needed.

**Models:**

| File | Notes |
|------|-------|
| `qwen2.py` | done in PR1 |
| `qwen3.py` | + `maybe_remap_kv_scale` |
| `qwen3_classification.py` | inherits Qwen3 |
| `llama.py` | FP8 suffix + kv scale via `RemapRegistry` |
| `mistral.py` | delegates → Llama |
| `glm4.py` | dot-prefix stacked |
| `gemma.py`, `gemma2.py`, `gemma3_causal.py`, `gemma3n_causal.py` | gemma2/3 + kv remap |
| `olmo.py`, `olmo2.py`, `internlm2.py`, `granite.py` | standard |
| `stablelm.py`, `starcoder2.py`, `opt.py`, `orion.py` | standard |
| `baichuan.py`, `xverse.py`, `exaone.py`, `exaone4.py` | standard / remap |
| `commandr.py`, `apertus.py`, `arcee.py`, `nemotron_nas.py` | + remap |
| `falcon_h1.py`, `lfm2.py`, `iquest_loopcoder.py`, `jet_nemotron.py` | standard |
| `mimo.py`, `gpt_j.py`, `sdar.py`, `dflash.py`, `afmoe.py` | standard / remap |
| `bert.py`, `roberta.py`, `clip.py`, `whisper.py`, `siglip2.py` | encoders |
| `llama_embedding.py`, `torch_native_llama.py` | variants |

Cherry-pick walker diffs from
[brayden/weight-loader-refactor-part-1](https://github.com/sgl-project/sglang/compare/main...brayden/weight-loader-refactor-part-1)
where possible.

**Tests:** parametrized CPU mapping tests; spot-check GPU equiv (`Llama-3.2-1B`, `Qwen3-1.7B-FP8`).

---

## PR3 — Standard MoE models (~22 files)

**Addresses:** Models using `FusedMoE.make_expert_params_mapping` with dense stacked layers.

**Coverage:** +~22 files (~36% cumulative).

**Work per file:** `ExpertParamsDispatch` on `FusedMoE` submodule; dense submodules as PR2;
`verify_expert_shards_complete` where all routed expert shards are mandatory (e.g. `laguna.py`).

**Models:**

| File | Notes |
|------|-------|
| `qwen2_moe.py`, `qwen3_moe.py` | standard MoE |
| `mixtral.py`, `glm4_moe.py`, `olmoe.py`, `phimoe.py` | standard |
| `cohere2_moe.py`, `exaone_moe.py`, `lfm2_moe.py` | standard |
| `hunyuan.py`, `hunyuan_v3.py`, `bailing_moe.py` | standard |
| `laguna.py` | + expert-shard completeness assert |
| `ernie4.py`, `grok.py`, `kimi_linear.py`, `llada2.py` | standard |
| `minimax_m2.py`, `sdar_moe.py`, `sarvam_moe.py`, `dbrx.py` | standard |
| `xverse_moe.py`, `granitemoe.py` | standard / delegates Mixtral |

**Tests:** `Qwen/Qwen1.5-MoE-A2.7B` equiv; extend `test_auto_loader_moe.py`.

---

## PR4 — MLA mixin models (~5 files)

**Addresses:** `DeepseekV2WeightLoaderMixin` / `do_load_weights` — shared expert preload, MLA
name remaps, and `post_load_weights` for `w_kc`/`w_vc` derivation.

**Coverage:** +~5 files (~38% cumulative).

**Work:** Replace mixin loop with walker + submodule loaders; wire shared-expert preload hook;
split `post_load_weights` from tensor load (PR1 protocol); shrink `deepseek_weight_loader.py`.

**Models:**

| File | Notes |
|------|-------|
| `deepseek_v2.py` | V2, V3, V32 |
| `deepseek_common/deepseek_weight_loader.py` | mixin refactor |
| `glm4_moe_lite.py` | GLM-4.7 |
| `bailing_moe_linear.py` | linear MoE variant |
| `kimi_k25_eagle3.py` | eagle MLA draft |

**Tests:** `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`; GLM-4.7 mapping units.

---

## PR5 — `qwen3_5.py`

**Addresses:** Fused MoE checkpoint format (`gate_up_proj` → w1 + w3 shards) and packed linear
shard fan-out.

**Coverage:** +1 file (~39% cumulative).

**Work:** Add `FusedExpertDispatch`; fan-out multi-shard records from single ckpt tensors;
migrate `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`.

**Tests:** `Qwen/Qwen3.5-4B`; fused-expert unit tests.

---

## PR6 — `qwen3_next.py`

**Addresses:** Hybrid mamba + MoE decoder — mixer and expert paths need separate submodule
`load_weights`.

**Coverage:** +1 file (~39% cumulative).

**Work:** Submodule overrides on mamba mixer and MoE block; standard dispatch elsewhere.

**Tests:** mapping unit tests (full-model GPU equiv impractical at default sizes).

---

## PR7 — `deepseek_v4.py`

**Addresses:** Large bespoke loader (~335 lines) with custom expert routing and `post_load_weights`.

**Coverage:** +1 file (~40% cumulative).

**Work:** Decompose into submodule loaders; reuse PR4 post_load protocol.

**Tests:** unit mapping tests; targeted manual validation.

---

## PR8 — Multimodal models (~25 files)

**Addresses:** Vision, audio, and OCR tower weights routed alongside text trunk.

**Coverage:** +~25 files (~54% cumulative).

**Work:** Text trunk uses PR2/PR3 walker; add `VisionTower.load_weights` /
`AudioTower.load_weights` submodule delegation; `skip_substrs` for optional towers. Reuse
`FusedExpertDispatch` from PR5 where VL-MoE uses fused expert ckpts.

**Models (representative):** `qwen2_vl.py`, `qwen2_5_vl.py`, `qwen3_vl.py`, `qwen3_vl_moe.py`,
`qwen3_omni_moe.py`, `qwen2_audio.py`, `qwen3_asr.py`, `deepseek_vl2.py`, `deepseek_ocr.py`,
`deepseek_janus_pro.py`, `glm4v.py`, `glm4v_moe.py`, `glm_ocr.py`, `glmasr.py`, `minicpmv.py`,
`internvl.py`, `kimi_vl.py`, `moss_vl.py`, `gemma3_mm.py`, `gemma4_mm.py`, `pixtral.py`, …

**Tests:** `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen3-VL-2B-Thinking`; per-tower CPU units.

---

## PR9 — Wrapper models (~35 files)

**Addresses:** Thin classes that only call `super().load_weights` (eagle, MTP, NextN,
classification, reward, delegates).

**Coverage:** +~35 wrapper files (~73% cumulative). No new loader implementation — inherits
v2 from parent. Smoke-test that `super().load_weights` forwards `WeightLoadResult` and
`run_post_load` correctly; fix NextN `__init__` bypass mapping gaps.

**Work:** Ensure mappings are available when `__init__` is bypassed (NextN pattern); forward
`run_post_load` to parent; smoke tests only.

**Models (representative):** `qwen2_eagle.py`, `qwen3_moe_mtp.py`, `qwen3_5_mtp.py`,
`qwen3_next_mtp.py`, `deepseek_nextn.py`, `deepseek_v4_nextn.py`, `llava.py`, `granitemoe.py`,
`mistral.py`, `qwen2_rm.py`, …

---

## Not supported today (open to community contribution)

The PR1–PR9 plan does **not** include these models. They stay on legacy `load_weights` until
someone contributes a dedicated migration. v2 remains opt-in via `SGLANG_ENABLE_WEIGHT_LOADER_V2`.

### Legacy architectures

| File | Reason |
|------|--------|
| `qwen.py` | Superseded by Qwen2+; hand-rolled loader |
| `deepseek.py` | Superseded by DeepSeek V2+; hand-rolled loader |

### Custom loaders (no `stacked_params_mapping` pattern)

| File | Reason |
|------|--------|
| `phi.py` | `packed_modules_mapping` driven |
| `phi3_small.py` | Custom loader loop |
| `solar.py` | `packed_modules_mapping` driven |
| `gpt2.py` | Direct param iteration |
| `gpt_bigcode.py` | Direct param iteration |
| `chatglm.py` | Custom GLM QKV routing |
| `minicpm.py` | Manual w1/w2/w3 expert table |
| `minicpm3.py` | Manual expert mapping |
| `parakeet.py` | Custom loader |
| `persimmon.py` | Custom loader |

### Bespoke loaders (need dedicated design per model)

| File | Reason |
|------|--------|
| `step3p5.py` | Custom expert name matching (~146L loader) |
| `step3_vl.py` | Custom expert matching + vision |
| `step3p5_mtp.py` | MTP variant of step3 loader |
| `step3_vl_10b.py` | VL delegate + step3 patterns |
| `step3p7.py` | Step family bespoke routing |
| `nemotron_h.py` | Non-standard expert `weight_loader` signature |
| `nemotron_h_mtp.py` | MTP wrapper on nemotron_h |
| `zaya.py` | MOD routing / custom MoE |
| `longcat_flash.py` | Hybrid loader (~180L) |
| `longcat_flash_nextn.py` | NextN + longcat patterns |
| `mimo_v2.py` | Custom qkv routing (~198L loader) |
| `mimo_v2_nextn.py` | MTP on mimo_v2 |
| `mimo_v2_asr.py` | ASR delegate on mimo_v2 |
| `gpt_oss.py` | Unique quant routing |
| `gemma4_causal.py` | Fused expert ckpt (needs `FusedExpertDispatch` beyond PR5 scope) |
| `granitemoehybrid.py` | Mamba + MoE hybrid (~150L loader) |
| `mindspore.py` | MindSpore cell loader path |

Pull requests that migrate any of the above to the v2 stack (walker + dispatch + records) are
welcome; add mapping unit tests and a manual or CI equiv test where a small checkpoint exists.

---

## Checklist

- [ ] PR1: infra + post_load protocol + `qwen2.py` + `transformers.py` (walker)
- [ ] PR2: standard dense (~41 files)
- [ ] PR3: standard MoE (~22 files)
- [ ] PR4: MLA mixin (~5 files)
- [ ] PR5: `qwen3_5.py`
- [ ] PR6: `qwen3_next.py`
- [ ] PR7: `deepseek_v4.py`
- [ ] PR8: multimodal (~25 files)
- [ ] PR9: wrappers (~35 files)

---

## References

- [brayden/weight-loader-refactor-part-1](https://github.com/sgl-project/sglang/compare/main...brayden/weight-loader-refactor-part-1) — cherry-pick dense walker diffs into PR2
- [PR #24256](https://github.com/sgl-project/sglang/pull/24256) — checksum for production equiv validation
