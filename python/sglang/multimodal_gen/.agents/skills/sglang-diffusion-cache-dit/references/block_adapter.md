# BlockAdapter Reference

When to read this: read this file when writing or extending a custom `BlockAdapter` for SGLang Diffusion, selecting a `ForwardPattern`, or diagnosing cache interception issues. Return to `../SKILL.md` for the high-level workflow.

> **SGLang scope rules (read first):**
> - **This document is a reference for ideas only.** It documents how cache-dit itself implements Cache adapters; do not copy its registration/build flow verbatim into sglang.
> - **All sglang-diffusion BlockAdapter code strictly lives in the sglang repo** — adapters are constructed directly in `runtime/cache/cache_dit_integration.py` (sglang transformers are third-party, non-diffusers modules). Do NOT register sglang adapters in the cache-dit repo (`BlockAdapterRegister` / `block_adapters/__init__.py`); the registration flow described in cache-dit does not apply to sglang.
> - **PatchFunctor is NOT recommended.** Monkey-patching `transformer.forward()` does not fit the current sglang diffusion design. Treat §1.6 as diagnostic background: if a sglang transformer hits one of those structural pitfalls, fix the call structure on the sglang side instead of wiring a PatchFunctor.
> - The parts that do transfer to sglang: `ForwardPattern` selection (§1.2), `BlockAdapter` parameters such as `check_forward_pattern` / `has_separate_cfg` (§1.3), construction templates (§1.4), and the third-party (non-diffusers) adapter rules (§1.5).

## 1. Cache Integration: BlockAdapter + ForwardPattern

### 1.1 Concept

cache-dit's caching engine works by intercepting the forward pass of DiT transformer blocks. To do this, it needs to know:

1. **Where the blocks are** — which `ModuleList` attribute holds the repeated transformer blocks.
2. **What goes in and out** — the block's `forward()` input/output signature ("forward pattern").
3. **Any model quirks** — separate CFG passes, special patching needs, etc.

All of this is described by a single `BlockAdapter` dataclass instance.

### 1.2 ForwardPattern — The 6 Block I/O Contracts

`ForwardPattern` is an enum in `src/cache_dit/caching/forward_pattern.py`. It captures the hidden-state ordering and forward-signature shape of a family of transformer blocks. Choose the pattern that matches your block's `forward()` signature:

| Pattern             | `forward()` inputs                       | `forward()` returns                      | Return_H_First | Return_H_Only | Forward_H_only | Typical Models                                                         |
| ------------------- | ------------------------------------------ | ------------------------------------------ | -------------- | ------------- | -------------- | ---------------------------------------------------------------------- |
| **Pattern_0** | `(hidden_states, encoder_hidden_states)` | `(hidden_states, encoder_hidden_states)` | `True`       | `False`     | `False`      | Mochi, CogVideoX, CogView4, HunyuanVideo, EasyAnimate                  |
| **Pattern_1** | `(hidden_states, encoder_hidden_states)` | `(encoder_hidden_states, hidden_states)` | `False`      | `False`     | `False`      | Flux transformer_blocks, QwenImage, SD3, VisualCloze                   |
| **Pattern_2** | `(hidden_states, encoder_hidden_states)` | `(hidden_states,)`                       | `False`      | `True`      | `False`      | Wan, Allegro, Cosmos, LTX-1                                            |
| **Pattern_3** | `(hidden_states,)`                       | `(hidden_states,)`                       | `False`      | `True`      | `True`       | Flux single_transformer_blocks, DiT, PixArt, Sana, Lumina2, SkyReelsV2 |
| **Pattern_4** | `(hidden_states,)`                       | `(hidden_states, encoder_hidden_states)` | `True`       | `False`     | `True`       | (rare)                                                                 |
| **Pattern_5** | `(hidden_states,)`                       | `(encoder_hidden_states, hidden_states)` | `False`      | `False`     | `True`       | (rare)                                                                 |

**How to determine the correct pattern for your model:**

1. Open the block's `forward()` method in diffusers source.
2. Check the parameter list: does it take only `hidden_states`, or also `encoder_hidden_states`? This determines `Forward_H_only`.
3. Check the return statement: does it return one tensor or two? In what order? This determines `Return_H_Only` / `Return_H_First`.
4. Match against the table above. If none fits exactly, open an issue.

### 1.3 BlockAdapter Parameters

Defined in `<cache_dit_dir>/src/cache_dit/caching/block_adapters/block_adapters.py`. Key parameters:

| Parameter                 | Type                                               | Description                                                                                                                                                                 |
| ------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pipe`                  | `DiffusionPipeline` or `FakeDiffusionPipeline` | The pipeline instance (or a placeholder if no pipeline is available).                                                                                                       |
| `transformer`           | `nn.Module` or `List[nn.Module]`               | The transformer module(s). Single module for most models; list of 2 for dual-transformer models (e.g., Wan 2.2 MoE).                                                        |
| `blocks`                | `nn.ModuleList` or `List[nn.ModuleList]`       | The block collection(s). Single ModuleList for most models; list of 2 for models with dual block types (e.g., Flux:`transformer_blocks` + `single_transformer_blocks`). |
| `forward_pattern`       | `ForwardPattern` or `List[ForwardPattern]`     | Must match`blocks` count. Single pattern for single block list; list of patterns for multiple block lists.                                                                |
| `check_forward_pattern` | `Optional[bool]`                                 | Validate that each block's I/O matches the declared pattern. If left `None` (default), cache-dit **auto-detects**: `True` for `diffusers` transformers, `False` for third-party ones (`maybe_skip_checks()`); it is also forced `False` when the transformer already has an `_hf_hook` / `_diffusers_hook`. Set explicitly for new models.                                                             |
| `check_num_outputs`     | `bool`                                           | If`True`, cache-dit additionally validates that each block returns the exact number of outputs the pattern declares. Needed for models whose blocks can return a variable tuple (e.g., HiDream, HunyuanVideo 1.0). Default `False`.                                             |
| `has_separate_cfg`      | `bool`                                           | Set `True` if the pipeline runs **two separate `transformer.forward()` calls** for the conditional and unconditional passes of Classifier-Free Guidance (CFG). Set `False` if the pipeline concatenates cond+uncond into a single batch and calls `transformer.forward()` **once**. See §1.3.1 for the decision guide and code patterns. |
| `patch_functor`         | `PatchFunctor` or `None`                       | Optional pre-patch logic. Used when the model needs structural modification before caching hooks are installed (e.g., Flux dummy block merging, DiT re-patching).           |
| `blocks_name`           | `str` or `List[str]`                           | Override block attribute names (advanced).                                                                                                                                  |
| `dummy_blocks_names`    | `List[str]`                                      | Names of blocks that should be treated as dummy/merged (advanced, e.g., Flux single_transformer_blocks when merged into transformer_blocks).                                |

### 1.3.1 `has_separate_cfg` — Decision Guide & Code Patterns

> **⚠️ This parameter is about the NUMBER of `transformer.forward()` calls per denoising step, NOT about whether CFG is enabled.** A model can use CFG (`guidance_scale > 1`) and still have `has_separate_cfg=False` if the pipeline batches cond+uncond into one forward call.

**Definition:**

| `has_separate_cfg` | Pipeline behavior per denoising step | Number of `transformer.forward()` calls |
|---|---|---|
| `True`  | Pipeline calls `transformer(...)` **twice**: once with cond embeddings, once with uncond embeddings. The two outputs are combined by `noise_pred = uncond + scale * (cond - uncond)`. | **2** |
| `False` | Pipeline concatenates `[latents, latents]` into one batch, calls `transformer(...)` **once** with `encoder_hidden_states=[uncond, cond]`, then splits the output via `chunk(2)`. | **1** |

**Why it matters for caching:** cache-dit caches transformer block outputs. When `has_separate_cfg=True`, the cond and uncond passes have **independent cache contexts** (`"cond"` / `"uncond"`) because their inputs differ. When `False`, there is only one forward pass and one cache context. Setting this incorrectly causes the cache to mix cond/uncond states → garbled output.

**How to decide — read the pipeline's `__call__` denoising loop:**

**Pattern A → `has_separate_cfg=True`** (two separate forward calls):

```python
# WanPipeline (diffusers) — TWO calls, one for cond, one for uncond
latent_model_input = latents.to(transformer_dtype)  # NOT concatenated

with current_model.cache_context("cond"):
    noise_pred = current_model(
        hidden_states=latent_model_input,
        encoder_hidden_states=prompt_embeds,          # cond embeddings
        ...
    )[0]

if self.do_classifier_free_guidance:
    with current_model.cache_context("uncond"):
        noise_uncond = current_model(                 # SECOND forward call
            hidden_states=latent_model_input,         # same latents, NOT batched
            encoder_hidden_states=negative_prompt_embeds,  # uncond embeddings
            ...
        )[0]
    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
```

**Tell-tale signs of Pattern A:**
- `latent_model_input` is NOT concatenated (no `torch.cat([latents] * 2)`).
- There are **two** `transformer(...)` / `current_model(...)` calls inside the loop.
- The second call uses `negative_prompt_embeds` / `negative_pooled_projections`.
- `cache_context("cond")` / `cache_context("uncond")` wrap the two calls.

**Models using Pattern A:** `Wan` (`has_separate_cfg=True`), `Flux` (with `do_true_cfg`), `QwenImage`, `CogView4`, `Cosmos`, `SkyReelsV2`, `Chroma`, `HunyuanImage`, `OvisImage`, `LongCatImage`, `GlmImage`, `Helios`, `ErnieImage`, `Krea2`, `JoyImage`, `BriaFibo`.

---

**Pattern B → `has_separate_cfg=False`** (single batched forward call):

```python
# AnyFlowPipeline (diffusers) — ONE call, cond+uncond batched together
latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
# latent_model_input shape: (2*B, ...) — cond and uncond stacked

noise_pred = self.transformer(
    hidden_states=latent_model_input,           # batched cond+uncond
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,        # [uncond_embeds, cond_embeds] stacked
    ...
)[0]

if self.do_classifier_free_guidance:
    noise_uncond, noise_pred = noise_pred.chunk(2)  # split the batched output
    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
```

**Tell-tale signs of Pattern B:**
- `torch.cat([latents] * 2)` or `torch.cat([latents, latents])` before the forward call.
- `encoder_hidden_states` is `torch.cat([negative_prompt_embeds, prompt_embeds])` (stacked).
- Only **one** `transformer(...)` call inside the loop.
- Output is split via `noise_pred.chunk(2)` after the forward.
- No `cache_context("cond")` / `cache_context("uncond")` — single context.

**Models using Pattern B:** `AnyFlow` (`has_separate_cfg=False`), `ErnieImage` (`has_separate_cfg=False`), `Krea2` (when `guidance_scale=0`, no CFG at all), distilled models with CFG folded into weights (`guidance_scale=1.0`).

---

**Pattern C → `has_separate_cfg=False`** (no CFG at all, `guidance_scale=1.0`):

```python
# Distilled model — no CFG, single forward, single batch
noise_pred = self.transformer(
    hidden_states=latents,                       # NOT concatenated
    encoder_hidden_states=prompt_embeds,          # only cond
    ...
)[0]
# No chunk(2), no noise_uncond, no CFG combination
```

**Tell-tale signs of Pattern C:**
- `guidance_scale=1.0` (or `0.0`).
- No `do_classifier_free_guidance` branch, no `torch.cat([latents]*2)`.
- Only one forward call with no cond/uncond splitting.

**Models using Pattern C:** AnyFlow (default `guidance_scale=1.0`, CFG folded into weights), ErnieImage Turbo, Krea2 Turbo (`guidance_scale=0.0`), ZImage Turbo (`guidance_scale=0.0`).

> **Note:** Pattern B and Pattern C both use `has_separate_cfg=False`. The difference is whether CFG is active (B: `guidance_scale > 1`, batched) or inactive (C: `guidance_scale <= 1`, single). In both cases there is only ONE `transformer.forward()` call, so cache-dit uses a single cache context.

**Quick decision flowchart:**

```
Read the pipeline __call__ denoising loop.
  │
  ├─ Does it call transformer(...) TWICE per step
  │  (once with cond, once with uncond)?
  │    └─ YES → has_separate_cfg = True
  │
  ├─ Does it torch.cat([latents]*2) and call transformer(...) ONCE,
  │  then chunk(2) the output?
  │    └─ YES → has_separate_cfg = False
  │
  └─ Does it call transformer(...) ONCE with no cat/chunk
     (guidance_scale <= 1, no CFG)?
        └─ YES → has_separate_cfg = False
```

### 1.4 Reference Templates: Constructing a BlockAdapter

These templates mirror how cache-dit's built-in adapters are written (`<cache_dit_dir>/src/cache_dit/caching/block_adapters/adapters.py`, read-only reference). The same `BlockAdapter(...)` construction applies in sglang's `runtime/cache/cache_dit_integration.py` — as plain construction code, **without** the `@BlockAdapterRegister.register(...)` decorator cache-dit uses internally.

#### Template A: Single block list (most common)

```python
adapter = BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,    # adjust to your model
    check_forward_pattern=True,
)
```

#### Template B: Dual block lists (like Flux)

```python
# Standard Flux: both block types use Pattern_1.
# For Flux2 / Nunchaku variants: single_transformer_blocks use Pattern_3 instead.

adapter = BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=[
        pipe.transformer.transformer_blocks,
        pipe.transformer.single_transformer_blocks,
    ],
    forward_pattern=[
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_1,
    ],
    check_forward_pattern=True,
)
```

#### Template C: Dual transformers (like Wan 2.2 MoE)

```python
adapter = BlockAdapter(
    pipe=pipe,
    transformer=[
        pipe.transformer,
        pipe.transformer_2,          # second transformer (MoE)
    ],
    blocks=[
        pipe.transformer.blocks,
        pipe.transformer_2.blocks,
    ],
    forward_pattern=[
        ForwardPattern.Pattern_2,
        ForwardPattern.Pattern_2,
    ],
    check_forward_pattern=True,
    has_separate_cfg=True,
)
```

### 1.5 Third-Party (Non-Diffusers) Models

If your model does **not** come from the official `diffusers` library (e.g., it is defined in `sglang` or another third-party package), follow these rules:

**Do NOT hardcode `from diffusers import ...`.** Instead, use `_safe_import` with name-based matching, or simply skip the diffusers-specific import entirely.

**`_relaxed_assert` is NOT mandatory.** The function (`<cache_dit_dir>/src/cache_dit/caching/block_adapters/adapters.py`) checks `transformer.__module__` — if it does not start with `"diffusers"`, the function logs a warning and skips the strict type check automatically. For third-party models, you can:

- Omit `_relaxed_assert` entirely, or
- Call it with `allow_classes=None` to rely on the automatic skip behavior.

**Example — third-party BlockAdapter without `_relaxed_assert`:**

```python
# No `from diffusers import ...` — the transformer type is resolved at runtime.
adapter = BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
)
```

The same principle applies to cache-dit's distributed planners (CP, TP, TE-P, VAE-P), for reference: they never hardcode diffusers class names for third-party models either — dispatch matches on a registered descriptive name instead. SGLang does not add planners to cache-dit; anything parallelism-related is wired on the sglang side.

### 1.6 Interception Pitfalls — PatchFunctor Background (NOT recommended for sglang)

> ⚠️ **Always check for these pitfalls before declaring the cache integration "done."** A BlockAdapter that looks correct on paper can silently produce wrong results if the `transformer.forward()` has any of the structural issues below. When in doubt, run a full inference with caching enabled and compare PSNR/SSIM against the uncached baseline.
>
> **SGLang:** read this section for diagnosis only. cache-dit's remedy for these pitfalls is a `PatchFunctor` (a monkey-patch of `transformer.forward()`), which does not fit the current sglang diffusion design. If a sglang transformer hits one of these pitfalls, fix the call structure in sglang code instead.

The `BlockAdapter` works by intercepting the block-loop inside `transformer.forward()`. It replaces the original `ModuleList` (e.g., `self.transformer_blocks`) with `UnifiedBlocks` — a wrapper that injects cache look-up/save logic around each block call. This interception is mechanical: it relies on `inspect.signature` to bind arguments and on the assumption that the for-loop body contains **nothing but a single block call**. When the model's `forward()` violates these assumptions, the cache produces wrong results silently (no crash, just corrupted output).

A **`PatchFunctor`** is cache-dit's escape hatch for these cases: a monkey-patch that rewrites `transformer.forward()` *before* the `BlockAdapter` is applied. The two pitfall categories below explain **why** a structural fix is needed; the fix itself should land in sglang code, not a PatchFunctor.

#### Pitfall A: Block call argument mismatch (keyword vs positional)

**Problem**: `transformer.forward()` calls blocks with **keyword arguments** (e.g., `block(hidden_states=x, encoder_hidden_states=e, temb=t)`), but the block's `forward()` signature defines those parameters as **positional**. When cache-dit's `UnifiedBlocks` wrapper intercepts the call, it uses `inspect.signature.bind()` to match arguments — keyword-to-positional mismatches cause `bind()` to fail or bind to the wrong parameters.

**Symptom**: `TypeError` from `inspect.signature.bind()`, or the cache silently feeds wrong tensors to the block.

**Fix idea (reference)**: cache-dit's `LTX2PatchFunctor` rewrites the call site so positional parameters are passed positionally (matching the block's actual signature), keeping only truly keyword-only parameters as keyword args. In sglang, apply the same idea directly in the sglang-side call site.

**Canonical example — `LTX2PatchFunctor`** (`<cache_dit_dir>/src/cache_dit/caching/patch_functors/functor_ltx2.py`):

The original diffusers code for LTX-2.0 passes all block arguments as keywords:

```python
# Original (diffusers) — ALL keyword args:
hidden_states, audio_hidden_states = block(
    hidden_states=hidden_states,
    audio_hidden_states=audio_hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    audio_encoder_hidden_states=audio_encoder_hidden_states,
    temb=temb,
    temb_audio=temb_audio,
    ...
)
```

The patched version converts the first four positional parameters to positional form, keeping the rest as keyword:

```python
# Patched — positional args match the block's forward(hidden_states, audio_hidden_states, ...):
hidden_states, audio_hidden_states = block(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    temb=temb,
    temb_audio=temb_audio,
    ...
)
```

**How to detect this pitfall**: Read the block's `forward()` signature in the diffusers source. Count how many parameters are positional (before any `*` or `*args`). Then check how `transformer.forward()` invokes the block — if it passes any of those positional params as keyword args, the call site needs the fix above.

#### Pitfall B: For-loop body has extra operations

**Problem**: The `for block in self.blocks:` loop in `transformer.forward()` contains operations *other than* the block call itself — such as `temb` reassignment, conditional checks, or tensor reshaping. After `CacheAdapter.apply()` replaces `self.blocks` with `UnifiedBlocks`, the caching wrapper **takes over the iteration** and only executes the block call; all extra operations inside the original loop body are **silently skipped**.

**Symptom**: Cache-enabled output is corrupted (low PSNR/SSIM, visual artifacts) because modulation parameters or intermediate tensors are stale or missing.

**Fix idea (reference)**: cache-dit's `ErnieImagePatchFunctor` moves the extra operations **outside** (before or after) the for-loop, so the loop body contains only the block call. In sglang, restructure the sglang-side forward the same way.

**Canonical example — `ErnieImagePatchFunctor`** (`<cache_dit_dir>/src/cache_dit/caching/patch_functors/functor_ernie_image.py`):

The original diffusers code reconstructs `temb` inside the loop body:

```python
# Original (diffusers) — temb reassigned INSIDE the for-loop:
for layer in self.layers:
    temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
    x = layer(x, rotary_pos_emb, temb, attention_mask=attention_mask)
```

After `CacheAdapter.apply()` replaces `self.layers` with `UnifiedBlocks`, the `temb = [...]` line is never executed — each block receives a stale or undefined `temb`. The patched version moves `temb` construction **before** the loop:

```python
# Patched — temb constructed ONCE before the loop:
temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
for layer in self.layers:
    x = layer(x, rotary_pos_emb, temb, attention_mask=attention_mask)
```

**How to detect this pitfall**: Inspect the for-loop body in `transformer.forward()`. If *any* line between `for ... in self.XXX:` and the actual block call does something other than a trivial `if torch.is_grad_enabled()` guard, the loop body needs restructuring.

#### Beyond the two canonical pitfalls

Pitfalls A and B are the two simplest cases (fix at the call site, or hoist one line out of the loop). Real models often need heavier structural fixes. cache-dit ships 13+ `PatchFunctor`s under `<cache_dit_dir>/src/cache_dit/caching/patch_functors/` — browse them as a **source of fix ideas only**. Recurring patterns include:

- **Per-block `forward()` replacement + block-id injection** — when the loop body has *per-block* extra operations that cannot simply be hoisted (they depend on the block index). The functor patches `transformer.forward()` **and** each block's `forward()`, and injects a `_block_id` / `_layer_id` onto every block so the patched block can look up per-block data (skip-connection lists, per-block encoder states, control hints). Examples: `HiDreamPatchFunctor`, `HunyuanDiTPatchFunctor`, `WanVACEPatchFunctor`, `ChromaPatchFunctor`, `GlmImagePatchFunctor`, `BriaFiboPatchFunctor`.
- **Block signature modification** — rewriting a block's `forward()` signature so the caching wrapper can bind it (e.g. `FluxPatchFunctor` adds an `encoder_hidden_states` parameter to `FluxSingleTransformerBlock` in older diffusers).
- **Block-list merge / dummy blocks** — structurally merging two `ModuleList`s into one for unified caching (e.g. `FluxPatchFunctor` merging `transformer_blocks` + `single_transformer_blocks` when `dummy_blocks_names` is set).

For sglang, whatever fix pattern you borrow, the resulting code must keep the exact same signature and produce identical output with caching disabled (verify via PSNR/SSIM) — and it lands in the sglang repo, not as a cache-dit PatchFunctor.


## More references

We recommend reading the following files for additional context:

- cache related source code: `<cache_dit_dir>/src/cache_dit/caching/`
