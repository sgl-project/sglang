---
name: add-diffusion-model
description: Use when adding a new diffusion model or pipeline variant to SGLang Diffusion (sglang.multimodal_gen).
---

# Add a Diffusion Model to SGLang

Use this skill when adding a new diffusion model or pipeline variant to
`sglang.multimodal_gen`.

The implementation guide is
`docs/docs/sglang-diffusion/support_new_models.mdx`. It owns the file map, the
code skeleton for each piece, the pipeline shapes, the registry, the
distributed and memory contract, and the PR checklist. Read it first. This
skill gives the order of work and the checks between steps; it does not repeat
the guide's code, and the guide wins where the two disagree.

Paths below are relative to `python/sglang/multimodal_gen/` unless they start
with `docs/`.

## 1. Read the reference before writing code

Work from the model's actual source: its Diffusers pipeline, the official
implementation, or the checkpoint's `model_index.json`. If the user gave a
Hugging Face model ID or repo, inspect it yourself first; ask only when the
reference is private, ambiguous, or unavailable. Do not infer the architecture
from the model name.

Write down the items listed under "Read the Reference First" in the guide
before creating files.

## 2. Pick the smallest change

Follow "Start With the Smallest Change" in the guide. Compare the model with
the families in `docs/docs/sglang-diffusion/compatibility_matrix.mdx` and with
`runtime/pipelines/` before adding anything:

- a new checkpoint of an existing family is a config variant, not a new
  pipeline class; FireRed-Image-Edit, for example, is registered on the
  Qwen-Image edit config
- a stage, VAE, text encoder, or scheduler that already covers most of what the
  model needs is extended, not copied
- a new file has to carry behavior that existing code cannot express

## 3. Choose the pipeline shape

Use the order in "Choose a Pipeline Shape" in the guide, and stop at the first
shape that fits:

1. native stages through the `add_standard_*` helpers, such as
   `add_standard_t2i_stages`, `add_standard_ti2i_stages`, and
   `add_standard_ti2v_stages`
2. a subclass of the one native stage that differs
3. a custom single-purpose stage inserted between native stages
4. an aggregated `BeforeDenoisingStage`, only as a last resort

Complex preprocessing is not a reason to jump to shape 4. An aggregated stage
hides several responsibilities in one place and bypasses the shared hooks for
offload, profiling, disaggregation, and batching, so the PR should say why
shapes 1-3 do not fit.

In-tree examples of each shape:

| Shape | Pipeline file under `runtime/pipelines/` (model-specific stage) |
| --- | --- |
| Native stages | `flux.py`, `flux_2.py`, `joy_image.py`, `wan_i2v_pipeline.py`, `wan_pipeline.py` |
| Native-stage subclass | `sana_video.py` (`SanaVideoTextEncodingStage(TextEncodingStage)`), `lingbot_video_moe.py` (`LingBotVideoTextEncodingStage(TextEncodingStage)`), `ideogram.py` (`Ideogram4TextEncodingStage`, `Ideogram4DenoisingStage`) |
| Custom single-purpose stage | `stable_diffusion_3.py` (`SD3ConditioningStage`), `longcat_image.py` (`LongCatPromptRewriteStage`), `ernie_image.py` (`PromptEnhancementStage`) |
| Aggregated stage | `krea2.py` (`Krea2BeforeDenoisingStage`), `glm_image.py`, `sana_wm_pipeline.py`, `hunyuan3d_pipeline.py`, `QwenImageLayeredPipeline` in `qwen_image.py` |

### Coupled multimodal models

One checkpoint may expose several tightly coupled modalities or request
profiles that the generic image and video sampling fields cannot describe.
MiniMax-H3 is the reference: `runtime/pipelines/minimax_h3_pipeline.py` selects
a partition from one root model ID, validates `task` / `conditions` / `target`
before queueing, packs text, video, and audio tokens into one denoise sequence,
and returns synchronized video and audio. `configs/sample/minimax_h3.py` owns
its request fields.

Such a model still uses `ComposedPipelineBase`. Keep request validation, media
materialization, packed-sequence construction, per-modality encode and decode,
and presentation as separate single-purpose stages under
`runtime/pipelines_core/stages/model_specific_stages/<model>/`. Do not force
coupled state into the standard `DenoisingStage` / `DecodingStage` contract to
resemble a simpler model. `cosmos3_pipeline.py` is a second example of a chain
made only of model-specific single-purpose stages.

- Take this path only with source evidence that the public API, the scheduler,
  or the joint latent state needs it.
- Offline `generate` and HTTP serving must build the same validated request
  object, and request-contract tests must cover it.
- The root checkpoint plus a variant flag must resolve the partition; users
  must not have to pick internal subdirectories.
- Reject an invalid task, condition role or order, target canvas or duration,
  or output container before queueing.
- Check every output stream. A valid video is not enough when the model also
  generates audio or actions.

For a pipeline that serves several tasks from one configuration, see "Declare
multiple task types" in the guide.

## 4. Implement the pieces

Follow "Implement the Pieces" in the guide for sampling params, pipeline
config, pipeline wiring, component loaders, distributed and memory
integration, and the registry. Checks the guide does not spell out:

- **Weights.** Keep the reference parameter names, or provide an explicit
  mapping. Audit missing and unexpected keys instead of accepting a partial
  load.
- **Shared layers.** Build the DiT on `runtime/layers/`: the attention backend
  selector, the parallel linear layers, and the fused norm layers. Check
  `existing-fast-paths.md` in the `sglang-diffusion-benchmark-profile` skill
  before writing a model-local version of a norm, RoPE, or modulation chain.
- **Config base.** There is no `VideoPipelineConfig`. Image models subclass
  `ImagePipelineConfig` or `SpatialImagePipelineConfig`; video models set a
  video `task_type` and subclass `PipelineConfig` or their family's base. Copy
  hook signatures, including the decode-side `get_decode_scale_and_shift`,
  `post_denoising_loop`, and `post_decoding`, from
  `configs/pipeline_configs/base.py`, not from another model.
- **Component configs.** Model the DiT and VAE config classes on a neighbor
  under `configs/models/dits/` and `configs/models/vaes/`.
- **Rotary style.** `is_neox_style=True` is the split-half rotation and
  `False` is the interleaved one. Confirm which one the reference uses.
- **VAE precision.** `vae_precision` defaults to `fp32`. Lower it only after
  comparing decode output against the reference.
- **Registry.** Prefer exact `hf_model_paths` for public checkpoints used in
  docs or tests. Add `model_detectors` only where local mirrors or renamed
  repos are common, and check that a detector does not also match an older
  family.
- **Shared stages.** When the model does not fit a shared stage, go back to
  step 3. Do not add model-specific branches to a shared stage.
- **Guide sections with no step here.** "Video encode hand-off", "Conditioning
  reuse", and the `check_cancelled` rule for custom sampling loops apply to
  every port; read them before writing a VAE, an encoder call, or a loop.

## 5. Verify the port

Run the steps in "Verify the Port" in the guide. Noisy or garbled output means
the port is wrong; do not ship it. Find the first diverging tensor instead of
judging the final image:

1. Run the reference and SGLang with the same prompt, seed, shape, and steps.
2. Compare the text encoder output and prompt embeddings.
3. Compare the initial latents, timesteps, and sigmas.
4. Compare the first denoiser prediction.
5. Compare VAE decode on the same latents.

The guide's "Common failure points" list covers the usual causes.

## 6. Decide the ComfyUI route (optional)

`docs/docs/sglang-diffusion/comfyui.mdx` describes the two modes and how to add
a model to integrated mode: a checkpoint spec under
`runtime/loader/comfyui_checkpoints/` and a `ComfyUIModelAdapter`
(`apps/ComfyUI_SGLDiffusion/executors/adapter.py`), with no extra pipeline
class.

- **Server mode** costs nothing when the request fits the existing image and
  video fields. Pass model-specific fields through `extra_fields`; the client
  in `apps/ComfyUI_SGLDiffusion/core/server_api.py` needs no per-model change.
  Add a node in `nodes.py` only when the inputs deserve ComfyUI widgets.
  `SGLDiffusionGenerateH3` is the worked example.
- **Integrated mode** keeps ComfyUI's text encoders, VAE, and sampler loop, and
  SGLang runs one DiT forward per sampler step. It fits a model whose
  conditioning and latents ComfyUI can already build and decode.

The deciding question is whether ComfyUI's sampler can drive the model's loop
unchanged. If reproducing the conditioning inside ComfyUI would duplicate
stages the server already runs, or the model needs inputs ComfyUI cannot
supply, use server mode.

## 7. Validate the model for Breakable CUDA Graph (optional)

Do this only after eager output matches the reference. A forward that captures
once is not enough.

`--enable-breakable-cuda-graph` runs on any model. The lists below record the
validated checkpoints; a model outside them stays enabled and logs
`[Diffusion BCG] not validated`. Marking a model validated has three parts:

1. the resolved pipeline config class in
   `BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS`
   (`runtime/server_args/server_args.py`)
2. the model IDs and safe basename aliases in
   `BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS` (same file); a config selected
   only by `--pipeline-class-name` goes in
   `BREAKABLE_CUDA_GRAPH_MODEL_ID_EXEMPT_PIPELINE_CONFIGS` instead
3. a prompt padder under `runtime/breakable_cuda_graph/model_padders/`, added
   to the imports in `_ensure_model_padders_registered`
   (`runtime/breakable_cuda_graph/prompt_padding.py`); a padder module that is
   not imported there never registers

The padder is model semantics, not a shape utility. Reuse
`pad_masked_prompt_kwargs` only when the model already consumes a real mask and
zero-padding every coupled text tensor leaves attention and RoPE unchanged.
The existing padders show the contracts:

- Qwen-Image pads embeddings, masks, text RoPE caches, and sequence-length
  metadata together, and synthesizes a mask when eager did not need one.
- Ideogram pads the combined text-image sequence and carries a replay-local
  `DynamicVarlenMaskMeta`.
- Z-Image keeps the native prompt length because extra tokens change its
  output.
- MiniMax-H3 buckets only within compatible packed-sequence alignment groups.
- LongCat-Image and default SANA-Video pass the prompt through. LongCat sends
  a 512-token prompt body, plus image tokens when editing, and padding it
  would change attention and RoPE. SANA-Video's default text stage always
  emits 300 tokens.

Keep mask construction active at batch size one; an `if batch > 1` shortcut
lets eager pass while the graph attends padded tokens. Rebuild anything that
depends on live lengths from the replay buffers on every replay. Do not bake
Python lists, varlen indices, or weakly referenced tensors from warmup into the
graph.

Warmup captures graphs only at its own `quality`, which is `lossless` unless
`--warmup-sampling-params` sets another. A request at a different level runs
eager and logs `has no captured breakable CUDA graphs`, so a run at that level
is not a BCG measurement.

A BCG change is validated when:

- warmup logs `[Diffusion BCG] captured`
- the log has none of `not validated`, `capture failed`,
  `warmup captured no breakable CUDA graphs`, `serving signature MISSED`, or
  `has no captured breakable CUDA graphs`
- eager and BCG output are compared for the same prompt, seed, shape, steps,
  guidance, dtype, topology, and `quality`. BCG is not guaranteed to be
  byte-identical ("Per-model tuning starting points" in
  `docs/docs/sglang-diffusion/performance-optimization.mdx`). Report whether
  the hashes match; when they do not, report SSIM and PSNR against eager and
  name the cause, such as a padder that lengthens the attended sequence
- short and long prompts exercise every intended bucket, and an over-limit
  prompt falls back deliberately
- for video, the captured frame count and conditioning shapes match the
  request: `--warmup-resolutions` sets width and height only, and
  `--warmup-num-frames` sets the frame count
- unit tests cover the aliases, the pipeline config list, fixed lengths, masks,
  RoPE and position tensors, and replay-local metadata

## 8. Put non-bit-exact fast paths behind `quality`

`quality` has three cumulative levels: `exact`, `lossless` (the default), and
`high`. "Numerical contracts and quality tiers" in
`docs/docs/sglang-diffusion/fused_kernels.mdx` defines them, and
`QUALITY_LEVELS` in `configs/sample/sampling_params.py` is the source of truth.

- A fast path that reproduces the reference rounding bit for bit mounts
  unconditionally behind a `BitExactFusionGate`.
- A fusion that keeps the reference math and the precision of every operand and
  accumulator, and only moves where the rounding happens, registers at
  `lossless`. It then also mounts at `high`. `exact` runs the reference chain.
- A fusion that quantizes or lowers an operand or accumulator precision
  registers at `high`. Sparse, caching, and other approximate paths are also
  `high`-only, and stay outside the fusion handlers below.

Integrate through the site framework in `sglang.kernels.ops.diffusion.sites`:
mark sites during model construction with a `QualityGatedFusion`, and register
it with its lowest level in `_QUALITY_FUSION_HANDLERS`
(`runtime/pipelines_core/stages/denoising.py`). Do not scatter request-policy
branches through the DiT.

- Coupled sites mount all or nothing.
- A site fails closed on an unsupported dtype, shape, layout, backend, BCG
  state, or compile state.
- A fusion registered at `lossless` passes the admission checks in
  `test/quality_tier_admission.py`.
- Each family has site-level guard and parity tests and one model wiring test.

## 9. Tests and performance data

Once the model produces coherent output, read
[references/testing-and-accuracy.md](references/testing-and-accuracy.md) before
adding GPU cases, component-accuracy hooks or skips, suite entries, or
benchmark claims.

For performance claims, use the `sglang-diffusion-benchmark-profile` skill. Its
`--quality-bcg-matrix` runs eager and BCG at `exact`, `lossless`, and `high` on
one GPU set, and the skill states the evidence a performance PR needs.

When users need a new launch command, add or update the model's page under
`docs/cookbook/diffusion/` with the `cookbook-add-model` skill.

## Before opening the PR

Go through "PR Checklist" in the guide, then confirm:

- [ ] The PR states the pipeline shape, and why an earlier shape did not fit if
      it is not shape 1.
- [ ] Output is compared against the reference with a fixed prompt and seed.
- [ ] The model is either marked BCG-validated in full (pipeline config,
      model ID, and registered padder) or not at all.
- [ ] When BCG is validated, capture and replay are proven at the served
      `quality`, and the output is compared against eager: matching hashes, or
      SSIM and PSNR with the cause of the difference.
- [ ] Every non-bit-exact fast path is registered at its `quality` level and
      `exact` still runs the reference chain.
- [ ] For a coupled multimodal model, every output stream is checked and both
      entry points share one request contract.
