# LTX One-Stage Native Scheduler Migration Report

This note records the current status of the LTX one-stage native scheduler migration, the validation performed so far, the multi-output smoke-test results, the issues encountered, and the proposed follow-up fixes for a PR write-up.

## Executive Summary

The LTX one-stage scheduler migration itself is numerically aligned with the previous pipeline-local scheduler.

For multi-output generation, the current status is more nuanced:

- The OpenAI-compatible `/v1/videos` single-request multi-output path is wired and can generate multiple outputs for LTX.
- LTX default CFG multi-output (`n=4`, default `guidance_scale=4.0`) is now fixed in the LTX-specific denoising stage by expanding text conditioning tensors / masks from prompt batch size to latent batch size before CFG concatenation.
- The earlier offline `DiffGenerator.generate(num_outputs_per_prompt=4)` failure is a different issue: it expands one request into a `list[Req]`, then the grouped executor passes that list into component residency code that expects a single `Req`.

So the accurate PR wording is:

> The native scheduler migration is complete and numerically aligned for LTX one-stage. LTX OpenAI API single-request multi-output now works for the tested LTX-2 `n=4` default-CFG smoke. Offline `DiffGenerator` grouped multi-output remains a separate execution-contract issue.

## What Changed

The scheduler migration moved LTX's one-stage flow-match scheduler out of the pipeline file and into the native scheduler directory.

Main implementation changes:

- Added `python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_ltx2_flow_match.py`.
- Removed the temporary pipeline-local `LTX2FlowMatchScheduler(diffusers.FlowMatchEulerDiscreteScheduler)` class from `python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py`.
- `_BaseLTX2Pipeline.initialize_pipeline()` now rebuilds the loaded diffusers scheduler config as the native `LTX2FlowMatchScheduler`.
- Added unit coverage for scheduler behavior and pipeline scheduler replacement.

The native scheduler preserves LTX-specific behavior:

- Custom `set_timesteps(sigmas=...)` path builds `timesteps = sigmas * num_train_timesteps` and appends terminal sigma `0`.
- `_time_shift_exponential()` keeps the previous ndarray behavior by computing through torch float32 and converting back to numpy.
- `step()` preserves the deterministic Euler update used by the previous LTX one-stage scheduler.
- `SchedulerRLMixin` remains available for rollout-related hooks.

## Validation Already Performed

### Unit Tests

Passed:

- `python -m pytest python/sglang/multimodal_gen/test/unit/test_ltx2_scheduler.py`
- `python -m pytest python/sglang/multimodal_gen/test/unit/test_ltx2_pipeline_scheduler.py`

Covered:

- Custom `sigmas` path.
- Dynamic shifting path.
- Euler `step()` update and `_step_index` progression.
- Presence of rollout mixin methods.
- `_BaseLTX2Pipeline.initialize_pipeline()` replacing the scheduler with native `LTX2FlowMatchScheduler`.

### Full Real Pipeline Per-Step Latent Dump

The real-pipeline comparison tested the old pipeline-local scheduler against the new native scheduler.

LTX-2.3 one-stage:

- 30 steps.
- I2V path.
- Per-step scheduler state and video/audio latents were dumped.
- 30/30 steps were bitwise identical for scheduler state and latents.
- Final raw RGB video was bitwise identical.
- Final decoded audio was not byte-identical, but cosine similarity was `0.9997002509748341`; since audio latents were bitwise identical, this was attributed to post-denoise audio encode/decode/container behavior, not scheduler drift.

LTX-2 one-stage:

- 40 steps.
- T2V path.
- 40/40 per-step JSON and tensor dumps matched.
- All tracked tensors were bitwise identical:
  - `timesteps`
  - `sigmas`
  - `sigma`
  - `sigma_next`
  - `video_before`
  - `video_after`
  - `audio_before`
  - `audio_after`
- Final raw RGB video SHA was identical.
- Final decoded f32 audio SHA was identical.

Conclusion:

> The scheduler migration itself does not introduce numerical drift in LTX one-stage denoising.

## Multi-Output Tests Performed

### Offline `DiffGenerator` Multi-Output Attempt

Test shape:

- Model: `Lightricks/LTX-2`
- Prompt: `"A small robot paints a red circle on a white wall while soft studio lights flicker."`
- 20 steps, 9 frames, 256x256, fps 8.
- Compared four single requests against `num_outputs_per_prompt=4`.

Observed result:

- Four separate single-output requests succeeded.
- `num_outputs_per_prompt=4` through offline `DiffGenerator` failed before denoising.

Failure:

```text
AttributeError: 'list' object has no attribute 'is_warmup'
```

Relevant path:

- `python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py:231`
  calls `expand_request_outputs(...)`.
- That creates a `list[Req]`.
- The grouped path eventually reaches component residency code that expects a single `Req` and reads `batch.is_warmup`.

Interpretation:

This is not a scheduler problem and not caused by the new native LTX scheduler. The failure happens before `LTX2FlowMatchScheduler.step()` is reached. It is a contract mismatch between the grouped request path and component residency management.

Proposed strategy:

- For a minimal LTX-only workaround, add an LTX-specific sequential fallback in `LTX2Pipeline.forward_batch()` or `_BaseLTX2Pipeline.forward_batch()`:

```python
def forward_batch(self, batches, server_args):
    if len(batches) == 1:
        return [self.forward(batches[0], server_args)]
    return [self.forward(batch, server_args) for batch in batches]
```

- This avoids the crash but does not provide true batch throughput improvement.
- For real throughput improvement, fix the grouped executor / component residency contract and then validate all LTX grouped stages.

### OpenAI `/v1/videos` Smoke: Why This Path Matters

The OpenAI video endpoint uses a different request shape than offline `DiffGenerator`.

Relevant code:

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:59-66`
  maps `n` or `num_outputs_per_prompt` into `num_outputs_per_prompt`.
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py:329`
  calls `scheduler_client.forward([batch])`.

This means OpenAI `/v1/videos` sends one `Req` with `num_outputs_per_prompt=4`, instead of a `list[Req]`. This avoids the earlier `list.is_warmup` grouped-path failure.

### OpenAI `/v1/videos`, `n=1`, Default CFG

Request:

- `n=1`
- `guidance_scale=4.0` default
- 4 inference steps
- 9 frames
- 256x256

Result:

- Success.
- Output:
  `/tmp/ltx_openai_smoke_outputs/6c415028-63c4-4945-9343-7a7daacb9ec9.mp4`
- `inference_time_s`: `1.419876734027639`
- `peak_memory_mb`: `48316.0`

Interpretation:

OpenAI `/v1/videos` works for normal single-output LTX requests with default CFG.

### OpenAI `/v1/videos`, `n=4`, CFG Disabled

Request:

- `n=4`
- `guidance_scale=1.0`
- seed list: `[42, 43, 44, 45]`
- 4 inference steps
- 9 frames
- 256x256

Result:

- Success.
- Four files generated:
  - `/tmp/ltx_openai_smoke_outputs/24ea86b4-9c94-457f-8c06-e6ded1d79bba_0.mp4`
  - `/tmp/ltx_openai_smoke_outputs/24ea86b4-9c94-457f-8c06-e6ded1d79bba_1.mp4`
  - `/tmp/ltx_openai_smoke_outputs/24ea86b4-9c94-457f-8c06-e6ded1d79bba_2.mp4`
  - `/tmp/ltx_openai_smoke_outputs/24ea86b4-9c94-457f-8c06-e6ded1d79bba_3.mp4`
- `inference_time_s`: `1.5994476759806275`
- `peak_memory_mb`: `48260.0`

Interpretation:

The OpenAI API single-`Req` multi-output route is functional for LTX when CFG is disabled. The native scheduler does not block multi-output generation.

### OpenAI `/v1/videos`, `n=4`, Default CFG

Request:

- `n=4`
- default `guidance_scale=4.0`
- seed list: `[42, 43, 44, 45]`
- 4 inference steps
- 9 frames
- 256x256

Result:

- Initially failed in `LTX2AVDenoisingStage` before the condition-batch fix.
- After the fix, succeeded and generated four files:
  - `/tmp/ltx_openai_cfg_fix_outputs/862892e1-047c-43e1-9a4b-7147bcd9aa49_0.mp4`
  - `/tmp/ltx_openai_cfg_fix_outputs/862892e1-047c-43e1-9a4b-7147bcd9aa49_1.mp4`
  - `/tmp/ltx_openai_cfg_fix_outputs/862892e1-047c-43e1-9a4b-7147bcd9aa49_2.mp4`
  - `/tmp/ltx_openai_cfg_fix_outputs/862892e1-047c-43e1-9a4b-7147bcd9aa49_3.mp4`
- `inference_time_s`: `3.117498349980451`
- `peak_memory_mb`: `48154.0`

Original error before the fix:

```text
The size of tensor a (8) must match the size of tensor b (2) at non-singleton dimension 0
```

Stack location:

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py:1742`
  calls `step.current_model(**model_kwargs)`.
- The error occurs inside LTX attention:
  `python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py:722`
  calls `scaled_dot_product_attention(...)`.

Interpretation:

This was the LTX CFG multi-output bug. The successful smoke confirms the LTX-specific condition/mask expansion now aligns text conditioning with the `2 * num_outputs_per_prompt` CFG model batch.

## Detailed CFG Failure Analysis

Default LTX-2 sampling uses `guidance_scale=4.0`. In `Req.validate()`, CFG is enabled when the effective CFG scale is greater than `1.0` and a negative prompt exists:

- `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py:316-323`

For `n=4`, the latent/model-input batch has 4 samples. In the two-branch CFG path, LTX duplicates the model input for unconditional and conditional branches:

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`

That makes the model-input batch size `8`:

```python
cfg_batch_size = batch_size * 2
model_kwargs = self._repeat_ltx2_model_kwargs_batch(model_kwargs, cfg_batch_size)
```

Before the fix, the text condition tensors and text attention masks in
`python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`
were only concatenated as `[negative, positive]`.

For a single prompt with `n=4`, this creates a mismatch:

- latent/model-input batch: `8` (`4 outputs * 2 CFG branches`)
- text condition / mask batch: `2` (`negative + positive`, not expanded per output)

That explains the observed attention error:

```text
tensor a (8) vs tensor b (2)
```

The key point is that the old LTX one-stage CFG path assumed either:

- one output per prompt, or
- condition tensors that have already been expanded to the output count.

The OpenAI `n=4` path gives one prompt and four output seeds, so LTX must explicitly tile the positive and negative conditioning to the output batch size before concatenating CFG branches.

The fix adds LTX-specific helpers in `ltx_2_denoising.py` that:

- expand condition tensors with `repeat_interleave` from prompt batch size to latent batch size,
- preserve per-prompt ordering for multi-prompt requests,
- concatenate CFG branches as `[negative for all outputs, positive for all outputs]`,
- apply the same expansion to audio text conditions and text attention masks.

## Why Qwen Does Not Hit This Same Failure

Qwen's supported multi-output path generally avoids the offline `list[Req]` grouped path:

- OpenAI/API usage sends one `Req`.
- Dynamic batching merges requests into one merged `Req`.

Also, Qwen's stage logic is designed around its existing single-`Req` batch abstraction, so its condition tensors are aligned with the requested batch shape. LTX has extra model-specific complexity:

- video + audio denoising,
- LTX-specific text connector outputs,
- explicit two-branch CFG logic,
- custom attention masks for LTX text/audio/video interactions.

The LTX failure was therefore not because native scheduler support was missing. It was because LTX-specific denoising had not expanded CFG conditioning tensors and masks for `num_outputs_per_prompt > 1`.

## Issues Encountered and Strategy

### Issue 1: Scheduler Used to Live in Pipeline File

Problem:

- LTX one-stage used a temporary scheduler subclass inside `ltx_2_pipeline.py`.
- That was inconsistent with the native scheduler registry/model-specific scheduler pattern.

Strategy:

- Move scheduler to `runtime/models/schedulers`.
- Implement it as an LTX-specific native scheduler class, not a thin subclass of the generic native FlowMatch scheduler.
- Keep only scheduler behavior in the scheduler file; keep LTX guider/audio/video/two-stage logic in pipeline/stages.

Status:

- Done.
- Numerically validated.

### Issue 2: Need Exact Alignment With Previous Scheduler

Problem:

- Scheduler changes can silently drift outputs.

Strategy:

- Unit-test scheduler internals.
- Compare old vs new scheduler using full real pipeline per-step dumps.
- Validate per-step scheduler states and video/audio latents.

Status:

- Done.
- LTX-2 and LTX-2.3 one-stage denoising matched bitwise at every tested step.

### Issue 3: Offline Multi-Output Crashes Before Denoising

Problem:

- `DiffGenerator.generate(num_outputs_per_prompt=4)` expands into `list[Req]`.
- The grouped executor path eventually passes a list to component residency code that expects a single `Req`.
- Failure:

```text
AttributeError: 'list' object has no attribute 'is_warmup'
```

Strategy options:

1. LTX-only low-risk fallback:
   - Implement `forward_batch()` in LTX pipeline to run each `Req` sequentially.
   - Pro: small blast radius.
   - Con: no true throughput gain.

2. Generic grouped execution fix:
   - Make component residency manager and grouped executor agree on a batch/list contract.
   - Pro: fixes the real abstraction mismatch.
   - Con: larger blast radius across diffusion pipelines.

3. Prefer OpenAI API single-`Req` multi-output path for this PR's smoke:
   - This avoids the grouped list path and matches the Qwen-style serving path more closely.

Status:

- Not fixed in this migration.
- OpenAI smoke was used to isolate the next LTX-specific issue.

### Issue 4: OpenAI `n=4` Default CFG Failed in LTX Denoising

Problem:

- OpenAI `/v1/videos` sends one `Req(num_outputs_per_prompt=4)`.
- That path reaches LTX denoising correctly.
- With default CFG enabled, LTX expands model inputs to batch `8`, but conditioning tensors/masks are still batch `2`.

Strategy:

- Fixed in LTX-specific denoising, not in the scheduler.
- Added helpers that expand per-prompt condition tensors and masks to match `batch_size` before building two-branch CFG tensors.
- For single prompt / multiple outputs, tile each of:
  - positive `encoder_hidden_states`,
  - negative `encoder_hidden_states`,
  - positive `audio_encoder_hidden_states`,
  - negative `audio_encoder_hidden_states`,
  - positive text attention mask,
  - negative text attention mask.
- Then build CFG order as:

```text
[negative for all outputs, positive for all outputs]
```

Expected shapes for `n=4`:

```text
negative text batch: 4
positive text batch: 4
CFG text batch:      8
latent CFG batch:    8
```

Status:

- Fixed and confirmed by smoke:
  - `n=4`, `guidance_scale=1.0` succeeds.
  - `n=4`, default `guidance_scale=4.0` succeeds.
  - `n=1`, default `guidance_scale=4.0` still succeeds.

### Issue 5: GPU Resource/OOM Noise During Smoke

Problem:

- Initial OpenAI smoke on GPU0 failed with CUDA OOM because external processes occupied nearly all visible memory.

Strategy:

- Verify GPU free memory using both `nvidia-smi` and PyTorch `torch.cuda.mem_get_info`.
- Move smoke to a less occupied GPU.

Status:

- Resolved for testing by running on GPU5.
- The meaningful failure after that was the CFG batch/mask mismatch, not OOM.

## Implemented Follow-Up

### Code Fix

Made an LTX-specific change in `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`.

Targeted the two-branch CFG path and stage-1 guider pass batching around:

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ltx_2_denoising.py`

Added helpers equivalent to:

```python
def _expand_condition_batch_for_ltx2_outputs(tensor, target_batch_size):
    if tensor is None:
        return None
    if tensor.shape[0] == target_batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat_interleave(target_batch_size, dim=0)
    raise ValueError(...)
```

Used it before CFG concatenation:

```python
negative_encoder = expand(negative_encoder, batch_size)
positive_encoder = expand(positive_encoder, batch_size)
encoder_hidden_states = torch.cat([negative_encoder, positive_encoder], dim=0)
```

Did the same for audio encoder states and text masks.

This should be kept in the LTX-specific denoising file because the bug is specific to LTX's custom audio/video CFG and attention mask construction.

### Tests Added / Re-run

1. Added unit-level shape tests in `python/sglang/multimodal_gen/test/unit/test_ltx2_denoising_cfg_batch.py`:
   - expands prompt-batch condition tensors to output-batch size with `repeat_interleave`,
   - preserves per-prompt output order,
   - concatenates CFG branches as `[negative all outputs, positive all outputs]`,
   - rejects unaligned prompt/output batch counts.

2. Re-ran scheduler and pipeline scheduler unit tests:
   - `python -m pytest python/sglang/multimodal_gen/test/unit/test_ltx2_scheduler.py python/sglang/multimodal_gen/test/unit/test_ltx2_pipeline_scheduler.py python/sglang/multimodal_gen/test/unit/test_ltx2_denoising_cfg_batch.py`
   - Result: `9 passed`.

3. OpenAI smoke:
   - `/v1/videos`
   - `n=4`
   - default `guidance_scale=4.0`
   - small dimensions / few steps
   - Result: success, `num_outputs == 4`, four output files exist.
   - `inference_time_s`: `3.117498349980451`
   - `peak_memory_mb`: `48154.0`

4. Regression smoke:
   - `/v1/videos`
   - `n=1`
   - default CFG
   - Result: success, `num_outputs == 1`.
   - `inference_time_s`: `1.2016618559719063`
   - `peak_memory_mb`: `48206.0`

5. Attempted LTX-2.3 OpenAI smoke:
   - `/v1/videos`
   - `n=4`
   - default LTX-2.3 CFG / stage-1 guider path
   - Result: blocked before serving by local disk exhaustion while materializing the LTX-2.3 overlay model.
   - Error:

```text
safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)
```

   - The failed materialization created `/root/.cache/sgl_diffusion/materialized_models/Lightricks__LTX-2.3-c24cea94ab17c493.tmp`.
   - That temporary directory was removed, recovering disk from roughly `3.5M` free to roughly `15G` free.
   - This is recorded as an environment/storage blocker, not a model execution failure.

6. Scheduler regression:
   - Keep existing scheduler unit tests.
   - Keep old-vs-new per-step latent comparison as migration evidence, not necessarily a required CI test because it is expensive.

### Optional Follow-Up

After CFG multi-output works through OpenAI API, decide separately whether to support offline `DiffGenerator.generate(num_outputs_per_prompt=4)` as true grouped execution or an LTX sequential fallback.

That should be treated as a separate execution-contract issue, not as part of the scheduler migration.

## Current PR Claim Boundary

Safe claims:

- LTX one-stage now uses a native SGLang scheduler class.
- The native scheduler matches the previous pipeline-local scheduler in tested one-stage trajectories.
- OpenAI `/v1/videos` is wired for LTX and can run LTX single-output requests.
- OpenAI `/v1/videos` can run LTX `n=4` when CFG is disabled.
- OpenAI `/v1/videos` can run the tested LTX-2 `n=4` default-CFG smoke after the LTX denoising condition-batch expansion fix.

Do not claim yet:

- Offline `DiffGenerator` multi-output grouped execution is fixed.
- LTX true batched multi-output throughput has been fully validated.
- LTX-2.3 OpenAI `n=4` default-CFG smoke has been completed in this environment; it was blocked by local disk space during overlay materialization.
