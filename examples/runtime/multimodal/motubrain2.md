# Motubrain 2.0 action inference

Install SGLang Diffusion and the Motubrain2 inference package in the same Python environment. This example requires access to the Motubrain runtime, robot profile, policy weights and the encoders used by the checkpoint. Astribot uses the DA3 DinoV2 geometry encoder. [Motubrain runtime and assets](https://github.com/shengshu-ai/Motubrain-2.0)

Run `sglang serve` on a compute node with the local model directory as its positional argument. Set `--model-type diffusion`, `--pipeline-class-name Motubrain2Pipeline`, `--num-gpus 1` and `--pipeline-config-path` with an absolute path to a JSON configuration. This pipeline ships in SGLang and is discovered automatically.

The JSON fields `wan`, `text_encoder` and `profile` identify the Wan VAE, T5Gemma encoder and robot profile. For Astribot, also provide `da3` and, for standalone weight files, `wan_config` and `da3_config`. Keep the profile's normalization assets beside the profile, or use its native `action_stats` override. The model directory may contain the released policy weights directly or a serving bundle with a `policy` directory and profile. The checkpoint supplies the sampler, diffusion shifts and default step count.

Read model metadata at `GET /v1/actions/metadata` for the required cameras, frame counts, state width and output shape. Send `POST /v1/actions/generations` with the instruction at `input.task` and the observed cameras, state and extrinsic at `input.observation`. The `images` dictionary accepts an RGB image or ordered RGB clip per camera. MessagePack can carry uint8 arrays directly. The response contains decoded controller actions at `data[0].action.values`.

Each named HTTP session and each Action WebSocket or OpenPI connection has its own random number generator. Omit `parameters.seed` to advance that generator from its configured starting seed (default 0). An explicit seed applies to a single request and preserves the session's generator state. Set `input.observation.reset` to restart the episode, including its generator and action prefix.

Enable realtime chunking in server configuration with `rtc_enabled`, `rtc_execute_prefix` and `rtc_delay`. HTTP clients must then send a distinct `input.observation.session_id` per episode. With RTC disabled, an HTTP request that omits the session ID starts from the configured seed each time. Set `parameters.num_inference_steps` to override the checkpoint step count.

To supply controller-space waypoints directly, send `input.observation.rtc` with `future` shaped `[N, action_dim]`, an integer `delay >= 0`, and an optional `anchor` shaped `[action_dim]`. In flat OpenPI requests use `rtc` at the top level. An explicit RTC constraint overrides the stored action prefix and also works with automatic RTC disabled. The policy uses the anchor as the controller pose and locks up to `min(delay, N, action_horizon)` future waypoints. A zero delay preserves the supplied pose while locking zero waypoints.

## Precision

The pipeline defaults to BF16. For video-stream MXFP8 the native loader accepts `video_mxfp8` and an optional `video_mxfp8_overlay`. SGLang pins FlashInfer 0.7.4, while the current Motubrain MXFP8 extra requires 0.7.5. The BF16 configuration is the tested installation for this integration.

## Optional DiT cache

The native policy can reuse velocity predictions across denoising steps within a single request. With `dit_cache_enabled` omitted or set to `null`, the released Base checkpoint enables the cache and the four-step Distill checkpoint leaves it off. The native loader identifies Base by the stored 30-step Euler recipe with video shift 6 and action shift 1. Request-level changes to the step count preserve the checkpoint default.

Set `dit_cache_enabled` to `true` in the pipeline JSON to enable caching for either model, or to `false` to force it off. Enabling the cache without explicit thresholds selects cosine similarities of 0.95 and 0.93. An explicit primary threshold also enables caching in automatic mode. To use a single threshold, supply only the primary value. Use a Motubrain2 package with checkpoint-aware DiT caching. This example specifies the enabled defaults.

```json
{
  "dit_cache_enabled": true,
  "dit_cache_threshold": 0.95,
  "dit_cache_threshold_2": 0.93,
  "dit_cache_skip_steps": 4,
  "dit_cache_skip_steps_2": 2,
  "dit_cache_damping": 0.25
}
```

The primary threshold takes precedence. The two skip intervals default to 4 and 2, and the extrapolation damping defaults to 0.25. Reusing velocities can change the video and action predictions. Compare the four native outputs and timing on the same recorded observation and seed before selecting settings for a robot.

## Return all four native tensors

Set the boolean `parameters.return_native_outputs` to `true` on an HTTP request. On a flat OpenPI request, put `return_native_outputs` alongside the observation fields. The same forward pass produces `video` (packed video latents), `action` (model coordinates), `decoded_action` (controller coordinates) and `decoded_valid` (boolean validity mask). Both JSON and MessagePack expose the archive at `extras.native_outputs`. The safetensors header preserves shapes and dtypes.

```python
import base64
from safetensors.torch import load

archive = response["extras"]["native_outputs"]
assert archive["format"] == "safetensors" and archive["encoding"] == "base64"
outputs = load(base64.b64decode(archive["data"], validate=True))
actions = outputs["decoded_action"]
```
