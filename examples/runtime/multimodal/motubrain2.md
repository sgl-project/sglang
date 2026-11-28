# Motubrain 2.0 action inference

Install the Motubrain2 inference package in the GPU server's Python environment. You will need the robot profile, policy weights and the encoders used by the checkpoint. Astribot uses the DA3 DinoV2 geometry encoder. [Motubrain runtime and assets](https://github.com/shengshu-ai/Motubrain-2.0)

Install SGLang Diffusion and its Motubrain backend registration from the same checkout. The following commands assume that checkout is at `/opt/sglang`. Substitute the absolute paths to your checkout, model and pipeline configuration on the GPU server.

```bash
python -m pip install -e '/opt/sglang/python[diffusion]' /opt/sglang/examples/runtime/multimodal/motubrain2_backend
sglang serve --model-type motubrain2 --help
```

Start the policy on a compute node. For an existing robot deployment, keep the model path expected by the bridge and reuse its configured server address and port. The example uses the Astribot S1 Distill deployment.

```bash
sglang serve /opt/motubrain-s1/models/Distill \
  --model-type motubrain2 --num-gpus 1 \
  --pipeline-config-path /opt/motubrain-s1/config/sglang.json --warmup-mode off \
  --host 127.0.0.1 --port 30000
```

The same command accepts Base weights. The action stage runs the Motubrain observation encoder and native policy. It can reuse encoded inputs when the instruction, camera frames, robot state, camera transform and history are identical.

The JSON fields `wan`, `text_encoder` and `profile` identify the Wan VAE, T5Gemma encoder and robot profile. For Astribot, also provide `da3` and, for standalone weight files, `wan_config` and `da3_config`. Keep the profile's normalization assets beside the profile, or use its native `action_stats` override. The model directory may contain the released policy weights directly or a serving bundle with a `policy` directory and profile. The checkpoint supplies the sampler, diffusion shifts and default step count.

Read model metadata at `GET /v1/actions/metadata` for the required cameras, frame counts, state width and output shape. Send `POST /v1/actions/generations` with the instruction at `input.task` and the observed cameras, state and extrinsic at `input.observation`. The `images` dictionary accepts an RGB image or ordered RGB clip per camera. MessagePack can carry uint8 arrays directly. The response contains decoded controller actions at `data[0].action.values`.

WebSocket clients use `/v1/actions/realtime` for the action envelope and `/openpi/policy` for raw observations and actions. Each named HTTP session and each WebSocket connection has its own random number generator. Omit `parameters.seed` to advance that generator from its configured starting seed (default 0). An explicit seed applies to a single request and preserves the session's generator state. Set `input.observation.reset` to restart the episode, including its generator and action prefix.

Enable realtime chunking in server configuration with `rtc_enabled`, `rtc_execute_prefix` and `rtc_delay`. HTTP clients must then send a distinct `input.observation.session_id` per episode. With RTC disabled, an HTTP request that omits the session ID starts from the configured seed each time. Set `parameters.num_inference_steps` to override the checkpoint step count.

To supply controller-space waypoints directly, send `input.observation.rtc` with `future` shaped `[N, action_dim]`, an integer `delay >= 0`, and an optional `anchor` shaped `[action_dim]`. In flat OpenPI requests use `rtc` at the top level. An explicit RTC constraint overrides the stored action prefix and also works with automatic RTC disabled. The policy uses the anchor as the controller pose and locks up to `min(delay, N, action_horizon)` future waypoints. A zero delay preserves the supplied pose while locking zero waypoints.

## Precision

The default BF16 policy loads `model.safetensors`. Pass `--mxfp8` on the serve command or set `"mxfp8": true` in the pipeline JSON to load `model-mxfp8.safetensors` from the same bundle. The bundle must include the matching `manifest.json`, `config.json` and `sampling.json`. On a CUDA Blackwell GPU, the native loader verifies the weight, model config and sampling config against the manifest. Leave compilation disabled for MXFP8. The HTTP and WebSocket request formats are identical for both precisions.

For a dedicated MXFP8 deployment, use FlashInfer 0.7.5 in the policy environment. SGLang's default environment pins 0.7.4. After installing SGLang, run the following command in that environment.

```bash
python -m pip install 'flashinfer-python[cu13]==0.7.5' 'motubrain2[mxfp8] @ git+https://github.com/shengshu-ai/Motubrain-2.0.git@308137cd0ce9de48ec5e6ed3bd0619859db22c9f'
```

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
