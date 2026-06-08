# Cosmos3 (NVIDIA Cosmos World Foundation Models)

SGLang-Diffusion serves NVIDIA Cosmos3-Nano (16B) and Cosmos3-Super (64B) through a
single Mixture-of-Transformers checkpoint. Text is embedded directly by the model's
Understanding (UND) pathway and the Generation (GEN) pathway cross-attends to it, so
there is no separate text encoder.

## Modes

| Mode | Trigger |
| --- | --- |
| Text-to-image (T2I) | `num_frames == 1` |
| Text-to-video (T2V) | default |
| Image-to-video (I2V) | a reference image is set |
| Video-to-video (V2V) | `image_path` is a video file |
| Joint video + sound | `generate_sound: true` (checkpoints with a sound head) |
| Action (policy / forward / inverse dynamics) | `action_mode` is set (checkpoints with an action head) |

V2V conditions on the first latent frames of the input video
(`condition_frame_indexes_vision`, default `[0, 1]`) and generates the rest. Action
modes condition on a robot observation and generate (or consume) an action chunk.
The embodiment selects the per-domain action heads.

The sampler is the native FlowUniPC rectified-flow schedule with a per-mode shift
(Nano T2I 3 / I2V 5 / T2V 10, Super a flat 5). T2I additionally applies CFG only in
the high-noise window `[400, 1000]`, skipping the unconditional pass at low noise.

## Serve

```shell
# Nano, single GPU
sglang serve --model-path nvidia/Cosmos3-Nano --model-type diffusion --attention-backend fa --port 8000

# Super (64B), 4 GPUs with CFG + Ulysses parallelism
sglang serve --model-path nvidia/Cosmos3-Super --model-type diffusion \
  --num-gpus 4 --cfg-parallel-size 2 --ulysses-degree 2 --port 8000
```

## Generate

```python
from sglang import DiffGenerator

gen = DiffGenerator.from_pretrained(model_path="nvidia/Cosmos3-Nano")

# Text-to-video with a joint audio track
gen.generate(sampling_params_kwargs=dict(
    prompt="A small warehouse robot moves a blue box across a clean floor.",
    width=1280, height=720, num_frames=189, fps=24,
    num_inference_steps=35, guidance_scale=6.0,
    generate_sound=True, save_output=True,
))
```

Sound is decoded by the bundled Oobleck/AVAE tokenizer and muxed into the MP4 as a
stereo 48 kHz AAC track. Sound generation runs on a single GPU.

```python
# Video-to-video: condition on the input video, regenerate with a new prompt
gen.generate(sampling_params_kwargs=dict(
    prompt="A robotic arm pours liquid into a glass on a white tabletop.",
    image_path="robot_pouring.mp4",
    width=1280, height=704, num_frames=45, fps=24,
    num_inference_steps=35, guidance_scale=6.0, save_output=True,
))

# Action: predict an action chunk from a robot observation (policy)
result = gen.generate(sampling_params_kwargs=dict(
    prompt="Put the pot to the left of the purple item.",
    image_path="observation.mp4",
    action_mode="policy", action_embodiment="bridge_orig_lerobot",
    action_chunk_size=16, fps=5, guidance_scale=1.0, num_inference_steps=30,
))
predicted_action = result.action  # [chunk, action_dim]
```

Action modes are `policy` (observation → action), `forward_dynamics` (observation +
action → video), and `inverse_dynamics` (full video → action). Action runs on a
single GPU. Without an `action_stats_path`, actions stay in the model's normalized
`[-1, 1]` space.

## Notes

- The full Cosmos3 omni checkpoint is loaded. The Qwen3-VL reasoner / prompt
  upsampling is a client-side step and is not bundled in the server.
- Guardrails are on by default when `cosmos-guardrail` is installed. Disable with
  `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`.
