# MiniMax-H3 video on V100

MiniMax-H3 generates a video and a matching audio track in one request.
This tree serves it on Volta: four V100 32 GB GPUs, fp16 compute (Volta has
no bf16), and the TileLang attention backend. Build the stack first.
[INSTALL.md](INSTALL.md) is the toolchain; this page is the video server.

The checkpoint is [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3).
It is split into weight partitions. One server process loads one partition.

```bash
pip install -U "huggingface_hub[cli]"
hf download MiniMaxAI/MiniMax-H3 --local-dir ~/models/MiniMax-H3
export H3_MODEL=~/models/MiniMax-H3
```

| `--model-variant` | Directory | Tasks |
|---|---|---|
| `fl2va` | `FL2VA/` | Text-to-video, and video conditioned on a first frame, a last frame, or both |
| `ref2va` | `Ref2VA/` | Reference image, reference video, reference audio, and video-to-video |

`i2v` and `l2v` are not task names. A first frame is `fl2va` with
`frame_index` 0. A last frame is `fl2va` with `frame_index` -1.

---

## Serve

Four GPUs that can see each other. Point `CUDA_VISIBLE_DEVICES` at them.
`ffmpeg` and `ffprobe` have to be on `PATH`; the server will not start
without them. The text encoder and both VAEs stay on the host; the diffusion
transformer stays on the GPUs. `--warmup-mode off` skips a startup generation.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST=7.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 4 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --performance-mode speed \
  --quantization v100_w4a16_awq \
  --attention-backend tilelang_fa_v100 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --enable-torch-compile false \
  --warmup-mode off \
  --host 0.0.0.0 \
  --port 30010
```

`scripts/serve_minimax_h3_v100.sh` is that `fl2va` command. Pass `ref2va`
for reference tasks. Override `H3_MODEL`, `H3_GPUS`, `H3_PORT`, and
`SGLANG_V100_VENV`. A local checkout must contain `FL2VA/`. Reference tasks
need `Ref2VA/` and a second start of the same script:

```bash
bash scripts/serve_minimax_h3_v100.sh ref2va
```

That process reads `Ref2VA/` and rejects `t2va` and `fl2va`. The `fl2va`
process rejects `ref2va`. Restart to switch.

W4A16 above is the shipped recipe. It is an on-load group-128 4-bit packing
of the transformer, under the name `v100_w4a16_awq`. The text encoder and
both VAEs stay in fp16.

Ready when the log says Uvicorn is listening:

```bash
curl -sf http://127.0.0.1:30010/health
```

TileLang on this path accepts fp16 and head size 128, which is the
transformer. The text encoder and the VAEs use PyTorch SDPA. That split is
automatic when `--attention-backend tilelang_fa_v100` is set.

---

## Request

`POST /v1/videos` returns immediately with an id. Poll `GET /v1/videos/{id}`
until `status` is `completed` or `failed`. Download
`GET /v1/videos/{id}/content`. The file is an MP4: H.264 video, AAC audio,
`yuv420p`.

`model` is the same string you passed to `--model-path`.

A plain sentence is a valid prompt. The checkpoint was trained on three
labeled fields, and keyframe tasks also want an alignment line in front of
them:

```text
integrated_multimodal_description: [Shot 1] Live-action. ...

overall_soundscape: ...

non_diegetic_music: None.
```

`integrated_multimodal_description` is the picture and the action.
`overall_soundscape` is what the soundtrack should contain.
`non_diegetic_music` is score that the scene itself does not hear. Write the
action so it fits the requested duration.

### Text to video

```bash
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "A tiger walks slowly through morning fog while birds and leaves are heard around it.",
    "task": "t2va",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5.0
    },
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 1101
  }'
```

`t2va` requires `conditions` to be an empty list.

### First frame, last frame, or both

`fl2va` takes one or two images. `frame_index` 0 is the first frame, -1 is
the last. Any other index is rejected. The file URI has to be readable on
every rank; a local `file://` absolute path is the reliable form.

First frame only:

```bash
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action. The clip starts on <Picture 1> and continues with calm motion.\n\noverall_soundscape: Quiet ambient sound from the scene.\n\nnon_diegetic_music: None.",
    "task": "fl2va",
    "conditions": [{
      "type": "image",
      "uri": "file:///absolute/path/first.png",
      "role": "keyframe",
      "frame_index": 0
    }],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5.0
    },
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 2101
  }'
```

Last frame only: the same body with `frame_index` -1, and an alignment line
that names the end of the clip. Both frames: two condition objects, indexes
`0` and `-1`, and an alignment line that names 0.00 seconds and the end.

`aspect_ratio` `"auto"` on `fl2va` follows the first keyframe. A last-frame-only
request has no first keyframe, so pass an explicit ratio there.

### Reference image, video, or audio

Start the server with `--model-variant ref2va`. Conditions use
`"role": "reference"`. Types:

| `type` | What it conditions |
|---|---|
| `image` | A still subject or style reference |
| `video` | A reference clip. Its soundtrack is the audio reference |
| `audio` | Audio only |
| `video_audio` | Picture and soundtrack together |

An image reference is resized on its own to a 2048px short edge, including
upscaling, and is not clamped to the target pixel budget. A `video` or
`video_audio` reference keeps its aspect ratio and is resized to the 768
short-edge canvas. Lowering `target.short_edge` shrinks the generated clip,
not that reference.

`start_time_seconds` is allowed on `video` and `video_audio` only. A ref2va
request can also carry one `fl2va`-style keyframe signature (`0`, `-1`, or
both) plus at least one reference.

A plain sentence is accepted. The checkpoint was trained on six sections, in
this order: `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`. Keep
labels consistent: `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`.
When a keyframe is also attached, put the alignment line in front of those
sections, the same way `fl2va` does.

If you omit `target.duration_seconds`, duration is taken from the single
audio-bearing reference (`audio`, `video`, or `video_audio`). Two of those
and no explicit duration is an error. The derived length still has to fall
in the 4–15 second range.

```bash
curl -sS -X POST http://127.0.0.1:30010/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "The subject in <Picture 1> walks through the same room.",
    "task": "ref2va",
    "conditions": [{
      "type": "image",
      "uri": "file:///absolute/path/subject.png",
      "role": "reference"
    }],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5.0
    },
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 3101
  }'
```

### Poll and save

```bash
while true; do
  status=$(curl -sS "http://127.0.0.1:30010/v1/videos/$video_id" | jq -r '.status')
  [ "$status" = completed ] && break
  [ "$status" = failed ] && { curl -sS "http://127.0.0.1:30010/v1/videos/$video_id"; exit 1; }
  sleep 5
done
curl -sS -L "http://127.0.0.1:30010/v1/videos/$video_id/content" -o clip.mp4
```

A failed job's `error` field is the reason. Admission errors (bad duration,
wrong task for this partition, a middle keyframe) come back as HTTP 400, or
as a job that fails before denoising.

---

## Picture size, length, and steps

`target` has three fields.

**`short_edge`.** 768 is the short edge MiniMax publishes recipes for.
Any other positive integer is accepted. The server scales it by the aspect
ratio, clamps the frame to a 768×1344 pixel budget, and rounds both sides to
a multiple of 32. A value other than 768 logs one warning and still runs.
At 16:9, 768 lands on 1344×768. A smaller edge such as 544 lands on 960×544
and costs less.

**`aspect_ratio`.** `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, or `auto`.
`auto` on `t2va` and `ref2va` is 16:9. `auto` on `fl2va` follows the first
keyframe.

**`duration_seconds`.** 4.0 through 15.0 inclusive. The frame count is
`round(duration × 24)`, then snapped up to the checkpoint's 17n+5 boundary.
Five seconds becomes 124 frames, which is 5.167 seconds in the file. The
soundtrack follows that same duration. Frame rate in the file is 24.

**`num_inference_steps`.** 50 is the setting the checkpoint is built around.
The minimum is 2. The sampler evaluates the model `steps - 1` times, so 50
steps is 49 evaluations. Fewer steps finish sooner and resolve less of the
image.

**`flow_shift` and `audio_flow_shift`.** Defaults are 12.0 and 3.0, and
those are the shifts to use. Both have to be positive finite numbers when
you set them. A video shift of 6 on a 12-step run left vertical bands across
the whole frame. The same canvas at shift 12, including shorter runs, did
not. That is the schedule, not a canvas or decode fault.

**`seed`.** A signed integer. It selects the initial noise for that request.

These fields are rejected. The checkpoint is guidance-distilled, with a
single positive branch, and the delivery format is the resolved canvas at
24 fps:

| Field | Why it is rejected |
|---|---|
| `fps`, `num_frames` | Timing comes from `target.duration_seconds` |
| `guidance_scale`, `guidance_scale_2`, `true_cfg_scale`, `negative_prompt` | No classifier-free guidance branch |
| `enable_frame_interpolation`, `enable_upscaling` | Output is the resolved canvas at 24 fps |
| `enable_teacache` | The packed video/audio loop has no TeaCache path |
| `quality=high` | That cache profile is an audited 4× H200 configuration. This server is V100 |

---

## Time

On 4× V100-32GB with the W4 recipe and `flow_shift` 12, measured 2026-09-28.
A step is one denoising evaluation (`num_inference_steps - 1` of them).

| Clip | Steps | Step time |
|---|---:|---:|
| 960×544, about 5 s, text or keyframe | 50 | 10–12 s |
| 1344×768, 5 s, text-to-video | 50 | about 30 s (1471 s total) |
| 960×544, 4.5 s, video reference | 12 | about 53 s |
| 1344×768, 4.5 s, image reference | 12 | about 42 s |
| 960×544, 15 s, image reference | 12 | about 84 s |

A video reference is resized to the 768 short-edge canvas even when the
generated clip is smaller, so those steps cost more than a text-only clip of
the same output size. The 15-second image reference peaked near 29.5 GB on
a 32 GB card. Decode after the loop is a few seconds.

---

## Volta behavior

The transformer weights are stored for bf16 hardware. On V100 the matmul and
attention kernels run in fp16, and the residual stream is kept in fp32 so a
deep stack of blocks does not overflow fp16 and turn the latents into NaNs.
That is specific to this port. A Hopper or Blackwell box serving the same
checkpoint in bf16 is a different numeric path.

Leave `quality` unset. The default is `lossless`, which does not enable the
H200 cache. `quality=high` is rejected on this server.
