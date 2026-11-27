---
title: LingBot Video MoE
description: Serve the native LingBot Video MoE 30B-A3B text-to-video model with SGLang Diffusion.
metatags:
    description: "Run robbyant/lingbot-video-moe-30b-a3b text-to-video generation with SGLang Diffusion."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';

<DiffusionModelTags tags={["video", "text-to-video", "mixture-of-experts", "30B-A3B"]} />

import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/LingBot/lingbot-video.jsx';

## 1. Quick start

Follow the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation) for your GPU platform.

<a id="2-installation" />

<Deployment config={config} />

### Optional: Prompt rewriter

Rather than writing the JSON by hand, use the rewriter the model was released with. It runs in two turns: expand the free-text prompt into a detailed caption on the base VLM, then map that caption to JSON with the rewriter LoRA enabled. The weights are `Qwen/Qwen3.6-27B` plus `robbyant/lingbot-video-rewriter-lora`, a second model rather than part of the checkpoint.

The rewriter has no CLI flags, so configure it in a config file:

```bash Command
cat > /tmp/lingbot_rewriter.json <<'EOF'
{
  "model_path": "robbyant/lingbot-video-moe-30b-a3b",
  "prompt": "A red fox trots across fresh snow in a pine forest.",
  "num_frames": 81,
  "rewriter_model_path": "Qwen/Qwen3.6-27B",
  "rewriter_adapter_path": "<path to lingbot-video-rewriter-lora>"
}
EOF

sglang generate --config /tmp/lingbot_rewriter.json --save-output
```

This loads the VLM in the diffusion process and switches the adapter between turns, which is what the reference implementation does. It holds roughly 54 GB in bf16 on top of the DiT and has no offload path. `rewriter_device_map` defaults to `"auto"`, which spreads it over every visible GPU; to keep it off the DiT GPUs, point it at GPUs of its own or use the served option below. An under-provisioned VLM degrades quietly rather than failing.

To serve the rewriter separately instead, launch it on its own port, for example `--port 30001`, set `rewriter_url` to it, and name a model per turn:

```json Example
{
  "rewriter_url": "http://127.0.0.1:30001",
  "rewriter_expand_model": "Qwen/Qwen3.6-27B",
  "rewriter_map_model": "lingbot-rewriter"
}
```

The mapping turn needs the adapter applied. `--lora-paths` does not accept this adapter: `Qwen/Qwen3.6-27B` is a `qwen3_5`-architecture checkpoint whose attention projections SGLang loads fused, while the adapter targets them unfused through a regex. So serve a checkpoint with the adapter already merged in, as its own endpoint via `rewriter_map_url` when one server cannot hold both.

Prompts that already start with `{` are passed through untouched. If the mapping turn returns nothing parseable the request fails rather than falling back to the free-text prompt, which the DiT would render as a broken video. Setting `rewriter_model_path` without `rewriter_adapter_path` is rejected at startup, since the mapping turn would otherwise silently run base weights. Leaving both `rewriter_url` and `rewriter_model_path` unset, the default, skips the stage.

Set `"rewriter_auto_negative": true` to also prune the negative prompt per request, which requires one of the two backends above. The shipped negative is deliberately over-complete and self-contradictory, so terms this caption legitimately wants push the sample away from the request. The rewriter deletes those terms and keeps the rest verbatim; a few, such as `underexposed` for a night scene and `motion blur` for a long exposure, are also deleted by rule afterwards because the model tends to leave them in. Because a merged request carries one negative prompt for several captions, enabling this turns dynamic batching off for the server.

<a id="1-model-introduction" />

## 2. Model capabilities

[LingBot Video MoE 30B-A3B](https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b)
is a text-to-video mixture-of-experts model served through a native SGLang
Diffusion pipeline. Its main advantage is the MoE economics: 30B total
parameters with 3B active per token, so a plain `sglang serve` invocation
covers the released 480×480, 81-frame profile without parallelism flags.

Prompting is the distinctive part of this checkpoint: it expects a structured
JSON caption rather than an unexpanded natural-language prompt, passed as the
request's `prompt` string. A compact 17-frame, 12-step smoke-test profile is
available for quick validation next to the released 40-step profile.

The public checkpoint served by the pipeline:

| Model ID | Task | Default output |
| --- | --- | --- |
| `robbyant/lingbot-video-moe-30b-a3b` | Text to video | 480x480, 81 frames at 16 FPS |

The checkpoint expects a structured JSON caption rather than an unexpanded
natural-language prompt. The JSON is passed as the request's `prompt` string;
it is not an `extra_params` object.

## 3. Serve LingBot Video MoE

Start the server with the Hugging Face model ID:

```bash Command
sglang serve \
  --model-path robbyant/lingbot-video-moe-30b-a3b \
  --port 30010
```

## 4. Generate a video

The following request uses the compact 17-frame, 12-step smoke-test profile.
Use the model defaults of 81 frames and 40 steps for the released generation
profile.

```python Python
import json
import time
from pathlib import Path

import requests

base_url = "http://127.0.0.1:30010"
prompt = json.dumps(
    {
        "comprehensive_description": {
            "scene_content_description": (
                "A small silver robot arm on a white table slowly reaches "
                "toward a red cube. The background is a softly lit laboratory wall."
            ),
            "camera_movement_description": (
                "The camera is static at eye level in a medium shot."
            ),
        },
        "camera_info": {
            "color": "Neutral",
            "frame_size": "Medium",
            "shot_type_angle": "Eye level",
            "lens_size": "Medium",
            "composition": "Center",
            "lighting": "Soft light",
            "lighting_type": "Artificial light",
        },
        "world_knowledge": [],
        "prominent_elements": [
            {
                "name": "robot arm",
                "description": "A small silver robot arm with a two-finger gripper.",
                "actions": [
                    {
                        "timestamp": "[0.0s - 1.0s]",
                        "action": "reaches toward the red cube",
                    }
                ],
                "location": "center of the frame",
                "relative_size": "dominant",
                "shape_and_color": "articulated silver metal arm",
                "texture": "brushed metal",
                "appearance_details": "two-finger gripper and visible joints",
                "relationship": "reaching toward the red cube on the table",
                "orientation": "upright, base on the table",
                "pose": "reaching",
            }
        ],
    },
    separators=(",", ":"),
)

response = requests.post(
    f"{base_url}/v1/videos",
    json={
        "model": "robbyant/lingbot-video-moe-30b-a3b",
        "prompt": prompt,
        "size": "640x384",
        "num_frames": 17,
        "fps": 16,
        "num_inference_steps": 12,
        "guidance_scale": 6.0,
        "flow_shift": 3.0,
        "seed": 0,
    },
    timeout=60,
)
response.raise_for_status()
video_id = response.json()["id"]

while True:
    job = requests.get(f"{base_url}/v1/videos/{video_id}", timeout=30).json()
    if job["status"] == "completed":
        break
    if job["status"] == "failed":
        raise RuntimeError(job.get("error") or "Video generation failed")
    time.sleep(1)

video = requests.get(
    f"{base_url}/v1/videos/{video_id}/content",
    timeout=300,
)
video.raise_for_status()
Path("lingbot_video_moe.mp4").write_bytes(video.content)
```

## 5. Request constraints

- `num_frames` must be `1` or `4n+1`; examples include 17 and 81.
- Width and height must both be multiples of 16.
- The native defaults are `guidance_scale=6.0`, `flow_shift=3.0`,
  `num_inference_steps=40`, and `fps=16`.
- Keep the prompt as serialized JSON. Raw free text is outside the
  checkpoint's expected caption format.
