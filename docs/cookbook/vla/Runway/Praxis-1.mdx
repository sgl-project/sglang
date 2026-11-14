---
title: Praxis-1
metatags:
    description: "Deploy Runway Praxis-1 (DROID single-arm and YAM bimanual), a video-action world model that denoises future video and robot actions jointly, with SGLang's native multimodal_gen runtime."
tag: dVLA
---

## 1. Model Introduction

Praxis-1 is a 27.2B-parameter sparse mixture-of-experts policy from Runway. From the camera frames, the end-effector state and a language instruction, it denoises a chunk of future video latents **jointly** with a chunk of continuous actions. The actions are returned; the predicted video can be decoded on request.

The model has three parts:

- **Video-action transformer**: a MoE backbone over video, text and action tokens with a windowed attention mask, plus an action expert with adaptive layer norm. Attention runs on FlexAttention.
- **Text encoder**: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).
- **Video VAE**: the Wan2.2 VAE (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`).

The sampler is a 2-step Euler window-joint sampler. Classifier-free guidance on the video stream costs a second, unguided pass per step, and each checkpoint sets its own recipe:

| Checkpoint | Flow shift | Video guidance | Action guidance |
| --- | ---: | ---: | ---: |
| DROID | `12.5` | `4.0` | `1.0` |
| YAM | `1.0` | `1.0` (off) | `1.0` |

Calls of one episode share a `session_id`: the first call has no context (one frame per view), later calls carry the 25 frames executed since the previous call and run with one context chunk.

SGLang runs it in the native `multimodal_gen` runtime, not through the reference code.

Supported checkpoints, both public Hugging Face repos in the LeRobot layout:

| Checkpoint | Embodiment | Cameras | State | Action chunk | Rate |
| --- | --- | --- | --- | --- | ---: |
| [`runwayml/praxis-1-droid-lerobot`](https://huggingface.co/runwayml/praxis-1-droid-lerobot) | Single-arm DROID | `exterior`, `wrist` | 8 values | 24 rows of 13 values | 15 Hz |
| [`runwayml/praxis-1-yam-lerobot`](https://huggingface.co/runwayml/praxis-1-yam-lerobot) | Bimanual YAM (two arms) | `exterior` (top), `wrist_left`, `wrist_right` | 16 values | 24 rows of 26 values | 15 Hz |

The server picks the embodiment from the checkpoint, so the same command serves either one. The weights of the two checkpoints are 54.4 GB each (bf16, 17 shards).

References:

- [Praxis-1 DROID (LeRobot checkpoint)](https://huggingface.co/runwayml/praxis-1-droid-lerobot)
- [Praxis-1 YAM (LeRobot checkpoint)](https://huggingface.co/runwayml/praxis-1-yam-lerobot)

## 2. Installation

```bash Command
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[diffusion]"
```

## 3. Model Deployment

Serve the DROID policy:

```bash Command
sglang serve runwayml/praxis-1-droid-lerobot \
  --model-type diffusion \
  --host 127.0.0.1 \
  --port 30000
```

Serve the YAM policy:

```bash Command
sglang serve runwayml/praxis-1-yam-lerobot \
  --model-type diffusion \
  --host 127.0.0.1 \
  --port 30000
```

The text encoder and the VAE are downloaded from their own Hub repositories on first start. A local copy of the policy (a directory holding `config.json` and the `.safetensors` shards) is detected automatically when passed as the model path. Pass `--revision` to pin a Hub revision.

Peak GPU memory on one GPU, as the driver reports it (`nvidia-smi`, including the CUDA context and the memory the allocator keeps reserved) over a run of calls that includes decoding the predicted video:

| Flags | DROID | YAM |
| --- | ---: | ---: |
| (none) | 62.1 GiB | 65.1 GiB |
| `--dit-layerwise-offload` | 23.9 GiB | 27.5 GiB |
| `--layerwise-offload-components all` | 22.5 GiB | 25.5 GiB |

Layerwise offload streams the DiT blocks (and, with `all`, the text encoder and VAE blocks) from host memory, so use it only when the resident configuration does not fit.

### 3.1 Multi-GPU

Praxis-1 runs on one GPU, or on two with CFG parallel, which computes the guided and unguided passes on separate GPUs. Tensor and sequence parallelism are not supported.

```bash Command
sglang serve runwayml/praxis-1-droid-lerobot \
  --model-type diffusion \
  --num-gpus 2 \
  --enable-cfg-parallel \
  --port 30000
```

CFG parallel only helps requests with guidance (`guidance_scale` other than `1.0`). The YAM policy serves without video guidance by default, so it gains nothing from a second GPU unless the request sets `guidance_scale` above `1.0`.


### 3.2 Action Request Schema

The DROID and YAM requests differ in `images` and `state`. Every other field is shared.

| Field | Type | Description |
| --- | --- | --- |
| `input.task` | string | Language instruction. The YAM checkpoint was trained on captions such as `Pick up the pen from the center of the table, then place the pen in the bag`, without a closing period; send prompts in that form. |
| `input.observation.images.exterior` / `wrist` | array | DROID. Frames since the previous call, oldest first, uint8 `[T, H, W, 3]` (or floats in `[0, 1]`). `T` is 1 on the first call of a session and 25 afterwards. |
| `input.observation.images.exterior` / `wrist_left` / `wrist_right` | array | YAM. The same frame stacks for three views; `exterior` is the top camera. Views of any 16:10 size are fitted into 256x416 tiles of one canvas. |
| `input.observation.state` | array | DROID. One row per frame `[x, y, z, qw, qx, qy, qz, grip]`, shape `[T, 8]`. |
| `input.observation.state` | array | YAM. One row per frame, the left arm then the right arm, each `[x, y, z, qw, qx, qy, qz, grip]` in the rig's world frame, shape `[T, 16]`. |
| `input.observation.session_id` | string, optional | Calls with the same id continue one episode. Without it every call is a first call. The server keeps the 32 most recently used sessions. |
| `input.observation.return_video` | boolean, optional | Also decode the predicted video into `extras.video`. Needs `output_format: "numpy"`. |
| `parameters.guidance_scale` | number, optional | Video guidance. Default `4.0` for DROID and `1.0` for YAM; `1.0` skips the unguided pass. |
| `parameters.guidance_scale_action` | number, optional | Action guidance. Default `1.0`. |
| `parameters.num_inference_steps` | integer, optional | Defaults to the checkpoint recipe (2). |
| `parameters.seed` | integer, optional | Noise seed of a new session. Defaults to the checkpoint's `seed`. |
| `runtime.output_format` | `"list"` or `"numpy"`, optional | Use `"numpy"` with msgpack clients. |

The response returns `data[0].action` with shape `[24, 13]` for DROID: position `xyz` (3), the rotation matrix row-major (9) and the gripper target (1). For YAM the shape is `[24, 26]`: those 13 values for the left arm, then for the right arm. On the first call of a YAM session `extras.commands` holds 21 rows, and 24 afterwards. `extras.commands` carries the same chunk as robot commands, and `timings` reports `policy_ms` (and `step_ms`, `decode_ms` when the video was decoded). `GET /v1/actions/metadata` reports the camera keys, the action layout and the sampler defaults of the served policy.

## 4. API Usage

Send msgpack with numpy arrays: a 25-frame stack of two views is too large to encode as JSON lists. `pack_msgpack` and `unpack_msgpack` come with SGLang.

```python Example
import numpy as np
import requests

from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    pack_msgpack,
    unpack_msgpack,
)

URL = "http://127.0.0.1:30000/v1/actions/generations"
HEADERS = {"Content-Type": "application/msgpack", "Accept": "application/msgpack"}


def step(frames: int, session_id: str) -> np.ndarray:
    pose = np.tile([0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0], (frames, 1))
    view = np.zeros((frames, 234, 416, 3), dtype=np.uint8)
    body = {
        "input": {
            "task": "Put the red block in the bowl.",
            "observation": {
                "images": {"exterior": view, "wrist": view},
                "state": pose.astype(np.float32),
                "session_id": session_id,
            },
        },
        "parameters": {"seed": 0},
        "runtime": {"output_format": "numpy"},
    }
    response = requests.post(URL, data=pack_msgpack(body), headers=HEADERS)
    response.raise_for_status()
    out = unpack_msgpack(response.content)
    print(out["timings"]["policy_ms"])
    return out["data"][0]["action"]["values"]


first = step(1, "episode-0")  # no context: one frame per view
later = step(25, "episode-0")  # the 25 frames executed since the first call
print(later.shape)  # (24, 13)
```

Predicted video: add `"return_video": True` to the observation and read `out["extras"]["video"]`, an array of frames `[T, H, W, 3]`.

For the YAM policy, send three views and a 16-value state. The server returns 26 values per row, 13 for each arm:

```python Example
import numpy as np
import requests

from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    pack_msgpack,
    unpack_msgpack,
)

URL = "http://127.0.0.1:30000/v1/actions/generations"
HEADERS = {"Content-Type": "application/msgpack", "Accept": "application/msgpack"}
ARM = [0.4, 0.2, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0]  # x, y, z, qw, qx, qy, qz, grip


def step(frames: int, session_id: str) -> np.ndarray:
    state = np.tile(ARM + ARM, (frames, 1)).astype(np.float32)  # left arm, right arm
    view = np.zeros((frames, 260, 416, 3), dtype=np.uint8)  # 16:10
    body = {
        "input": {
            "task": "Pick up the pen from the center of the table, then place the pen in the bag",
            "observation": {
                "images": {"exterior": view, "wrist_left": view, "wrist_right": view},
                "state": state,
                "session_id": session_id,
            },
        },
        "parameters": {"seed": 0},
        "runtime": {"output_format": "numpy"},
    }
    response = requests.post(URL, data=pack_msgpack(body), headers=HEADERS)
    response.raise_for_status()
    out = unpack_msgpack(response.content)
    print(out["timings"]["policy_ms"])
    return out["data"][0]["action"]["values"]


first = step(1, "episode-0")  # no context: one frame per view
later = step(25, "episode-0")  # the 25 frames executed since the first call
print(later.shape)  # (24, 26)
```
