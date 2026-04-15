# Alpamayo-R1

[Alpamayo-R1](https://huggingface.co/nvidia/Alpamayo-R1-10B) is NVIDIA's Vision-Language-Action (VLA) model for autonomous driving. Built on top of Qwen3-VL-8B, it takes multi-camera images and ego-vehicle trajectory history as input, performs chain-of-thought reasoning about the driving scene, and outputs future trajectory waypoints.

Key features:
- Multi-camera input (4 cameras, 4 frames each)
- Trajectory history conditioning via special `<|traj_history|>` tokens
- Chain-of-thought reasoning before trajectory prediction
- Outputs 64 future waypoints (6.4s at 10Hz) as (x, y, z) coordinates

## Launch Server

```{note}
Currently only the `triton` attention backend is supported for Alpamayo-R1. Other backends (`flashinfer`, `torch_native`, `trtllm_mha`) are not yet compatible.
```

The `--tokenizer-path` is optional. If omitted, SGLang will automatically download the tokenizer from `Qwen/Qwen3-VL-8B-Instruct`. You can also specify it explicitly if you have a local copy:

```shell
python3 -m sglang.launch_server \
  --model-path nvidia/Alpamayo-R1-10B \
  --tokenizer-path Qwen/Qwen3-VL-8B-Instruct \
  --port 30000 \
  --tp 1 \
  --disable-cuda-graph \
  --attention-backend triton
```


## Inference Example

The following script sends multi-camera images and trajectory history to the server via the `/generate` endpoint, then extracts the predicted trajectory from the response.

### Dependencies

Alpamayo-R1 requires the [physical-ai-av](https://huggingface.co/datasets/nvidia/physical-ai-av) dataset SDK for loading driving data:

```shell
pip install physical-ai-av einops scipy
```

### Inference Script

This example uses the OpenAI-compatible `/v1/chat/completions` endpoint.

```python
import base64
import io
import os
import pathlib

import numpy as np
import physical_ai_av
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from openai import OpenAI
from PIL import Image


def _default_cache_dir():
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        return pathlib.Path(hf_home) / "hub"
    return None


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi=None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
):
    """Load data from physical_ai_av for AlpamayoR1 model inference."""
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=_default_cache_dir())

    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0, "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2, "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4, "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    features_to_load = [avdi.features.LABELS.EGOMOTION] + list(camera_features)
    avdi.download_clip_features(clip_id, features_to_load)
    egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=maybe_stream)

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2, time_step * 1_000_000,
    ).astype(np.int64)
    future_offsets_us = np.arange(
        time_step * 1_000_000, (num_future_steps + 0.5) * time_step * 1_000_000, time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = (t0_us + future_offsets_us)
    ego_max_us = int(egomotion.timestamps[-1] - egomotion.timestamps[0]) - 1
    future_timestamps = future_timestamps[future_timestamps <= ego_max_us]

    ego_history = egomotion(t0_us + history_offsets_us)
    ego_future = egomotion(future_timestamps)

    t0_rot = spt.Rotation.from_quat(ego_history.pose.rotation.as_quat()[-1])
    t0_rot_inv = t0_rot.inv()
    t0_xyz = ego_history.pose.translation[-1]

    def to_local_xyz(xyz):
        return torch.from_numpy(t0_rot_inv.apply(xyz - t0_xyz)).float().unsqueeze(0).unsqueeze(0)

    def to_local_rot(quat):
        return torch.from_numpy((t0_rot_inv * spt.Rotation.from_quat(quat)).as_matrix()).float().unsqueeze(0).unsqueeze(0)

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)], dtype=np.int64
    )
    image_frames_list, camera_indices_list, timestamps_list = [], [], []
    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=maybe_stream)
        frames, frame_ts = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")
        cam_name = (cam_feature.split("/")[-1] if isinstance(cam_feature, str) else cam_feature).lower()
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(camera_name_to_index.get(cam_name, 0))
        timestamps_list.append(torch.from_numpy(frame_ts.astype(np.int64)))

    image_frames = torch.stack(image_frames_list)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list)
    sort_order = torch.argsort(camera_indices)

    return {
        "image_frames": image_frames[sort_order],
        "camera_indices": camera_indices[sort_order],
        "ego_history_xyz": to_local_xyz(ego_history.pose.translation),
        "ego_history_rot": to_local_rot(ego_history.pose.rotation.as_quat()),
        "ego_future_xyz": to_local_xyz(ego_future.pose.translation),
        "ego_future_rot": to_local_rot(ego_future.pose.rotation.as_quat()),
        "relative_timestamps": (all_timestamps[sort_order] - all_timestamps.min()).float() * 1e-6,
        "t0_us": t0_us, "clip_id": clip_id,
    }


def encode_image_to_base64(image: torch.Tensor) -> str:
    """Encode an RGB tensor (C,H,W) or (H,W,C) to a JPEG base64 string."""
    tensor = image.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D tensor (C,H,W), got shape={tuple(tensor.shape)}")

    if tensor.shape[0] == 3:
        tensor_chw = tensor
    elif tensor.shape[-1] == 3:
        tensor_chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB with 3 channels, got shape={tuple(tensor.shape)}")

    if tensor_chw.dtype != torch.uint8:
        tensor_chw = (tensor_chw.clamp(0, 1) * 255).to(torch.uint8)

    array_hwc = tensor_chw.permute(1, 2, 0).numpy()
    img = Image.fromarray(array_hwc)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Load data and encode images ---
clip_id = "06b483cf-6d9c-4b18-b54b-4429c80867e3"
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

frames = data["image_frames"].flatten(0, 1)  # (N_cameras * num_frames, C, H, W)
images_b64 = [encode_image_to_base64(frame) for frame in frames]

history_traj = {
    "ego_history_xyz": data["ego_history_xyz"].tolist(),
    "ego_history_rot": data["ego_history_rot"].tolist(),
}

# --- Build prompt via OpenAI-compatible chat API ---
num_traj_token = 48
hist_traj_placeholder = (
    f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
)

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

messages = [
    {
        "role": "system",
        "content": "You are a driving assistant that generates safe and accurate actions.",
    },
    {
        "role": "user",
        "content": [
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                for b64 in images_b64
            ],
            {
                "type": "text",
                "text": (
                    f"{hist_traj_placeholder}"
                    "output the chain-of-thought reasoning of the driving process, "
                    "then output the future trajectory"
                ),
            },
        ],
    },
    {"role": "assistant", "content": "<|cot_start|>"},
]

resp = client.chat.completions.create(
    model="nvidia/Alpamayo-R1-10B",
    messages=messages,
    max_tokens=256,
    temperature=0.6,
    top_p=0.98,
    extra_body={"history_traj": history_traj, "continue_final_message": True},
)

# --- Extract results ---
print("Generated text:", resp.choices[0].message.content)

sglext = getattr(resp, "sglext", None)
if sglext is None:
    sglext = getattr(resp, "model_extra", {}).get("sglext")

# pred_traj is a list of per-decode-step dicts; take the last non-None entry
pred_traj = next((x for x in reversed(sglext["pred_traj"]) if x is not None), None)
traj_xyz = np.asarray(pred_traj["traj_xyz"])

# Normalize to (n_samples, n_waypoints, 3)
while traj_xyz.ndim > 3:
    traj_xyz = traj_xyz[0]
if traj_xyz.ndim == 2:
    traj_xyz = traj_xyz[None]
pred_xy_all = traj_xyz[:, :, :2]

# --- Compute ADE (Average Displacement Error) ---
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
min_steps = min(pred_xy_all.shape[1], gt_xy.shape[0])
pred_xy_all = pred_xy_all[:, :min_steps, :]
gt_xy = gt_xy[:min_steps, :]

ade_each = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=-1).mean(axis=-1)
best_idx = int(np.argmin(ade_each))
print(f"Best trajectory index: {best_idx}")
print(f"ADE: {ade_each[best_idx]:.4f} meters")
```
