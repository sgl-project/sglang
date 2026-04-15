"""Manual test for AlpamayoR1 model inference via both /generate and OpenAI chat APIs.

Usage:
    python -m pytest test/manual/models/test_alpamayo_r1.py -v
    python test/manual/models/test_alpamayo_r1.py
"""

import base64
import io
import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_IDS = [
    "030c760c-ae38-49aa-9ad8-f5650a545d26",
    "74d763f1-9c47-416e-a840-a3e4bd8cc6a7",
]

MODEL_PATH = os.environ.get("ALPAMAYO_MODEL_PATH", "nvidia/Alpamayo-R1-10B")
TOKENIZER_PATH = os.environ.get("ALPAMAYO_TOKENIZER_PATH", "Qwen/Qwen3-VL-8B-Instruct")

NUM_TRAJ_TOKEN = 48
HIST_TRAJ_PLACEHOLDER = (
    f"<|traj_history_start|>{'<|traj_history|>' * NUM_TRAJ_TOKEN}<|traj_history_end|>"
)

OUTPUT_DIR = os.environ.get("ALPAMAYO_OUTPUT_DIR", "./alpamayo_test_output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode_rgb_tensor_to_base64(
    image: torch.Tensor, fmt: str = "JPEG", quality: int = 85, target_h: int = 200
) -> str:
    from PIL import Image

    tensor = image.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor (C,H,W), got shape={tuple(tensor.shape)}")

    if tensor.shape[0] == 3:
        tensor_chw = tensor
    elif tensor.shape[-1] == 3:
        tensor_chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(
            f"Expected RGB with 3 channels, got shape={tuple(tensor.shape)}"
        )

    if tensor_chw.dtype != torch.uint8:
        tensor_chw = (tensor_chw.clamp(0, 1) * 255).to(torch.uint8)

    array_hwc = tensor_chw.permute(1, 2, 0).numpy()
    img = Image.fromarray(array_hwc)

    if target_h and img.height != target_h:
        target_w = max(1, round(img.width * target_h / img.height))
        img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _default_cache_dir():
    import pathlib

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
    import physical_ai_av
    import scipy.spatial.transform as spt
    from einops import rearrange

    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            cache_dir=_default_cache_dir(),
        )

    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    features_to_load = [avdi.features.LABELS.EGOMOTION] + list(camera_features)
    avdi.download_clip_features(clip_id, features_to_load)

    egomotion = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=maybe_stream,
    )

    assert (
        t0_us > num_history_steps * time_step * 1_000_000
    ), "t0_us must be greater than the history time range"

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2, time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_max_us = int(egomotion.timestamps[-1] - egomotion.timestamps[0]) - 1
    future_timestamps = future_timestamps[future_timestamps <= ego_max_us]
    if future_timestamps.size == 0:
        raise ValueError(
            f"t0_us={t0_us} is too close to clip end (max={ego_max_us}us), no future steps available"
        )

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)
    ).as_matrix()
    ego_future_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)
    ).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=maybe_stream)
        frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        if isinstance(cam_feature, str):
            cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
            cam_name = cam_name.lower()
        else:
            raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
        cam_idx = camera_name_to_index.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


def _prepare_data(clip_id: str, t0_us: int = 5_100_000):
    """Load dataset and encode images. Returns (data, images_b64, history_traj)."""
    print(f"  Loading dataset for clip_id={clip_id} ...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    print("  Dataset loaded.")

    frames = data["image_frames"].flatten(0, 1)
    images_b64 = [_encode_rgb_tensor_to_base64(frame) for frame in frames]

    history_traj = {
        "ego_history_xyz": data["ego_history_xyz"].tolist(),
        "ego_history_rot": data["ego_history_rot"].tolist(),
    }
    return data, images_b64, history_traj


def _rotate_90cc(xy: np.ndarray) -> np.ndarray:
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


def _compute_ade_and_plot(
    clip_id: str,
    pred_xyz: np.ndarray,
    gt_future_xyz: torch.Tensor,
    output_dir: str,
    api_label: str,
) -> float:
    """Compute ADE and save trajectory plot. Returns the best ADE value."""
    while pred_xyz.ndim > 3:
        pred_xyz = pred_xyz[0]
    if pred_xyz.ndim == 2:
        pred_xyz = pred_xyz[None]

    pred_xy_all = pred_xyz[:, :, :2]
    gt_xy = gt_future_xyz.cpu()[0, 0, :, :2].numpy()

    min_steps = min(pred_xy_all.shape[1], gt_xy.shape[0])
    pred_xy_all = pred_xy_all[:, :min_steps, :]
    gt_xy = gt_xy[:min_steps, :]

    ade_each = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=-1).mean(axis=-1)
    best_idx = int(np.argmin(ade_each))
    best_ade = float(ade_each[best_idx])
    pred_xy = pred_xy_all[best_idx]

    print(f"  ADE: {best_ade:.4f} meters (best traj index={best_idx})")

    os.makedirs(output_dir, exist_ok=True)
    pred_xy_rot = _rotate_90cc(pred_xy)
    gt_xy_rot = _rotate_90cc(gt_xy)

    plt.figure(figsize=(8, 8))
    plt.plot(pred_xy_rot[:, 0], pred_xy_rot[:, 1], "o-", label="Predicted")
    plt.plot(gt_xy_rot[:, 0], gt_xy_rot[:, 1], "r-", label="Ground Truth")
    plt.ylabel("y (meters)")
    plt.xlabel("x (meters)")
    plt.legend(loc="best")
    plt.axis("equal")
    plt.title(f"{api_label} | clip={clip_id[:8]}... | ADE={best_ade:.4f}m")
    plt.tight_layout()

    pic_path = os.path.join(output_dir, f"{clip_id}_{api_label}_trajectory.png")
    plt.savefig(pic_path, dpi=200)
    plt.close()
    print(f"  Saved plot to {pic_path}")
    return best_ade


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestAlpamayoR1(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = {
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        }
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--tokenizer-path",
                TOKENIZER_PATH,
                "--tp",
                "1",
                "--disable-cuda-graph",
                "--attention-backend",
                "triton",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_api(self):
        for clip_id in CLIP_IDS:
            with self.subTest(clip_id=clip_id):
                ade = self._run_generate(clip_id)
                self.assertIsNotNone(ade, "pred_traj.traj_xyz was empty in response")
                print(f"  [generate] {clip_id[:12]}... ADE={ade:.4f}m")

    def test_chat_api(self):
        for clip_id in CLIP_IDS:
            with self.subTest(clip_id=clip_id):
                ade = self._run_chat(clip_id)
                self.assertIsNotNone(ade, "sglext.pred_traj.traj_xyz not found in response")
                print(f"  [chat] {clip_id[:12]}... ADE={ade:.4f}m")

    # ---- generate API ----
    def _run_generate(self, clip_id):
        print(f"\n[generate] clip_id={clip_id}")
        data, images_b64, history_traj = _prepare_data(clip_id)

        image_data = [f"data:image/jpeg;base64,{b64}" for b64 in images_b64]
        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"

        user_text = (
            f"{image_placeholder * len(image_data)}"
            f"{HIST_TRAJ_PLACEHOLDER}"
            "output the chain-of-thought reasoning of the driving process, "
            "then output the future trajectory"
        )
        prompt = (
            "<|im_start|>system\n"
            "You are a driving assistant that generates safe and accurate actions."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<|cot_start|>"
        )

        payload = {
            "text": prompt,
            "image_data": image_data,
            "history_traj": history_traj,
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 0.98,
                "max_new_tokens": 256,
            },
            "stream": False,
        }

        t0 = time.perf_counter()
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        resp.raise_for_status()
        ret = resp.json()
        print(f"  Request took {time.perf_counter() - t0:.2f}s")

        if isinstance(ret, list):
            ret = ret[0]

        print(f"  Generated text: {ret.get('text', '')[:200]}...")

        meta_info = ret.get("meta_info", {})
        _pred_traj = next((x for x in reversed(meta_info.get("pred_traj", [])) if x is not None), None)
        pred_xyz = np.asarray(_pred_traj["traj_xyz"] if _pred_traj else [])
        if pred_xyz.size == 0:
            return None

        return _compute_ade_and_plot(
            clip_id, pred_xyz, data["ego_future_xyz"], OUTPUT_DIR, "generate"
        )

    # ---- chat API ----
    def _run_chat(self, clip_id):
        from openai import OpenAI

        print(f"\n[chat] clip_id={clip_id}")
        data, images_b64, history_traj = _prepare_data(clip_id)

        client = OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY")
        img_mime = "image/jpeg"

        messages = [
            {
                "role": "system",
                "content": "You are a driving assistant that generates safe and accurate actions.",
            },
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_mime};base64,{b64}"},
                        }
                        for b64 in images_b64
                    ],
                    {
                        "type": "text",
                        "text": (
                            f"{HIST_TRAJ_PLACEHOLDER}"
                            "output the chain-of-thought reasoning of the driving process, "
                            "then output the future trajectory"
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": "<|cot_start|>"},
        ]

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=TOKENIZER_PATH,
            messages=messages,
            max_tokens=256,
            extra_body={"history_traj": history_traj, "continue_final_message": True},
            temperature=0.6,
            top_p=0.98,
        )
        print(f"  Request took {time.perf_counter() - t0:.2f}s")

        text = resp.choices[0].message.content or ""
        print(f"  Generated text: {text[:200]}...")

        sglext = getattr(resp, "sglext", None)
        if sglext is None:
            sglext = getattr(resp, "model_extra", {}).get("sglext")

        if not sglext or "pred_traj" not in sglext:
            return None

        _pred_traj = next((x for x in reversed(sglext["pred_traj"]) if x is not None), None)
        if _pred_traj is None:
            return None
        pred_xyz = np.asarray(_pred_traj["traj_xyz"])
        if pred_xyz.size == 0:
            return None

        return _compute_ade_and_plot(
            clip_id, pred_xyz, data["ego_future_xyz"], OUTPUT_DIR, "chat"
        )


if __name__ == "__main__":
    unittest.main()
