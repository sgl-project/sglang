# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from: https://github.com/Robbyant/lingbot-world

# SPDX-License-Identifier: Apache-2.0
import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits import LingBotWorldVideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_I2V_A14B_Config
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.utils.camera_geometry import (
    camera_poses_to_plucker,
    compute_relative_poses,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def lingbot_prompt_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class _LingBotWorldCameraState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.action_history: list[list[str]] = []
        self.last_actions: list[str] = []

    def reset_camera_actions(self):
        self.action_history.clear()
        self.last_actions = []

    def append_camera_actions(self, camera_actions: list[list[str]]) -> None:
        for actions in camera_actions:
            normalized = list(actions)
            self.action_history.append(normalized)
            self.last_actions = normalized

    def dispose(self):
        super().dispose()
        self.reset_camera_actions()


def _validate_actions(actions: Any) -> list[list[str]]:
    if not isinstance(actions, list):
        raise TypeError("actions must be a list[list[str]]")
    result: list[list[str]] = []
    for frame_actions in actions:
        if not isinstance(frame_actions, list):
            raise TypeError("actions must be a list[list[str]]")
        result.append(list(frame_actions))
    return result


def _pad_actions_to_chunk(
    action_history: list[list[str]], chunk_size: int
) -> list[list[str]]:
    if len(action_history) >= chunk_size:
        return action_history
    fill_item = action_history[-1] if action_history else []
    return action_history + [
        list(fill_item) for _ in range(chunk_size - len(action_history))
    ]


def _get_rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    def calculate_c_s():
        return np.cos(angle_rad), np.sin(angle_rad)

    if axis == "x":
        c, s = calculate_c_s()
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        c, s = calculate_c_s()
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        c, s = calculate_c_s()
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def _actions_to_c2ws(action_history: list[list[str]]) -> list[np.ndarray]:
    move_speed = 0.05
    rotate_speed_rad_ik = np.deg2rad(4.0)
    rotate_speed_rad_jl = np.deg2rad(6.0)

    current_c2w = np.eye(4)
    current_pitch = 0.0
    pitch_limit = np.deg2rad(85)
    all_matrices = [current_c2w]

    for frame_keys in action_history:
        R = current_c2w[:3, :3]
        T = current_c2w[:3, 3]

        pitch_delta = 0.0
        if "i" in frame_keys:
            pitch_delta += rotate_speed_rad_ik
        if "k" in frame_keys:
            pitch_delta -= rotate_speed_rad_ik

        new_pitch = current_pitch + pitch_delta
        if -pitch_limit <= new_pitch <= pitch_limit:
            current_pitch = new_pitch
        else:
            pitch_delta = 0.0

        yaw_delta = 0.0
        if "j" in frame_keys:
            yaw_delta -= rotate_speed_rad_jl
        if "l" in frame_keys:
            yaw_delta += rotate_speed_rad_jl

        R_pitch = _get_rotation_matrix("x", pitch_delta)
        R_yaw = _get_rotation_matrix("y", yaw_delta)
        R_new = R_yaw @ R @ R_pitch

        vec_right = R_new[:, 0]
        vec_forward = R_new[:, 2]
        forward_flat = np.array([vec_forward[0], 0, vec_forward[2]])
        right_flat = np.array([vec_right[0], 0, vec_right[2]])

        f_norm = np.linalg.norm(forward_flat)
        r_norm = np.linalg.norm(right_flat)
        if f_norm > 0:
            forward_flat = forward_flat / (f_norm + 1e-6)
        if r_norm > 0:
            right_flat = right_flat / (r_norm + 1e-6)

        move_vec = np.zeros(3)
        if "w" in frame_keys:
            move_vec += forward_flat * move_speed
        if "s" in frame_keys:
            move_vec -= forward_flat * move_speed
        if "d" in frame_keys:
            move_vec += right_flat * move_speed
        if "a" in frame_keys:
            move_vec -= right_flat * move_speed

        T_new = T + move_vec
        current_c2w = np.eye(4)
        current_c2w[:3, :3] = R_new
        current_c2w[:3, 3] = T_new
        all_matrices.append(current_c2w)

    return all_matrices


def _get_camera_control(
    action_history: list[list[str]],
    *,
    chunk_size: int,
    width: int,
    height: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2ws_list = _actions_to_c2ws(action_history)
    c2ws_np = np.stack(c2ws_list[1:])
    c2ws = torch.from_numpy(c2ws_np).to(device=device, dtype=dtype)
    Ks = torch.tensor(
        [[500.0, 500.0, width / 2, height / 2]],
        device=device,
        dtype=dtype,
    ).repeat(chunk_size, 1)
    logger.debug("prefix c2ws shape: %s, Ks shape: %s", c2ws.shape, Ks.shape)
    return c2ws, Ks


def _build_camera_condition(
    *,
    action_history: list[list[str]],
    width: int,
    height: int,
    spatial_scale: int,
    device: torch.device | str,
    dtype: torch.dtype,
    tail_chunk_size: int,
) -> torch.Tensor:
    action_history = _pad_actions_to_chunk(action_history, tail_chunk_size)
    c2ws_prefix, Ks = _get_camera_control(
        action_history,
        chunk_size=tail_chunk_size,
        width=width,
        height=height,
        device=device,
        dtype=dtype,
    )
    c2ws_prefix = compute_relative_poses(c2ws_prefix, framewise=True)
    c2ws_prefix = c2ws_prefix[-tail_chunk_size:]

    return camera_poses_to_plucker(
        c2ws=c2ws_prefix,
        Ks=Ks,
        height=height,
        width=width,
        spatial_scale=spatial_scale,
        device=device,
        dtype=dtype,
    )


def _prepare_lingbot_world_condition(
    *,
    batch,
    pipeline_config,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if batch.c2ws_plucker_emb is not None:
        return batch.c2ws_plucker_emb.to(device=device, dtype=dtype)

    actions = batch.condition_inputs.get("camera_actions")
    if actions is None:
        return None

    spatial_scale = pipeline_config.vae_config.arch_config.spatial_compression_ratio
    chunk_size = batch.realtime_chunk_size or max(
        1,
        int(pipeline_config.dit_config.arch_config.num_frames_per_block),
    )

    normalized_actions = _validate_actions(actions)
    if len(normalized_actions) == 0:
        return None

    if batch.session is None:
        action_history = normalized_actions
    else:
        state = batch.session.get_or_create_state(_LingBotWorldCameraState)
        if batch.block_idx == 0:
            state.reset_camera_actions()
        state.append_camera_actions(normalized_actions)
        action_history = state.action_history

    if len(action_history) == 0:
        return None

    c2ws_plucker_emb = _build_camera_condition(
        action_history=action_history,
        width=int(batch.width),
        height=int(batch.height),
        spatial_scale=spatial_scale,
        device=device,
        dtype=dtype,
        tail_chunk_size=chunk_size,
    )
    logger.debug(
        "LingBot action condition prepared: session_id=%s, block_idx=%s, new_action_count=%s, total_history=%s",
        batch.realtime_session_id,
        batch.block_idx,
        len(normalized_actions),
        len(action_history),
    )
    return c2ws_plucker_emb


@dataclass
class LingBotWorldI2VConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorldVideoConfig)
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (lingbot_prompt_clean,)
    )

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        kwargs = super().prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)
        if batch.c2ws_plucker_emb is not None:
            kwargs["c2ws_plucker_emb"] = batch.c2ws_plucker_emb.to(
                device=device, dtype=dtype
            )
        return kwargs

    def preprocess_realtime_condition_image(self, batch, _vae_image_processor) -> bool:
        if batch.condition_image is None:
            return False
        if isinstance(batch.condition_image, list):
            batch.condition_image = batch.condition_image[0]

        width = int(batch.width or 832)
        height = int(batch.height or 480)
        batch.condition_image = batch.condition_image.resize((width, height))
        batch.width = width
        batch.height = height
        return True

    def prepare_world_condition(self, batch, device, dtype):
        c2ws_plucker_emb = _prepare_lingbot_world_condition(
            batch=batch,
            pipeline_config=self,
            device=device,
            dtype=dtype,
        )
        if c2ws_plucker_emb is None:
            return None
        return {"c2ws_plucker_emb": c2ws_plucker_emb}


@dataclass
class LingBotWorldCausalDMDConfig(LingBotWorldI2VConfig):
    is_causal: bool = True
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 821, 642, 321]
    )
    warp_denoising_step: bool = True
    realtime_causal_sink_size: int | None = None
    realtime_causal_kv_cache_num_frames: int | None = None
    interactive_kv_window_enable: bool = True
    interactive_kv_still_window: int = 3
    interactive_kv_moving_window: int | None = 12
    interactive_kv_still_chunks: int = 2

    def postprocess_image_latent(self, latent_condition, batch):
        """Build condition tensor aligned to chunk_size (num_frames_per_block).

        Matches lingbot_fast_server's _prepare_latents_causal:
        condition = [mask(temporal_ratio ch), latent(z_dim ch)] -> 20ch total,
        with temporal dim aligned to chunk_size.
        """
        vae_arch = self.vae_config.arch_config
        temporal_ratio = vae_arch.temporal_compression_ratio
        spatial_ratio = vae_arch.spatial_compression_ratio
        chunk_size = self.dit_config.arch_config.num_frames_per_block

        latent_height = batch.height // spatial_ratio
        latent_width = batch.width // spatial_ratio

        # Align num_latent_frames to chunk_size
        num_latent_frames = latent_condition.shape[2]
        num_latent_frames = num_latent_frames - (num_latent_frames % chunk_size)
        latent_condition = latent_condition[:, :, :num_latent_frames, :, :]

        # Number of initial frames that have actual image content
        # (latent_condition from VAE encode of [image, zeros...])
        # First frame is real, rest are zero-padded
        initial_latent_frames = 1  # single image -> 1 latent frame

        # Build mask: [B, temporal_ratio, num_latent_frames, H, W]
        mask = torch.ones(
            1,
            temporal_ratio,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=latent_condition.dtype,
            device=latent_condition.device,
        )
        # Zero out mask for frames beyond the initial image
        if initial_latent_frames < num_latent_frames:
            mask[:, :, initial_latent_frames:] = 0

        return torch.cat([mask, latent_condition], dim=1)
