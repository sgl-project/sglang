# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""LingBot-World camera-control conditioning utilities."""

from __future__ import annotations

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    rot = T[:, :3, :3]
    trans = T[:, :3, 3:]
    r_inv = rot.transpose(-1, -2)
    t_inv = -torch.bmm(r_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(
        T.shape[0], 1, 1
    )
    T_inv[:, :3, :3] = r_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = se3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise and len(relative_poses) > 1:
        relative_poses_framewise = torch.bmm(
            se3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    *,
    bias: float = 0.5,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    return grid_xy[None, ...].repeat(n_frames, 1, 1)


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)
    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)
    rays_o = c2ws_mat[:, :3, 3][:, None, :].expand_as(rays_d)
    plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)
    return plucker_embeddings.view([n_frames, height, width, 6])


def get_rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def actions_to_c2ws(action_history: list[list[str]]) -> list[np.ndarray]:
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

        R_pitch = get_rotation_matrix("x", pitch_delta)
        R_yaw = get_rotation_matrix("y", yaw_delta)
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


def get_camera_control(
    action_history: list[list[str]],
    *,
    chunk_size: int,
    width: int,
    height: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2ws_list = actions_to_c2ws(action_history)
    c2ws_np = np.stack(c2ws_list[1:])
    c2ws = torch.from_numpy(c2ws_np).to(device=device, dtype=dtype)
    Ks = torch.tensor(
        [[500.0, 500.0, width / 2, height / 2]],
        device=device,
        dtype=dtype,
    ).repeat(chunk_size, 1)
    logger.debug("prefix c2ws shape: %s, Ks shape: %s", c2ws.shape, Ks.shape)
    return c2ws, Ks


def camera_poses_to_plucker(
    *,
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    spatial_scale: int = 8,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    plucker = get_plucker_embeddings(c2ws, Ks, height, width)
    latent_height = height // spatial_scale
    latent_width = width // spatial_scale
    plucker = plucker.view(
        c2ws.shape[0],
        latent_height,
        spatial_scale,
        latent_width,
        spatial_scale,
        6,
    )
    plucker = plucker.permute(0, 1, 3, 5, 2, 4).contiguous()
    plucker = plucker.view(
        c2ws.shape[0],
        latent_height,
        latent_width,
        6 * spatial_scale * spatial_scale,
    )
    return (
        plucker.permute(3, 0, 1, 2)
        .contiguous()
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
