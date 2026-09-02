# SPDX-License-Identifier: Apache-2.0

"""Camera pose and Plucker ray utilities."""

from __future__ import annotations

import torch


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
