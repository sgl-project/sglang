# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterable, List, Optional, Tuple, TypeVar, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from transformers import PretrainedConfig

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.utils import logger

_action_logger = logging.getLogger(__name__)

# ==============================================================================
# Inlined from alpamayo/rotation.py
# ==============================================================================

TensorOrNDArray = TypeVar("TensorOrNDArray", torch.Tensor, np.ndarray)


def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def so3_to_yaw_np(rot_mat: np.ndarray) -> np.ndarray:
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return np.arctan2(cos_th_sin_phi, cos_th_cos_phi)


def euler_2_so3(
    euler_angles: np.ndarray, degrees: bool = True, seq: str = "xyz"
) -> np.ndarray:
    return (
        R.from_euler(seq=seq, angles=euler_angles, degrees=degrees)
        .as_matrix()
        .astype(np.float32)
    )


def angle_wrap(radians: TensorOrNDArray) -> TensorOrNDArray:
    return (radians + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle: Union[float, np.ndarray]) -> np.ndarray:
    batch_dims = 0
    if isinstance(angle, np.ndarray):
        batch_dims = angle.ndim
    rotmat: np.ndarray = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return rotmat.transpose(*np.arange(2, batch_dims + 2), 0, 1)


def rotation_matrix_torch(angle: torch.Tensor) -> torch.Tensor:
    rotmat: torch.Tensor = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )
    return rotmat


def transform_coords_2d_np(
    coords: np.ndarray,
    offset: Optional[np.ndarray] = None,
    angle: Optional[np.ndarray] = None,
    rot_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rot_mat is None and angle is not None:
        rot_mat = rotation_matrix(angle)
    if rot_mat is not None:
        coords = np.einsum("...ij,...j->...i", rot_mat, coords)
    if offset is not None:
        coords += offset
    return coords


def stable_gramschmidt(M: torch.Tensor) -> torch.Tensor:
    EPS = 1e-7
    x = M[..., 0]
    y = M[..., 1]
    x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), EPS)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), EPS)
    z = torch.cross(x, y, dim=-1)
    _R = torch.stack((x, y, z), dim=-1)
    return _R


def rot_3d_to_2d(rot):
    xu = rot[..., :2, 0]
    yu = rot[..., :2, 1]
    EPS = 1e-6
    xu = xu / (torch.norm(xu, dim=-1, keepdim=True) + EPS)
    yu = yu - torch.sum(xu * yu, dim=-1, keepdim=True) * xu
    yu = yu / (torch.norm(yu, dim=-1, keepdim=True) + EPS)
    return torch.stack((xu, yu), dim=-1)


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    rot = torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(
                rot.shape[:-2] + (1, 1)
            ),
        ],
        dim=-2,
    )
    return rot


def ratan2(s, c, eps=1e-4):
    sign = (c >= 0).float() * 2 - 1
    eps = eps * (c.abs() < eps).type(c.dtype) * sign
    return torch.arctan2(s, c + eps)


def round_2pi(x: np.ndarray) -> np.ndarray:
    return np.atan2(np.sin(x), np.cos(x))


def round_2pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


# ==============================================================================
# Inlined from alpamayo/action_space_base.py
# ==============================================================================


class ActionSpace(ABC, nn.Module):
    """Action space base class for the trajectory generation."""

    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor: ...

    @abstractmethod
    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]: ...

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        num_action_dims = len(self.get_action_space_dims())
        batch_shape = (
            action.shape[:-num_action_dims] if num_action_dims > 0 else action.shape
        )
        return torch.ones(batch_shape, dtype=torch.bool, device=action.device)


# ==============================================================================
# Inlined from alpamayo/action_space_utils.py
# ==============================================================================


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    d = torch.diff(phi, dim=-1)
    d = round_2pi_torch(d)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def first_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def second_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def third_order_D(
    N: int,
    lead_shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def construct_DTD(
    N: int,
    lead: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)
    if w_smooth1 is not None:
        lam_1 = lam / dt**2
        if isinstance(w_smooth1, float):
            w_smooth1_tensor = torch.full(
                (*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device
            )
        else:
            w_smooth1_tensor = w_smooth1
        D1 = first_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_1 * einops.einsum(
            D1 * w_smooth1_tensor.unsqueeze(-1), D1, "... i j, ... i k -> ... j k"
        )

    if w_smooth2 is not None:
        lam_2 = lam / dt**4
        if isinstance(w_smooth2, float):
            w_smooth2_tensor = torch.full(
                (*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device
            )
        else:
            w_smooth2_tensor = w_smooth2
        D2 = second_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_2 * einops.einsum(
            D2 * w_smooth2_tensor.unsqueeze(-1), D2, "... i j, ... i k -> ... j k"
        )

    if w_smooth3 is not None:
        lam_3 = lam / dt**6
        if isinstance(w_smooth3, float):
            w_smooth3_tensor = torch.full(
                (*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device
            )
        else:
            w_smooth3_tensor = w_smooth3
        D3 = third_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_3 * einops.einsum(
            D3 * w_smooth3_tensor.unsqueeze(-1), D3, "... i j, ... i k -> ... j k"
        )

    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    if N <= 0:
        raise ValueError("x_mid must have a positive last-dimension length N.")
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

    x = torch.cat([x_init.unsqueeze(-1), x], dim=-1)
    return x


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)
    if w_data.shape != y.shape:
        raise ValueError("w_data must have the same shape as y")

    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )

    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(
                *lead, N, N
            )
            lhs = ATA + DTD + ridge_term
            if rhs.dtype != lhs.dtype:
                rhs = rhs.to(lhs.dtype)
            L = torch.linalg.cholesky(lhs)
        except RuntimeError as e:
            _action_logger.error(f"Error in cholesky decomposition: {e}", exc_info=True)
            ridge *= 10
            _action_logger.warning(f"Resolving singularity using ridge {ridge}")

    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy

    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )

    ridge_term = v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(
        *lead, N + 1, N + 1
    )
    lhs = ATA + DTD + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

    return y


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy

    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)

    ridge_term = v_ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

    return torch.cat([v0.unsqueeze(-1), y], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float = 1.0,
    theta_lambda: float = 1e-4,
    theta_ridge: float = 1e-4,
) -> torch.Tensor:
    theta = so3_to_yaw_torch(traj_future_rot)
    theta = unwrap_angle(theta)
    theta_init = torch.zeros_like(theta[..., 0])
    return solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )


# ==============================================================================
# Inlined from alpamayo/action_in_proj.py
# ==============================================================================


class MLPEncoder(nn.Module):
    """Basic MLP encoder."""

    def __init__(
        self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int
    ):
        super().__init__()
        assert 1 <= num_enc_layers, f"{num_enc_layers=} must be >= 1"

        enc_layers = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, outdim),
                    ]
                )

        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    """Improved Fourier feature encoder with logarithmically-spaced frequencies."""

    def __init__(self, dim: int, max_freq: float = 100.0, persistent: bool = True):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :], persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProjV2(torch.nn.Module):
    """Improved per-waypoint action input projection module."""

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
        fourier_persistent: bool = True,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        sinus = []
        for _ in range(in_dims[-1]):
            sinus.append(
                FourierEncoderV2(
                    dim=num_fourier_feats, max_freq=max_freq, persistent=fourier_persistent
                )
            )
        self.sinus = nn.ModuleList(sinus)
        self.timestep_fourier_encoder = FourierEncoderV2(
            dim=num_fourier_feats, max_freq=max_freq, persistent=fourier_persistent
        )
        num_input_feats = (
            sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        )
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        action_feats = torch.cat(
            [s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1
        )
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, T, 1)
        x = torch.cat((action_feats, timestep_feats), dim=-1)

        x = x.to(dtype=self.encoder.trunk[0].weight.dtype)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))


# ==============================================================================
# Inlined from alpamayo/unicycle_accel_curvature.py
# ==============================================================================


class UnicycleAccelCurvatureActionSpace(ActionSpace):
    """Unicycle Kinematic Model with acceleration and curvature as control inputs."""

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        accel = action[..., 0]
        kappa = action[..., 1]
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean
        is_accel_within_bounds = (accel >= self.accel_bounds[0]) & (
            accel <= self.accel_bounds[1]
        )
        is_kappa_within_bounds = (kappa >= self.curvature_bounds[0]) & (
            kappa <= self.curvature_bounds[1]
        )
        return torch.all(is_accel_within_bounds & is_kappa_within_bounds, dim=-1)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        dv = (v[..., 1:] - v[..., :-1]) / self.dt
        a = solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
        )
        return a

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        dtheta = theta[..., 1:] - theta[..., :-1]
        dt = self.dt
        s = dt * v[..., :-1] + (dt**2) / 2.0 * a

        w = torch.ones_like(dtheta)
        return solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=w,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self, traj_history_xyz: torch.Tensor, traj_history_rot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        full_xy = traj_history_xyz[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = so3_to_yaw_torch(traj_history_rot)
        theta = unwrap_angle(theta)

        v = dxy_theta_to_v_without_v0(
            dxy=dxy,
            theta=theta,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        v_t0 = v[..., -1]
        return {"v": v_t0}

    @torch.no_grad()
    @torch._dynamo.disable()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
        output_all_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(
                f"future trajectory must have length {self.n_waypoints} "
                f"but got {traj_future_xyz.shape[-2]}"
            )

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[
            ..., :2
        ]

        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = theta_smooth(
            traj_future_rot=traj_future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )

        v0 = t0_states["v"]
        v = dxy_theta_to_v(
            dxy=dxy,
            theta=theta,
            v0=v0,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )

        accel = self._v_to_a(v)
        kappa = self._theta_v_a_to_kappa(theta, v, accel)

        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std

        if not output_all_states:
            return torch.stack([accel, kappa], dim=-1)
        else:
            return torch.stack([accel, kappa], dim=-1), torch.stack(
                [v[:, :-1], accel, theta[:, :-1]], dim=-1
            )

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        accel, kappa = action[..., 0], action[..., 1]

        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        v0 = t0_states["v"]
        dt = self.dt

        dt_2_term = 0.5 * (self.dt**2)
        velocity = torch.cat(
            [
                v0.unsqueeze(-1),
                (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)),
            ],
            dim=-1,
        )
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                (
                    initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
                ),
            ],
            dim=-1,
        )
        half_dt_term = 0.5 * dt
        initial_x = torch.zeros_like(v0)
        initial_y = torch.zeros_like(v0)
        x = (
            initial_x.unsqueeze(-1)
            + torch.cumsum(
                velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term, dim=-1
            )
            + torch.cumsum(
                velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt_term, dim=-1
            )
        )
        y = (
            initial_y.unsqueeze(-1)
            + torch.cumsum(
                velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt_term, dim=-1
            )
            + torch.cumsum(
                velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt_term, dim=-1
            )
        )
        batch_dim = traj_history_xyz.shape[:-2]
        traj_future_xyz = torch.zeros(
            *batch_dim,
            self.n_waypoints,
            3,
            device=traj_history_xyz.device,
            dtype=traj_history_xyz.dtype,
        )
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]

        traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))

        return traj_future_xyz, traj_future_rot


class AlpamayoR1LogitsProcessor(LogitsProcessor):
    """Masks out Alpamayo trajectory token logits."""

    def __init__(self, config, traj_token_start_idx, traj_vocab_size):
        super().__init__(config)
        self.traj_mask_start = traj_token_start_idx
        self.traj_mask_end = traj_token_start_idx + traj_vocab_size

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        logits_metadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = super()._get_logits(
            hidden_states, lm_head, logits_metadata, embedding_bias
        )
        logits[:, self.traj_mask_start : self.traj_mask_end] = float("-inf")
        return logits


class AlpamayoR1(nn.Module):
    """AlpamayoR1: VLM + expert diffusion model for trajectory prediction.

    Wraps Qwen3VLForConditionalGeneration as VLM and a separate Qwen3Model
    as the expert branch for flow-matching-based action generation.
    """

    _fourier_persistent: bool = True  # R1 checkpoint includes freqs buffers

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        # Store config for later use
        self.config = config
        qwen_config = config
        # we increaset vocab size to match Alpamayo's tokenizer, which may have additional special tokens compared to the base Qwen3-VL config
        qwen_config.text_config.vocab_size = config.vocab_size

        # Initialize internal Qwen3-VL model as 'vlm' (matching alpamayo naming)
        self.vlm = Qwen3VLForConditionalGeneration(
            qwen_config,
            quant_config=quant_config,
        )

        # override the logits processor to mask out trajectory tokens during generation
        self.vlm.logits_processor = AlpamayoR1LogitsProcessor(
            self.config,
            traj_token_start_idx=config.traj_token_start_idx,
            traj_vocab_size=config.traj_vocab_size,
        )

        logger.info("AlpamayoR1: Successfully initialized Qwen3-VL as self.vlm")

        # Build expert from text_config only (same as AutoModel.from_config(text_config)).
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if getattr(config, "expert_cfg", None) is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = Qwen3Model(expert_config, quant_config=quant_config)
        # Expert branch consumes continuous action embeddings, so token embedding is not needed.
        if hasattr(self.expert, "embed_tokens"):
            del self.expert.embed_tokens

        # Build action projection modules from Alpamayo config to match checkpoint shapes.
        action_in_proj_cfg = config.action_in_proj_cfg
        traj_tokenizer_cfg = config.traj_tokenizer_cfg
        action_space_cfg = config.traj_tokenizer_cfg["action_space_cfg"]
        n_waypoints = action_space_cfg["n_waypoints"]
        action_dim = len(traj_tokenizer_cfg["dims_max"])

        # Instantiate action space (UnicycleAccelCurvatureActionSpace)
        action_space_kwargs = {
            k: v
            for k, v in action_space_cfg.items()
            if k not in ("_target_", "_recursive_", "n_waypoints")
        }
        self.action_space = UnicycleAccelCurvatureActionSpace(
            **action_space_kwargs,
        )

        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=[n_waypoints, action_dim],
            out_dim=expert_config.hidden_size,
            hidden_size=action_in_proj_cfg["hidden_size"],
            num_enc_layers=action_in_proj_cfg["num_enc_layers"],
            max_freq=action_in_proj_cfg["max_freq"],
            num_fourier_feats=action_in_proj_cfg["num_fourier_feats"],
            fourier_persistent=self._fourier_persistent,
        )
        self.action_out_proj = torch.nn.Linear(expert_config.hidden_size, action_dim)

        self.traj_future_start_token_id = 155681  # <|traj_future_start|>
        self.traj_force_stop_token_id = 151645  # <|im_end|>

        # Set expert attention layers to ENCODER_ONLY:
        #   - Bidirectional (non-causal) attention among action tokens
        #   - Reads VLM's KV cache via shared layer_ids (both 0..N-1)
        #   - Does NOT write to KV cache (FlashInfer auto-skips set_kv_buffer for ENCODER_ONLY)
        for layer in self.expert.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
                layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY
                # Force save_kv_cache=False
                original_forward = layer.self_attn.attn.forward

                def _patched_forward(
                    q,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache=False,
                    _orig=original_forward,
                    **kwargs,
                ):
                    return _orig(q, k, v, forward_batch, save_kv_cache=False, **kwargs)

                layer.self_attn.attn.forward = _patched_forward
        # Flow matching parameters
        self.n_diffusion_tokens = n_waypoints
        self.action_dims = [n_waypoints, action_dim]
        diffusion_cfg = getattr(config, "diffusion_cfg", {}) or {}
        self.num_inference_steps = diffusion_cfg.get("num_inference_steps", 10)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.LongTensor,
        forward_batch: "ForwardBatch",
        **kwargs,
    ):
        ret = self.vlm(input_ids, positions, forward_batch, **kwargs)

        if forward_batch.forward_mode.is_decode():
            bstar = int(input_ids.shape[0])
            active_indices = []
            trajs = forward_batch.history_trajs
            for i in range(bstar):
                has_history_traj = (
                    trajs is not None and i < len(trajs) and trajs[i] is not None
                )

                should_trigger_flow_matching = input_ids[
                    i
                ] == self.traj_future_start_token_id or (
                    has_history_traj and input_ids[i] == self.traj_force_stop_token_id
                )

                if should_trigger_flow_matching:
                    # Force generation to stop immediately
                    ret.next_token_logits[i, :] = float("-inf")
                    ret.next_token_logits[i, self.traj_force_stop_token_id] = 0.0
                    active_indices.append(i)

            if active_indices:
                sampled_actions = self._run_flow_matching(active_indices, forward_batch)
                # Convert sampled actions → trajectories and write to ret.customized_info
                self._attach_traj_to_reqs(
                    sampled_actions, active_indices, forward_batch, ret, bstar
                )

        return ret

    def _attach_traj_to_reqs(
        self,
        sampled_actions: torch.Tensor,
        active_indices: List[int],
        forward_batch: "ForwardBatch",
        logits_output,
        bstar: int,
    ) -> None:
        """Convert sampled actions to trajectories and write to logits_output.customized_info.

        Args:
            sampled_actions: (bstar, n_waypoints, action_dim) on GPU.
            active_indices: batch-slot indices of active requests.
            forward_batch: ForwardBatch carrying per-req history_trajs.
            logits_output: LogitsProcessorOutput to write customized_info into.
            bstar: total batch size.
        """
        trajs = forward_batch.history_trajs
        if trajs is None:
            logger.warning(
                "_attach_traj_to_reqs: forward_batch.history_trajs is None; skipping action_to_traj"
            )
            return

        device = sampled_actions.device
        pred_traj_list: List = [None] * bstar

        for j, slot_i in enumerate(active_indices):
            history_traj = trajs[slot_i] or {}

            hist_xyz_raw = history_traj.get("ego_history_xyz")
            hist_rot_raw = history_traj.get("ego_history_rot")

            if hist_xyz_raw is None or hist_rot_raw is None:
                logger.warning(
                    f"_attach_traj_to_reqs: slot {slot_i} missing history_traj; "
                    "skipping action_to_traj for this request"
                )
                continue

            # Convert to tensor if needed and move to GPU.
            # Use float32 to match action_space precision (not bfloat16).
            if not isinstance(hist_xyz_raw, torch.Tensor):
                hist_xyz = torch.tensor(
                    hist_xyz_raw, dtype=torch.float32, device=device
                )
            else:
                hist_xyz = hist_xyz_raw.to(dtype=torch.float32, device=device)

            if not isinstance(hist_rot_raw, torch.Tensor):
                hist_rot = torch.tensor(
                    hist_rot_raw, dtype=torch.float32, device=device
                )
            else:
                hist_rot = hist_rot_raw.to(dtype=torch.float32, device=device)

            # Ensure shape: (T, 3) and (T, 3, 3) – add batch dim for action_to_traj
            if hist_xyz.dim() == 2:  # (T, 3)
                hist_xyz = hist_xyz.unsqueeze(0)  # (1, T, 3)
            if hist_rot.dim() == 3:  # (T, 3, 3)
                hist_rot = hist_rot.unsqueeze(0)  # (1, T, 3, 3)

            # sampled_actions[j]: (n_waypoints, action_dim) → unsqueeze batch
            action_j = sampled_actions[j].unsqueeze(0).float()  # (1, n_waypoints, 2)

            with torch.no_grad():
                pred_xyz, pred_rot = self.action_space.action_to_traj(
                    action_j, hist_xyz, hist_rot
                )  # (1, n_waypoints, 3), (1, n_waypoints, 3, 3)

            pred_traj_list[slot_i] = {
                "traj_xyz": pred_xyz[0].cpu().tolist(),
                "traj_rot": pred_rot[0].cpu().tolist(),
            }

        logits_output.customized_info = {"pred_traj": pred_traj_list}

    def _build_expert_forward_batch(
        self,
        active_indices: List[int],
        forward_batch: ForwardBatch,
        mrope_positions: torch.Tensor,
    ) -> ForwardBatch:
        """Build an EXTEND-mode ForwardBatch for the expert model.

        The expert reads the VLM's KV cache (shared layer_ids) and processes
        n_diffusion_tokens new action embeddings per request, using bidirectional
        attention (ENCODER_ONLY) without writing to KV cache.

        Args:
            active_indices: Indices of requests that need flow matching.
            forward_batch: The original decode-mode ForwardBatch from VLM.
            mrope_positions: Multimodal RoPE positions for action tokens,
                shape (3, bstar * n_diff).

        Returns:
            A ForwardBatch configured for EXTEND mode with the expert.
        """
        device = forward_batch.seq_lens.device
        bstar = len(active_indices)
        n_diff = self.n_diffusion_tokens
        idx_tensor = torch.tensor(active_indices, device=device, dtype=torch.long)

        # VLM's current seq_lens for active requests (tokens already in KV cache)
        vlm_seq_lens = forward_batch.seq_lens[idx_tensor]

        # Expert sees: prefix (VLM cached) + new action tokens.
        expert_seq_lens = vlm_seq_lens + n_diff
        extend_prefix_lens = vlm_seq_lens.to(torch.int32)
        extend_seq_lens = torch.full((bstar,), n_diff, dtype=torch.int32, device=device)

        # Cumulative start locations for the new tokens (each request contributes n_diff)
        extend_start_loc = torch.arange(
            0, bstar * n_diff, n_diff, dtype=torch.int32, device=device
        )

        # Dummy out_cache_loc — ENCODER_ONLY skips KV cache writes,
        # but the tensor must exist with the correct shape.
        out_cache_loc = torch.zeros(bstar * n_diff, dtype=torch.int32, device=device)

        expert_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bstar,
            input_ids=torch.zeros(bstar * n_diff, dtype=torch.long, device=device),
            req_pool_indices=forward_batch.req_pool_indices[idx_tensor],
            seq_lens=expert_seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(expert_seq_lens.sum()),
            extend_num_tokens=bstar * n_diff,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=extend_seq_lens.cpu().tolist(),
            extend_prefix_lens_cpu=extend_prefix_lens.cpu().tolist(),
            seq_lens_cpu=expert_seq_lens.cpu(),
            # Share the VLM's KV cache pool and attention backend
            req_to_token_pool=forward_batch.req_to_token_pool,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            attn_backend=forward_batch.attn_backend,
            # Multimodal RoPE positions for the action tokens
            mrope_positions=mrope_positions,
        )
        return expert_batch

    @torch.no_grad()
    def _run_flow_matching(
        self,
        active_indices: List[int],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Run flow matching (Euler integration) for requests that generated
        <traj_future_start>.

        This implements the same logic as the original alpamayo step_fn:
        1. Sample Gaussian noise x ~ N(0, I) in action space
        2. For each Euler step t_i -> t_{i+1}:
            a. action_in_proj(x, t) -> token embeddings [bstar, n_diff, hidden]
            b. expert forward (bidirectional, reads VLM KV cache) -> hidden states
            c. action_out_proj -> predicted velocity field v
            d. x = x + dt * v
        3. Return final x as the sampled action

        Args:
            active_indices: Indices of requests needing flow matching.
            forward_batch: The decode-mode ForwardBatch.

        Returns:
            Sampled actions of shape (bstar, n_waypoints, action_dim).
        """
        device = forward_batch.seq_lens.device
        bstar = len(active_indices)
        n_diff = self.n_diffusion_tokens

        # --- 1. Compute mRoPE positions for the action tokens ---
        # During decode, mrope_positions has shape (3, total_batch_size)
        # For each active request, action tokens get positions starting after
        # the current decode position (which is <traj_future_start>).
        positions_list = []
        for idx in active_indices:
            # Current mrope position of the <traj_future_start> token: shape (3,)
            current_mrope = forward_batch.mrope_positions[:, idx]  # (3,)
            # Action tokens: pos+1, pos+2, ..., pos+n_diff
            action_pos = (
                current_mrope.unsqueeze(1)
                + 1
                + torch.arange(n_diff, device=device).unsqueeze(0)
            )  # (3, n_diff)
            positions_list.append(action_pos)
        mrope_positions = torch.cat(positions_list, dim=1)  # (3, bstar * n_diff)

        # --- 2. Backend check ---
        backend = forward_batch.attn_backend
        if not isinstance(backend, TritonAttnBackend):
            raise RuntimeError(
                f"Alpamayo flow matching requires triton backend, "
                f"got {type(backend).__name__}"
            )

        # --- 3. Build the expert ForwardBatch (EXTEND mode) ---
        expert_batch = self._build_expert_forward_batch(
            active_indices, forward_batch, mrope_positions
        )

        backend.init_forward_metadata(expert_batch)

        # --- 3. Euler integration loop ---
        # Match reference FlowMatching._euler: x is fp32 (default dtype)
        # so Euler accumulation x = x + dt * v stays in fp32 throughout.

        x = torch.randn(
            bstar,
            *self.action_dims,
            device=device,
        )

        time_steps = torch.linspace(
            0.0, 1.0, self.num_inference_steps + 1, device=device
        )

        for step_i in range(self.num_inference_steps):
            dt = time_steps[step_i + 1] - time_steps[step_i]
            t = time_steps[step_i].view(1, 1, 1).expand(bstar, 1, 1)

            # --- step_fn start ---
            # a. Project noisy action + timestep -> expert token embeddings
            #    x: (bstar, n_waypoints, action_dim)
            #    t: (bstar, 1, 1)
            #    output: (bstar, n_diff, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(bstar, n_diff, -1)

            # b. Flatten for sglang's model forward: (bstar * n_diff, hidden_size)
            input_embeds_flat = future_token_embeds.reshape(bstar * n_diff, -1)

            # c. Run expert: bidirectional attention over action tokens
            expert_hidden = self.expert(
                input_ids=None,
                positions=mrope_positions,
                forward_batch=expert_batch,
                input_embeds=input_embeds_flat,
            )  # (bstar * n_diff, hidden_size)

            # d. Project to action space
            expert_hidden = expert_hidden.view(
                bstar, n_diff, -1
            )  # (bstar, n_diff, hidden_size)
            pred = self.action_out_proj(expert_hidden)  # (bstar, n_diff, action_dim)
            pred = pred.view(bstar, *self.action_dims)
            # --- step_fn end ---

            # Euler update: x_{i+1} = x_i + dt * v(x_i, t_i)
            x = x + dt * pred
        return x  # (bstar, n_waypoints, action_dim)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.vlm.pad_input_ids(input_ids, mm_inputs)

    def _load_expert_weights(
        self,
        expert_weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        """Load expert (Qwen3 text backbone) weights.

        Returns:
            The number of checkpoint tensors loaded.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.expert.named_parameters())
        expected_param_names = set(params_dict.keys())
        loaded_full_params = set()
        loaded_stacked_shards = defaultdict(set)
        unexpected_ckpt_keys = []
        loaded_cnt = 0

        for name, loaded_weight in expert_weights:
            # Keep compatibility with checkpoints that include an extra "model." prefix.
            if name.startswith("model."):
                name = name[len("model.") :]

            # embed_tokens is intentionally removed for expert branch.
            if name.startswith("embed_tokens."):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.expert.start_layer or layer_id >= self.expert.end_layer
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Expert parameter {name} not found; skipping")
                    unexpected_ckpt_keys.append(name)
                    break

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_shards[name].add(shard_id)
                loaded_cnt += 1
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Expert parameter {name} not found; skipping")
                    unexpected_ckpt_keys.append(name)
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_full_params.add(name)
                loaded_cnt += 1

        missing_params = []
        for pname in expected_param_names:
            if "qkv_proj" in pname:
                if loaded_stacked_shards.get(pname, set()) != {"q", "k", "v"}:
                    missing_params.append(pname)
                continue
            if "gate_up_proj" in pname:
                if loaded_stacked_shards.get(pname, set()) != {0, 1}:
                    missing_params.append(pname)
                continue
            if pname not in loaded_full_params:
                missing_params.append(pname)

        logger.info(f"AlpamayoR1: loaded {loaded_cnt} expert tensors")
        if missing_params or unexpected_ckpt_keys:
            raise RuntimeError(
                "AlpamayoR1 expert load failed: "
                f"missing={len(missing_params)}, "
                f"unexpected={len(unexpected_ckpt_keys)}. "
                f"Sample missing={missing_params[:8]}, "
                f"sample unexpected={unexpected_ckpt_keys[:8]}"
            )
        return loaded_cnt

    def _load_plain_module_weights(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        state_dict = {name: tensor for name, tensor in module_weights}
        if not state_dict:
            raise RuntimeError(f"AlpamayoR1: no weights found for {module_name}")
        module.load_state_dict(state_dict, strict=True)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        strict: bool = True,
    ):
        """
        Load checkpoint weights for AlpamayoR1, routing them to the correct
        submodules (vlm, expert, action_in_proj, action_out_proj, action_space).
        """
        vlm_weights = []
        expert_weights = []
        action_in_proj_weights = []
        action_out_proj_weights = []
        action_space_weights = []

        for name, tensor in weights:
            if name.startswith("vlm."):
                vlm_weights.append((name[len("vlm.") :], tensor))
                continue

            if name.startswith("expert."):
                expert_weights.append((name[len("expert.") :], tensor))
                continue

            if name.startswith("action_in_proj."):
                action_in_proj_weights.append((name[len("action_in_proj.") :], tensor))
                continue

            if name.startswith("action_out_proj."):
                action_out_proj_weights.append(
                    (name[len("action_out_proj.") :], tensor)
                )
                continue

            if name.startswith("action_space."):
                action_space_weights.append((name[len("action_space.") :], tensor))
                continue

            # Keep compatibility for checkpoints without explicit "vlm." prefix.
            vlm_weights.append((name, tensor))

        # 1) Load VLM weights.
        if strict and not vlm_weights:
            raise RuntimeError(
                "AlpamayoR1 strict load failed: no checkpoint weights were routed to vlm."
            )
        self.vlm.load_weights(iter(vlm_weights))
        logger.info(f"AlpamayoR1: loaded {len(vlm_weights)} vlm tensors")

        # 2) Load expert weights.
        expert_loaded_cnt = self._load_expert_weights(expert_weights)

        # 3) Load action space buffers (accel_mean, accel_std, etc.).
        # IMPORTANT: convert action_space to float32 BEFORE loading weights,
        # so that the float32 checkpoint values are not truncated to bfloat16.
        # The original alpamayo code keeps action_space in float32 (keep_same_dtype
        # only applies to diffusion, action_in_proj, action_out_proj).
        self.action_space = self.action_space.float()
        if action_space_weights:
            self._load_plain_module_weights(
                self.action_space,
                "action_space",
                action_space_weights,
            )

        # 4) Load action projection modules.
        self._load_plain_module_weights(
            self.action_in_proj,
            "action_in_proj",
            action_in_proj_weights,
        )
        self._load_plain_module_weights(
            self.action_out_proj,
            "action_out_proj",
            action_out_proj_weights,
        )

        # 5) Match reference: convert action_in_proj / action_out_proj to
        #    the same dtype as the expert (bf16).  This is critical because
        #    FourierEncoderV2.freqs is a non-persistent buffer that is NOT
        #    loaded from the checkpoint -- without .to() it stays in fp32,
        #    producing different Fourier features than training time.
        expert_dtype = next(self.expert.parameters()).dtype
        keep_same_dtype = getattr(self.vlm.config, "keep_same_dtype", True)
        if keep_same_dtype:
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        logger.info(
            "AlpamayoR1 load summary: "
            f"strict={strict}, "
            f"vlm_ckpt_tensors={len(vlm_weights)}, "
            f"expert_ckpt_tensors={len(expert_weights)}, "
            f"expert_loaded_tensors={expert_loaded_cnt}, "
            f"action_in_proj_ckpt_tensors={len(action_in_proj_weights)}, "
            f"action_out_proj_ckpt_tensors={len(action_out_proj_weights)}, "
            f"action_space_ckpt_tensors={len(action_space_weights)}, "
        )


# Architecture name alias used in HuggingFace config (architectures: ['Alpamayo1_5'])
class Alpamayo1_5(AlpamayoR1):
    _fourier_persistent: bool = False  # 1.5 recomputes freqs at runtime


# Entry point for SGLang model registry
EntryClass = [AlpamayoR1, Alpamayo1_5]
