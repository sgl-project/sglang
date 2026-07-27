# SPDX-License-Identifier: Apache-2.0
"""Position encoding and patch-merging helpers for Kimi-K3 MoonViT."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    x_shape=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del x_shape
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_complex = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], -1, 2)
    )
    xk_complex = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], -1, 2)
    )
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def tpool_patch_merger(
    hidden_states: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    hidden_size = hidden_states.size(-1)
    outputs = []
    offset = 0
    merge_h, merge_w = merge_kernel_size
    for t, h, w in grid_thws.tolist():
        sequence = hidden_states[offset : offset + t * h * w]
        new_h, new_w = h // merge_h, w // merge_w
        sequence = sequence.view(
            t, new_h, merge_h, new_w, merge_w, hidden_size
        )
        sequence = (
            sequence.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .mean(dim=0)
        )
        outputs.append(
            sequence.view(new_h * new_w, merge_h * merge_w, hidden_size)
        )
        offset += t * h * w
    return outputs


def _get_1d_sincos_pos_embed(embed_dim: int, size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32).reshape(-1)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    values = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(values), np.cos(values)], axis=1)


class KimiK3Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(dim, num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )
        nn.init.normal_(self.weight)

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        position_embeddings = []
        for t, h, w in grid_thws.tolist():
            if t > self.num_frames:
                raise ValueError(f"t={t} exceeds num_frames={self.num_frames}")
            if (h, w) == self.weight.shape[:-1]:
                spatial = self.weight.flatten(end_dim=1)
            else:
                spatial = (
                    F.interpolate(
                        self.weight.permute(2, 0, 1).unsqueeze(0),
                        size=(h, w),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(end_dim=1)
                )
            if t == 1:
                spatiotemporal = spatial
            else:
                spatiotemporal = (
                    spatial.unsqueeze(0).repeat(t, 1, 1)
                    + self.time_weight[:t]
                )
            position_embeddings.append(
                spatiotemporal.reshape(-1, spatiotemporal.shape[-1])
            )
        return hidden_states + torch.cat(position_embeddings)


class KimiK3Rope2DPosEmbRepeated(nn.Module):
    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: int = 10000,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("K3 vision RoPE dimension must be divisible by 4")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        flat_positions = torch.arange(
            self.max_height * self.max_width,
            device=device,
            dtype=torch.float32,
        )
        x_positions = flat_positions % self.max_width
        y_positions = flat_positions // self.max_width
        dim_range = torch.arange(
            0, self.dim, 4, device=device, dtype=torch.float32
        )[: self.dim // 4]
        frequencies = 1.0 / (
            self.theta_base ** (dim_range / self.dim)
        )
        x_freqs = torch.outer(x_positions, frequencies)
        y_freqs = torch.outer(y_positions, frequencies)
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        return torch.cat(
            [x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1
        ).reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis",
                self._precompute_freqs_cis(device),
                persistent=False,
            )
        frequencies = []
        for t, h, w in grid_thws.tolist():
            if not (
                1 <= h <= self.max_height and 1 <= w <= self.max_width
            ):
                raise ValueError(
                    f"Invalid K3 vision grid {(t, h, w)} for "
                    f"{self.max_height}x{self.max_width} RoPE"
                )
            frequencies.append(
                self.freqs_cis[:h, :w]
                .reshape(-1, self.dim // 2)
                .repeat(t, 1)
            )
        return torch.cat(frequencies, dim=0)
