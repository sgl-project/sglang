from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.configs.inkling import InklingVisionConfig
from sglang.srt.models.inkling_common.norm import RMSNorm


def _prime_factors(n: int) -> list[int]:
    """Return the prime factors of ``n`` in ascending order."""
    if n < 1:
        raise ValueError("n must be a positive integer")

    factors: list[int] = []

    while n % 2 == 0:
        factors.append(2)
        n //= 2

    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2

    if n > 1:
        factors.append(n)
    return factors


def plan_out_scales(
    temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int = 3
) -> list[tuple[int, int, int, int]]:
    """Plan the ``(time, height, width, channels)`` scale at each HMLP layer."""
    if patch_size <= 1:
        raise ValueError(
            "patch_size must be greater than 1, otherwise this doesn't make sense"
        )

    def _round_up(x: int) -> int:
        return int(np.ceil(x / 64)) * 64

    last_h_scale = 1
    scales: list[tuple[int, int, int, int]] = [(1, 1, 1, n_channels)]
    for pscale in _prime_factors(patch_size)[::-1]:
        last_h_scale *= pscale
        scales.append(
            (
                1,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels),
            )
        )
    last_t_scale = 1
    for tscale in _prime_factors(temporal_patch_size)[::-1]:
        last_t_scale *= tscale
        scales.append(
            (
                last_t_scale,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels * last_t_scale),
            )
        )

    size_reduction = np.prod(np.array(scales)[:, :-1], 1)

    log_ideal_scales = np.linspace(
        0,
        np.log(patch_size * patch_size * temporal_patch_size * n_channels),
        n_layers + 1,
    )
    cost_matrix = np.abs(log_ideal_scales[:, None] - np.log(size_reduction)[None])

    if n_layers >= len(scales):
        idxs = np.argmin(cost_matrix, axis=1)
    else:
        from scipy.optimize import linear_sum_assignment

        idxs = linear_sum_assignment(cost_matrix)[1]

    assert len(idxs) >= 2
    idxs[0] = 0
    idxs[-1] = len(scales) - 1

    return [scales[i] for i in idxs]


def fold_timespace_to_depth(
    vision_patches_bthwc: torch.Tensor, t_fold: int, hw_fold: int
) -> torch.Tensor:
    """Fold temporal and spatial neighborhoods into the channel dimension."""
    B, T, H, W, C = vision_patches_bthwc.shape

    assert T % t_fold == 0, f"Temporal dimension {T} must be divisible by {t_fold}"
    assert H % hw_fold == 0, f"Height dimension {H} must be divisible by {hw_fold}"
    assert W % hw_fold == 0, f"Width dimension {W} must be divisible by {hw_fold}"

    t_new = T // t_fold
    h_new = H // hw_fold
    w_new = W // hw_fold

    x = vision_patches_bthwc.reshape(
        B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C
    )

    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)

    x = x.reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)

    return x


class HMLPPatchEncoder(nn.Module):
    def __init__(
        self,
        config: InklingVisionConfig,
    ):
        super().__init__()
        self.decoder_dmodel = config.decoder_dmodel
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.n_channels = config.n_channels
        self.n_layers = config.n_layers
        self.use_vision_norm = config.use_vision_norm

        self.scales: list[tuple[int, int, int, int]] = plan_out_scales(
            self.temporal_patch_size, self.patch_size, self.n_layers, self.n_channels
        )
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for i, (start_scale, end_scale) in enumerate(
            zip(self.scales[:-1], self.scales[1:])
        ):
            shuffle_mult = (
                (end_scale[0] // start_scale[0])
                * (end_scale[1] // start_scale[1])
                * (end_scale[2] // start_scale[2])
            )
            if i == self.n_layers - 1:
                self.layers[f"linear_{i}"] = nn.Linear(
                    start_scale[3] * shuffle_mult, self.decoder_dmodel, bias=False
                )
            else:
                self.layers[f"linear_{i}"] = nn.Linear(
                    start_scale[3] * shuffle_mult, end_scale[3], bias=False
                )
                self.layers[f"norm_{i}"] = RMSNorm(end_scale[3])

        self.final_norm: RMSNorm | None = None
        if self.use_vision_norm:
            self.final_norm = RMSNorm(self.decoder_dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_patches, T, H, W, C = x.shape
        for i, (start_scale, end_scale) in enumerate(
            zip(self.scales[:-1], self.scales[1:])
        ):
            t_fold = end_scale[0] // start_scale[0]
            hw_fold = end_scale[1] // start_scale[1]
            if hw_fold > 1 or t_fold > 1:
                x = fold_timespace_to_depth(x, t_fold, hw_fold)
            assert x.shape[1:-1] == (
                T // end_scale[0],
                H // end_scale[1],
                W // end_scale[2],
            )
            x = self.layers[f"linear_{i}"](x)
            if i < self.n_layers - 1:
                norm = cast(RMSNorm, self.layers[f"norm_{i}"])
                x = norm(x)
                x = F.gelu(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        x = x.reshape(num_patches, -1)
        return x
