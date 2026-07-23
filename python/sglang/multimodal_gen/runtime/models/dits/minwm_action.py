# SPDX-License-Identifier: Apache-2.0
"""MinWM primitive-token action ontology and conditioner."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_CLASSES_PER_AXIS = 9
NUM_ACTION_CLASSES = 81
PRIMITIVE_BIT_WIDTH = 8
PRIMITIVE_VOCAB_SIZE = 5
BITS_PER_SUBSET = 4

_KEY_TO_BIT = {
    "w": 0,
    "a": 1,
    "s": 2,
    "d": 3,
    "i": 4,
    "j": 5,
    "k": 6,
    "l": 7,
    "up": 4,
    "left": 5,
    "down": 6,
    "right": 7,
}
_TRANS_BITS_TO_LABEL = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 0, 1, 0): 2,
    (0, 1, 0, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 1, 0, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 0, 1, 1): 8,
}
_ROT_BITS_TO_LABEL = {
    (0, 0, 0, 0): 0,
    (0, 0, 0, 1): 1,
    (0, 1, 0, 0): 2,
    (1, 0, 0, 0): 3,
    (0, 0, 1, 0): 4,
    (1, 0, 0, 1): 5,
    (0, 0, 1, 1): 6,
    (1, 1, 0, 0): 7,
    (0, 1, 1, 0): 8,
}

_LABEL_TO_BITS = torch.zeros(NUM_ACTION_CLASSES, PRIMITIVE_BIT_WIDTH)
for trans_bits, trans_label in _TRANS_BITS_TO_LABEL.items():
    for rot_bits, rot_label in _ROT_BITS_TO_LABEL.items():
        _LABEL_TO_BITS[trans_label * ACTION_CLASSES_PER_AXIS + rot_label] = (
            torch.tensor((*trans_bits, *rot_bits), dtype=torch.float32)
        )


def key_state_to_action_label(keys: list[str]) -> int:
    """Convert one held-key state to the checkpoint's exact 81-class label."""
    bits = [0] * PRIMITIVE_BIT_WIDTH
    for raw_key in keys:
        key = str(raw_key).lower().strip()
        if key not in _KEY_TO_BIT:
            raise ValueError(
                f"unknown MinWM action key {raw_key!r}; valid keys: "
                f"{sorted(_KEY_TO_BIT)}"
            )
        bits[_KEY_TO_BIT[key]] = 1
    trans = tuple(bits[:BITS_PER_SUBSET])
    rot = tuple(bits[BITS_PER_SUBSET:])
    if trans not in _TRANS_BITS_TO_LABEL or rot not in _ROT_BITS_TO_LABEL:
        raise ValueError(
            f"unsupported MinWM action combination {keys!r}; use at most one "
            "forward/backward, one strafe, one pitch, and one yaw key"
        )
    return (
        _TRANS_BITS_TO_LABEL[trans] * ACTION_CLASSES_PER_AXIS + _ROT_BITS_TO_LABEL[rot]
    )


def validate_action_labels(labels, *, expected_frames: int | None = None) -> list[int]:
    if not isinstance(labels, list):
        raise ValueError("action_labels must be a list[int]")
    result = []
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError("action_labels must be a list[int]")
        if not 0 <= label < NUM_ACTION_CLASSES:
            raise ValueError("MinWM action labels must be in [0, 80]")
        result.append(label)
    if expected_frames is not None and len(result) != expected_frames:
        raise ValueError(
            f"expected {expected_frames} MinWM action labels, got {len(result)}"
        )
    return result


def validate_action_weights(
    weights, *, expected_frames: int | None = None
) -> list[list[float]]:
    """Validate per-decoded-frame ``[w,a,s,d,i,j,k,l]`` weights in ``[0,1]``."""
    if not isinstance(weights, list):
        raise ValueError("action_weights must be a list of 8-value rows")
    result = []
    for row in weights:
        if not isinstance(row, list) or len(row) != PRIMITIVE_BIT_WIDTH:
            raise ValueError("each action_weights row must contain 8 values")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("action_weights values must be numbers in [0, 1]")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("action_weights values must be finite and in [0, 1]")
            values.append(value)
        result.append(values)
    if expected_frames is not None and len(result) != expected_frames:
        raise ValueError(
            f"expected {expected_frames} MinWM action weight rows, got {len(result)}"
        )
    return result


def action_labels_to_primitive_bits(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.long)
    if torch.any((labels < 0) | (labels >= NUM_ACTION_CLASSES)):
        raise ValueError("MinWM action labels must be in [0, 80]")
    return _LABEL_TO_BITS.to(device=labels.device)[labels]


class CausalActionTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(out_channels)
        self.causal_pad = kernel_size - 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (self.causal_pad, 0))
        hidden_states = self.conv(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        return F.silu(hidden_states).transpose(1, 2)


class PrimitiveTokenResidualActionEncoder(nn.Module):
    """Exact `primitive_token_residual` module used by minWM main."""

    def __init__(
        self,
        dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.move_embedding = nn.Embedding(PRIMITIVE_VOCAB_SIZE, embed_dim)
        self.look_embedding = nn.Embedding(PRIMITIVE_VOCAB_SIZE, embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
        )
        self.encode_1 = CausalActionTemporalBlock(
            embed_dim * 2, hidden_dim, kernel_size
        )
        self.encode_2 = CausalActionTemporalBlock(hidden_dim, hidden_dim, kernel_size)
        self.proj = nn.Linear(hidden_dim, dim)

    @staticmethod
    def _pool(weights: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        active_count = (weights > 0).sum(dim=-1, keepdim=True).to(dtype=weights.dtype)
        pooled = weights @ embedding.weight[1:]
        pooled = pooled / active_count.clamp_min(1).sqrt()
        noop = embedding.weight[0].view(1, 1, -1).expand_as(pooled)
        return torch.where(active_count > 0, pooled, noop)

    def frame_states(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 2:
            weights = action_labels_to_primitive_bits(action)
        elif action.ndim == 4 and action.shape[-1] == PRIMITIVE_BIT_WIDTH:
            weights = action
            if not torch.isfinite(weights).all():
                raise ValueError("MinWM action weights must be finite")
            if torch.any((weights < 0) | (weights > 1)):
                raise ValueError("MinWM action weights must be in [0, 1]")
        else:
            raise ValueError(
                "MinWM action must have shape [B, F] labels or [B, F, S, 8] weights"
            )
        weights = weights.to(
            device=self.move_embedding.weight.device,
            dtype=self.move_embedding.weight.dtype,
        )
        move = self._pool(weights[..., :BITS_PER_SUBSET], self.move_embedding)
        look = self._pool(weights[..., BITS_PER_SUBSET:], self.look_embedding)
        hidden_states = torch.cat([move, look], dim=-1)
        hidden_states = hidden_states + self.fuse(hidden_states)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.mean(dim=2)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.encode_2(self.encode_1(hidden_states)).transpose(1, 2)
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype))

    def token_residual(
        self,
        action: torch.Tensor,
        *,
        num_current_frames: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action.ndim not in (2, 4):
            raise ValueError(
                "MinWM action tensor must have shape [B, F] or [B, F, S, 8]"
            )
        if action.shape[1] < num_current_frames:
            raise ValueError(
                "MinWM action window is shorter than the current latent chunk"
            )
        states = self.frame_states(action)[:, -num_current_frames:]
        return (
            states[:, :, None]
            .expand(-1, -1, tokens_per_frame, -1)
            .reshape(states.shape[0], num_current_frames * tokens_per_frame, -1)
            .to(dtype=dtype)
            .contiguous()
        )
