"""Target KV feature math shared by offline training and draft serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Literal

import msgspec
import torch
from sglang.srt.training_capture.protocol import (
    DTYPES,
    ContractError,
    KVSpec,
    Positive,
    StrictStruct,
    validate_kv_spec,
)


class StandardRopeConfig(StrictStruct):
    type: Literal["default"]
    theta: Annotated[float, msgspec.Meta(gt=0)]
    rotary_dim: Positive
    interleaved: bool
    scaling: None

    @classmethod
    def from_kv(cls, kv: KVSpec):
        validate_kv_spec(kv)
        rope = msgspec.convert(kv.rope_config, type=cls, strict=True)
        if not math.isfinite(rope.theta) or rope.rotary_dim % 2:
            raise ContractError(
                "standard RoPE requires finite theta and even rotary_dim"
            )
        if any(layer.key_head_dim < rope.rotary_dim for layer in kv.layers):
            raise ContractError("rotary_dim exceeds a selected key head dimension")
        return rope


def inverse_standard_rope(
    key: torch.Tensor, positions: torch.Tensor, rope: StandardRopeConfig
) -> torch.Tensor:
    """Invert the declared rotation in FP32, preserving any unrotated suffix.

    This cannot recover precision lost when the source pool rounded its K.
    Target K norm is already present and is deliberately not inverted.
    """
    if key.ndim != 3 or positions.shape != key.shape[:1]:
        raise ContractError("K and actual positions must cover the same token rows")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ContractError("positions must be integer token positions")
    if positions.device != key.device:
        raise ContractError("K and positions must be on the same device")
    dim = rope.rotary_dim
    if dim > key.shape[-1]:
        raise ContractError("rotary_dim exceeds key head dimension")
    # Construct frequencies in FP32 here so model.to(bfloat16) cannot narrow
    # an inverse-frequency buffer and silently change the codec.
    frequencies = 1.0 / (
        rope.theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=key.device) / dim)
    )
    angles = positions.float()[:, None] * frequencies[None]
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    source = key.float()
    rotated = source[..., :dim]
    if rope.interleaved:
        left, right = rotated[..., 0::2], rotated[..., 1::2]
        restored = torch.stack(
            (left * cos + right * sin, right * cos - left * sin), dim=-1
        ).flatten(-2)
    else:
        left, right = rotated.chunk(2, dim=-1)
        restored = torch.cat(
            (left * cos + right * sin, right * cos - left * sin), dim=-1
        )
    return torch.cat((restored, source[..., dim:]), dim=-1)


def target_kv_features(
    kv: KVSpec,
    tensors: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    *,
    feature_k_stage: Literal["pre_rope", "post_rope"] = "pre_rope",
) -> torch.Tensor:
    """Concatenate layer-ordered K then V, with head/dim inside each component.

    Source tensors are constants; gradients belong to the downstream encoder.
    The caller supplies only the committed context rows, excluding the anchor.
    """
    rope = StandardRopeConfig.from_kv(kv)
    if feature_k_stage not in ("pre_rope", "post_rope"):
        raise ContractError("unknown target K feature stage")
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ContractError("positions must be a one-dimensional integer tensor")
    if kv.source_k_stage == "pre_rope" and feature_k_stage == "post_rope":
        raise ContractError("pre-RoPE sources cannot be used as post-RoPE features")
    expected_names = {
        f"target_{component}.{layer.layer_id}"
        for layer in kv.layers
        for component in ("k", "v")
    }
    if set(tensors) != expected_names:
        raise ContractError("KV features require exactly the selected layer components")
    features = []
    for layer in kv.layers:
        for component, dim in (("k", layer.key_head_dim), ("v", layer.value_head_dim)):
            value = tensors[f"target_{component}.{layer.layer_id}"]
            if (
                tuple(value.shape) != (positions.numel(), layer.num_kv_heads, dim)
                or value.dtype != DTYPES[kv.dtype]
                or value.device != positions.device
            ):
                raise ContractError("source KV tensor disagrees with feature contract")
            value = value.detach()
            if (
                component == "k"
                and kv.source_k_stage == "post_rope"
                and feature_k_stage == "pre_rope"
            ):
                value = inverse_standard_rope(value, positions, rope)
            features.append(value.float().flatten(1))
    return torch.cat(features, dim=-1)
