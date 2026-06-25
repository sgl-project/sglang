# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import (
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import torch

FP8_MAX_E4M3 = 448.0
FP8_SCALE_EPS = 1.0e-12
COSMOS_FP8_ACTIVATION_SCALE_SITES: Tuple[str, ...] = (
    "sa_normed",
    "sa_q",
    "sa_k",
    "sa_v",
    "sa_attn_out",
    "ca_normed",
    "ca_q",
    "ca_attn_out",
    "ffn_normed",
    "ffn1_gelu",
)


_COSMOS_BLOCK_FP8_LINEAR_KEYS: Tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
)

_COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS: Tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
)

_COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY = "self_attn.qkv_proj.weight"
_COSMOS_FP8_PREPARED_WEIGHT_SUFFIX = "_fp8_prepared"
_COSMOS_FP8_PREPARED_SCALE_SUFFIX = "_fp8_prepared_scale"

_COSMOS_FP8_LINEAR_POLICY_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "sa_qkv": _COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS,
    "sa_out": ("self_attn.output_proj.weight",),
    "ca_q": ("cross_attn.q_proj.weight",),
    "ca_out": ("cross_attn.output_proj.weight",),
    "ffn1": ("mlp.layer1.weight",),
    "ffn2": ("mlp.layer2.weight",),
}

_COSMOS_FP8_LINEAR_POLICIES: Mapping[str, Tuple[str, ...]] = {
    "all": tuple(_COSMOS_FP8_LINEAR_POLICY_GROUPS.keys()),
    "attn_only": ("sa_qkv", "sa_out", "ca_q", "ca_out"),
    "mlp_only": ("ffn1", "ffn2"),
    "no_ffn2": ("sa_qkv", "sa_out", "ca_q", "ca_out", "ffn1"),
    "no_out_proj": ("sa_qkv", "ca_q", "ffn1"),
}


def cosmos_block_fp8_linear_keys(num_blocks: int) -> Tuple[str, ...]:
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    return tuple(
        f"blocks.{block_idx}.{rel_key}"
        for block_idx in range(num_blocks)
        for rel_key in _COSMOS_BLOCK_FP8_LINEAR_KEYS
    )


def cosmos_block_fp8_fused_qkv_keys(num_blocks: int) -> Tuple[str, ...]:
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    return tuple(
        f"blocks.{block_idx}.{_COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY}"
        for block_idx in range(num_blocks)
    )


def scale_key_for_weight(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected a .weight key, got {weight_key!r}")
    return f"{weight_key}_scale"


def fp8_prepared_weight_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected a .weight key, got {weight_key!r}")
    return f"{weight_key}{_COSMOS_FP8_PREPARED_WEIGHT_SUFFIX}"


def fp8_prepared_scale_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected a .weight key, got {weight_key!r}")
    return f"{weight_key}{_COSMOS_FP8_PREPARED_SCALE_SUFFIX}"


def cosmos_fp8_linear_policy_groups() -> Tuple[str, ...]:
    return tuple(_COSMOS_FP8_LINEAR_POLICY_GROUPS.keys())


def cosmos_fp8_linear_policies() -> Tuple[str, ...]:
    return tuple((*_COSMOS_FP8_LINEAR_POLICIES.keys(), "custom"))


def _normalize_group_names(groups: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
    if groups is None:
        return ()
    if isinstance(groups, str):
        raw = groups.replace(";", ",").split(",")
    else:
        raw = list(groups)
    normalized = tuple(group.strip().lower() for group in raw if group.strip())
    unknown = tuple(
        group for group in normalized if group not in _COSMOS_FP8_LINEAR_POLICY_GROUPS
    )
    if unknown:
        valid = ", ".join(cosmos_fp8_linear_policy_groups())
        raise ValueError(
            f"unknown Cosmos FP8 linear policy group {unknown[0]!r}; expected one of: {valid}"
        )
    return normalized


def resolve_cosmos_fp8_linear_policy(
    policy: str = "all",
    *,
    custom_groups: Optional[Sequence[str] | str] = None,
) -> Tuple[str, ...]:
    """Return relative Cosmos block linear keys selected for FP8 quantization."""

    normalized = policy.strip().lower()
    if normalized == "custom":
        groups = _normalize_group_names(custom_groups)
        if not groups:
            raise ValueError(
                "custom Cosmos FP8 linear policy requires at least one group"
            )
    elif normalized in _COSMOS_FP8_LINEAR_POLICIES:
        if custom_groups is not None:
            raise ValueError("custom_groups is only valid when policy='custom'")
        groups = _COSMOS_FP8_LINEAR_POLICIES[normalized]
    else:
        valid = ", ".join(cosmos_fp8_linear_policies())
        raise ValueError(
            f"unknown Cosmos FP8 linear policy {policy!r}; expected one of: {valid}"
        )

    selected: list[str] = []
    for group in groups:
        for rel_key in _COSMOS_FP8_LINEAR_POLICY_GROUPS[group]:
            if rel_key not in selected:
                selected.append(rel_key)
    return tuple(selected)


def quantize_fp8_per_out_channel(
    weight: torch.Tensor,
    *,
    scale_dtype: torch.dtype = torch.float16,
    eps: float = FP8_SCALE_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a PyTorch Linear weight [out_features, in_features] to E4M3 bytes.

    The returned byte tensor intentionally keeps the original row-major
    [out, in] shape. Cosmos CUTLASS RCR GEMMs consume that exact memory as a
    column-major [in, out] matrix, matching ``input @ weight.T``.
    """

    if weight.dim() != 2:
        raise ValueError(
            f"linear weight must be 2D [out, in], got {tuple(weight.shape)}"
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for Cosmos FP8 quantization"
        )

    weight_f32 = weight.detach().to(torch.float32)
    amax = weight_f32.abs().amax(dim=1)
    scale_f32 = torch.clamp(amax / FP8_MAX_E4M3, min=eps)
    q_f32 = torch.clamp(weight_f32 / scale_f32[:, None], -FP8_MAX_E4M3, FP8_MAX_E4M3)
    q_u8 = q_f32.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return q_u8, scale_f32.to(device=weight.device, dtype=scale_dtype).contiguous()


def fp8_activation_scale(
    activation: torch.Tensor,
    *,
    mode: str = "dynamic",
    static_scale: Optional[torch.Tensor | float] = None,
    calibrated_amax: Optional[torch.Tensor | float] = None,
    scale_dtype: torch.dtype = torch.float32,
    eps: float = FP8_SCALE_EPS,
) -> torch.Tensor:
    """Return a finite per-tensor E4M3 activation scale.

    Supported modes are explicit because the kernel ABI must not hide how Q/K/V
    ranges were chosen:

    - ``dynamic``: scale from the runtime tensor amax.
    - ``calibrated``: scale from a caller-provided calibration amax.
    - ``static``: use a caller-provided scale directly.
    """

    mode = mode.lower()
    device = activation.device
    if mode == "dynamic":
        scale = activation.detach().to(torch.float32).abs().amax() / FP8_MAX_E4M3
    elif mode == "calibrated":
        if calibrated_amax is None:
            raise ValueError("calibrated mode requires calibrated_amax")
        scale = (
            torch.as_tensor(calibrated_amax, device=device, dtype=torch.float32)
            / FP8_MAX_E4M3
        )
    elif mode == "static":
        if static_scale is None:
            raise ValueError("static mode requires static_scale")
        scale = torch.as_tensor(static_scale, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown FP8 activation scale mode {mode!r}")

    scale = torch.clamp(scale, min=eps)
    if not torch.isfinite(scale).all():
        raise ValueError("FP8 activation scale must be finite")
    return scale.to(dtype=scale_dtype).contiguous()


def quantize_fp8_activation_per_tensor(
    activation: torch.Tensor,
    *,
    mode: str = "dynamic",
    static_scale: Optional[torch.Tensor | float] = None,
    calibrated_amax: Optional[torch.Tensor | float] = None,
    scale_dtype: torch.dtype = torch.float32,
    eps: float = FP8_SCALE_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation tensor to raw E4M3 bytes with one tensor scale."""

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for Cosmos FP8 quantization"
        )
    scale = fp8_activation_scale(
        activation,
        mode=mode,
        static_scale=static_scale,
        calibrated_amax=calibrated_amax,
        scale_dtype=scale_dtype,
        eps=eps,
    )
    q_f32 = torch.clamp(
        activation.detach().to(torch.float32) / scale.to(torch.float32),
        -FP8_MAX_E4M3,
        FP8_MAX_E4M3,
    )
    q_u8 = q_f32.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return q_u8, scale


def assert_fp8_scale_shape(
    scale: torch.Tensor, expected_shape: Tuple[int, ...], *, name: str
) -> None:
    if tuple(scale.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(scale.shape)}"
        )
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not torch.isfinite(scale.float()).all():
        raise ValueError(f"{name} must contain only finite values")


def validate_cosmos_fp8_activation_calibration(
    calibration: Mapping[str, object],
    *,
    num_blocks: int,
) -> Dict[str, torch.Tensor]:
    """Validate and normalize a per-layer activation calibration artifact.

    The accepted Python representation is intentionally simple:
    ``{"amax": {site: sequence_of_num_blocks_values}}``.
    """

    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    raw_amax = calibration.get("amax")
    if not isinstance(raw_amax, Mapping):
        raise ValueError("Cosmos FP8 activation calibration requires an 'amax' mapping")
    raw_amax_map = cast(Mapping[str, object], raw_amax)

    normalized: Dict[str, torch.Tensor] = {}
    for site in COSMOS_FP8_ACTIVATION_SCALE_SITES:
        if site not in raw_amax_map:
            raise ValueError(
                f"Cosmos FP8 activation calibration is missing site {site!r}"
            )
        tensor = torch.as_tensor(raw_amax_map[site], dtype=torch.float32)
        if tensor.shape != (num_blocks,):
            raise ValueError(
                f"Cosmos FP8 activation calibration site {site!r} must have shape "
                f"({num_blocks},), got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"Cosmos FP8 activation calibration site {site!r} must be finite"
            )
        if (tensor <= 0).any():
            raise ValueError(
                f"Cosmos FP8 activation calibration site {site!r} must be positive"
            )
        normalized[site] = tensor.contiguous()
    return normalized


def cosmos_fp8_activation_scale_tensor(
    calibration: Mapping[str, object],
    *,
    num_blocks: int,
    device: Optional[torch.device | str] = None,
    scale_dtype: torch.dtype = torch.float32,
    eps: float = FP8_SCALE_EPS,
) -> torch.Tensor:
    """Return runtime `[num_blocks, num_sites]` activation scales.

    The C++ streaming path consumes scales, not raw amax values. Columns follow
    ``COSMOS_FP8_ACTIVATION_SCALE_SITES`` exactly.
    """

    normalized = validate_cosmos_fp8_activation_calibration(
        calibration,
        num_blocks=num_blocks,
    )
    stacked_amax = torch.stack(
        [normalized[site] for site in COSMOS_FP8_ACTIVATION_SCALE_SITES], dim=1
    )
    scales = torch.clamp(stacked_amax / FP8_MAX_E4M3, min=eps)
    if device is not None:
        scales = scales.to(device=device)
    return scales.to(dtype=scale_dtype).contiguous()


def fuse_cosmos_self_attn_qkv_fp8_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    drop_split_self_attn_qkv: bool = False,
    include_missing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Add optional fused self-attention QKV FP8 weights.

    Cosmos checkpoints store Q/K/V projections as separate Linear weights.
    The FP8 runtime can avoid three small GEMMs when a caller provides a
    pre-concatenated ``self_attn.qkv_proj.weight`` with per-output scales.
    Split tensors are kept by default so existing callers retain the fallback.
    """

    out: Dict[str, torch.Tensor] = dict(weights)
    for block_idx in range(num_blocks):
        prefix = f"blocks.{block_idx}."
        split_keys = tuple(
            prefix + rel_key for rel_key in _COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS
        )
        split_scale_keys = tuple(scale_key_for_weight(key) for key in split_keys)
        fused_key = prefix + _COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY
        fused_scale_key = scale_key_for_weight(fused_key)

        missing = [key for key in (*split_keys, *split_scale_keys) if key not in out]
        if missing:
            if include_missing:
                continue
            raise KeyError(
                f"missing Cosmos split QKV FP8 tensors for fusion: {missing[0]!r}"
            )

        q_weight, k_weight, v_weight = (out[key] for key in split_keys)
        q_scale, k_scale, v_scale = (out[key] for key in split_scale_keys)
        out[fused_key] = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()
        out[fused_scale_key] = torch.cat(
            (q_scale, k_scale, v_scale), dim=0
        ).contiguous()

        if drop_split_self_attn_qkv:
            for key in (*split_keys, *split_scale_keys):
                out.pop(key, None)

    return out


def quantize_cosmos_fp8_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    device: Optional[torch.device | str] = None,
    include_missing: bool = False,
    fuse_self_attn_qkv: bool = True,
    drop_split_self_attn_qkv: bool = False,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
) -> Dict[str, torch.Tensor]:
    """Return a shallow-copied weights dict with Cosmos block linears in FP8.

    For each quantized weight key ``K``, the output dict contains:

    - ``K``: raw E4M3 bytes as ``torch.uint8`` in the original [out, in] layout
    - ``f"{K}_scale"``: per-output-channel dequantization scale

    Non-quantized tensors are carried through unchanged.
    """

    out: Dict[str, torch.Tensor] = dict(weights)
    target_device = torch.device(device) if device is not None else None
    selected_rel_keys = set(
        resolve_cosmos_fp8_linear_policy(
            linear_policy,
            custom_groups=custom_linear_groups,
        )
    )

    for block_idx in range(num_blocks):
        prefix = f"blocks.{block_idx}."
        for rel_key in selected_rel_keys:
            key = prefix + rel_key
            if key not in weights:
                if include_missing:
                    continue
                raise KeyError(f"missing Cosmos FP8 linear weight {key!r}")
            weight = weights[key]
            if target_device is not None:
                weight = weight.to(target_device)
            q_u8, scale = quantize_fp8_per_out_channel(weight)
            out[key] = q_u8
            out[scale_key_for_weight(key)] = scale

    for key in cosmos_block_fp8_linear_keys(num_blocks):
        if key not in weights:
            if include_missing:
                continue
            raise KeyError(f"missing Cosmos FP8 linear weight {key!r}")
        if key not in out:
            continue

    has_all_self_qkv = all(
        rel_key in selected_rel_keys for rel_key in _COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS
    )
    if fuse_self_attn_qkv and has_all_self_qkv:
        out = fuse_cosmos_self_attn_qkv_fp8_weights(
            out,
            num_blocks=num_blocks,
            drop_split_self_attn_qkv=drop_split_self_attn_qkv,
            include_missing=include_missing,
        )

    elif drop_split_self_attn_qkv:
        raise ValueError(
            "drop_split_self_attn_qkv=True requires a policy that quantizes all self-attention Q/K/V weights"
        )

    return out


def add_cosmos_fp8_prepared_aliases(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    include_missing: bool = False,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
) -> Dict[str, torch.Tensor]:
    """Add explicit prepared FP8 aliases for the streaming runtime.

    The aliases deliberately do not use the BF16 ``.weight_prepared`` suffix:
    BF16 prepared tensors are transposed for the prepared BF16 GEMM path, while
    Cosmos FP8 kernels consume raw E4M3 bytes in the original [out, in] layout.
    """

    out: Dict[str, torch.Tensor] = dict(weights)
    selected_rel_keys = set(
        resolve_cosmos_fp8_linear_policy(
            linear_policy,
            custom_groups=custom_linear_groups,
        )
    )
    qkv_rel_keys = set(_COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS)
    has_all_self_qkv = qkv_rel_keys.issubset(selected_rel_keys)

    for block_idx in range(num_blocks):
        prefix = f"blocks.{block_idx}."
        fused_key = prefix + _COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY
        fused_ready = (
            isinstance(out.get(fused_key), torch.Tensor)
            and out[fused_key].dtype == torch.uint8
            and isinstance(out.get(scale_key_for_weight(fused_key)), torch.Tensor)
        )
        rel_keys: list[str] = [
            rel_key
            for rel_key in _COSMOS_BLOCK_FP8_LINEAR_KEYS
            if rel_key in selected_rel_keys
        ]
        if has_all_self_qkv and fused_key in out:
            rel_keys.append(_COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY)

        for rel_key in rel_keys:
            key = prefix + rel_key
            if rel_key in qkv_rel_keys and key not in out and fused_ready:
                continue
            scale_key = scale_key_for_weight(key)
            if key not in out or scale_key not in out:
                if include_missing:
                    continue
                missing = key if key not in out else scale_key
                raise KeyError(
                    f"missing Cosmos FP8 tensor for prepared alias: {missing!r}"
                )

            weight = out[key]
            scale = out[scale_key]
            if not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8:
                raise ValueError(
                    f"{key} must be torch.uint8 raw E4M3 bytes for FP8 prepared aliases"
                )
            if not isinstance(scale, torch.Tensor):
                raise ValueError(
                    f"{scale_key} must be a tensor for FP8 prepared aliases"
                )
            if scale.dim() != 1 or scale.numel() != weight.shape[0]:
                raise ValueError(
                    f"{scale_key} must have shape [{weight.shape[0]}], got {tuple(scale.shape)}"
                )

            out[fp8_prepared_weight_key(key)] = weight.contiguous()
            out[fp8_prepared_scale_key(key)] = scale.contiguous()

    return out


def prepare_cosmos_quantized_streaming_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    device: Optional[torch.device | str] = None,
    include_missing: bool = False,
    fuse_self_attn_qkv: bool = True,
    drop_split_self_attn_qkv: Optional[bool] = None,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
    add_fp8_prepared: bool = True,
    validate: bool = True,
) -> Dict[str, torch.Tensor]:
    """Prepare Cosmos streaming weights and quantized FP8 artifacts together.

    This composes the BF16 streaming preparation from ``common.cosmos_weights``
    with the FP8 quantization contract in this module. The result contains the
    existing BF16 ``.weight_prepared`` tensors plus opt-in FP8 prepared aliases
    that can be selected by ``cosmos_quantized_prepared=True`` at runtime.

    By default, the prepared artifact drops split self-attention Q/K/V FP8
    tensors whenever fused self-QKV is available. The streaming runtime consumes
    the fused tensor directly, so keeping both layouts only increases artifact
    size. Pass ``drop_split_self_attn_qkv=False`` to retain split FP8 Q/K/V
    fallback tensors.
    """

    selected_rel_keys = set(
        resolve_cosmos_fp8_linear_policy(
            linear_policy,
            custom_groups=custom_linear_groups,
        )
    )
    has_all_self_qkv = set(_COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS).issubset(
        selected_rel_keys
    )
    if drop_split_self_attn_qkv is None:
        drop_split_self_attn_qkv = fuse_self_attn_qkv and has_all_self_qkv

    prepared_bf16 = prepare_cosmos_streaming_weights(weights)
    quantized = quantize_cosmos_fp8_weights(
        prepared_bf16,
        num_blocks=num_blocks,
        device=device,
        include_missing=include_missing,
        fuse_self_attn_qkv=fuse_self_attn_qkv,
        drop_split_self_attn_qkv=drop_split_self_attn_qkv,
        linear_policy=linear_policy,
        custom_linear_groups=custom_linear_groups,
    )
    if add_fp8_prepared:
        quantized = add_cosmos_fp8_prepared_aliases(
            quantized,
            num_blocks=num_blocks,
            include_missing=include_missing,
            linear_policy=linear_policy,
            custom_linear_groups=custom_linear_groups,
        )
    if device is not None:
        target_device = torch.device(device)
        quantized = {
            key: value.to(device=target_device).contiguous()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in quantized.items()
        }
    else:
        quantized = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in quantized.items()
        }
    if validate:
        assert_cosmos_fp8_ready(
            quantized,
            num_blocks=num_blocks,
            linear_policy=linear_policy,
            custom_linear_groups=custom_linear_groups,
        )
        if add_fp8_prepared:
            assert_cosmos_fp8_prepared_ready(
                quantized,
                num_blocks=num_blocks,
                linear_policy=linear_policy,
                custom_linear_groups=custom_linear_groups,
            )
    return quantized


def missing_cosmos_fp8_keys(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
) -> Tuple[str, ...]:
    missing = []
    selected_rel_keys = set(
        resolve_cosmos_fp8_linear_policy(
            linear_policy,
            custom_groups=custom_linear_groups,
        )
    )
    split_qkv_rel_keys = set(_COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS)
    has_all_self_qkv = split_qkv_rel_keys.issubset(selected_rel_keys)
    for block_idx in range(num_blocks):
        prefix = f"blocks.{block_idx}."
        fused_key = prefix + _COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY
        fused_scale_key = scale_key_for_weight(fused_key)
        fused_weight = weights.get(fused_key)
        fused_scale = weights.get(fused_scale_key)
        fused_qkv_ready = (
            isinstance(fused_weight, torch.Tensor)
            and fused_weight.dtype == torch.uint8
            and isinstance(fused_scale, torch.Tensor)
        )
        if has_all_self_qkv and fused_key in weights and not fused_qkv_ready:
            if (
                not isinstance(fused_weight, torch.Tensor)
                or fused_weight.dtype != torch.uint8
            ):
                missing.append(fused_key)
            if not isinstance(fused_scale, torch.Tensor):
                missing.append(fused_scale_key)
        for rel_key in _COSMOS_BLOCK_FP8_LINEAR_KEYS:
            if rel_key not in selected_rel_keys:
                continue
            if rel_key in split_qkv_rel_keys and fused_qkv_ready:
                continue
            key = prefix + rel_key
            scale_key = scale_key_for_weight(key)
            if key not in weights:
                missing.append(key)
            elif weights[key].dtype != torch.uint8:
                missing.append(key)
            if scale_key not in weights:
                missing.append(scale_key)
    return tuple(missing)


def missing_cosmos_fp8_prepared_keys(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
) -> Tuple[str, ...]:
    missing = list(
        missing_cosmos_fp8_keys(
            weights,
            num_blocks=num_blocks,
            linear_policy=linear_policy,
            custom_linear_groups=custom_linear_groups,
        )
    )
    selected_rel_keys = set(
        resolve_cosmos_fp8_linear_policy(
            linear_policy,
            custom_groups=custom_linear_groups,
        )
    )
    qkv_rel_keys = set(_COSMOS_BLOCK_FP8_SELF_ATTN_QKV_KEYS)
    has_all_self_qkv = qkv_rel_keys.issubset(selected_rel_keys)

    for block_idx in range(num_blocks):
        prefix = f"blocks.{block_idx}."
        fused_key = prefix + _COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY
        fused_prepared_weight_key = fp8_prepared_weight_key(fused_key)
        fused_prepared_scale_key = fp8_prepared_scale_key(fused_key)
        fused_prepared_ready = (
            isinstance(weights.get(fused_key), torch.Tensor)
            and weights[fused_key].dtype == torch.uint8
            and isinstance(weights.get(scale_key_for_weight(fused_key)), torch.Tensor)
            and isinstance(weights.get(fused_prepared_weight_key), torch.Tensor)
            and weights[fused_prepared_weight_key].dtype == torch.uint8
            and isinstance(weights.get(fused_prepared_scale_key), torch.Tensor)
        )

        rel_keys: list[str] = [
            rel_key
            for rel_key in _COSMOS_BLOCK_FP8_LINEAR_KEYS
            if rel_key in selected_rel_keys
        ]
        if has_all_self_qkv and fused_key in weights:
            rel_keys.append(_COSMOS_BLOCK_FP8_FUSED_SELF_ATTN_QKV_KEY)

        for rel_key in rel_keys:
            key = prefix + rel_key
            if rel_key in qkv_rel_keys and key not in weights and fused_prepared_ready:
                continue
            weight = weights.get(key)
            if not isinstance(weight, torch.Tensor) or weight.dtype != torch.uint8:
                continue

            prepared_weight_key = fp8_prepared_weight_key(key)
            prepared_scale_key = fp8_prepared_scale_key(key)
            prepared_weight = weights.get(prepared_weight_key)
            prepared_scale = weights.get(prepared_scale_key)
            if (
                not isinstance(prepared_weight, torch.Tensor)
                or prepared_weight.dtype != torch.uint8
            ):
                missing.append(prepared_weight_key)
            elif (
                prepared_weight.shape != weight.shape
                or not prepared_weight.is_contiguous()
            ):
                missing.append(prepared_weight_key)
            if not isinstance(prepared_scale, torch.Tensor):
                missing.append(prepared_scale_key)
            else:
                scale_key = scale_key_for_weight(key)
                scale = weights.get(scale_key)
                if isinstance(scale, torch.Tensor) and (
                    prepared_scale.shape != scale.shape
                    or not prepared_scale.is_contiguous()
                ):
                    missing.append(prepared_scale_key)

    return tuple(dict.fromkeys(missing))


def assert_cosmos_fp8_ready(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
) -> None:
    missing = missing_cosmos_fp8_keys(
        weights,
        num_blocks=num_blocks,
        linear_policy=linear_policy,
        custom_linear_groups=custom_linear_groups,
    )
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... +{len(missing) - 8} more"
        raise ValueError(f"Cosmos FP8 weights are incomplete: {preview}{suffix}")


def assert_cosmos_fp8_prepared_ready(
    weights: Mapping[str, torch.Tensor],
    *,
    num_blocks: int,
    linear_policy: str = "all",
    custom_linear_groups: Optional[Sequence[str] | str] = None,
) -> None:
    missing = missing_cosmos_fp8_prepared_keys(
        weights,
        num_blocks=num_blocks,
        linear_policy=linear_policy,
        custom_linear_groups=custom_linear_groups,
    )
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... +{len(missing) - 8} more"
        raise ValueError(
            f"Cosmos FP8 prepared weights are incomplete: {preview}{suffix}"
        )


def move_cosmos_fp8_weights_(
    weights: MutableMapping[str, torch.Tensor],
    *,
    device: torch.device | str,
    keys: Optional[Iterable[str]] = None,
) -> None:
    selected = tuple(keys) if keys is not None else tuple(weights.keys())
    target = torch.device(device)
    for key in selected:
        value = weights.get(key)
        if isinstance(value, torch.Tensor):
            weights[key] = value.to(device=target).contiguous()

# ============================================================================
# Cosmos streaming weight prep (folded from omnidreams_cosmos_weights.py)
# ============================================================================
# SPDX-License-Identifier: Apache-2.0

"""Cosmos streaming weight preparation helpers."""


from collections.abc import Mapping

import torch
from torch import Tensor

_PREPARED_SUFFIXES = (
    "self_attn.qkv_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
)


def prepare_cosmos_streaming_weights(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Augment a Cosmos state dict with per-block fused self-attention QKV weights.

    Adds ``blocks.{i}.self_attn.qkv_proj.weight`` as
    ``cat([q_proj, k_proj, v_proj], dim=0)`` while preserving the original
    separate Q/K/V keys used by other paths.
    """
    weights = dict(state_dict)
    q_suffix = "self_attn.q_proj.weight"
    for q_key, q_weight in list(weights.items()):
        if not q_key.endswith(q_suffix):
            continue

        prefix = q_key[: -len(q_suffix)]
        fused_key = prefix + "self_attn.qkv_proj.weight"
        if fused_key in weights:
            weights[fused_key] = weights[fused_key].contiguous()
            continue

        k_key = prefix + "self_attn.k_proj.weight"
        v_key = prefix + "self_attn.v_proj.weight"
        if k_key not in weights or v_key not in weights:
            raise KeyError(f"Missing K/V weights for fused QKV key {fused_key!r}")

        weights[fused_key] = torch.cat(
            [q_weight, weights[k_key], weights[v_key]], dim=0
        ).contiguous()

    for key, weight in list(weights.items()):
        if key.endswith(".weight_prepared"):
            weights[key] = weight.contiguous()
            continue
        if key.endswith(_PREPARED_SUFFIXES):
            weights[f"{key}_prepared"] = weight.t().contiguous()

    return {k: v.contiguous() for k, v in weights.items()}
