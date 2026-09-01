# Copyright 2023-2024 SGLang Team
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
"""SM90 FP8/FP4 Mega-MoE forward paths and expert-weight preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.utils import _device_sm

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a recently added SGLang env option across image revisions."""
    option = getattr(envs, name, None)
    return default if option is None else bool(option.get())


def is_sm90_fp8_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return (
        hasattr(deep_gemm, "fp8_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and getattr(experts, "_mega_moe_sm90_fp8_weights", False)
    )


def is_sm90_fp4_mega_moe_available(experts) -> bool:
    if _device_sm != 90:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    extension = getattr(deep_gemm, "_C", None)
    return (
        callable(getattr(deep_gemm, "fp8_fp4_mega_moe", None))
        and callable(getattr(extension, "fp8_fp4_mega_moe_sm90", None))
        and callable(
            getattr(deep_gemm, "transform_weights_for_mega_moe_sm90_fp4", None)
        )
        and callable(getattr(deep_gemm, "get_symm_buffer_for_mega_moe", None))
        and getattr(experts, "_mega_moe_sm90_fp4_weights", False)
    )


def run_sm90_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    use_fp4_weights = getattr(moe.experts, "_mega_moe_sm90_fp4_weights", False)

    # Both SM90 kernels use FP8 activations with per-128 scales. Enabling FP4
    # activations changes the symmetric-buffer layout and would silently feed
    # incompatible scales to either SM90 kernel.
    if _env_bool("SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS"):
        raise RuntimeError(
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is incompatible with "
            "SM90 MegaMoE. H20 uses FP8 activations for both FP8 and FP4 "
            "weights; disable the flag or use an SM100 path."
        )

    if moe.experts.should_fuse_routed_scaling_factor_in_topk:
        routed_scaling_factor = 1.0
    else:
        routed_scaling_factor = float(moe.routed_scaling_factor)

    mega_moe_pre_dispatch(
        hidden_states,
        topk_ids,
        topk_weights,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        quant_group_size=128,
    )

    y = torch.empty(
        (max(num_tokens, 1), moe.config.hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if use_fp4_weights:
        extension = getattr(deep_gemm, "_C", None)
        if not callable(getattr(extension, "fp8_fp4_mega_moe_sm90", None)):
            raise RuntimeError(
                "DeepGEMM build lacks the Hopper FP8xFP4 MegaMoE kernel "
                "_C.fp8_fp4_mega_moe_sm90"
            )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=getattr(moe.config, "swiglu_limit", None),
            fast_math=True,
        )
    else:
        deep_gemm.fp8_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=getattr(moe.config, "swiglu_limit", None),
            fast_math=True,
        )
    if routed_scaling_factor != 1.0:
        y.mul_(routed_scaling_factor)
    y = y[:num_tokens]

    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)


def _validate_sm90_fp4_input_pair(
    name: str, pair: object, *, interleaved: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError(f"{name} must be a (packed_weight, fp32_scale) tuple")
    weight, scale = pair
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} entries must both be torch.Tensor objects")
    if weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(
            f"{name} packed weight must use int8/uint8 storage, got {weight.dtype}"
        )
    if scale.dtype != torch.float32:
        raise TypeError(f"{name} scale must be float32, got {scale.dtype}")
    if weight.ndim != 3 or scale.ndim != 3:
        raise ValueError(
            f"{name} tensors must be rank 3, got weight={weight.ndim}, "
            f"scale={scale.ndim}"
        )
    if weight.device != scale.device:
        raise ValueError(
            f"{name} weight and scale must share a device, got "
            f"{weight.device}/{scale.device}"
        )
    expected_scale_shape = (*weight.shape[:2], weight.shape[2] // 16)
    if weight.shape[2] % 16 != 0 or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"{name} requires packed [E,N,K/2] weights and [E,N,K/32] "
            f"scales; got {tuple(weight.shape)} and {tuple(scale.shape)}"
        )
    # Four per-32 UE8M0 scale bytes are packed into one int32 word.
    if scale.shape[2] % 4 != 0:
        raise ValueError(
            f"{name} scale K/32 dimension must be divisible by 4 for the "
            f"SM90 packed layout, got {scale.shape[2]}"
        )
    if interleaved and weight.shape[1] % 16 != 0:
        raise ValueError(
            f"{name} gate/up N dimension must be divisible by 16 for "
            f"8-row interleaving, got {weight.shape[1]}"
        )
    # A decoded UE8M0 scale is an exponent in an fp32 container. Any mantissa
    # bits would be silently discarded by DeepGEMM's pack helper.
    if torch.any(scale.view(torch.int32).bitwise_and((1 << 23) - 1) != 0).item():
        raise ValueError(f"{name} scale contains non-zero fp32 mantissa bits")
    return weight, scale


def _validate_sm90_fp4_transform_output(
    transformed: object,
    expected_shapes: tuple[
        tuple[torch.Size, torch.Size], tuple[torch.Size, torch.Size]
    ],
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise TypeError(
            "SM90 FP4 weight transform must return an (l1_pair, l2_pair) tuple"
        )
    validated = []
    for name, pair, (weight_shape, scale_shape) in zip(
        ("l1 output", "l2 output"), transformed, expected_shapes
    ):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{name} must be a (packed_weight, packed_scale) tuple")
        weight, scale = pair
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name} entries must both be torch.Tensor objects")
        if weight.dtype != torch.int8:
            raise TypeError(f"{name} weight must be int8, got {weight.dtype}")
        if scale.dtype != torch.int32:
            raise TypeError(f"{name} packed scale must be int32, got {scale.dtype}")
        if tuple(weight.shape) != tuple(weight_shape):
            raise ValueError(
                f"{name} weight shape mismatch: got {tuple(weight.shape)}, "
                f"expected {tuple(weight_shape)}"
            )
        if tuple(scale.shape) != tuple(scale_shape):
            raise ValueError(
                f"{name} scale shape mismatch: got {tuple(scale.shape)}, "
                f"expected {tuple(scale_shape)}"
            )
        if weight.device != scale.device:
            raise ValueError(
                f"{name} weight and scale must share a device, got "
                f"{weight.device}/{scale.device}"
            )
        if not weight.is_contiguous() or not scale.is_contiguous():
            raise ValueError(f"{name} tensors must be contiguous")
        validated.append((weight, scale))
    return validated[0], validated[1]


def _resolve_sm90_fp4_weight_transform(deep_gemm) -> Callable:
    extension = getattr(deep_gemm, "_C", None)
    if not callable(getattr(deep_gemm, "fp8_fp4_mega_moe", None)) or not callable(
        getattr(extension, "fp8_fp4_mega_moe_sm90", None)
    ):
        raise RuntimeError(
            "DeepGEMM does not provide the Hopper FP8xFP4 MegaMoE runtime; "
            "required symbol: _C.fp8_fp4_mega_moe_sm90"
        )
    transform = getattr(deep_gemm, "transform_weights_for_mega_moe_sm90_fp4", None)
    if not callable(transform):
        raise RuntimeError(
            "DeepGEMM build lacks transform_weights_for_mega_moe_sm90_fp4; "
            "install a build containing the Hopper implementation from d6b9815"
        )
    return transform


def _transpose_scale_for_utccp(scale: torch.Tensor) -> torch.Tensor:
    """Transpose DeepGEMM's packed scale rows for the SM90 UTCCP path."""
    if scale.dtype != torch.int32:
        raise TypeError(f"SM90 MegaMoE scales must be int32, got {scale.dtype}")
    num_groups, mn, packed_k = scale.shape
    if mn % 128 != 0:
        raise ValueError(f"SM90 MegaMoE scale M dimension must be 128-aligned, got {mn}")
    return (
        scale.reshape(num_groups, -1, 4, 32, packed_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_k)
        .contiguous()
    )


def build_sm90_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError(f"SM90 FP8 MegaMoE weights must be float8_e4m3fn, got {w13.dtype}/{w2.dtype}")

    num_groups, n1, k1 = w13.shape
    _, n2, k2 = w2.shape
    scale_group_mn, scale_group_k = 128, 128

    if k1 % scale_group_k != 0 or k2 % scale_group_k != 0:
        raise ValueError(f"invalid SM90 mega-moe K/group_size: k1={k1}, k2={k2}, group_k={scale_group_k}")
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    expected = ((expected_n_groups_1, expected_k_groups_1), (expected_n_groups_2, expected_k_groups_2))
    actual = ((w13_sf_fp32.shape[1], w13_sf_fp32.shape[2]), (w2_sf_fp32.shape[1], w2_sf_fp32.shape[2]))
    if actual != expected:
        raise ValueError(f"SM90 FP8 scale groups mismatch: got {actual}, expected {expected}")

    if _env_bool("SGLANG_OPT_FIX_MEGA_MOE_MEMORY"):
        import deep_gemm

        w13_sf = deep_gemm.transform_sf_into_required_layout(
            w13_sf_fp32, mn=n1, k=k1, recipe=(1, 32), num_groups=num_groups, disable_ue8m0_cast=False
        )
        w2_sf = deep_gemm.transform_sf_into_required_layout(
            w2_sf_fp32, mn=n2, k=k2, recipe=(1, 32), num_groups=num_groups, disable_ue8m0_cast=False
        )
        w13_interleaved = _interleave_l1_weight_only(w13)
        w13_sf_interleaved = _interleave_l1_weight_only(w13_sf)
        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            _transpose_scale_for_utccp(w13_sf_interleaved),
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            _transpose_scale_for_utccp(w2_sf),
        )
    else:
        import deep_gemm

        w13_sf = deep_gemm.transform_sf_into_required_layout(
            w13_sf_fp32,
            mn=n1,
            k=k1,
            recipe=(128, 128),
            num_groups=num_groups,
            disable_ue8m0_cast=True,
        )
        w2_sf = deep_gemm.transform_sf_into_required_layout(
            w2_sf_fp32,
            mn=n2,
            k=k2,
            recipe=(128, 128),
            num_groups=num_groups,
            disable_ue8m0_cast=True,
        )
        l1_pair, l2_pair = deep_gemm.transform_weights_for_mega_moe_sm90(
            (w13, w13_sf), (w2, w2_sf)
        )
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp8_weights = True
    experts._mega_moe_weights_built = True


def build_sm90_fp4_mega_moe_experts_weights(experts) -> None:
    """Transform packed E2M1 weights and raw per-32 E8M0 scales for H20."""
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    import deep_gemm

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    _validate_sm90_fp4_input_pair(
        "w13", (w13, w13_sf_fp32), interleaved=True
    )
    _validate_sm90_fp4_input_pair("w2", (w2, w2_sf_fp32), interleaved=False)

    transform = _resolve_sm90_fp4_weight_transform(deep_gemm)
    transformed = transform((w13, w13_sf_fp32), (w2, w2_sf_fp32))
    l1_pair, l2_pair = _validate_sm90_fp4_transform_output(
        transformed,
        (
            (w13.shape, torch.Size((*w13_sf_fp32.shape[:2], w13_sf_fp32.shape[2] // 4))),
            (w2.shape, torch.Size((*w2_sf_fp32.shape[:2], w2_sf_fp32.shape[2] // 4))),
        ),
    )

    if _env_bool("SGLANG_OPT_FIX_MEGA_MOE_MEMORY"):
        # The FP4 MegaMoE layout cannot serve as a generic grouped-GEMM
        # fallback. Replace the checkpoint-layout parameters so KV sizing can
        # reclaim them, then make the request path fail closed on a cap miss.
        experts.w13_weight.data = l1_pair[0]
        experts.w2_weight.data = l2_pair[0]
        experts.w13_weight_scale_inv.data = l1_pair[1]
        experts.w2_weight_scale_inv.data = l2_pair[1]
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True
        experts.mega_l1_weights = (
            experts.w13_weight.data,
            experts.w13_weight_scale_inv.data,
        )
        experts.mega_l2_weights = (
            experts.w2_weight.data,
            experts.w2_weight_scale_inv.data,
        )
    else:
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_sm90_fp4_weights = True
    experts._mega_moe_weights_built = True
