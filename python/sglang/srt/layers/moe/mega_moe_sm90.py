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
"""SM90 FP8/FP4 Mega-MoE forward paths and expert-weight prep."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.async_probe import maybe_sync_eagle_cuda_debug

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


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
    native_module = getattr(deep_gemm, "_C", None)
    return (
        hasattr(deep_gemm, "fp8_fp4_mega_moe")
        and hasattr(deep_gemm, "mega_moe_pre_dispatch_sm90")
        and native_module is not None
        and hasattr(native_module, "fp8_fp4_mega_moe_sm90")
        and getattr(experts, "_mega_moe_sm90_fp4_weights", False)
    )


def run_sm90_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf: SymmBuffer,
    num_tokens: int,
    forward_batch=None,
) -> torch.Tensor:
    import deep_gemm

    use_fp4_weights = getattr(moe.experts, "_mega_moe_sm90_fp4_weights", False)

    # Both SM90 kernels use FP8 activations with per-128 scales. Enabling FP4
    # activations changes the symmetric-buffer layout and is incompatible with
    # the H20 kernels, including the FP4-weight kernel.
    if get_exec().moe.enable_w4a4_mxfp4_megamoe:
        raise RuntimeError(
            "--enable-w4a4-mxfp4-megamoe is incompatible with SM90 "
            "MegaMoE. H20 uses FP8 activations for both FP8 and FP4 weights; "
            "disable the flag or use an SM100 path."
        )

    if moe.experts.should_fuse_routed_scaling_factor_in_topk:
        routed_scaling_factor = 1.0
    else:
        routed_scaling_factor = float(moe.routed_scaling_factor)

    deep_gemm.mega_moe_pre_dispatch_sm90(
        hidden_states,
        topk_ids,
        topk_weights,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=128,
        routed_scaling_factor=routed_scaling_factor,
    )
    maybe_sync_eagle_cuda_debug(forward_batch, "after_megamoe_pre_dispatch")

    y = torch.empty(
        (max(num_tokens, 1), moe.config.hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if use_fp4_weights:
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
    maybe_sync_eagle_cuda_debug(forward_batch, "after_megamoe_routed")
    y = y[:num_tokens]

    return y


def _interleave_l1_weight_only(weight: torch.Tensor, gran: int = 8) -> torch.Tensor:
    num_groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(num_groups, half // gran, gran, *rest)
    up = weight[:, half:].reshape(num_groups, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)


def _transform_weights_for_mega_moe_sm90_fp4_compat(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Compatibility port of the DeepGEMM SM90 FP4 weight transform."""

    def _interleave_one(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
        num_groups, n, *rest = t.shape
        half = n // 2
        gate = t[:, :half].reshape(num_groups, half // gran, gran, *rest)
        up = t[:, half:].reshape(num_groups, half // gran, gran, *rest)
        return torch.empty_like(t).copy_(
            torch.stack([gate, up], dim=2).reshape(num_groups, n, *rest)
        )

    def _pack_fp32_sf_to_ue8m0_kmajor(sf_fp32: torch.Tensor) -> torch.Tensor:
        assert sf_fp32.dtype == torch.float32, f"unexpected SF dtype {sf_fp32.dtype}"
        num_experts, n, k_groups = sf_fp32.shape
        assert k_groups % 4 == 0, f"K/32={k_groups} must be a multiple of 4"
        bits = sf_fp32.view(torch.int32)
        ue8m0 = (bits.bitwise_right_shift(23).bitwise_and(0xFF)).to(torch.uint8)
        ue8m0 = ue8m0.contiguous().view(num_experts, n, k_groups // 4, 4)
        return (
            ue8m0.view(torch.int32).reshape(num_experts, n, k_groups // 4).contiguous()
        )

    def _as_packed_fp4_storage(fp4: torch.Tensor) -> torch.Tensor:
        assert fp4.dtype in (
            torch.int8,
            torch.uint8,
        ), f"unexpected FP4 dtype {fp4.dtype}"
        return fp4.contiguous().view(torch.int8)

    l1_fp4, l1_sf_fp32 = l1_weights
    l2_fp4, l2_sf_fp32 = l2_weights
    l1_fp4 = _interleave_one(_as_packed_fp4_storage(l1_fp4))
    l2_fp4 = _as_packed_fp4_storage(l2_fp4)
    l1_sf_fp32 = _interleave_one(l1_sf_fp32)
    return (
        (l1_fp4, _pack_fp32_sf_to_ue8m0_kmajor(l1_sf_fp32)),
        (l2_fp4, _pack_fp32_sf_to_ue8m0_kmajor(l2_sf_fp32)),
    )


def _resolve_sm90_fp4_weight_transform(deep_gemm):
    missing_kernels = [
        name
        for name in ("fp8_fp4_mega_moe", "mega_moe_pre_dispatch_sm90")
        if not hasattr(deep_gemm, name)
    ]
    if missing_kernels:
        raise RuntimeError(
            "DeepGEMM does not provide the SM90 FP4 MegaMoE runtime; missing "
            + ", ".join(missing_kernels)
        )
    native_module = getattr(deep_gemm, "_C", None)
    if native_module is None or not hasattr(native_module, "fp8_fp4_mega_moe_sm90"):
        raise RuntimeError(
            "DeepGEMM does not provide the SM90 FP4 MegaMoE native kernel; "
            "missing _C.fp8_fp4_mega_moe_sm90"
        )

    transform = getattr(deep_gemm, "transform_weights_for_mega_moe_sm90_fp4", None)
    if transform is not None:
        return transform

    logger.warning(
        "DeepGEMM provides the SM90 FP4 MegaMoE kernels but not "
        "transform_weights_for_mega_moe_sm90_fp4; using SGLang's "
        "compatibility transform."
    )
    return _transform_weights_for_mega_moe_sm90_fp4_compat


def build_sm90_mega_moe_experts_weights(experts) -> None:
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    assert w13.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    num_groups, n1, k1 = w13.shape
    _, n2, k2 = w2.shape
    scale_group_mn, scale_group_k = 128, 128

    assert k1 % scale_group_k == 0 and k2 % scale_group_k == 0, (
        f"invalid SM90 mega-moe K/group_size: k1={k1}, k2={k2}, "
        f"group_k={scale_group_k}"
    )
    expected_n_groups_1 = (n1 + scale_group_mn - 1) // scale_group_mn
    expected_n_groups_2 = (n2 + scale_group_mn - 1) // scale_group_mn
    expected_k_groups_1 = k1 // scale_group_k
    expected_k_groups_2 = k2 // scale_group_k
    assert w13_sf_fp32.shape[1] == expected_n_groups_1, (
        f"w13 scale N groups mismatch: got {w13_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_1} (n1={n1}, group_mn={scale_group_mn})"
    )
    assert w2_sf_fp32.shape[1] == expected_n_groups_2, (
        f"w2 scale N groups mismatch: got {w2_sf_fp32.shape[1]}, "
        f"expected {expected_n_groups_2} (n2={n2}, group_mn={scale_group_mn})"
    )
    assert w13_sf_fp32.shape[2] == expected_k_groups_1, (
        f"w13 scale K groups mismatch: got {w13_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_1} (k1={k1}, group_k={scale_group_k})"
    )
    assert w2_sf_fp32.shape[2] == expected_k_groups_2, (
        f"w2 scale K groups mismatch: got {w2_sf_fp32.shape[2]}, "
        f"expected {expected_k_groups_2} (k2={k2}, group_k={scale_group_k})"
    )

    w13_interleaved = _interleave_l1_weight_only(w13)
    experts.w13_weight.data = w13_interleaved
    experts.mega_l1_weights = (
        experts.w13_weight.data,
        experts.w13_weight_scale_inv.data,
    )
    experts.mega_l2_weights = (
        experts.w2_weight.data,
        experts.w2_weight_scale_inv.data,
    )

    experts._mega_moe_sm90_fp8_weights = True
    experts._mega_moe_weights_built = True


def build_sm90_fp4_mega_moe_experts_weights(experts) -> None:
    """Transform packed E2M1 weights and fp32 per-32 UE8M0 scales for H20."""
    if getattr(experts, "_mega_moe_weights_built", False):
        return

    import deep_gemm

    transform = _resolve_sm90_fp4_weight_transform(deep_gemm)
    scale_suffix = (
        "weight_scale_inv"
        if hasattr(experts, "w13_weight_scale_inv")
        else "weight_scale"
    )
    w13_scale = getattr(experts, f"w13_{scale_suffix}")
    w2_scale = getattr(experts, f"w2_{scale_suffix}")
    w13 = experts.w13_weight.data
    w2 = experts.w2_weight.data

    assert w13.dtype in (torch.int8, torch.uint8)
    assert w2.dtype in (torch.int8, torch.uint8)
    assert w13_scale.dtype == torch.float32
    assert w2_scale.dtype == torch.float32

    l1_pair, l2_pair = transform(
        (w13, w13_scale.data),
        (w2, w2_scale.data),
    )

    # The SM90 FP4 layout has no compatible grouped-GEMM fallback. Repoint the
    # checkpoint parameters to avoid retaining both expert layouts per layer.
    experts.w13_weight.data = l1_pair[0]
    experts.w2_weight.data = l2_pair[0]
    w13_scale.data = l1_pair[1]
    w2_scale.data = l2_pair[1]
    w13_scale.format_ue8m0 = True
    w2_scale.format_ue8m0 = True
    experts.mega_l1_weights = (experts.w13_weight.data, w13_scale.data)
    experts.mega_l2_weights = (experts.w2_weight.data, w2_scale.data)
    experts._mega_moe_sm90_fp4_weights = True
    experts._mega_moe_weights_built = True
