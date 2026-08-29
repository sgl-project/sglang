"""Ascend DeviceOperator: runtime-visible generation contracts for NPU.

Following RFC #35709, standalone kernel / provider differences belong to
sgl-kernel-npu, while differences that are visible to the SGLang runtime
(dtype / layout / metadata / operator contract) are adapted here. Feature
code calls semantic methods instead of branching on the device generation.

This first contract is the MXFP per-token scale runtime layout used by the
MoE routing path: ``npu_moe_init_routing_v2(quant_mode=3)`` emits a flat 2D
e8m0 block scale, while the A5 grouped matmul consumes the pair-split 3D
representation.

The factory is intentionally lazy: importing this module never probes the
device, the first ``get_device_operator()`` call does.
"""

import functools
from enum import Enum
from typing import Optional

import torch


class AscendDeviceGeneration(str, Enum):
    """Ascend device generations distinguished by runtime-visible contracts."""

    LEGACY = "legacy"
    A5 = "a5"


@functools.lru_cache(maxsize=None)
def get_npu_device_generation(
    device_id: Optional[int] = None,
) -> AscendDeviceGeneration:
    """Detect the Ascend device generation (cached per device).

    Returns LEGACY on non-NPU environments. When ``device_id`` is omitted the
    current device is used (never a hardcoded 0, so multi-rank processes do
    not probe the wrong device).
    """
    from sglang.srt.utils import is_npu

    if not is_npu():
        return AscendDeviceGeneration.LEGACY

    if device_id is None:
        device_id = torch.npu.current_device()

    device_name = torch.npu.get_device_name(device_id)

    if device_name.startswith("Ascend950"):
        return AscendDeviceGeneration.A5

    return AscendDeviceGeneration.LEGACY


class BaseDeviceOperator:
    """Runtime contract for existing (pre-A5) Ascend devices."""

    @staticmethod
    def normalize_mxfp_scale_layout(
        scale: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Legacy MXFP scale runtime representation: the producer's native layout."""
        return scale


class A5DeviceOperator(BaseDeviceOperator):
    """Runtime contract for Ascend A5 (Ascend950) devices."""

    @staticmethod
    def normalize_mxfp_scale_layout(
        scale: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Convert a flat 2D e8m0 block scale ``[N, M]`` into pair-split ``[N, M//2, 2]``.

        ``npu_moe_init_routing_v2(quant_mode=3)`` emits the scale flat while the
        grouped matmul wants the pair-split view. Already-3D scales (what
        ``npu_dynamic_mx_quant`` returns) and ``None`` pass through untouched.
        """
        if scale is None or scale.ndim != 2:
            return scale

        return scale.reshape(
            scale.shape[0],
            scale.shape[1] // 2,
            2,
        )


@functools.lru_cache(maxsize=None)
def get_device_operator(
    device_id: Optional[int] = None,
) -> BaseDeviceOperator:
    """Return the DeviceOperator for the given (or current) device."""
    generation = get_npu_device_generation(device_id)

    if generation == AscendDeviceGeneration.A5:
        return A5DeviceOperator()

    return BaseDeviceOperator()
