# SPDX-License-Identifier: Apache-2.0
"""SiLU followed by clamping, with a single BF16 output conversion."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend

if TYPE_CHECKING:
    import torch


def _prepare_output(input: torch.Tensor, out: Optional[torch.Tensor]) -> torch.Tensor:
    import torch

    if (
        input.layout != torch.strided
        or input.device.type not in ("cpu", "cuda")
        or input.dtype != torch.bfloat16
        or input.ndim != 2
        or input.shape[1] == 0
        or input.shape[1] % 2
        or input.stride(1) != 1
        or (input.shape[0] > 1 and input.stride(0) < input.shape[1])
        or input.requires_grad
    ):
        raise ValueError(
            "input must be inference-only BF16 [tokens, 2*width] on CPU/CUDA, "
            "with positive width, contiguous columns and nonoverlapping rows"
        )
    shape = (input.shape[0], input.shape[1] // 2)
    if out is None:
        return torch.empty(shape, dtype=input.dtype, device=input.device)
    if (
        out.layout != torch.strided
        or out.shape != shape
        or out.dtype != input.dtype
        or out.device != input.device
        or out.stride(1) != 1
        or (out.shape[0] > 1 and out.stride(0) < out.shape[1])
        or out.requires_grad
    ):
        raise ValueError(
            "out must match input device/dtype with shape [tokens, width], "
            "contiguous columns, nonoverlapping rows and no gradients"
        )
    if input.numel():
        input_storage = input.untyped_storage()
        output_storage = out.untyped_storage()
        input_start = input_storage.data_ptr()
        output_start = output_storage.data_ptr()
        if max(input_start, output_start) < min(
            input_start + input_storage.nbytes(),
            output_start + output_storage.nbytes(),
        ):
            raise ValueError("nonempty input and out must not share storage memory")
    return out


class SiluClampAndMulOp(BaseFusedOp):
    """``min(sigmoid(g)*g, L) * clamp(up, -L, L)`` with FP32 intermediates.

    Input is BF16 ``[tokens, 2*width]``; output is BF16 ``[tokens, width]``.
    Both require contiguous columns and nonoverlapping rows; row padding and
    storage offsets are supported. Nonempty input/output must use distinct
    storage address ranges, including otherwise disjoint views of one allocation
    and overlapping external-memory wrappers with different base pointers. Only
    the logical output is written. Empty token rows return without a kernel launch.
    This is an inference-only operator; gradients are unsupported.

    ``limit`` is fixed at construction and specialized by Triton. It must be a
    positive finite, exactly representable FP32 Python scalar. Input values are
    assumed finite; validation reads metadata only, without a device reduction.
    The final product is converted to BF16 once, with no BF16 SiLU intermediate.
    Native torch preserves these cast edges but its sigmoid is not a device
    arithmetic oracle. Compiled torch is inherited and is not byte-qualified.
    """

    op = "activation.silu_clamp_and_mul"
    priority = (KernelBackend.TRITON, KernelBackend.TORCH)
    capabilities = {KernelBackend.TRITON: frozenset({CapabilityRequirement.CUDA})}
    format_signature = FormatSignature(
        supported_dtypes=("bfloat16",),
        description="2D, contiguous columns, padded rows; disjoint input/out storage",
    )
    descriptions = {
        KernelBackend.TRITON: "FP32 SiLU, post-SiLU clamp, multiply, BF16 store.",
        KernelBackend.TORCH: "FP32 cast-edge reference; not a GPU arithmetic oracle.",
    }

    def __init__(self, *, limit: float = 7.0) -> None:
        super().__init__()
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            raise TypeError("limit must be a Python float or int")
        if limit <= 0 or limit > 3.4028234663852886e38 or not math.isfinite(limit):
            raise ValueError("limit must be positive and finite in FP32")
        if struct.unpack("f", struct.pack("f", limit))[0] != limit:
            raise ValueError("limit must be exactly representable in FP32")
        self._limit = float(limit)

    def forward_native(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        import torch

        out = _prepare_output(input=input, out=out)
        if input.shape[0] == 0:
            return out
        gate, up = input.float().chunk(2, dim=-1)
        gate = (torch.sigmoid(gate) * gate).clamp(max=self._limit)
        up = up.clamp(min=-self._limit, max=self._limit)
        out.copy_((gate * up).to(input.dtype))
        return out

    def forward_triton(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        import torch

        if input.device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("the Triton backend requires an NVIDIA CUDA tensor")
        out = _prepare_output(input=input, out=out)
        if input.shape[0] == 0:
            return out
        from sglang.kernels.ops.activation.silu_clamp_and_mul_triton import (
            _silu_clamp_and_mul,
        )

        _silu_clamp_and_mul(input=input, out=out, limit=self._limit)
        return out
