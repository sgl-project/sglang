from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _pack_flashinfer_scales_module() -> Module:
    return load_jit(
        "pack_flashinfer_moe_scales",
        cuda_files=["moe/pack_flashinfer_scales.cuh"],
        cuda_wrappers=[
            ("pack_flashinfer_moe_scales", "PackFlashInferMoeScalesKernel::run"),
            (
                "shuffle_rows_and_pack_flashinfer_moe_scales",
                "ShuffleRowsAndPackFlashInferMoeScalesKernel::run",
            ),
            (
                "silu_and_mul_quant_pack_flashinfer_moe",
                "SiluAndMulQuantPackFlashInferMoeKernel::run",
            ),
        ],
    )


def pack_flashinfer_moe_scales(
    scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack row-major FP32 scales into FlashInfer's padded expert layout."""
    assert scales.ndim == 2 and scales.dtype == torch.float32
    assert expert_offsets.ndim == 1 and expert_offsets.dtype == torch.int32
    assert scales.is_contiguous() and expert_offsets.is_contiguous()

    total_rows, k_blocks = scales.shape
    num_experts = expert_offsets.numel() - 1
    padded_rows = ((total_rows + 3 * num_experts) // 4) * 4
    if out is None:
        out = torch.empty(
            (k_blocks, padded_rows), device=scales.device, dtype=scales.dtype
        )
    else:
        assert out.shape == (k_blocks, padded_rows)
        assert out.dtype == scales.dtype and out.device == scales.device
        assert out.is_contiguous()

    _pack_flashinfer_scales_module().pack_flashinfer_moe_scales(
        scales, expert_offsets, out
    )
    return out


def shuffle_rows_and_pack_flashinfer_moe_scales(
    values: torch.Tensor,
    scales: torch.Tensor,
    row_map: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle FP8 rows and pack their scales for FlashInfer in one launch."""
    assert values.ndim == scales.ndim == 2
    assert values.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert values.shape[0] == scales.shape[0]
    assert row_map.ndim == expert_offsets.ndim == 1
    assert row_map.dtype == expert_offsets.dtype == torch.int32
    assert all(
        tensor.is_contiguous() for tensor in (values, scales, row_map, expert_offsets)
    )

    output_rows = row_map.numel()
    num_experts = expert_offsets.numel() - 1
    padded_rows = ((output_rows + 3 * num_experts) // 4) * 4
    if out is None:
        out = torch.empty(
            (output_rows, values.shape[1]), dtype=values.dtype, device=values.device
        )
    else:
        assert out.shape == (output_rows, values.shape[1])
        assert out.dtype == values.dtype and out.device == values.device
        assert out.is_contiguous()
    if out_scales is None:
        out_scales = torch.empty(
            (scales.shape[1], padded_rows), dtype=scales.dtype, device=scales.device
        )
    else:
        assert out_scales.shape == (scales.shape[1], padded_rows)
        assert out_scales.dtype == scales.dtype and out_scales.device == scales.device
        assert out_scales.is_contiguous()
    _pack_flashinfer_scales_module().shuffle_rows_and_pack_flashinfer_moe_scales(
        values, scales, row_map, expert_offsets, out, out_scales
    )
    return out, out_scales


def silu_and_mul_quant_pack_flashinfer_moe(
    gate_up: torch.Tensor,
    expert_offsets: torch.Tensor,
    swiglu_limit: float,
    out: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply clamped SwiGLU, FP8-quantize, and pack scales in one launch."""
    assert gate_up.ndim == 2 and gate_up.dtype == torch.bfloat16
    assert gate_up.shape[1] % 256 == 0
    assert expert_offsets.ndim == 1 and expert_offsets.dtype == torch.int32
    assert gate_up.is_contiguous() and expert_offsets.is_contiguous()

    rows, gate_up_size = gate_up.shape
    hidden_size = gate_up_size // 2
    num_experts = expert_offsets.numel() - 1
    padded_rows = ((rows + 3 * num_experts) // 4) * 4
    if out is None:
        out = torch.empty(
            (rows, hidden_size), dtype=torch.float8_e4m3fn, device=gate_up.device
        )
    else:
        assert out.shape == (rows, hidden_size)
        assert out.dtype == torch.float8_e4m3fn and out.device == gate_up.device
        assert out.is_contiguous()
    if out_scales is None:
        out_scales = torch.empty(
            (hidden_size // 128, padded_rows),
            dtype=torch.float32,
            device=gate_up.device,
        )
    else:
        assert out_scales.shape == (hidden_size // 128, padded_rows)
        assert out_scales.dtype == torch.float32 and out_scales.device == gate_up.device
        assert out_scales.is_contiguous()

    _pack_flashinfer_scales_module().silu_and_mul_quant_pack_flashinfer_moe(
        gate_up, expert_offsets, out, out_scales, swiglu_limit
    )
    return out, out_scales
