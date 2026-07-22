from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_OUTPUT_DTYPES = (torch.float8_e4m3fn, torch.int8)
_SUPPORTED_GROUP_SIZES = (16, 32, 64, 128, 256)


@cache_once
def _jit_module(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    group_size: int,
    scale_ue8m0: bool,
    row_major: bool,
    aligned: bool,
    fuse_silu_and_mul: bool,
    masked_layout: bool,
    use_pdl: bool,
) -> Module:
    assert in_dtype in _SUPPORTED_INPUT_DTYPES
    assert out_dtype in _SUPPORTED_OUTPUT_DTYPES
    assert group_size in _SUPPORTED_GROUP_SIZES
    trait_args = make_cpp_args(
        in_dtype,
        out_dtype,
        group_size,
        scale_ue8m0,
        row_major,
        aligned,
        fuse_silu_and_mul,
        use_pdl,
    )
    launcher = (
        "PerTokenGroupQuantMaskedKernel"
        if masked_layout
        else "PerTokenGroupQuantFlatKernel"
    )
    return load_jit(
        "per_token_group_quant",
        *trait_args,
        "masked" if masked_layout else "flat",
        cuda_files=["gemm/per_token_group_quant.cuh"],
        cuda_wrappers=[("per_token_group_quant", f"{launcher}<{trait_args}>::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def _infer_scale_layout(
    output_s: torch.Tensor, scale_ue8m0: bool, num_groups: int
) -> Tuple[bool, bool]:
    """Return ``(row_major, aligned)`` for ``output_s``.

    Column-major (transposed) scale buffers have token stride 1 and a larger
    group stride; row-major buffers are contiguous.
    """
    row_major = output_s.stride(-2) >= output_s.stride(-1)
    if output_s.dtype == torch.int32:
        if not scale_ue8m0:
            raise ValueError("int32-packed scale buffers require scale_ue8m0=True")
        aligned = num_groups % 4 == 0
        return row_major, aligned
    if output_s.dtype == torch.float32:
        if scale_ue8m0:
            raise ValueError("scale_ue8m0=True requires an int32-packed output_s")
        return row_major, True
    raise ValueError(f"Unsupported output_s dtype {output_s.dtype}")


@register_custom_op(
    op_name="per_token_group_quant",
    mutates_args=["output_q", "output_s"],
)
def _per_token_group_quant_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
) -> None:
    num_groups = output_q.shape[-1] // group_size
    row_major, aligned = _infer_scale_layout(output_s, scale_ue8m0, num_groups)
    module = _jit_module(
        input.dtype,
        output_q.dtype,
        int(group_size),
        bool(scale_ue8m0),
        row_major,
        aligned,
        bool(fuse_silu_and_mul),
        masked_m is not None,
        is_arch_support_pdl(),
    )
    if masked_m is not None:
        module.per_token_group_quant(
            input, output_q, output_s, masked_m, int(expected_m or -1)
        )
    else:
        module.per_token_group_quant(input, output_q, output_s)


def _allocate_outputs(
    input: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype,
    scale_ue8m0: bool,
    column_major_scales: bool,
    fuse_silu_and_mul: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate ``(output_q, output_s)`` in the requested major mode / scale
    format, selected by ``(column_major_scales, scale_ue8m0)``."""
    hidden = input.shape[-1] // (2 if fuse_silu_and_mul else 1)
    out_shape = (*input.shape[:-1], hidden)
    output_q = torch.empty(out_shape, device=input.device, dtype=out_dtype)

    num_groups = hidden // group_size
    if scale_ue8m0 and not column_major_scales:
        # Row-major packed UE8M0: int32 [..., ceil(ng/4)] contiguous (an
        # unaligned ng leaves a partially-used last int32 that the kernel zero-
        # pads). The shared create_*_output_scale helper does not produce this
        # layout.
        output_s = torch.empty(
            (*out_shape[:-1], (num_groups + 3) // 4),
            device=input.device,
            dtype=torch.int32,
        )
    else:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        output_s = create_per_token_group_quant_fp8_output_scale(
            x_shape=out_shape,
            device=input.device,
            group_size=group_size,
            column_major_scales=column_major_scales,
            scale_tma_aligned=column_major_scales,
            scale_ue8m0=scale_ue8m0,
        )
    return output_q, output_s


@debug_kernel_api
def per_token_group_quant(
    input: torch.Tensor,
    output_q: Optional[torch.Tensor] = None,
    output_s: Optional[torch.Tensor] = None,
    group_size: int = 128,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    expected_m: Optional[int] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group quantization. Returns ``(output_q, output_s)``.

    ``output_q`` / ``output_s`` are optional: pass them to quantize into
    caller-owned buffers, or omit both to have them allocated per ``out_dtype``
    (default fp8_e4m3), ``scale_ue8m0`` and ``column_major_scales``. Either way
    the two tensors are returned.

    Input / output shapes:
      vanilla:            input [T, hidden],      output_q [T, hidden]
      fuse_silu_and_mul:  input [T, hidden*2],    output_q [T, hidden]
      masked (+ above):   input [E, T_pad, ...],  output_q [E, T_pad, hidden],
                          masked_m [E] int32
    ``output_s`` scale layouts (inferred from a supplied buffer's dtype/strides,
    or allocated to match when omitted):
      float32 contiguous  -> row-major fp32 scales
      float32 transposed  -> col-major fp32 scales (TMA-aligned view)
      int32 transposed    -> col-major UE8M0 bytes packed 4-per-int32
      int32 contiguous    -> row-major UE8M0 bytes packed 4-per-int32
    The packed layouts require ``scale_ue8m0=True``.

    ``expected_m`` (masked only) is an optional expected-tokens-per-expert hint.

    Inputs are bf16/fp16; group size is one of 16/32/64/128/256; the quant range
    follows ``output_q.dtype`` (fp8_e4m3: +-448, int8: [-128, 127]).
    """
    if output_q is None:
        assert output_s is None
        output_q, output_s = _allocate_outputs(
            input,
            group_size,
            out_dtype or torch.float8_e4m3fn,
            scale_ue8m0,
            column_major_scales,
            fuse_silu_and_mul,
        )
    else:
        assert output_s is not None
        assert out_dtype is None or out_dtype == output_q.dtype
    _per_token_group_quant_custom_op(
        input=input,
        output_q=output_q,
        output_s=output_s,
        group_size=group_size,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        expected_m=expected_m,
    )
    return output_q, output_s
