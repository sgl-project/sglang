# Copyright 2024 SGLang Team
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

import logging
from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.quantization.fp8_utils import fp8_dtype_to_triton
from sglang.srt.utils import (
    ceil_align,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_gfx1250_supported,
    is_hip,
    is_musa,
    is_xpu,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu = is_cpu()
_is_musa = is_musa()
_is_gfx1250 = is_gfx1250_supported()
_is_xpu = is_xpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8
    from sglang.kernels.ops.quantization.per_tensor_quant_fp8 import (
        per_tensor_quant_fp8 as sgl_per_tensor_quant_fp8,
    )
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant,
    )
elif _is_xpu:
    from sgl_kernel import sgl_per_tensor_quant_fp8, sgl_per_token_quant_fp8

if _is_musa:
    from sgl_kernel import sgl_per_token_quant_fp8

    from sglang.kernels.ops.quantization.per_tensor_quant_fp8 import (
        per_tensor_quant_fp8 as sgl_per_tensor_quant_fp8,
    )

if _is_musa:
    # per_token_group_quant is CUDA-only JIT; MUSA keeps the AOT v2 group-quant op.
    from sglang.kernels.ops.quantization import sgl_per_token_group_quant_8bit

if _is_hip and _use_aiter:
    try:
        from aiter import (  # v0.1.3
            dynamic_per_tensor_quant,
            dynamic_per_token_scaled_quant,
            static_per_tensor_quant,
        )
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")

if _is_musa:

    @register_fake_if_exists("sgl_kernel::sgl_per_token_group_quant_8bit_v2")
    def _(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        fuse_silu_and_mul,
        masked_m,
    ):
        return


logger = logging.getLogger(__name__)


@lru_cache()
def is_fp8_fnuz() -> bool:
    if _is_hip:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
    return False


if is_fp8_fnuz():
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = 224.0
else:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_8bit_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id.to(tl.int64) * group_size
    y_q_ptr += g_id.to(tl.int64) * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    if SCALE_UE8M0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def _per_token_group_quant_8bit_raw(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = fp8_dtype,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    if _is_hip:
        if dtype == torch.int8:
            bit8_max = 127.0
        else:
            # fp8 range is device-dependent on ROCm: e4m3fnuz (max 224.0) on
            # gfx94x vs e4m3fn (max 448.0) on gfx95x. Use the device-resolved
            # module constant instead of hardcoding the gfx94x value.
            bit8_max = fp8_max
        bit8_min = -bit8_max  # TODO incorrect for int8
    else:
        if dtype == torch.int8:
            info = torch.iinfo(dtype)
        else:
            info = torch.finfo(dtype)
        bit8_max = info.max
        bit8_min = info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x.shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=False,
    )

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_8bit_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x_s.stride(1),
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
            SCALE_UE8M0=scale_ue8m0,
        )
    else:
        assert not scale_ue8m0
        _per_token_group_quant_8bit[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            N,
            eps,
            bit8_min=bit8_min,
            bit8_max=bit8_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if scale_ue8m0:
        from deep_gemm import transform_sf_into_required_layout

        x_s = transform_sf_into_required_layout(
            x_s,
            num_groups=None,
            mn=x_q.shape[0],
            k=x_q.shape[1],
            recipe=(1, group_size, group_size),
            is_sfa=True,
        )

    return x_q, x_s


def _per_token_group_quant_8bit_fuse_silu_and_mul(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
    masked_m: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Another way to implement (can be used in e.g. comparison tests)
    # from sgl_kernel import silu_and_mul
    # x_after_silu_and_mul = silu_and_mul(x)
    # return per_token_group_quant_fp8(
    #     x_after_silu_and_mul,
    #     group_size=group_size,
    #     eps=eps,
    #     column_major_scales=column_major_scales,
    #     scale_tma_aligned=scale_tma_aligned,
    #     scale_ue8m0=scale_ue8m0,
    # )

    from deep_gemm import transform_sf_into_required_layout

    from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_post_quant_fwd

    assert column_major_scales
    assert scale_tma_aligned
    assert scale_ue8m0

    needs_unsqueeze = x.dim() == 2
    if needs_unsqueeze:
        num_tokens, _ = x.shape
        x = x.unsqueeze(0)
        assert masked_m is None
        masked_m = torch.tensor([num_tokens], device=x.device, dtype=torch.int32)

    # Use `zeros` for easier testing
    output = torch.zeros(
        (*x.shape[:-1], x.shape[-1] // 2),
        device=x.device,
        dtype=dst_dtype,
    )
    # Use `zeros` for easier testing
    output_scale_for_kernel = torch.zeros(
        (*x.shape[:-1], x.shape[-1] // 2 // group_size),
        device=x.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        input=x,
        output=output,
        output_scale=output_scale_for_kernel,
        quant_group_size=group_size,
        masked_m=masked_m,
        scale_ue8m0=scale_ue8m0,
    )

    output_scale = transform_sf_into_required_layout(
        output_scale_for_kernel,
        num_groups=output.shape[0],
        mn=output.shape[-2],
        k=output.shape[-1],
        recipe=(1, group_size, group_size),
        is_sfa=True,
    )

    if needs_unsqueeze:
        output = output.squeeze(0)
        output_scale = output_scale.squeeze(0)

    return output, output_scale


def per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fuse_silu_and_mul:
        return _per_token_group_quant_8bit_fuse_silu_and_mul(
            x=x,
            group_size=group_size,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
            masked_m=masked_m,
        )
    else:
        return _per_token_group_quant_8bit_raw(
            x=x,
            group_size=group_size,
            eps=eps,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
            dtype=dst_dtype,
        )


def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        if column_major_scales and scale_tma_aligned:
            *x_batch, x_q_mn, x_q_k = x_shape
            x_s_mn, x_s_k = x_q_mn, x_q_k // group_size
            aligned_mn = ceil_align(x_s_mn, 4)
            aligned_k = ceil_align(x_s_k, 4)
            # TODO(FIXME): Fix cuda kernel and recover here to empty.
            return torch.empty(
                (*x_batch, aligned_k // 4, aligned_mn),
                device=device,
                dtype=torch.int,
            ).transpose(-1, -2)[..., :x_s_mn, :]
        else:
            assert not column_major_scales, (
                "column_major_scales requires scale_tma_aligned=True "
                "when scale_ue8m0 is enabled"
            )
            # Row-major UE8M0 keeps the scale as float32 power-of-two values,
            # matching deep_gemm.ceil_to_ue8m0 and deep_gemm.fp8_einsum.
            return torch.empty(
                x_shape[:-1] + (x_shape[-1] // group_size,),
                device=device,
                dtype=torch.float32,
            )
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            # `...` so batched (e.g. masked [E, T, H]) shapes slice the token
            # axis, not dim 0.
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[..., : x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )


# AOT v2 (the MUSA path) runtime-switches on these; the JIT
# per_token_group_quant kernel also templates on 256.
_MUSA_KERNEL_SUPPORTED_GROUP_SIZES = (16, 32, 64, 128)
_V3_KERNEL_SUPPORTED_GROUP_SIZES = (16, 32, 64, 128, 256)


def _run_per_token_group_quant_8bit_kernel(
    x: torch.Tensor,
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    *,
    scale_ue8m0: bool,
    fuse_silu_and_mul: bool,
    masked_m: Optional[torch.Tensor],
) -> None:
    """Quantize into caller-owned ``x_q`` / ``x_s``.

    CUDA routes to the JIT per_token_group_quant kernel; MUSA stays on the AOT
    v2 op (the JIT kernel is CUDA-only), and the fp32-pow-2 storage flavor of
    row-major UE8M0 (float32 ``x_s``, deep_gemm ``ceil_to_ue8m0`` convention)
    stays on the JIT v2 baseline — per_token_group_quant only packs UE8M0 as
    int32. The kernel bakes the quant constants in at compile time, so
    drifted constants are rejected loudly here instead of silently quantizing
    with different ones; unsupported shapes/layouts error inside its host
    checks. Whole-row (per-token) quantization is a different op:
    ``sglang_per_token_quant_fp8``.
    """
    if scale_ue8m0 and x_s.dtype == torch.float32 and not _is_musa:
        from sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2 import (
            per_token_group_quant_8bit_v2,
        )

        per_token_group_quant_8bit_v2(
            input=x,
            output_q=x_q,
            output_s=x_s,
            group_size=group_size,
            eps=eps,
            min_8bit=fp8_min,
            max_8bit=fp8_max,
            scale_ue8m0=scale_ue8m0,
            fuse_silu_and_mul=fuse_silu_and_mul,
            masked_m=masked_m,
        )
        return

    if _is_musa:
        sgl_per_token_group_quant_8bit(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
            enable_v2=True,
        )
        return

    assert eps == 1e-10, (
        f"per_token_group_quant bakes the absmax floor in at 1e-10, got {eps}"
    )
    expected_range = (-448.0, 448.0) if x_q.dtype == fp8_dtype else (-128.0, 127.0)
    assert (fp8_min, fp8_max) == expected_range, (
        f"per_token_group_quant bakes the {x_q.dtype} quant range in at {expected_range}, "
        f"got ({fp8_min}, {fp8_max})"
    )
    per_token_group_quant(
        x,
        x_q,
        x_s,
        group_size,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
):
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    if (
        group_size == x.shape[-1]
        and x.dim() == 2
        and not (column_major_scales or scale_ue8m0 or fuse_silu_and_mul)
        and masked_m is None
    ):
        # Whole-row group quant is per-token quant; route to the dedicated
        # kernel (same [T, 1] scale shape) instead of a group kernel that
        # would need arbitrary group sizes.
        return sglang_per_token_quant_fp8(x)

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        _run_per_token_group_quant_8bit_kernel(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0=scale_ue8m0,
            fuse_silu_and_mul=fuse_silu_and_mul,
            masked_m=masked_m,
        )

    return x_q, x_s


def sglang_per_token_group_quant_fp8_row_padded(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    row_alignment: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group quant writing into row-padded buffers (col-major scales).

    The cutlass fp8_blockwise_scaled_mm wrapper pads mat_a / scales_a to a
    multiple of 4 rows on every call (a zeros fill + a cat for each of mat_a
    and scales_a). Allocating the quant outputs with rows already aligned to
    ``row_alignment`` makes the wrapper's pad_tensor() short-circuit (pad_rows
    == 0), removing 2x fill + 2x cat kernels per GEMM. Rows in [m, m_pad) are
    zero-filled to match the legacy pad_tensor contract so the padded GEMM is
    bit-exact; the caller still slices the GEMM output back to m.
    """
    assert x.dim() == 2, "row-padded quant expects a 2D input"
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` must be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    supported_group_sizes = (
        _MUSA_KERNEL_SUPPORTED_GROUP_SIZES
        if _is_musa
        else _V3_KERNEL_SUPPORTED_GROUP_SIZES
    )
    if group_size not in supported_group_sizes:
        # Keep the legacy unpadded path and let the GEMM wrapper do the padding.
        return sglang_per_token_group_quant_fp8(
            x, group_size, eps, column_major_scales=True
        )

    m, k = x.shape
    m_pad = ceil_align(m, row_alignment)
    # mat_a buffer: (m_pad, k) row-major fp8
    x_q = torch.empty((m_pad, k), device=x.device, dtype=fp8_dtype)
    # scales_a buffer: column-major (stride(0) == 1), shape (m_pad, k // group)
    x_s = torch.empty(
        (k // group_size, m_pad), device=x.device, dtype=torch.float32
    ).transpose(0, 1)
    if m > 0:
        _run_per_token_group_quant_8bit_kernel(
            x,
            x_q[:m],
            x_s[:m],
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0=False,
            fuse_silu_and_mul=False,
            masked_m=None,
        )
    if m_pad != m:
        # Tail rows feed the cutlass GEMM's padded region; zero them so the padded
        # GEMM stays bit-exact with the legacy pad_tensor path (torch.empty is garbage).
        x_q[m:].zero_()
        x_s[m:].zero_()
    return x_q, x_s


def sglang_per_token_group_quant_fp8_ue8m0(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0, (
        f"hidden ({x.shape[-1]}) must be divisible by group_size ({group_size})"
    )
    assert x.is_contiguous(), "x must be contiguous"

    *x_batch, x_q_mn, x_q_k = x.shape
    x_q = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)

    x_s_mn = x_q_mn
    x_s_k = x_q_k // group_size
    aligned_mn = ceil_align(x_s_mn, 4)
    aligned_k = ceil_align(x_s_k, 4)
    x_s = torch.empty(
        (*x_batch, aligned_k // 4, aligned_mn),
        device=x.device,
        dtype=torch.int,
    ).transpose(-1, -2)[..., :x_s_mn, :]

    if x.shape[0] > 0:
        _run_per_token_group_quant_8bit_kernel(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0=True,
            fuse_silu_and_mul=False,
            masked_m=None,
        )

    return x_q, x_s


# TODO maybe unify int8 and fp8 code later
def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
):
    from sglang.kernels.ops.quantization.int8_kernel import (
        sglang_per_token_group_quant_int8,
    )

    if dst_dtype == torch.int8:
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not fuse_silu_and_mul
        assert masked_m is None
        return sglang_per_token_group_quant_int8(
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
        )

    return sglang_per_token_group_quant_fp8(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )


def sglang_per_token_quant_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = fp8_dtype,
):
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[0],
        1,
        device=x.device,
        dtype=torch.float32,
    )

    sgl_per_token_quant_fp8(x, x_q, x_s)

    return x_q, x_s


if _is_cuda:
    per_token_group_quant_fp8 = sglang_per_token_group_quant_fp8
elif _is_cpu:

    def per_token_group_quant_fp8(
        x: torch.Tensor,
        group_size: int,
        eps: float = 1e-10,
        dtype: torch.dtype = fp8_dtype,
        column_major_scales: bool = False,
        scale_tma_aligned: bool = False,
        scale_ue8m0: bool = False,
        fuse_silu_and_mul: bool = False,
        masked_m: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert dtype == torch.float8_e4m3fn
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not scale_ue8m0
        assert not fuse_silu_and_mul
        assert masked_m is None
        return torch.ops.sgl_kernel.per_token_group_quant_fp8_cpu(x, group_size, eps)
else:
    per_token_group_quant_fp8 = _per_token_group_quant_8bit_raw


@triton.jit
def _static_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_s_repeat_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    REPEAT_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    """A Triton-accelerated function to perform quantization using the given scale on a
    tensor

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    if REPEAT_SCALE:
        y_s_repeat_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y_s = tl.load(y_s_ptr).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(FP8_DTYPE)

    tl.store(y_q_ptr + cols, y_q.to(tl.uint8, bitcast=True), mask=mask)
    if REPEAT_SCALE:
        tl.store(y_s_repeat_ptr, y_s)


def static_quant_fp8(
    x: torch.Tensor,
    x_s: torch.Tensor,
    repeat_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform static quantization using the given scale on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        x_s: The quantization scale.
        repeat_scale: Whether to broadcast per-tensor scale to per-channel scale.
        dtype: The dype of output tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert x.is_contiguous(), "`x` is not contiguous"
    assert x_s.numel() == 1, "only supports per-tensor scale"

    if _is_cpu:
        x_q, scale = torch.ops.sgl_kernel.scaled_fp8_quant_cpu(x, x_s, 0, False)
        if repeat_scale:
            scale = scale.repeat(x.numel() // x.shape[-1]).view(-1, 1)
        return x_q, scale

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    if repeat_scale:
        x_s_repeat = torch.empty(
            (M, 1),
            device=x.device,
            dtype=torch.float32,
        )
    else:
        x_s_repeat = None

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    pdl_kwargs = {"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _static_quant_fp8[(M,)](
        x,
        x_q.view(torch.uint8),
        x_s,
        x_s_repeat,
        N,
        N,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        FP8_DTYPE=fp8_dtype_to_triton(fp8_dtype),
        REPEAT_SCALE=repeat_scale,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_kwargs,
    )
    x_s = x_s_repeat if repeat_scale else x_s
    return x_q, x_s


# universal entry point, for testing purposes


@triton.jit
def _per_tensor_quant_mla_fp8_stage1(
    x_ptr,
    x_s_ptr,
    head_size,
    x_stride_h,
    x_stride_s,
    eps,
    fp8_max,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < head_size

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)

    tl.atomic_max(x_s_ptr, _absmax / fp8_max)


@triton.jit
def _per_tensor_quant_mla_fp8_stage2(
    x_ptr,
    x_s_ptr,
    x_q_ptr,
    num_seq,
    head_size,
    x_stride_h,
    x_stride_s,
    fp8_min,
    fp8_max,
    FP8_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < head_size

    x_s = tl.load(x_s_ptr)
    x_s_inv = 1.0 / x_s

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s
    x_q_ptr += head_id * num_seq * head_size + seq_id * head_size

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * x_s_inv, fp8_min, fp8_max).to(FP8_DTYPE)
    tl.store(x_q_ptr + offset, x_q.to(tl.uint8, bitcast=True), mask=mask)


def per_tensor_quant_mla_fp8(
    x: torch.Tensor, x_s_out: torch.Tensor, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function quantizes input values to float8 values with tensor-wise quantization
    and specialized for mla absorbed case.
    """
    assert x.dim() == 3, "`x` is not a 3d-tensor"
    assert (
        x_s_out.shape == (1,)
        and x_s_out.dtype == torch.float32
        and x_s_out.device == x.device
    )

    x_q = x.new_empty(x.size(), dtype=fp8_dtype)

    num_head, num_seq, head_size = x.shape
    BLOCK_SIZE = triton.next_power_of_2(head_size)
    grid = (num_seq, num_head)

    _per_tensor_quant_mla_fp8_stage1[grid](
        x,
        x_s_out,
        head_size,
        x.stride(0),
        x.stride(1),
        eps,
        fp8_max,
        BLOCK_SIZE,
    )
    _per_tensor_quant_mla_fp8_stage2[grid](
        x,
        x_s_out,
        x_q.view(torch.uint8),
        num_seq,
        head_size,
        x.stride(0),
        x.stride(1),
        fp8_min,
        fp8_max,
        fp8_dtype_to_triton(fp8_dtype),
        BLOCK_SIZE,
    )

    return x_q, x_s_out


@triton.jit
def _per_token_group_quant_mla_deep_gemm_masked_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    masked_m_ptr,
    group_size,
    y_stride_b,
    y_stride_t,
    y_q_stride_b,
    y_q_stride_t,
    y_s_stride_b,
    y_s_stride_g,
    eps,
    fp8_min,
    fp8_max,
    FP8_DTYPE: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor for deep_gemm grouped_gemm_masked.
    This function converts the tensor values into float8 values.
    y and y_q: (b, t, k)
    y_s: (b, k//group_size, t)
    """
    t_id = tl.program_id(0)
    b_id = tl.program_id(1)

    y_ptr += b_id * y_stride_b + t_id * y_stride_t
    y_q_ptr += b_id * y_q_stride_b + t_id * y_q_stride_t
    y_s_ptr += b_id * y_s_stride_b + t_id

    if t_id == 0:
        tl.store(masked_m_ptr + b_id, tl.num_programs(0))

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    for gid in range(NUM_GROUP):
        y = tl.load(y_ptr + gid * group_size + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max
        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(FP8_DTYPE)

        tl.store(
            y_q_ptr + gid * group_size + cols,
            y_q.to(tl.uint8, bitcast=True),
            mask=mask,
        )
        tl.store(y_s_ptr + gid * y_s_stride_g, y_s)


def per_token_group_quant_mla_deep_gemm_masked_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-12,
    dtype: torch.dtype = fp8_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function quantizes input values to float8 values with per-token-group-quantization
    for deep_gemm grouped_gemm_masked and specialized for mla absorbed case.
    """
    assert x.dim() == 3, "`x` is not a 3d-tensor"

    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256  # 256 is the max block_m of the gemm kernel
    num_tiles_k = k // group_size
    assert num_tiles_k * group_size == k, f"k % {group_size} must be zero"

    x_q = x.new_empty((b, aligned_m, k), dtype=dtype)
    x_s = x.new_empty((b, num_tiles_k, aligned_m), dtype=torch.float32)
    masked_m = x.new_empty((b,), dtype=torch.int32)

    BLOCK_SIZE = triton.next_power_of_2(group_size)
    grid = (m, b)

    _per_token_group_quant_mla_deep_gemm_masked_fp8[grid](
        x,
        x_q.view(torch.uint8),
        x_s,
        masked_m,
        group_size,
        x.stride(0),
        x.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        x_s.stride(0),
        x_s.stride(1),
        eps,
        -fp8_max,
        fp8_max,
        fp8_dtype_to_triton(dtype),
        num_tiles_k,
        BLOCK_SIZE,
    )

    return x_q, x_s.transpose(1, 2), masked_m, m, aligned_m


"""
Quantize input tensor to FP8 (8-bit floating point) format.

Args:
    input (torch.Tensor): Input tensor to be quantized
    scale (Optional[torch.Tensor]): Pre-computed scaling factor for static quantization.
        If None, scales will be computed dynamically.
    num_token_padding (Optional[int]): If specified, pad the first dimension
        of the output to at least this value.
    use_per_token_if_dynamic (bool): When using dynamic scaling (scale=None),
        determines the quantization granularity:
        - True: compute scale per token
        - False: compute single scale per tensor

Returns:
    Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - quantized_tensor: The FP8 quantized version of input
        - scale_tensor: The scaling factors used for quantization

Raises:
    AssertionError: If input is not 2D or if static scale's numel != 1
"""
if _is_hip:

    def _native_dynamic_per_token_quant_fp8(output, input, scale):
        """Native PyTorch fallback for dynamic per-token FP8 quantization when AITER is disabled."""
        M, N = input.shape
        eps = 1e-12
        # Compute per-token scale
        absmax = input.abs().max(dim=1, keepdim=True).values
        absmax = torch.clamp(absmax, min=eps)
        scale_val = absmax / fp8_max
        scale.copy_(scale_val)
        # Quantize
        output_data = torch.clamp(input / scale_val, fp8_min, fp8_max).to(fp8_dtype)
        output.copy_(output_data)

    def _native_dynamic_per_tensor_quant_fp8(output, input, scale):
        """Native PyTorch fallback for dynamic per-tensor FP8 quantization when AITER is disabled."""
        eps = 1e-12
        absmax = input.abs().max()
        absmax = torch.clamp(absmax, min=eps)
        scale_val = absmax / fp8_max
        # Use copy_ instead of fill_ with .item() to avoid CPU-GPU sync
        scale.view(-1).copy_(scale_val.view(-1))
        # Quantize
        output_data = torch.clamp(input / scale_val, fp8_min, fp8_max).to(fp8_dtype)
        output.copy_(output_data)

    def _native_static_quant_fp8(output, input, scale):
        """Native PyTorch fallback for static FP8 quantization when AITER is disabled."""
        # Use tensor directly instead of .item() to avoid CPU-GPU sync
        output_data = torch.clamp(input / scale, fp8_min, fp8_max).to(fp8_dtype)
        output.copy_(output_data)

    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        use_per_token_if_dynamic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                if _use_aiter:
                    dynamic_per_token_scaled_quant(output, input, scale)
                else:
                    _native_dynamic_per_token_quant_fp8(output, input, scale)
            else:
                scale = torch.empty(1, device=input.device, dtype=torch.float32)
                if _use_aiter:
                    dynamic_per_tensor_quant(output, input, scale)
                else:
                    _native_dynamic_per_tensor_quant_fp8(output, input, scale)
        else:
            # Static scaling
            assert scale.numel() == 1, (
                f"Expected scalar scale, got numel={scale.numel()}"
            )
            if _use_aiter:
                static_per_tensor_quant(output, input, scale)
            else:
                _native_static_quant_fp8(output, input, scale)

        return output, scale

else:

    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        use_per_token_if_dynamic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        if _is_cpu:
            return torch.ops.sgl_kernel.scaled_fp8_quant_cpu(
                input,
                scale,
                0 if num_token_padding is None else num_token_padding,
                use_per_token_if_dynamic,
            )

        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                sgl_per_token_quant_fp8(input, output, scale)
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                sgl_per_tensor_quant_fp8(
                    input, output, scale, is_static=False
                )  # False for dynamic
        else:
            # Static scaling
            assert scale.numel() == 1, (
                f"Expected scalar scale, got numel={scale.numel()}"
            )
            sgl_per_tensor_quant_fp8(
                input, output, scale, is_static=True
            )  # True for static

        return output, scale


fp8_autotune = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps)
        for block_m in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=["K", "BLOCK_K", "M_ALIGNMENT"],
)


@triton.jit
def _per_token_group_quant_fp8_hopper_moe_mn_major(
    a,  # (M, K):(K, 1)
    expert_offsets,  # (num_experts,)
    problem_sizes,  # (num_experts, 3)
    a_fp8,  # (M, K):(K, 1)
    sfa,  # (M, k)
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_ALIGNMENT: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tune
):
    k_offset = tl.program_id(0)
    expert_id = tl.program_id(1)

    m = tl.load(problem_sizes + expert_id * 3)
    current_expert_offset = tl.load(expert_offsets + expert_id).to(tl.int64)
    tl.multiple_of(m, M_ALIGNMENT)
    tl.multiple_of(current_expert_offset, M_ALIGNMENT)

    coord_k = k_offset * BLOCK_K + tl.arange(0, BLOCK_K)
    for i in tl.range(tl.cdiv(m, BLOCK_M)):
        coord_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        a_ptrs = a + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
        a_mask = (coord_m < m)[:, None] & (coord_k < K)[None, :]

        inp = tl.load(a_ptrs, mask=a_mask).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        inp_amax = tl.max(tl.abs(inp), axis=1)  # [BLOCK_M,]
        inp_amax = tl.clamp(inp_amax, min=1e-4, max=float("inf"))
        inp_fp8 = (inp * (448.0 / inp_amax[:, None])).to(FP8_DTYPE)

        # Store fp8
        a_fp8_ptrs = (
            a_fp8 + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
        )
        tl.store(a_fp8_ptrs, inp_fp8.to(tl.uint8, bitcast=True), mask=a_mask)

        # Store sfa
        k = tl.cdiv(K, BLOCK_K)
        sfa_ptrs = (
            sfa + current_expert_offset * k + k_offset * m + coord_m
        )  # MN-Major with sfa
        tl.store(sfa_ptrs, inp_amax / 448.0, mask=coord_m < m)


if not _is_cpu:
    _per_token_group_quant_fp8_hopper_moe_mn_major = fp8_autotune(
        _per_token_group_quant_fp8_hopper_moe_mn_major
    )


def per_token_group_quant_fp8_hopper_moe_mn_major(
    A: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    group_size: int,
    expert_tokens_alignment: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert A.dim() == 2
    assert A.is_contiguous(), "`A` is not contiguous"
    assert A.shape[-1] % group_size == 0, (
        "the last dimension of `A` cannot be divisible by `group_size`"
    )

    a_q = torch.empty_like(A, device=A.device, dtype=fp8_dtype)
    M, K = A.shape[0], A.shape[1]
    k = K // group_size
    sfa = torch.empty((M, k), device=A.device, dtype=torch.float32)
    num_experts = problem_sizes.shape[0]
    grid = (k, num_experts)
    _per_token_group_quant_fp8_hopper_moe_mn_major[grid](
        A,
        expert_offsets,
        problem_sizes,
        a_q.view(torch.uint8),
        sfa,
        K,
        group_size,
        expert_tokens_alignment,
        fp8_dtype_to_triton(fp8_dtype),
    )
    return a_q, sfa


@triton.jit
def _per_group_transpose(
    data_ptr: torch.Tensor,
    trans_data_ptr: torch.Tensor,
    expert_offsets: torch.Tensor,
    k: int,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_id = tl.program_id(1)
    k_id = tl.program_id(2)

    curr_expert_offset = tl.load(expert_offsets + expert_id)
    next_expert_offset = tl.load(expert_offsets + expert_id + 1)
    num_tokens_of_expert = next_expert_offset - curr_expert_offset
    tl.multiple_of(curr_expert_offset, M_ALIGNMENT)
    tl.multiple_of(next_expert_offset, M_ALIGNMENT)

    data_start_ptr = data_ptr + curr_expert_offset * k
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k

    k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_coord < k
    for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * tl.num_programs(1)):
        m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_coord < num_tokens_of_expert
        off = m_coord[:, None] * k + k_coord[None, :]
        trans_off = m_coord[:, None] + k_coord[None, :] * num_tokens_of_expert
        mask = m_mask[:, None] & k_mask[None, :]

        data = tl.load(data_start_ptr + off, mask=mask)
        tl.store(trans_data_start_ptr + trans_off, data, mask=mask)


def per_group_transpose(
    a: torch.Tensor,
    expert_offsets: torch.Tensor,
    M_ALIGNMENT: int = 1,
) -> torch.Tensor:
    assert a.dim() == 2
    assert a.is_contiguous(), "`a` is not contiguous"

    m, k = a.size()
    trans_a = torch.empty_like(a)
    num_experts = expert_offsets.size(0) - 1

    grid = lambda META: (
        num_experts,
        triton.cdiv((m + num_experts - 1) // num_experts, META["BLOCK_SIZE_M"]),
        triton.cdiv(k, META["BLOCK_SIZE_K"]),
    )
    _per_group_transpose[grid](
        a, trans_a, expert_offsets, k, M_ALIGNMENT, BLOCK_SIZE_M=16, BLOCK_SIZE_K=8
    )
    return trans_a


# input  - [M, K]
# weight - [K, N]
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
