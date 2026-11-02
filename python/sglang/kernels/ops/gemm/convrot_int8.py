"""ConvRot INT8 W8A8 linear (arXiv:2512.03673), JIT-compiled for the local GPU.

One entry rotates a BF16 activation group-wise with the regular Hadamard,
quantizes it per row to INT8 and runs the dense INT8 GEMM with the per-row x
per-column dequant (and bias) fused into the epilogue; the rotate step alone
is the offline transform for the [N, K] weight. CUTLASS 3.x WGMMA on CC 9.0,
tcgen05 on CC 10.0 and the CUTLASS 2.x mma.sync path on CC 12.0 / 12.1; the
module compiles once per machine on first use and is cached by the JIT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, get_jit_cuda_arch, load_jit
from sglang.kernels.jit.utils.common import is_hip_runtime, is_musa_runtime
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Exact compute capabilities the entry header carries a GEMM path for: sm_90a
# and sm_100a hold the WGMMA / tcgen05 kernels, CC 12.0 / 12.1 the mma.sync
# path. CC 10.3 is left out on purpose (INT8 tensor-core rate); the quant
# method reports the reason.
SUPPORTED_CAPABILITIES: frozenset[tuple[int, int]] = frozenset(
    {(9, 0), (10, 0), (12, 0), (12, 1)}
)
# Hadamard group widths the rotate kernel is instantiated for.
GROUP_SIZES: tuple[int, ...] = (64, 128, 256, 512)


def convrot_int8_supported_capabilities() -> frozenset[tuple[int, int]]:
    """(major, minor) pairs the convrot_int8 ops compile for."""
    return SUPPORTED_CAPABILITIES


def _convrot_int8_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        # Only the row-scale division and denormal handling see this flag; a
        # precise build rounds a few INT8 codes one step away from the
        # validated numerics. The GELU tanh is pinned to libdevice __nv_tanhf.
        "--use_fast_math",
    ]


def convrot_int8_local_capability() -> tuple[int, int] | None:
    """(major, minor) the JIT build would target, or None off CUDA (ROCm and
    MUSA capability numbers are not SM versions)."""
    if is_hip_runtime() or is_musa_runtime():
        return None
    arch = get_jit_cuda_arch()
    return (arch.major, arch.minor)


@cache_once
def _jit_convrot_int8_module() -> Module:
    """Compile and cache the convrot_int8 module for the local GPU."""
    capability = convrot_int8_local_capability()
    if capability not in SUPPORTED_CAPABILITIES:
        supported = ", ".join(f"{a}.{b}" for a, b in sorted(SUPPORTED_CAPABILITIES))
        where = (
            "a non-CUDA device"
            if capability is None
            else f"CC {capability[0]}.{capability[1]}"
        )
        raise RuntimeError(
            f"convrot_int8: no kernel for {where} (supported: {supported})"
        )
    return load_jit(
        "convrot_int8",
        cuda_files=["gemm/convrot_int8/convrot_int8_entry.cuh"],
        cuda_wrappers=[
            (
                "convrot_rotate_quantize_activation",
                "convrot_rotate_quantize_activation",
            ),
            ("convrot_int8_fused_linear", "convrot_int8_fused_linear"),
            ("convrot_int8_linear_prequant", "convrot_int8_linear_prequant"),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_convrot_int8_cuda_flags(),
    )


def load_convrot_int8_module() -> Module:
    """Build (first use) or load the cached module; raises when the GPU is
    unsupported, the CUDA toolchain is missing or the build fails."""
    return _jit_convrot_int8_module()


# The custom ops below exist for torch.compile: dynamo needs an opaque op. Eager
# calls go straight to the module instead, because the dispatcher round trip of
# a Python-implemented op costs ~9 us on top of the ~10 us tvm-ffi call
# (measured on GB200) and a Qwen-Image DiT forward issues ~850 of these calls.
@register_custom_op(
    op_name="convrot_rotate_quantize_activation",
    mutates_args=["x_q", "x_scale"],
)
def _convrot_rotate_quantize_activation_op(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    x: torch.Tensor,
    group_size: int,
) -> None:
    module = _jit_convrot_int8_module()
    module.convrot_rotate_quantize_activation(x_q, x_scale, x, group_size)


@register_custom_op(op_name="convrot_int8_fused_linear", mutates_args=["out"])
def _convrot_int8_fused_linear_op(
    out: torch.Tensor,
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    gelu_input: bool,
) -> None:
    module = _jit_convrot_int8_module()
    module.convrot_int8_fused_linear(
        out, x, weight_q, weight_scale, bias, group_size, gelu_input
    )


@register_custom_op(op_name="convrot_int8_linear_prequant", mutates_args=["out"])
def _convrot_int8_linear_prequant_op(
    out: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> None:
    module = _jit_convrot_int8_module()
    module.convrot_int8_linear_prequant(
        out, xq, xs, weight_q, weight_scale, bias, group_size
    )


@debug_kernel_api
def convrot_rotate_quantize_activation(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group Hadamard rotation + per-row INT8 quant of a contiguous BF16 [M, K].

    Returns (x_q int8 [M, K], x_scale float32 [M]). Also the offline transform
    for a [N, K] weight, yielding (weight_q, weight_scale).
    """
    x_q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    x_scale = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    if torch.compiler.is_compiling():
        _convrot_rotate_quantize_activation_op(x_q, x_scale, x, group_size)
    else:
        _jit_convrot_int8_module().convrot_rotate_quantize_activation(
            x_q, x_scale, x, group_size
        )
    return x_q, x_scale


@debug_kernel_api
def convrot_int8_fused_linear(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    out: torch.Tensor | None = None,
    gelu_input: bool = False,
) -> torch.Tensor:
    """BF16 [M, K] x int8 [N, K] -> BF16 [M, N]; x is rotated and quantized in-kernel.

    ``out`` must be a contiguous BF16 [M, N] when given (a row slice of a larger
    buffer qualifies). ``gelu_input`` applies GELU(tanh) to x inside the rotate
    kernel, bitwise equal to the eager ``F.gelu(x, approximate="tanh")`` first.
    """
    if out is None:
        out = torch.empty(
            (x.shape[0], weight_q.shape[0]), dtype=torch.bfloat16, device=x.device
        )
    if torch.compiler.is_compiling():
        _convrot_int8_fused_linear_op(
            out, x, weight_q, weight_scale, bias, group_size, gelu_input
        )
    else:
        _jit_convrot_int8_module().convrot_int8_fused_linear(
            out, x, weight_q, weight_scale, bias, group_size, gelu_input
        )
    return out


@debug_kernel_api
def convrot_int8_linear_prequant(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GEMM + dequant on (xq int8 [M, K], xs float32 [M]) from
    convrot_rotate_quantize_activation; bitwise equal to the fused op."""
    if out is None:
        out = torch.empty(
            (xq.shape[0], weight_q.shape[0]), dtype=torch.bfloat16, device=xq.device
        )
    if torch.compiler.is_compiling():
        _convrot_int8_linear_prequant_op(
            out, xq, xs, weight_q, weight_scale, bias, group_size
        )
    else:
        _jit_convrot_int8_module().convrot_int8_linear_prequant(
            out, xq, xs, weight_q, weight_scale, bias, group_size
        )
    return out
