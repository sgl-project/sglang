from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

try:
    from vllm import _custom_ops as ops

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
    static_quant_fp8,
    triton_scaled_mm,
    w8a8_block_fp8_matmul_deepgemm,
    w8a8_block_fp8_matmul_triton,
)
from sglang.srt.utils import (
    ceil_align,
    ceil_div,
    get_bool_env_var,
    get_cuda_version,
    get_device_capability,
    is_blackwell_supported,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_sm90_supported,
    offloader,
)

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    import aiter

    # from aiter import gemm_a8w8_blockscale, gemm_a8w8_bpreshuffle, get_hip_quant
    from aiter import gemm_a8w8_bpreshuffle, get_hip_quant
    from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)

if _is_cuda:
    from sgl_kernel import fp8_blockwise_scaled_mm, fp8_scaled_mm

    @torch.library.register_fake("sgl_kernel::fp8_scaled_mm")
    def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
        # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].
        M = mat_a.shape[-2]
        N = mat_b.shape[-1]
        return mat_a.new_empty((M, N), dtype=out_dtype)


use_vllm_cutlass_w8a8_fp8_kernel = get_bool_env_var("USE_VLLM_CUTLASS_W8A8_FP8_KERNEL")
use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def use_rowwise_torch_scaled_mm():
    _TORCH_VERSION = torch.__version__.split("+")[0]
    try:
        _TORCH_VERSION_TUPLE = tuple(map(int, _TORCH_VERSION.split(".")[:3]))
    except ValueError:
        _TORCH_VERSION_TUPLE = (0, 0, 0)
    if _is_hip:
        # The condition to determine if it is on a platform that supports
        # torch._scaled_mm rowwise feature.
        # The condition is determined once as the operations
        # are time consuming.
        return get_device_capability() >= (9, 4) and _TORCH_VERSION_TUPLE >= (2, 7, 0)
    return False


USE_ROWWISE_TORCH_SCALED_MM = use_rowwise_torch_scaled_mm()


def cutlass_fp8_supported():
    if not _is_cuda:
        return False
    major, minor = get_device_capability()
    cuda_version = get_cuda_version()
    if major >= 9:
        return cuda_version >= (12, 0)
    elif major == 8 and minor == 9:
        return cuda_version >= (12, 4)
    return False


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


class Fp8GemmRunnerBackend(Enum):
    """Enum for FP8 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER = "flashinfer_trtllm"
    CUTLASS = "cutlass"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    AITER = "aiter"

    def is_auto(self) -> bool:
        return self == Fp8GemmRunnerBackend.AUTO

    def is_flashinfer(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER

    def is_cutlass(self) -> bool:
        return self == Fp8GemmRunnerBackend.CUTLASS

    def is_deep_gemm(self) -> bool:
        return self == Fp8GemmRunnerBackend.DEEP_GEMM

    def is_triton(self) -> bool:
        return self == Fp8GemmRunnerBackend.TRITON

    def is_aiter(self) -> bool:
        return self == Fp8GemmRunnerBackend.AITER


FP8_GEMM_RUNNER_BACKEND: Fp8GemmRunnerBackend | None = None


def _check_cutlass_block_fp8_hardware_support() -> bool:
    """Return True if CUTLASS block FP8 is supported (Hopper or newer with CUDA 12.0+)."""
    return is_sm90_supported() or is_blackwell_supported()


if is_blackwell_supported() and is_flashinfer_available():
    from flashinfer.gemm import gemm_fp8_nt_groupwise


def dispatch_w8a8_block_fp8_linear() -> Callable:
    """
    Dispatch to the appropriate FP8 block linear implementation.

    This function selects the backend based on:
    1. The --fp8-gemm-backend server argument (preferred)
    2. Auto-detection based on hardware capabilities
    """
    backend = get_fp8_gemm_runner_backend()

    # Handle explicit backend selection via --fp8-gemm-backend
    if not backend.is_auto():
        return _dispatch_explicit_backend(backend)

    # Auto mode: Select based purely on hardware/backend availability
    return _dispatch_auto_backend()


def _dispatch_explicit_backend(backend: Fp8GemmRunnerBackend) -> Callable:
    """Dispatch based on explicitly selected backend."""
    if backend.is_flashinfer():
        if not (is_blackwell_supported() and is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer FP8 GEMM requires Blackwell GPUs and FlashInfer to be installed."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_cutlass():
        if not _check_cutlass_block_fp8_hardware_support():
            raise RuntimeError(
                "CUTLASS block FP8 requested via --fp8-gemm-backend=cutlass, "
                "but hardware does not support it. CUTLASS block FP8 requires "
                "Hopper (SM90+) GPUs with CUDA 12.0+."
            )
        return cutlass_w8a8_block_fp8_linear_with_fallback

    elif backend.is_aiter():
        if not _use_aiter:
            raise RuntimeError(
                "AITER backend requested via --fp8-gemm-backend=aiter, "
                "but AITER is not available. AITER requires AMD GPUs with "
                "SGLANG_USE_AITER=1 environment variable set."
            )
        return aiter_w8a8_block_fp8_linear

    elif backend.is_deep_gemm():
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise RuntimeError(
                "DeepGEMM backend requested via --fp8-gemm-backend=deep_gemm, "
                "but DeepGEMM is not available. This usually means the deep_gemm package "
                "is not installed or has been disabled via SGLANG_ENABLE_JIT_DEEPGEMM=0."
            )
        return deepgemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_triton():
        return triton_w8a8_block_fp8_linear

    else:
        raise ValueError(f"Unknown FP8 GEMM backend: {backend}")


def _dispatch_auto_backend() -> Callable:
    """Auto-select the best backend based on hardware capabilities."""
    # Priority order for auto selection:
    # 1. DeepGEMM (if enabled and available)
    # 2. FlashInfer TRTLLM (if Blackwell GPU and FlashInfer available)
    # 3. CUTLASS (if Hopper+ GPU and CUDA 12.0+)
    # 4. AITER (if AMD GPU with AITER enabled)
    # 5. Triton (fallback)

    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return deepgemm_w8a8_block_fp8_linear_with_fallback
    elif is_blackwell_supported() and is_flashinfer_available():
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
    elif _check_cutlass_block_fp8_hardware_support():
        return cutlass_w8a8_block_fp8_linear_with_fallback
    elif _use_aiter:
        return aiter_w8a8_block_fp8_linear
    else:
        return triton_w8a8_block_fp8_linear


def initialize_fp8_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP8 GEMM configuration."""
    global FP8_GEMM_RUNNER_BACKEND

    backend = server_args.fp8_gemm_runner_backend

    # TODO(brayden): Remove env-based overrides in v0.5.7, they will be fully removed in v0.5.7.
    # Only check environment variables when the server args is not set, server args should take priority.
    if backend == "auto":
        if envs.SGLANG_ENABLE_FLASHINFER_FP8_GEMM.get():
            backend = "flashinfer_trtllm"
        elif envs.SGLANG_SUPPORT_CUTLASS_BLOCK_FP8.get():
            backend = "cutlass"
    else:
        if (
            envs.SGLANG_ENABLE_FLASHINFER_FP8_GEMM.get()
            or envs.SGLANG_SUPPORT_CUTLASS_BLOCK_FP8.get()
        ):
            logger.warning(
                f"FP8 GEMM backend set to '{backend}' via --fp8-gemm-backend overrides "
                "environment variables SGLANG_ENABLE_FLASHINFER_FP8_GEMM and "
                "SGLANG_SUPPORT_CUTLASS_BLOCK_FP8. Using server argument value."
            )

    FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend(backend)


def get_fp8_gemm_runner_backend() -> Fp8GemmRunnerBackend:
    """Get the current FP8 GEMM runner backend."""
    global FP8_GEMM_RUNNER_BACKEND
    if FP8_GEMM_RUNNER_BACKEND is None:
        FP8_GEMM_RUNNER_BACKEND = Fp8GemmRunnerBackend.AUTO
    return FP8_GEMM_RUNNER_BACKEND


def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    # FlashInfer TRTLLM backend requires K dimension >= 256
    # Check shape before quantizing, otherwise we run into Flashinfer assertion.
    # TODO(brayden): make a better fallback here, maybe to cutlass backend?
    input_2d = input.view(-1, input.shape[-1])
    k_dim = input_2d.shape[1]  # K dimension

    if k_dim < 256:
        # Fallback to Triton for shapes that don't meet TRTLLM constraint.
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=True
    )
    # TRTLLM requires column-major scaling factors
    output = gemm_fp8_nt_groupwise(
        q_input,
        weight,
        x_scale,
        weight_scale,
        out_dtype=input_2d.dtype,
        backend="trtllm",
    )

    if bias is not None:
        output += bias

    return output.to(dtype=input_2d.dtype).view(*output_shape)


def cutlass_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    # TODO: add more robust shape check here
    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0

    if not shape_supported:
        # fallback to triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=True
    )
    output = fp8_blockwise_scaled_mm(
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=input_2d.dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


def deepgemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    output_dtype = input.dtype
    dtype_supported = output_dtype == torch.bfloat16

    # TODO: https://github.com/sgl-project/sglang/pull/6890#issuecomment-2943395737
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0

    if not (shape_supported and dtype_supported):
        # fall back to triton
        # If weight_scale is in UE8M0 packed format (int32), convert back to float32
        # UE8M0 format has shape (N, K//block_k//4) with dtype int32
        # Triton expects shape (N//block_n, K//block_k) with dtype float32
        if weight_scale.dtype == torch.int32:
            weight_scale = _unpack_ue8m0_scale_for_triton(
                weight_scale, weight.shape, block_size
            )
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )

    output = w8a8_block_fp8_matmul_deepgemm(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


def _unpack_ue8m0_scale_for_triton(
    sf_packed: torch.Tensor,
    weight_shape: Tuple[int, int],
    block_size: List[int],
) -> torch.Tensor:
    """
    Unpack UE8M0 packed scale tensor back to float32 format for triton kernel.

    The UE8M0 format packs scales as:
    - Shape: (N, K//block_k//4) with dtype int32
    - Each int32 contains 4 uint8 scale values

    Triton expects:
    - Shape: (N//block_n, K//block_k) with dtype float32

    Args:
        sf_packed: Packed scale tensor with shape (N, packed_k_groups) and dtype int32
        weight_shape: (N, K) shape of the weight tensor
        block_size: [block_n, block_k] quantization block size

    Returns:
        Unpacked scale tensor with shape (n_groups, k_groups) and dtype float32
    """
    assert sf_packed.dtype == torch.int32
    assert len(sf_packed.shape) == 2

    N, K = weight_shape
    block_n, block_k = block_size
    n_groups = ceil_div(N, block_n)
    k_groups = ceil_div(K, block_k)

    mn_repeat, k_div_4 = sf_packed.shape
    k_packed = k_div_4 * 4

    # Unpack int32 -> 4x uint8 -> float32
    # Each uint8 represents an exponent in UE8M0 format
    sf_u8 = sf_packed.contiguous().view(torch.uint8).view(mn_repeat, k_packed)
    sf_fp32 = (sf_u8.to(torch.int32) << 23).view(torch.float32)

    # Handle row dimension - may have 128x replication or direct mapping
    if mn_repeat == N:
        # Rows are replicated 128 times, take every 128th row
        # sf_fp32 shape: (N, k_packed) -> (n_groups, k_packed)
        # Select representative rows at indices 0, 128, 256, ...
        indices = torch.arange(0, N, block_n, device=sf_packed.device)
        sf_fp32 = sf_fp32.index_select(0, indices)
    elif mn_repeat == n_groups:
        # Already in the correct n_groups format
        pass
    else:
        raise ValueError(
            f"Unexpected scale shape: sf_packed.shape={sf_packed.shape}, "
            f"weight_shape={weight_shape}, block_size={block_size}"
        )

    # Crop k dimension to expected size (remove padding if any)
    sf_fp32 = sf_fp32[:, :k_groups].contiguous()

    return sf_fp32


def aiter_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    # if input_scale not None, input is quanted
    if input_scale is not None:
        q_input = input_2d
        x_scale = input_scale

    else:
        q_input, x_scale = aiter_per1x128_quant(input_2d, quant_dtype=aiter.dtypes.fp8)

    output = gemm_a8w8_blockscale(
        q_input,
        weight,
        x_scale,
        weight_scale,
        dtype=torch.bfloat16 if input_scale is not None else input.dtype,
    )

    if bias is not None:
        output += bias

    return output.to(
        dtype=torch.bfloat16 if input_scale is not None else input_2d.dtype
    ).view(*output_shape)


def triton_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=False
    )
    output = w8a8_block_fp8_matmul_triton(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input_2d.dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


def dequant_mxfp4(
    w_block: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype,
) -> torch.Tensor:
    """
    :param w_block: (batch, n, k, 16), uint8, pack two mxfp4 into one byte
    :param w_scale: (batch, n, k), uint8
    :return: (batch, n, k * 32), float32
    """

    assert w_block.dtype == torch.uint8
    assert w_scale.dtype == torch.uint8

    batch, n, k, pack_dim = w_block.shape
    batch_, n_, k_ = w_scale.shape
    assert pack_dim == 16
    assert batch == batch_
    assert n == n_
    assert k == k_

    out_raw = MXFP4QuantizeUtil.dequantize(
        quantized_data=w_block, scale=w_scale, dtype=out_dtype, block_sizes=[32]
    )
    return out_raw.reshape(batch, n, k * 32)


def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype = fp8_dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).float().clamp(min=1e-12)

    if _is_fp8_fnuz:
        dtype = fp8_dtype
        fp_max = fp8_max
    else:
        finfo = torch.finfo(dtype)
        fp_max = finfo.max

    scale = fp_max / amax
    x_scl_sat = (x.float() * scale).clamp(min=-fp_max, max=fp_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise quantization.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise quantization scale.
    Note only float8 is supported for now.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [
        [
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]

    x_q_tensor, scale = (
        scaled_fp8_quant(x_dq_block)
        if _is_cuda
        else input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    )
    return x_q_tensor, scale


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """This function converts block-wise quantization to unquantized.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The output is an unquantized tensor with dtype.
    """
    block_n, block_k = block_size[0], block_size[1]
    *_, n, k = x_q_block.shape

    # ... n_scale k_scale -> ... (n_scale block_n) (k_scale block_k)
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]

    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)


def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):
    assert isinstance(weight, torch.nn.Parameter)
    assert isinstance(weight_scale_inv, torch.nn.Parameter)

    new_weight, new_weight_scale_inv = requant_weight_ue8m0(
        weight.to(weight_scale_inv.device), weight_scale_inv, weight_block_size
    )

    offloader.update_param(weight, new_weight)
    weight_scale_inv.data = new_weight_scale_inv


def requant_weight_ue8m0(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]

    *_, n, k = weight.shape

    weight_dequant = block_quant_dequant(
        weight,
        weight_scale_inv,
        weight_block_size,
        torch.bfloat16,
    )

    out_w, out_s = quant_weight_ue8m0(
        weight_dequant=weight_dequant,
        weight_block_size=weight_block_size,
    )

    out_s = transform_scale_ue8m0(out_s, mn=out_w.shape[-2])

    return out_w, out_s


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]
    assert (
        weight_dequant.dtype == torch.bfloat16
    ), f"{weight_dequant.dtype=} {weight_dequant.shape=}"

    *batch_dims, n, k = weight_dequant.shape

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat)

    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s


def transform_scale_ue8m0_inplace(param, mn):
    param.data = transform_scale_ue8m0(param.data, mn=mn)


# NOTE copy and modified from DeepGEMM
def transform_scale_ue8m0(sf, mn, use_torch_impl: bool = False):
    import deep_gemm.utils.layout

    get_mn_major_tma_aligned_packed_ue8m0_tensor = (
        _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
        if use_torch_impl
        else deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor
    )

    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


# Copied from DeepGEMM tests
def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: torch.Tensor,
) -> torch.Tensor:
    from deep_gemm.utils import align, get_tma_aligned_size

    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)

    # Finally, transpose
    transposed = torch.zeros(
        (b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int
    ).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def inverse_transform_scale_ue8m0(sf_packed, mn):
    sf_fp32 = _inverse_transform_scale_ue8m0_impl(sf_packed)
    # Can call consistency check every time since this is only called on startup
    sf_packed_recreated = transform_scale_ue8m0(sf_fp32, mn=mn, use_torch_impl=True)
    assert torch.all(
        sf_packed == sf_packed_recreated
    ), f"{sf_packed=} {sf_packed_recreated=} {sf_fp32=}"
    return sf_fp32


# Inverse impl can refer to DeepGEMM's torch impl in get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
def _inverse_transform_scale_ue8m0_impl(sf_packed):
    """
    NOTE: We assume k is aligned
    :param sf_packed: (scale_mn, scale_k/4) int32
    :return: (scale_mn, scale_k), float32
    """
    if len(sf_packed.shape) == 3:
        return torch.stack(
            [_inverse_transform_scale_ue8m0_impl(x) for x in sf_packed], dim=0
        )

    block_size = 128
    assert len(sf_packed.shape) == 2, f"{sf_packed.shape=}"
    assert sf_packed.dtype == torch.int32

    mn_repeat_128, k_div_4 = sf_packed.shape
    mn = mn_repeat_128 // block_size
    k = k_div_4 * 4

    # packed u8 -> fp32
    sf_u8 = sf_packed.contiguous().flatten().view(torch.uint8).view(mn_repeat_128, k)
    sf_fp32 = (sf_u8.to(torch.int32) << 23).view(torch.float32)

    # remove repeat
    sf_reshaped = sf_fp32.view(mn, block_size, k)
    sf_unrepeated = sf_reshaped[:, 0:1, :]
    if not torch.all(sf_unrepeated == sf_reshaped):
        from sglang.srt.debug_utils.dumper import get_tensor_info

        raise AssertionError(
            f"sf_unrepeated != sf_reshaped ({get_tensor_info(sf_unrepeated)=} {get_tensor_info(sf_reshaped)=})"
        )
    sf_unrepeated = sf_unrepeated.squeeze(1).contiguous()

    assert sf_unrepeated.shape == (mn, k)
    return sf_unrepeated


# COPIED FROM DeepGEMM
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_align(m, 128), ceil_align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


# COPIED FROM DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def channel_quant_to_tensor_quant(
    x_q_channel: torch.Tensor,
    x_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_dq_channel = x_q_channel.to(torch.float32) * x_s
    x_q_tensor, scale = (
        scaled_fp8_quant(x_dq_channel)
        if _is_cuda
        else input_to_float8(x_dq_channel, dtype=x_q_channel.dtype)
    )
    return x_q_tensor, scale


def _process_scaled_mm_output(output, input_2d_shape, output_shape):
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)


def _apply_fallback_scaled_mm(
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d_shape,
    output_shape,
    bias,
    input_dtype,
):
    global TORCH_DEVICE_IDENTITY
    if TORCH_DEVICE_IDENTITY is None:
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32, device=weight.device)

    output = torch._scaled_mm(
        qinput,
        weight,
        scale_a=TORCH_DEVICE_IDENTITY,
        scale_b=TORCH_DEVICE_IDENTITY,
        out_dtype=torch.float32,
    )

    output = _process_scaled_mm_output(output, input_2d_shape, output_shape)
    x_scale = torch.narrow(x_scale, 0, 0, input_2d_shape[0])

    output = output * x_scale * weight_scale.t()
    if bias is not None:
        output = output + bias
    return output.to(dtype=input_dtype)


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),
    use_per_token_if_dynamic: bool = False,
    pad_output: Optional[bool] = None,
    compressed_tensor_quant: bool = False,
) -> torch.Tensor:
    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:
        pad_output = (
            not get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE")
            and not cutlass_fp8_supported
        )
    output_padding = 17 if pad_output else None

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    if compressed_tensor_quant:
        # Maybe apply padding to output, see comment in __init__
        num_token_padding = output_padding
        if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
            num_token_padding = None
        qinput, x_scale = scaled_fp8_quant(
            input_2d,
            input_scale,
            num_token_padding=num_token_padding,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
        )
    else:
        # cutlass w8a8 fp8 sgl-kernel only supports per-token scale
        if input_scale is not None:
            assert input_scale.numel() == 1
            # broadcast per-tensor scale to per-token scale when supporting cutlass
            qinput, x_scale = static_quant_fp8(
                input_2d, input_scale, repeat_scale=cutlass_fp8_supported
            )
        else:
            # default use per-token quantization if dynamic
            if _is_cuda:
                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
            else:
                # TODO(kkhuang): temporarily enforce per-tensor activation scaling if weight is per-tensor scaling
                # final solution should be: 1. add support to per-tensor activation scaling.
                # 2. solve the torch.compile error from weight_scale.numel() == 1 and x_scale.numel() > 1 (below line#308)
                if _is_hip and weight_scale.numel() == 1:
                    qinput, x_scale = scaled_fp8_quant(
                        input_2d,
                        input_scale,
                        use_per_token_if_dynamic=use_per_token_if_dynamic,
                    )
                else:
                    qinput, x_scale = per_token_group_quant_fp8(
                        input_2d, group_size=input_2d.shape[1]
                    )

    if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
        # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
        # for sgl-kernel fp8_scaled_mm, it support per channel W now
        if VLLM_AVAILABLE and use_vllm_cutlass_w8a8_fp8_kernel:
            # Fall back to vllm cutlass w8a8 fp8 kernel
            output = ops.cutlass_scaled_mm(
                qinput,
                weight,
                out_dtype=input.dtype,
                scale_a=x_scale,
                scale_b=weight_scale,
                bias=bias,
            )
        else:
            cutlass_compatible_b = (
                weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
            )
            if not cutlass_compatible_b or use_triton_w8a8_fp8_kernel:
                # Massage the input to be 2D
                qinput = qinput.view(-1, qinput.shape[-1])
                output = triton_scaled_mm(
                    qinput, weight, x_scale, weight_scale, input.dtype, bias
                )
            else:
                output = fp8_scaled_mm(
                    qinput,
                    weight,
                    x_scale,
                    weight_scale,
                    out_dtype=input.dtype,
                    bias=bias,
                )
        return output.view(*output_shape)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    per_tensor_weights = weight_scale.numel() == 1
    per_tensor_activations = x_scale.numel() == 1

    if (
        use_per_token_if_dynamic
        and not per_tensor_weights
        and not per_tensor_activations
        and (USE_ROWWISE_TORCH_SCALED_MM or _use_aiter)
    ):
        # into this sector means use dynamic per-token-per-channel quant
        # per-token scale quant for input matrix, every row(one token) have one scale factor
        # per-channel scale quant for weight matrix, every col(one channel) have one scale factor
        if _use_aiter:
            # gemm_a8w8_bpreshuffle(XQ, WQ, x_scale, w_scale, dtype)
            # XQ -> input tensor, shape = (m, k)
            # WQ -> weight tensor, shape = (n, k), with preshuffe get better perf
            # x_scale -> input scale tensor, shape = (m, 1)
            # w_scale -> weight scale tensor, shape = (n ,1)
            # dtype -> output dtype
            output = gemm_a8w8_bpreshuffle(
                XQ=qinput,
                WQ=weight.T,
                x_scale=x_scale,
                w_scale=weight_scale,
                dtype=input.dtype,
            )
            if bias is not None:
                output += bias
            return _process_scaled_mm_output(output, input_2d.shape, output_shape)
        else:
            # For now validated on ROCm platform
            # fp8 rowwise scaling in torch._scaled_mm is introduced in
            # https://github.com/pytorch/pytorch/pull/144432 using hipBLASLt
            # and ROCm 6.3, which only exists in torch 2.7 and above.
            # For CUDA platform please validate if the
            # torch._scaled_mm support rowwise scaled GEMM
            # Fused GEMM_DQ Rowwise GEMM
            output = torch._scaled_mm(
                qinput,
                weight,
                out_dtype=input.dtype,
                scale_a=x_scale,
                scale_b=weight_scale.t(),
                bias=bias,
            )
            return _process_scaled_mm_output(output, input_2d.shape, output_shape)

    if per_tensor_weights and per_tensor_activations:
        # Fused GEMM_DQ
        output = torch._scaled_mm(
            qinput,
            weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
        )
        return _process_scaled_mm_output(output, input_2d.shape, output_shape)

    # Fallback for channelwise case, where we use unfused DQ
    # due to limitations with scaled_mm

    # Symmetric quantized GEMM by definition computes the following:
    #   C = (s_x * X) (s_w * W) + bias
    # This is equivalent to dequantizing the weights and activations
    # before applying a GEMM.
    #
    # In order to compute quantized operands, a quantized kernel
    # will rewrite the above like so:
    #   C = s_w * s_x * (X * W) + bias
    #
    # For the scaled_mm fallback case, we break this down, since it
    # does not support s_w being a vector.
    return _apply_fallback_scaled_mm(
        qinput,
        weight,
        x_scale,
        weight_scale,
        input_2d.shape,
        output_shape,
        bias,
        input.dtype,
    )


def can_auto_enable_marlin_fp8() -> bool:
    try:
        major, minor = get_device_capability()
        sm = major * 10 + minor
        return 80 <= sm < 89
    except Exception:
        return False


def apply_fp8_ptpc_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),
    use_per_token_if_dynamic: bool = False,
    pad_output: Optional[bool] = None,
    compressed_tensor_quant: bool = False,
) -> torch.Tensor:
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])

    # weight is transposed (K, N)
    output_shape = [*input.shape[:-1], weight.shape[1]]

    q_input, x_scale = aiter.per_token_quant_hip(input_2d, quant_dtype=aiter.dtypes.fp8)

    per_tensor_weights = (weight_scale.numel() == 1) and weight_scale.dim() < 2
    per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2

    if not (per_tensor_weights and per_tensor_activations):
        # weight is in (N, K)
        output_shape = [*input.shape[:-1], weight.shape[0]]

    output = aiter.gemm_a8w8_bpreshuffle(
        q_input, weight, x_scale, weight_scale, None, input.dtype
    )
    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


def validate_fp8_block_shape(
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    block_size: list[int],
) -> None:
    """Validate block quantization shapes for tensor parallelism."""
    from sglang.srt.distributed import get_tensor_model_parallel_world_size

    tp_size = getattr(layer, "tp_size", get_tensor_model_parallel_world_size())
    block_n, block_k = block_size[0], block_size[1]

    # Required by row parallel
    if (
        tp_size > 1
        and input_size // input_size_per_partition == tp_size
        and input_size_per_partition % block_k != 0
    ):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by weight quantization block_k = {block_k}."
        )

    # Required by column parallel or enabling merged weights
    is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size
    is_merged_gemm = len(output_partition_sizes) > 1
    if is_tp_split or is_merged_gemm:
        sizes_to_check = output_partition_sizes
        if not is_tp_split and is_merged_gemm:
            # In case of merged matrices, we allow the last
            # matrix to not be a multiple of block size
            sizes_to_check = output_partition_sizes[:-1]
        for output_partition_size in sizes_to_check:
            if output_partition_size % block_n != 0:
                raise ValueError(
                    f"Weight output_partition_size = "
                    f"{output_partition_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
