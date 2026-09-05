from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache, partial
from typing import Callable, List, Optional, Tuple, Union

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
    get_w8a8_channelwise_fp8_config,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_group_quant_fp8,
    sglang_per_token_group_quant_fp8_row_padded,
    sglang_per_token_quant_fp8,
    static_quant_fp8,
    triton_scaled_mm,
    w8a8_block_fp8_matmul_deepgemm,
    w8a8_block_fp8_matmul_triton,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_platform,
)
from sglang.srt.utils import (
    ceil_align,
    ceil_div,
    get_bool_env_var,
    get_cuda_version,
    get_device_capability,
    get_device_sm,
    get_hip_version,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_gfx1250_supported,
    is_hip,
    is_musa,
    is_xpu,
    offloader,
)
from sglang.srt.utils.common import torch_release
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_xpu = is_xpu()
_is_fp8_fnuz = is_fp8_fnuz()
_is_gfx95_supported = is_gfx95_supported()
_is_gfx1250_supported = is_gfx1250_supported()
_is_musa = is_musa()

# gfx1250 (RDNA4) cannot compile the AITER CK quant/GEMM kernels, and even when
# CK builds it lacks the MFMA/WMMA instructions those kernels rely on. Force the
# pure-triton block-fp8 path on gfx1250.
_use_aiter = (
    get_bool_env_var("SGLANG_USE_AITER") and _is_hip and not _is_gfx1250_supported
)
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
# ROCm 7.0 hipcc miscompiles gemm_a8w8_blockscale_bpreshuffle on gfx95 (#23319).
_use_aiter_bpreshuffle_gfx95 = _use_aiter_gfx95 and get_hip_version() >= (7, 2, 0)
# gfx95 + ROCm < 7.2: bpreshuffle CK is disabled (above), and the non-bpreshuffle
# fallback ck_gemm_a8w8_blockscale returns NaN above a per-shape M for some shapes
# (measured NaN onset: (2560,4096)@M>=4096, (4096,1024)@M>=8192 at TP8; at TP4 the
# attn proj (4608,4096)@M>=2048 and o_proj (4096,2048)@M>=4096), corrupting prefill.
# Map each affected (n, k) to the largest M for which CK is confirmed correct
# (conservative = last verified-safe M). Keep the faster CK path at/below that M and
# fall back to the numerically-correct Triton FP8 GEMM above it. Fixed in ROCm 7.2.
_AITER_GFX95_CK_W8A8_MAX_SAFE_M = {
    (2560, 4096): 2048,
    (4096, 1024): 4096,
    (4096, 2048): 2048,  # TP4 o_proj (TP8 shape was (4096, 1024))
    (4608, 4096): 512,  # TP4 attn qkv/gate proj: CK NaN at M>=2048 on ROCm 7.0
}


class _MXFP4QuantizedData(MXFP4QuantizeUtil):
    def __init__(
        self,
        original_shape: torch.Size,
        original_dtype: torch.dtype,
        quantized_data: torch.Tensor,
    ):
        self.original_shape = original_shape
        self.original_dtype = original_dtype
        self.quantized_data = quantized_data


# Force CK bpreshuffle (not Triton) for the dense w8a8-block GEMMs (MLA q/kv/o
# projections), to match ATOM (CK preshuffle; Triton FP8 blockscale is slower).
# Default OFF; DeepseekV4 enables it via set_force_ck_w8a8(True). The env var
# SGLANG_FORCE_CK_W8A8=1 still works as an override.
_FORCE_CK_W8A8: bool = False


def set_force_ck_w8a8(enabled: bool = True) -> None:
    global _FORCE_CK_W8A8
    _FORCE_CK_W8A8 = enabled


def materialize_bpreshuffle_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Materialize the physical scale layout consumed by gfx95 bpreshuffle GEMM."""
    return scale.t().contiguous().t() if scale.dim() == 2 else scale


def view_aiter_fused_rms_transposed_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Zero-copy view of a ``transpose_scale=True`` fp8 group scale.

    Producer-neutral counterpart of ``materialize_bpreshuffle_fp8_scale``. When an
    AITER quant/fused-RMS kernel is asked for ``transpose_scale=True`` it writes the
    per-token group scale directly in physical ``[num_groups, tokens]`` byte order
    behind a row-major-looking ``[tokens, num_groups]`` tensor. Swapping the strides
    restores logical ``[M, G]`` indexing over those same bytes -- i.e. the
    column-major layout the gfx95 bpreshuffle GEMM consumes -- with no copy. Callers
    that instead take the row-major (``transpose_scale=False``) path relayout via
    ``materialize_bpreshuffle_fp8_scale``; this is the bit-identical no-copy path.

    Only valid for M(tokens) >= 2. At M == 1 the ``[1, G]`` and ``[G, 1]`` byte
    orders coincide, so producers keep ``transpose_scale=False`` and materialize;
    the stride swap here would be a no-op on shape but is never taken at M == 1.
    Non-2-D scales (e.g. per-tensor) pass through unchanged.
    """
    if scale.dim() != 2:
        return scale
    return torch.as_strided(scale, scale.shape, (1, scale.shape[0]))


def materialize_bpreshuffle_fp8_scale_tuple(
    value: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Materialize the scale slot in FP8 ``(q_input, x_scale, ...)`` tuples."""
    return (
        value[0],
        materialize_bpreshuffle_fp8_scale(value[1]),
        *value[2:],
    )


def view_aiter_fused_rms_transposed_fp8_scale_tuple(
    value: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Zero-copy scale reinterpret for FP8 ``(q_input, x_scale, ...)`` tuples."""
    return (value[0], view_aiter_fused_rms_transposed_fp8_scale(value[1]), *value[2:])


def emit_transposed_bpreshuffle_scale(m: int, *, on_bpreshuffle_gfx95: bool) -> bool:
    """Whether a producer should emit its fp8 scale already transposed.

    Producer sites choose between two equivalent gfx95 bpreshuffle scale layouts:
    ``transpose_scale=True`` + zero-copy ``view_aiter_fused_rms_transposed_fp8_scale`` (this
    predicate True), or row-major ``transpose_scale=False`` +
    ``materialize_bpreshuffle_fp8_scale`` (this predicate False). The transposed
    zero-copy path is only taken on gfx95 bpreshuffle and only for M(tokens) >= 2:
    at M == 1 the ``[1, G]`` and ``[G, 1]`` byte orders coincide, so the transposed
    emit buys nothing and the materialize path is used. Centralizes the gate shared
    by the MoE-down and MLA o_proj producer sites.
    """
    return on_bpreshuffle_gfx95 and m >= 2


def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) -> bool:
    if _FORCE_CK_W8A8:
        return False
    return (n, k) in [
        (1024, 8192),
        (16384, 1536),
        (2112, 7168),
        (3072, 1536),
        (32768, 8192),
        (4096, 7168),
        (4608, 7168),
        (512, 7168),
        (7168, 2048),
        (7168, 2304),
        (7168, 16384),
        (7168, 256),
        (8192, 1024),
        (8192, 32768),
    ]


if _use_aiter:
    import aiter
    from aiter import gemm_a8w8_blockscale as ck_gemm_a8w8_blockscale
    from aiter import (
        gemm_a8w8_blockscale_bpreshuffle,
        gemm_a8w8_bpreshuffle,
        get_hip_quant,
    )
    from aiter.ops.triton.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
    )

    aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)


if _is_cuda:
    from sglang.kernels.ops.gemm import fp8_scaled_mm
    from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
    from sglang.srt.utils.patch_torch import register_fake_if_exists

    @register_fake_if_exists("sgl_kernel::fp8_scaled_mm")
    def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
        # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].
        M = mat_a.shape[-2]
        N = mat_b.shape[-1]
        return mat_a.new_empty((M, N), dtype=out_dtype)

    from flashinfer import bmm_fp8 as _raw_bmm_fp8_batched

    @register_custom_op(op_name="flashinfer_bmm_fp8_batched", mutates_args=["out"])
    def _bmm_fp8_batched_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8_batched(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched (3D) per-tensor-scale FP8 matmul, via flashinfer's cuBLAS backend."""
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_batched_op(A, B, out, A_scale, B_scale)
        return out


use_triton_w8a8_fp8_kernel = get_bool_env_var("USE_TRITON_W8A8_FP8_KERNEL")

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def use_rowwise_torch_scaled_mm():
    if _is_hip:
        # The condition to determine if it is on a platform that supports
        # torch._scaled_mm rowwise feature.
        # The condition is determined once as the operations
        # are time consuming.
        return get_device_capability() >= (9, 4) and torch_release >= (2, 7)
    return False


USE_ROWWISE_TORCH_SCALED_MM = use_rowwise_torch_scaled_mm()


@lru_cache(maxsize=1)
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
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_DEEPGEMM = "flashinfer_deepgemm"
    CUTLASS = "cutlass"
    DEEP_GEMM = "deep_gemm"
    TRITON = "triton"
    AITER = "aiter"

    def is_auto(self) -> bool:
        return self == Fp8GemmRunnerBackend.AUTO

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_deepgemm(self) -> bool:
        return self == Fp8GemmRunnerBackend.FLASHINFER_DEEPGEMM

    def is_cutlass(self) -> bool:
        return self == Fp8GemmRunnerBackend.CUTLASS

    def is_deep_gemm(self) -> bool:
        return self == Fp8GemmRunnerBackend.DEEP_GEMM

    def is_triton(self) -> bool:
        return self == Fp8GemmRunnerBackend.TRITON

    def is_aiter(self) -> bool:
        return self == Fp8GemmRunnerBackend.AITER


class Mxfp8DenseGemmBackend(Enum):
    """Enum for MXFP8 dense linear backend selection, resolved separately from
    `Fp8GemmRunnerBackend`."""

    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    DEEP_GEMM = "deep_gemm"
    GFX95_DOT_SCALED = "gfx95_dot_scaled"
    UNSUPPORTED = "unsupported"

    def is_flashinfer_cutlass(self) -> bool:
        return self == Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_trtllm(self) -> bool:
        return self == Mxfp8DenseGemmBackend.FLASHINFER_TRTLLM

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def is_deep_gemm(self) -> bool:
        return self == Mxfp8DenseGemmBackend.DEEP_GEMM

    def is_gfx95_dot_scaled(self) -> bool:
        return self == Mxfp8DenseGemmBackend.GFX95_DOT_SCALED

    def is_unsupported(self) -> bool:
        return self == Mxfp8DenseGemmBackend.UNSUPPORTED


FP8_GEMM_RUNNER_BACKEND: Fp8GemmRunnerBackend | None = None


@lru_cache(maxsize=1)
def flashinfer_per_tensor_fp8_supported() -> bool:
    return is_flashinfer_available() and (
        get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
    )


if flashinfer_per_tensor_fp8_supported():
    from flashinfer import bmm_fp8 as _raw_flashinfer_bmm_fp8

    @register_custom_op(
        op_name="flashinfer_bmm_fp8",
        mutates_args=[],
        fake_impl=lambda q_input, weight, x_scale, weight_scale, out_dtype: (
            q_input.new_empty((q_input.shape[0], weight.shape[1]), dtype=out_dtype)
        ),
    )
    def flashinfer_bmm_fp8(
        q_input: torch.Tensor,  # [M, K] fp8 e4m3
        weight: torch.Tensor,  # [K, N] fp8 e4m3, column-major
        x_scale: torch.Tensor,  # per-tensor scalar
        weight_scale: torch.Tensor,  # per-tensor scalar
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        m, n = q_input.shape[0], weight.shape[1]
        return _raw_flashinfer_bmm_fp8(
            q_input.unsqueeze(0),
            weight.unsqueeze(0),
            x_scale.reshape(1),
            weight_scale.reshape(1),
            out_dtype,
            backend="cublas",
        ).view(m, n)


def _fake_flashinfer_mxfp8_quantize(
    input: torch.Tensor,
    _is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    backend: str = "cute-dsl",
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = input.numel() // input.shape[-1]
    k_aligned = ((input.shape[-1] + alignment - 1) // alignment) * alignment
    q_input = input.new_empty((m, k_aligned), dtype=torch.float8_e4m3fn)
    sf_columns = k_aligned // 32
    if _is_sf_swizzled_layout:
        padded_rows = ((m + 127) // 128) * 128
        padded_sf_columns = ((sf_columns + 3) // 4) * 4
        scale_size = padded_rows * padded_sf_columns
    else:
        scale_size = m * sf_columns
    scale = input.new_empty((scale_size,), dtype=torch.uint8)
    return q_input, scale


if get_platform().is_blackwell and is_flashinfer_available():
    from flashinfer import SfLayout
    from flashinfer import mm_mxfp8 as _raw_flashinfer_mm_mxfp8
    from flashinfer import mxfp8_quantize as _raw_flashinfer_mxfp8_quantize
    from flashinfer.gemm import gemm_fp8_nt_groupwise as _raw_gemm_fp8_nt_groupwise

    @lru_cache(maxsize=1)
    def _get_flashinfer_groupwise_backend() -> str:
        if get_fp8_gemm_runner_backend().is_flashinfer_cutlass():
            return "cutlass"
        if get_fp8_gemm_runner_backend().is_flashinfer_trtllm():
            return "trtllm"

        major, minor = get_device_capability()
        # SM120/121: CUTLASS only.
        # SM100/103: TRTLLM only.
        if major >= 12:
            return "cutlass"
        return "trtllm"

    # Wrap gemm_fp8_nt_groupwise as a custom op so torch.compile does not trace
    # into flashinfer's JIT compilation code (pathlib/cubin_loader ops).
    @register_custom_op(
        op_name="flashinfer_gemm_fp8_nt_groupwise",
        mutates_args=[],
        fake_impl=lambda q_input, weight, x_scale, weight_scale, out_dtype: (
            q_input.new_empty((q_input.shape[0], weight.shape[0]), dtype=out_dtype)
        ),
    )
    def gemm_fp8_nt_groupwise(
        q_input: torch.Tensor,
        weight: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        backend = _get_flashinfer_groupwise_backend()
        if backend == "cutlass":
            # FlashInfer CUTLASS groupwise kernel requires contiguous scale tensors
            x_scale = x_scale.contiguous()
            weight_scale = weight_scale.contiguous()
            return _raw_gemm_fp8_nt_groupwise(
                q_input,
                weight,
                x_scale,
                weight_scale,
                out_dtype=out_dtype,
                backend="cutlass",
                scale_major_mode="MN",
            )
        return _raw_gemm_fp8_nt_groupwise(
            q_input,
            weight,
            x_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend=backend,
        )

    # Wrap MXFP8 ops as custom ops so torch.compile does not trace into
    # flashinfer's JIT compilation path (filesystem checks/cubin loader).
    @register_custom_op(
        op_name="flashinfer_mxfp8_quantize",
        mutates_args=[],
        fake_impl=_fake_flashinfer_mxfp8_quantize,
    )
    def flashinfer_mxfp8_quantize(
        input: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        alignment: int = 32,
        backend: str = "cute-dsl",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _raw_flashinfer_mxfp8_quantize(
            input,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment,
            backend=backend,
            sf_swizzle_layout=(
                SfLayout.layout_128x4
                if is_sf_swizzled_layout
                else SfLayout.layout_linear
            ),
        )

    @register_custom_op(
        op_name="flashinfer_mm_mxfp8",
        mutates_args=[],
        fake_impl=lambda q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False, backend="auto": (
            q_input.new_empty((q_input.shape[0], weight_t.shape[1]), dtype=out_dtype)
        ),
    )
    def flashinfer_mm_mxfp8(
        q_input: torch.Tensor,
        weight_t: torch.Tensor,
        x_scale_u8: torch.Tensor,
        weight_scale_t: torch.Tensor,
        out_dtype: torch.dtype,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
    ) -> torch.Tensor:
        return _raw_flashinfer_mm_mxfp8(
            q_input,
            weight_t,
            x_scale_u8,
            weight_scale_t,
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )


if get_platform().is_sm90 and is_flashinfer_available():
    # FlashInfer SM90 DeepGEMM with automatic swapAB optimization for small M
    from flashinfer.gemm import fp8_blockscale_gemm_sm90


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


def resolve_mxfp8_dense_gemm_backend() -> Mxfp8DenseGemmBackend:
    """Pick the MXFP8 dense linear backend, honoring `--fp8-gemm-backend` only when it
    names a backend that owns an MXFP8 dense kernel."""
    backend = get_fp8_gemm_runner_backend()

    if backend.is_flashinfer_trtllm():
        if not (get_platform().is_sm100 and is_flashinfer_available()):
            raise RuntimeError(
                "MXFP8 dense GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but that kernel requires SM100/SM103 GPUs and FlashInfer."
            )
        return Mxfp8DenseGemmBackend.FLASHINFER_TRTLLM

    if backend.is_flashinfer_cutedsl():
        if not (
            get_platform().is_blackwell
            and is_flashinfer_available()
            and _raw_flashinfer_mm_mxfp8.is_backend_supported(
                "cute-dsl", get_device_sm()
            )
        ):
            raise RuntimeError(
                "MXFP8 dense GEMM requested via --fp8-gemm-backend=flashinfer_cutedsl, "
                "but that kernel requires an SM100/SM103 GPU and FlashInfer."
            )
        return Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL

    if backend.is_flashinfer_cutlass():
        if not (get_platform().is_blackwell and is_flashinfer_available()):
            raise RuntimeError(
                "MXFP8 dense GEMM requested via --fp8-gemm-backend=flashinfer_cutlass, "
                "but that kernel requires Blackwell GPUs and FlashInfer."
            )
        return Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS

    if backend.is_deep_gemm():
        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise RuntimeError(
                "MXFP8 dense GEMM requested via --fp8-gemm-backend=deep_gemm, but "
                "DeepGEMM is not available (package missing or "
                "SGLANG_ENABLE_JIT_DEEPGEMM=0)."
            )
        return Mxfp8DenseGemmBackend.DEEP_GEMM

    if _is_hip and _is_gfx95_supported:
        return Mxfp8DenseGemmBackend.GFX95_DOT_SCALED

    if get_platform().is_blackwell and is_flashinfer_available():
        if _raw_flashinfer_mm_mxfp8.is_backend_supported("cute-dsl", get_device_sm()):
            return Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        return Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS

    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return Mxfp8DenseGemmBackend.DEEP_GEMM

    return Mxfp8DenseGemmBackend.UNSUPPORTED


def _unsupported_mxfp8_linear(*args, **kwargs) -> torch.Tensor:
    raise RuntimeError(
        "No MXFP8 dense GEMM kernel is available on this device. MXFP8 dense linear "
        "requires Blackwell (SM100/SM103/SM110/SM120) with FlashInfer, Hopper (SM90) "
        "with DeepGEMM, or ROCm gfx95."
    )


def dispatch_w8a8_mxfp8_linear() -> Callable:
    backend = resolve_mxfp8_dense_gemm_backend()
    if backend.is_deep_gemm():
        return _deepgemm_w8a8_mxfp8_linear_with_fallback
    elif backend.is_flashinfer_trtllm():
        return partial(flashinfer_mxfp8_blockscaled_linear, backend="trtllm")
    elif backend.is_flashinfer_cutlass():
        return partial(flashinfer_mxfp8_blockscaled_linear, backend="cutlass")
    elif backend.is_flashinfer_cutedsl():
        return partial(flashinfer_mxfp8_blockscaled_linear, backend="cute-dsl")
    elif backend.is_unsupported():
        return _unsupported_mxfp8_linear

    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        dot_scaled_mxfp8_blockscaled_linear,
    )

    return dot_scaled_mxfp8_blockscaled_linear


def _deepgemm_w8a8_mxfp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    weight_scale_swizzled: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
        w8a8_mxfp8_matmul_deepgemm,
    )

    assert input_scale is None
    output_dtype = input.dtype

    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0
    dtype_supported = output_dtype == torch.bfloat16

    if not (shape_supported and dtype_supported):
        if weight_scale_swizzled is None:
            raise RuntimeError(
                f"DeepGEMM cannot serve this MXFP8 GEMM ({shape_supported=}, "
                f"{dtype_supported=}) and no FlashInfer fallback scale was prepared "
                "for this layer. Re-run with --fp8-gemm-backend=flashinfer_cutlass."
            )
        return flashinfer_mxfp8_blockscaled_linear(
            input=input,
            weight=weight,
            weight_scale=weight_scale_swizzled,
            input_scale=input_scale,
            bias=bias,
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d,
        32,
        column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )

    # weight_scale format is set per-backend by _process_mxfp8_linear_weight_scale
    # (int32 packed TMA-aligned on Blackwell, float32 on Hopper); NOT uint8 — Triton form is routed to the fallback above.

    output = w8a8_mxfp8_matmul_deepgemm(
        q_input, weight, x_scale, weight_scale, output_dtype=output_dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


def _dispatch_explicit_backend(backend: Fp8GemmRunnerBackend) -> Callable:
    """Dispatch based on explicitly selected backend."""
    if backend.is_flashinfer_trtllm():
        if not (get_platform().is_sm100 and is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer TRTLLM FP8 GEMM requires SM100/SM103 GPUs and FlashInfer."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_flashinfer_cutlass():
        if not (get_platform().is_blackwell and is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_cutlass, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer CUTLASS FP8 GEMM requires Blackwell GPUs and FlashInfer."
            )
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_flashinfer_deepgemm():
        if not (get_platform().is_sm90 and is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer DeepGEMM with swapAB requested via --fp8-gemm-backend=flashinfer_deepgemm, "
                "but it's not available. This backend requires Hopper (SM90) GPUs and FlashInfer "
                "to be installed."
            )
        return flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_cutlass():
        if not get_platform().is_sm120:
            raise RuntimeError(
                "--fp8-gemm-backend=cutlass is deprecated on this hardware. "
                "Please switch to DeepGEMM or FlashInfer TRTLLM on SM90/SM100."
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
    # 3. CUTLASS (if SM120 GPU and CUDA 12.8+)
    # 4. AITER (if AMD GPU with AITER enabled)
    # 5. Triton (fallback)

    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return deepgemm_w8a8_block_fp8_linear_with_fallback
    elif get_platform().is_blackwell and is_flashinfer_available():
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
    elif get_platform().is_sm120:
        return cutlass_w8a8_block_fp8_linear_with_fallback
    elif _use_aiter:
        return aiter_w8a8_block_fp8_linear
    else:
        return triton_w8a8_block_fp8_linear


def initialize_fp8_gemm_config() -> None:
    """Initialize FP8 GEMM configuration."""
    global FP8_GEMM_RUNNER_BACKEND

    backend = get_exec().kernel.fp8_gemm_runner_backend
    if backend == "auto" and get_platform().is_sm120:
        backend = "cutlass"

    backend = Fp8GemmRunnerBackend(backend)

    FP8_GEMM_RUNNER_BACKEND = backend


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

    input_2d = input.view(-1, input.shape[-1])
    backend = _get_flashinfer_groupwise_backend()
    # Fall back to triton for non-supported formats.
    # TODO: Check if flashinfer supports other output dtypes besides bf16.
    if backend == "trtllm" and (
        input_2d.shape[1] < 256 or input_2d.dtype != torch.bfloat16
    ):
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    output_shape = [*input.shape[:-1], weight.shape[0]]

    # TRTLLM uses the existing SGLang column-major scale layout.
    # CUTLASS with scale_major_mode="MN" expects (k//block_k, m), so we normalize below.
    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=(backend == "trtllm")
    )
    if backend == "cutlass":
        block_n, block_k = block_size
        m, k = input_2d.shape
        n = weight.shape[0]
        expected_x_scale_shape = (k // block_k, m)
        expected_weight_scale_shape = (k // block_k, n // block_n)
        if x_scale.shape == (m, k // block_k):
            x_scale = x_scale.transpose(-1, -2).contiguous()
        if weight_scale.shape == (n // block_n, k // block_k):
            weight_scale = weight_scale.transpose(-1, -2).contiguous()
        assert x_scale.shape == expected_x_scale_shape, (
            "FlashInfer CUTLASS groupwise FP8 expects A scale layout "
            f"(k//block_k, m) for scale_major_mode='MN', got {tuple(x_scale.shape)}; "
            f"expected {expected_x_scale_shape}. "
            f"strides={x_scale.stride()} is_contiguous={x_scale.is_contiguous()} "
            f"m={m} n={n} k={k} block_size={block_size}"
        )
        assert weight_scale.shape == expected_weight_scale_shape, (
            "FlashInfer CUTLASS groupwise FP8 expects B scale layout "
            f"(k//block_k, n//block_n) for scale_major_mode='MN', got {tuple(weight_scale.shape)}; "
            f"expected {expected_weight_scale_shape}. "
            f"strides={weight_scale.stride()} is_contiguous={weight_scale.is_contiguous()} "
            f"m={m} n={n} k={k} block_size={block_size}"
        )
        assert x_scale.dtype == torch.float32, (
            "FlashInfer CUTLASS groupwise FP8 expects x_scale dtype float32, "
            f"got {x_scale.dtype}."
        )
        assert weight_scale.dtype == torch.float32, (
            "FlashInfer CUTLASS groupwise FP8 expects weight_scale dtype float32, "
            f"got {weight_scale.dtype}."
        )
    # TRTLLM path continues using the original quantized scale layout.
    output = gemm_fp8_nt_groupwise(
        q_input,
        weight,
        x_scale,
        weight_scale,
        out_dtype=input_2d.dtype,
    )

    if bias is not None:
        output += bias

    return output.to(dtype=input_2d.dtype).view(*output_shape)


def flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashInfer DeepGEMM backend for SM90 (Hopper) with swapAB optimization.

    Uses flashinfer.gemm.fp8_blockscale_gemm_sm90 which automatically selects
    the swapAB kernel for small M dimensions (M < 32) for better performance
    during decoding/low batch size scenarios.

    For SM90 (Hopper), this uses the DeepGEMM JIT with automatic swapAB selection.
    """
    assert input_scale is None

    output_dtype = input.dtype
    dtype_supported = output_dtype == torch.bfloat16

    # fp8_blockscale_gemm_sm90 requires: N % 64 == 0, K % 128 == 0
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0

    # Keep this backend to 1 <= M < 32, mirroring vLLM's
    # FlashInferFp8DeepGEMMDynamicBlockScaledKernel. fp8_blockscale_gemm_sm90 is
    # one entry point over two kernels and only the M < 32 swapAB half is worth
    # taking:
    #   M >= 32 picks the non-swapAB kernel, which is slower than DeepGEMM (worst
    #     just above the threshold) and, on some checkpoints, less accurate.
    #   M == 0 hard-fails inside the kernel ("Check failed: (input_ptr !=
    #     nullptr)"). Empty batches are a normal steady-state input, not an edge
    #     case: DP attention hands an idle rank a zero-token forward so the
    #     collectives stay in sync (ScheduleBatch.prepare_for_idle).
    # Same shape of guard as the gfx95 CK M bound below.
    m_supported = 1 <= input.view(-1, input.shape[-1]).shape[0] < 32

    if not m_supported and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        # DeepGEMM covers both ends and falls back to triton on its own for
        # shapes it cannot serve.
        return deepgemm_w8a8_block_fp8_linear_with_fallback(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    if not (shape_supported and dtype_supported and m_supported):
        if weight_scale.dtype == torch.int32:
            weight_scale = _unpack_ue8m0_scale_for_triton(
                weight_scale, weight.shape, block_size
            )
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    # - input: (M, K) BF16 or FP8
    # - weight: (N, K) FP8 with weight_scale
    # - weight_scale: (N, K//128) for per-token or (N//128, K//128) for per-block

    output = fp8_blockscale_gemm_sm90(
        input_2d,
        weight,
        input_scale=None,  # BF16 input, internal quantization
        weight_scale=weight_scale,
        out_dtype=output_dtype,
    )

    if bias is not None:
        output += bias
    return output.view(*output_shape)


def cutlass_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # TODO: add more robust shape check here
    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0

    if input_scale is not None:
        # Pre-quantized activation (SGLANG_OPT_MOE_QUANT_ONCE): ``input`` is
        # the fp8 per-token-group-128 q (rows possibly padded to a multiple
        # of 4), ``input_scale`` the matching column-major scales
        # (stride(0) == 1). Output keeps the (padded) row count; the caller
        # slices back to the true token count.
        assert shape_supported, (
            "pre-quantized fp8 input requires cutlass-supported weight shapes "
            f"(got {tuple(weight.shape)})"
        )
        assert input.dtype == torch.float8_e4m3fn
        input_2d = input.view(-1, input.shape[-1])
        output = fp8_blockwise_scaled_mm(
            input_2d, weight.T, input_scale, weight_scale.T, out_dtype=torch.bfloat16
        )
        if bias is not None:
            output += bias
        return output.view(*input.shape[:-1], weight.shape[0])

    if not shape_supported:
        # fallback to triton
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    # Quantize into row-padded buffers so the sgl-kernel wrapper's per-call
    # pad_tensor() on mat_a / scales_a short-circuits (saves 2x fill + 2x cat
    # kernels per GEMM). weight_scale.T is left as a K-major view because the
    # kernel requires scales_b.stride(0) == 1 and materializes it internally.
    q_input, x_scale = sglang_per_token_group_quant_fp8_row_padded(
        input_2d, block_size[1]
    )
    output = fp8_blockwise_scaled_mm(
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=input_2d.dtype
    )
    if output.shape[0] != input_2d.shape[0]:
        # GEMM ran on the row-padded buffer; drop the padding rows.
        output = output[: input_2d.shape[0]]
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
    if input_scale is not None:
        # Pre-quantized activation (SGLANG_OPT_MOE_QUANT_ONCE): ``input`` is
        # the fp8 per-token-group-128 q with rows padded to a multiple of 4
        # and ``input_scale`` the matching column-major fp32 scales
        # (stride == (1, padded_rows)) -- identical to the MN-major
        # TMA-aligned layout this path's own quant would produce below.
        # Output keeps the padded row count; the caller slices back.
        # UE8M0 packed scales (Blackwell DeepGEMM) use a different layout;
        # the caller gates on it.
        assert not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        assert input.dtype == torch.float8_e4m3fn
        assert weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0, (
            "pre-quantized fp8 input requires DeepGEMM-supported weight shapes "
            f"(got {tuple(weight.shape)})"
        )
        input_2d = input.view(-1, input.shape[-1])
        output = w8a8_block_fp8_matmul_deepgemm(
            input_2d,
            weight,
            input_scale,
            weight_scale,
            block_size,
            output_dtype=torch.bfloat16,
        )
        if bias is not None:
            output += bias
        return output.view(*input.shape[:-1], weight.shape[0])

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

    if not _is_musa:
        q_input, x_scale = sglang_per_token_group_quant_fp8(
            input_2d,
            block_size[1],
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
    else:
        q_input, x_scale = sglang_per_token_group_quant_fp8(
            input_2d,
            block_size[1],
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

    n, k = weight.shape

    if _use_aiter_bpreshuffle_gfx95:
        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)
    elif _use_aiter_gfx95:
        # gfx95 on ROCm < 7.2: keep the (faster) CK path at/below the per-shape
        # CK-safe M bound; above it, ck_gemm_a8w8_blockscale returns NaN, so use
        # Triton. Unlisted shapes keep their original decision. Fixed in ROCm 7.2.
        _ck_safe_m = _AITER_GFX95_CK_W8A8_MAX_SAFE_M.get((n, k))
        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k) or (
            _ck_safe_m is not None and input_2d.shape[0] > _ck_safe_m
        )
    else:
        use_triton = True

    # if input_scale not None, input is quanted
    if input_scale is not None:
        q_input = input_2d
        x_scale = input_scale
        if _use_aiter_bpreshuffle_gfx95 and not use_triton:
            x_scale = materialize_bpreshuffle_fp8_scale(x_scale)
        # On ROCm >= 7.2, scale is in bpreshuffle's transposed layout.
        # Triton needs a row-major view, so adjust strides only. No copy.
        elif use_triton and _use_aiter_bpreshuffle_gfx95:
            x_scale = view_aiter_fused_rms_transposed_fp8_scale(x_scale)
    else:
        materialize_bpreshuffle_scale = _use_aiter_bpreshuffle_gfx95 and not use_triton
        # No-copy bpreshuffle scale: emit it already transposed and stride-reinterpret
        # to the column-major bpreshuffle layout, instead of a .t().contiguous().t()
        # copy. Bit-identical for M>=2; M==1 keeps materialize (there the [1,G] and
        # [G,1] byte orders coincide, so materialize is a no-op view anyway).
        emit_bpreshuffle_scale = (
            materialize_bpreshuffle_scale and input_2d.shape[0] >= 2
        )
        q_input, x_scale = aiter_per1x128_quant(
            input_2d,
            quant_dtype=aiter.dtypes.fp8,
            transpose_scale=emit_bpreshuffle_scale,
        )
        if emit_bpreshuffle_scale:
            x_scale = view_aiter_fused_rms_transposed_fp8_scale(x_scale)
        elif materialize_bpreshuffle_scale:
            x_scale = materialize_bpreshuffle_fp8_scale(x_scale)

    if use_triton:
        gemm_a8w8_blockscale_op = triton_gemm_a8w8_blockscale
    elif _use_aiter_bpreshuffle_gfx95:
        gemm_a8w8_blockscale_op = gemm_a8w8_blockscale_bpreshuffle
    else:
        gemm_a8w8_blockscale_op = ck_gemm_a8w8_blockscale

    output = gemm_a8w8_blockscale_op(
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
    if input_scale is not None:
        # Pre-quantized input: ``input`` is already fp8 and ``input_scale`` is
        # its per-group scale (row-major (M, cdiv(K, 128))). Produced on HIP by
        # fused act/rmsnorm+quant ops (e.g. fused_clamp_act_mul) that feed the
        # GEMM directly. Skip re-quantization and emit bf16.
        q_input = input.view(-1, input.shape[-1])
        x_scale = input_scale
        output_dtype = torch.bfloat16
        output_shape = [*input.shape[:-1], weight.shape[0]]
    else:
        input_2d = input.view(-1, input.shape[-1])
        output_dtype = input_2d.dtype
        output_shape = [*input.shape[:-1], weight.shape[0]]
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False
        )

    output = w8a8_block_fp8_matmul_triton(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


@lru_cache(maxsize=1)
def _get_triton_mxfp8_downcast():
    try:
        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    except Exception as err:
        raise RuntimeError(
            "MXFP8 quantization requires triton_kernels with MXFP8 support."
        ) from err
    return downcast_to_mxfp


def mxfp8_group_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D contiguous tensor to MXFP8 with UE8M0 scales per group (32)."""
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    assert x.is_contiguous(), "MXFP8 quantization requires a contiguous 2D tensor."
    _, k = x.shape
    assert k % 32 == 0, f"{k=} must be divisible by 32"
    if _is_hip and _is_gfx95_supported:
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            mxfp8_e4m3_quantize,
        )

        return mxfp8_e4m3_quantize(x)
    downcast_to_mxfp = _get_triton_mxfp8_downcast()
    q_input, scale_u8 = downcast_to_mxfp(x, torch.float8_e4m3fn, axis=1)
    return q_input.contiguous(), scale_u8.contiguous()


def flashinfer_mxfp8_blockscaled_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    backend: str = "cutlass",
) -> torch.Tensor:
    """MXFP8 dense linear via FlashInfer mm_mxfp8. `weight_scale` must be the layout
    the backend expects, prepared at load time."""
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    k = input_2d.shape[1]
    k_w = weight.shape[1]
    if k != k_w:
        raise ValueError(f"Input K={k} does not match weight K={k_w}.")
    if k % 32 != 0:
        raise ValueError(f"K={k} must be divisible by 32 for MXFP8.")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError("MXFP8 weight must be FP8 E4M3.")

    if input_scale is None:
        q_input, x_scale_u8 = flashinfer_mxfp8_quantize(
            input_2d, is_sf_swizzled_layout=True, alignment=32
        )
    else:
        q_input = input_2d
        x_scale_u8 = input_scale

    if output_dtype is None:
        if input_2d.dtype in (torch.float16, torch.bfloat16, torch.float32):
            output_dtype = input_2d.dtype
        else:
            output_dtype = torch.bfloat16

    if backend == "trtllm":
        weight_scale_t = weight_scale.view(-1)
    else:
        weight_scale_t = weight_scale.t() if weight_scale.ndim == 2 else weight_scale

    output = flashinfer_mm_mxfp8(
        q_input,
        weight.t(),
        x_scale_u8,
        weight_scale_t,
        out_dtype=output_dtype,
        use_8x4_sf_layout=False,
        backend=backend,
    )

    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)


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

    # NOTE: This is very memory inefficient, results in *16384 memory requirement for scales
    # with block_size = [128, 128].
    # ... n_scale k_scale -> ... (n_scale block_n) (k_scale block_k)
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]

    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)


def quantize_block_fp8_weight_to_mxfp4(
    fp8_weight: torch.Tensor,
    fp8_scale: torch.Tensor,
    weight_block_size: List[int],
    mxfp4_block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_weight_dequant = block_quant_dequant(
        fp8_weight,
        fp8_scale.to(torch.float32),
        weight_block_size,
        torch.bfloat16,
    )
    fp4_weight, fp4_scale = _MXFP4QuantizedData.quantize(
        fp8_weight_dequant, block_size=mxfp4_block_size
    )
    fp4_weight = fp4_weight.quantized_data
    fp4_weight = fp4_weight.contiguous().view(torch.int8)
    fp4_scale = fp4_scale.view(
        *fp8_weight_dequant.shape[:-1],
        fp8_weight_dequant.shape[-1] // mxfp4_block_size,
    )
    return fp4_weight, fp4_scale.contiguous().view(torch.float8_e8m0fnu)


def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):
    assert isinstance(weight, torch.nn.Parameter)
    assert isinstance(weight_scale_inv, torch.nn.Parameter)

    new_weight, new_weight_scale_inv = requant_weight_ue8m0(
        weight.to(weight_scale_inv.device), weight_scale_inv, weight_block_size
    )

    offloader.update_param(weight, new_weight)
    weight_scale_inv.data = new_weight_scale_inv


def requant_block_scale_ue8m0_for_deepgemm(
    weight: torch.nn.Parameter,
    weight_scale: torch.nn.Parameter,
    weight_block_size: Optional[List[int]],
    use_deepgemm_runner: bool,
    output_dtype: Optional[torch.dtype] = None,
    weight_shape=None,
) -> bool:
    """Requantize block-FP8 weight scales to UE8M0 in place for DeepGEMM.

    No-op (returns False) unless the caller selected the DeepGEMM runner, the
    block size is 128x128 (the only layout the requant kernel supports), the
    scales are not already UE8M0, and DeepGEMM can run the layer (bf16 output,
    aligned shape). Returns True when it requantizes.
    """
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0

    if (
        not use_deepgemm_runner
        or weight_block_size != [128, 128]
        or getattr(weight_scale, "format_ue8m0", False)
        or not should_deepgemm_weight_requant_ue8m0(
            weight_block_size=weight_block_size,
            output_dtype=output_dtype,
            weight_shape=weight_shape,
        )
    ):
        return False

    requant_weight_ue8m0_inplace(weight, weight_scale, weight_block_size)
    weight_scale.format_ue8m0 = True
    return True


def requant_weight_ue8m0(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]

    # 3D+ weights stack multiple experts (e.g. MoE); requant each group separately.
    # 2D weights are a single matrix and fall through to the direct path below.
    if weight.dim() > 2:
        return _requant_weight_ue8m0_grouped(
            weight, weight_scale_inv, weight_block_size
        )

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


def _requant_weight_ue8m0_grouped(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    weight_block_size: List[int],
):
    *group_dims, n, k = weight.shape
    w_groups = weight.reshape(-1, n, k)
    s_groups = weight_scale_inv.reshape(-1, *weight_scale_inv.shape[-2:])
    num_groups = w_groups.shape[0]

    out_w = None
    out_s = None
    for g in range(num_groups):
        weight_dequant = block_quant_dequant(
            w_groups[g],
            s_groups[g],
            weight_block_size,
            torch.bfloat16,
        )
        w_g, s_g = quant_weight_ue8m0(
            weight_dequant=weight_dequant,
            weight_block_size=weight_block_size,
        )
        if out_w is None:
            out_w = torch.empty(
                (num_groups, *w_g.shape), dtype=w_g.dtype, device=w_g.device
            )
            out_s = torch.empty(
                (num_groups, *s_g.shape), dtype=s_g.dtype, device=s_g.device
            )
        out_w[g] = w_g
        out_s[g] = s_g

    out_w = out_w.view(*group_dims, n, k)
    out_s = out_s.view(*group_dims, *out_s.shape[-2:])

    out_s = transform_scale_ue8m0(out_s, mn=n)

    return out_w, out_s


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]
    assert weight_dequant.dtype == torch.bfloat16, (
        f"{weight_dequant.dtype=} {weight_dequant.shape=}"
    )

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

    # In sgl-deep-gemm, the C++ deepgemm path returns through DLPack which collapses the stride
    # of size-1 trailing dims to 1 (happens when packed_sf_k == 1, i.e.
    # K <= block_k * 4). Restore the TMA-aligned stride so the deepgemm
    # assertion sf.stride(-1) == get_tma_aligned_size(mn, element_size) holds.
    if not use_torch_impl and sf.shape[-1] == 1:
        from deep_gemm.utils import get_tma_aligned_size

        aligned_mn = get_tma_aligned_size(sf.shape[-2], sf.element_size())
        if sf.stride(-1) != aligned_mn:
            new_stride = list(sf.stride())
            new_stride[-1] = aligned_mn
            sf = sf.as_strided(sf.shape, tuple(new_stride))
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
    assert torch.all(sf_packed == sf_packed_recreated), (
        f"{sf_packed=} {sf_packed_recreated=} {sf_fp32=}"
    )
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
    bits = x.abs().float().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    return (exp << 23).view(torch.float32)


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


def apply_fp8_linear_bmm_flashinfer(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-tensor static fp8 linear via flashinfer bmm_fp8 (SM90 and newer)."""
    output_shape = [*input.shape[:-1], weight.shape[1]]
    input_2d = input.view(-1, input.shape[-1])
    qinput, x_scale = static_quant_fp8(input_2d, input_scale, repeat_scale=False)
    output = flashinfer_bmm_fp8(qinput, weight, x_scale, weight_scale, input.dtype)
    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


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
    pre_quant_output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:
        pad_output = not cutlass_fp8_supported and not get_bool_env_var(
            "SGLANG_ENABLE_TORCH_COMPILE"
        )
    output_padding = 17 if pad_output else None

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    # A pre-quantized fp8 activation (e.g. from a fused RMSNorm+quant kernel)
    # carries no original dtype: skip re-quant, reuse the supplied per-tensor
    # input_scale, and emit ``pre_quant_output_dtype`` (the model's activation
    # dtype, propagated by the producer) or bf16 if it was not provided.
    input_prequantized = input_2d.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    if input_prequantized:
        output_dtype = pre_quant_output_dtype or torch.bfloat16
    else:
        output_dtype = input.dtype

    channelwise_cutlass = (
        cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]
    )
    cutlass_compatible_b = weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
    use_cutlass_channelwise_gemm = (
        channelwise_cutlass and cutlass_compatible_b and not use_triton_w8a8_fp8_kernel
    )
    # Consider a tuned Triton tile only where the shape would otherwise go to
    # CUTLASS (that is the path the offline sweep tuned against). On by default;
    # SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0 is the kill switch.
    use_tuned_triton_channelwise = (
        use_cutlass_channelwise_gemm and envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.get()
    )
    native_scalar_a_scale = use_cutlass_channelwise_gemm and (
        get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
    )

    if input_prequantized:
        assert input_scale is not None and input_scale.numel() == 1
        qinput = input_2d
        if channelwise_cutlass and not native_scalar_a_scale:
            # Unsupported CUTLASS epilogues require one A scale per row.
            x_scale = input_scale.repeat(input_2d.shape[0]).view(-1, 1)
        else:
            x_scale = input_scale
    elif compressed_tensor_quant:
        # Maybe apply padding to output, see comment in __init__
        num_token_padding = output_padding
        if channelwise_cutlass or (_is_xpu and weight_scale.numel() == weight.shape[1]):
            # On XPU, sgl-kernel-xpu's native quant kernels require output_q
            # to exactly match input's shape; padded output isn't supported.
            num_token_padding = None
        # For static per-tensor activation scales when using inductor compiler,
        # use pure PyTorch ops instead of the opaque sgl_kernel quant kernel.
        # Inductor fuses these with surrounding ops (RMSNorm, residual add),
        # eliminating a separate kernel launch per linear layer.
        # weight_scale shape does not matter here -- it is only used in the
        # GEMM epilogue, not in the activation quant fusion. Only activates when
        # cuda_graph_config[prefill].tc_compiler=inductor; eager PCG and
        # decode both use the faster custom kernel.

        if (
            input_scale is not None
            and input_scale.numel() == 1
            and get_exec().graph.cuda_graph_config.prefill.tc_compiler == "inductor"
        ):
            qinput = (
                (input_2d * input_scale.reciprocal())
                .clamp(min=fp8_min, max=fp8_max)
                .to(fp8_dtype)
            )
            x_scale = input_scale
        else:
            qinput, x_scale = scaled_fp8_quant(
                input_2d,
                input_scale,
                num_token_padding=num_token_padding,
                use_per_token_if_dynamic=use_per_token_if_dynamic,
            )
        if (
            input_scale is not None
            and channelwise_cutlass
            and not native_scalar_a_scale
        ):
            x_scale = input_scale.repeat(input_2d.shape[0]).view(-1, 1)
    else:
        if input_scale is not None:
            assert input_scale.numel() == 1
            qinput, x_scale = static_quant_fp8(
                input_2d,
                input_scale,
                repeat_scale=channelwise_cutlass and not native_scalar_a_scale,
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

    if channelwise_cutlass:
        # A tuned config exists only for shapes where tuned Triton beat the
        # CUTLASS dispatch on this GPU; otherwise this is None and the backend
        # choice below is unchanged. weight is [K, N] here.
        tuned_config = (
            get_w8a8_channelwise_fp8_config(
                N=weight.shape[1], K=weight.shape[0], M=qinput.shape[0]
            )
            if use_tuned_triton_channelwise
            else None
        )
        if not use_cutlass_channelwise_gemm:
            # Massage the input to be 2D
            qinput = qinput.view(-1, qinput.shape[-1])
            output = triton_scaled_mm(
                qinput, weight, x_scale, weight_scale, output_dtype, bias
            )
        elif tuned_config is not None:
            qinput = qinput.view(-1, qinput.shape[-1])
            output = triton_scaled_mm(
                qinput,
                weight,
                x_scale,
                weight_scale,
                output_dtype,
                bias,
                block_size_m=tuned_config["BLOCK_SIZE_M"],
                block_size_n=tuned_config["BLOCK_SIZE_N"],
                block_size_k=tuned_config["BLOCK_SIZE_K"],
                use_heuristic=False,
                num_warps=tuned_config["num_warps"],
                num_stages=tuned_config["num_stages"],
            )
        else:
            output = fp8_scaled_mm(
                qinput,
                weight,
                x_scale,
                weight_scale,
                out_dtype=output_dtype,
                bias=bias,
            )
        return output.view(*output_shape)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    per_tensor_weights = weight_scale.numel() == 1
    # When the number of token is 1,
    # per-token scale has shape (1, 1), per-tensor scale has shape (1) or ().
    per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2

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
                dtype=output_dtype,
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
                out_dtype=output_dtype,
                scale_a=x_scale,
                scale_b=weight_scale.t(),
                bias=bias,
            )
            return _process_scaled_mm_output(output, input_2d.shape, output_shape)

    if per_tensor_weights and per_tensor_activations:
        # Fused GEMM_DQ; _scaled_mm with torch.compile requires len(weight_scale.shape) == len(x_scale.shape)
        if weight_scale.ndim == 0 and x_scale.ndim == 1:
            weight_scale = weight_scale.unsqueeze(0)
        output = torch._scaled_mm(
            qinput,
            weight,
            out_dtype=output_dtype,
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
        output_dtype,
    )


def can_auto_enable_marlin_fp8() -> bool:
    try:
        major, minor = get_device_capability()
        sm = major * 10 + minor
        return 80 <= sm < 89
    except Exception:
        return False


def apply_fp8_ptpc_linear(
    input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
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
    """FP8 per-token per-channel linear. Only used with the aiter (ROCm) backend."""
    # Handle pre-quantized (fp8_tensor, scale) tuple from fused RMSNorm+Quant
    if isinstance(input, tuple):
        q_input, x_scale = input
        q_input = q_input.view(-1, q_input.shape[-1])
        output_shape = [*q_input.shape[:-1], weight.shape[0]]
        output = aiter.gemm_a8w8_bpreshuffle(
            q_input, weight, x_scale, weight_scale, None, torch.bfloat16
        )
        if bias is not None:
            output = output + bias
        return output.view(*output_shape)

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

    # Lazy: a ``getattr`` default would read the published bag even for a
    # layer that carries its own tp_size.
    tp_size = layer.tp_size if hasattr(layer, "tp_size") else get_parallel().tp_size
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
