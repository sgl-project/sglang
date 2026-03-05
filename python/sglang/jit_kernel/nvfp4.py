from __future__ import annotations

import importlib.util
import os
import pathlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_FLOAT4_E2M1_MAX = 6.0
_FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _find_package_root(package: str) -> Optional[pathlib.Path]:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return pathlib.Path(spec.origin).resolve().parent


def _resolve_cutlass_include_paths() -> list[str]:
    include_paths: list[str] = []

    flashinfer_root = _find_package_root("flashinfer")
    if flashinfer_root is not None:
        candidates = [
            flashinfer_root / "data" / "cutlass" / "include",
            flashinfer_root / "data" / "cutlass" / "tools" / "util" / "include",
        ]
        for path in candidates:
            if path.exists():
                include_paths.append(str(path))

    deep_gemm_root = _find_package_root("deep_gemm")
    if deep_gemm_root is not None:
        candidate = deep_gemm_root / "include"
        if candidate.exists():
            include_paths.append(str(candidate))

    # De-duplicate while preserving order.
    unique_paths = []
    seen = set()
    for path in include_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _nvfp4_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DFLASHINFER_ENABLE_F16",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-extended-lambda",
    ]


def _parse_cuda_version() -> tuple[int, int]:
    v = torch.version.cuda
    if not v:
        return (0, 0)
    parts = v.split(".")
    if len(parts) < 2:
        return (0, 0)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return (0, 0)


def _get_nvfp4_cuda_arch_list() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 JIT kernels require CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"NVFP4 JIT kernels require compute capability >= 10.0, got {major}.{minor}."
        )
    # NVFP4 kernels use architecture-family-specific instructions and must be
    # compiled for `sm_*a` targets (e.g. sm_100a), not plain sm_100.
    archs = [f"{major}.{minor}a"]
    cuda_major, _cuda_minor = _parse_cuda_version()
    if cuda_major >= 13 and "10.3a" not in archs:
        # Match sgl-kernel AOT fatbin behavior on CUDA 13+ for Blackwell.
        archs.append("10.3a")
    # Preserve order while de-duplicating.
    seen = set()
    ordered_archs: list[str] = []
    for arch in archs:
        if arch in seen:
            continue
        seen.add(arch)
        ordered_archs.append(arch)
    return " ".join(ordered_archs)


@contextmanager
def _nvfp4_arch_env():
    key = "TVM_FFI_CUDA_ARCH_LIST"
    old_val = os.environ.get(key)
    os.environ[key] = _get_nvfp4_cuda_arch_list()
    try:
        yield
    finally:
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val


@cache_once
def _jit_nvfp4_quant_module() -> Module:
    extra_include_paths = _resolve_cutlass_include_paths()
    if not extra_include_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for NVFP4 JIT quantization. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )

    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_quant",
            cuda_files=[
                "gemm/nvfp4/nvfp4_quant_kernels.cuh",
            ],
            cuda_wrappers=[
                ("scaled_fp4_quant", "scaled_fp4_quant_sm100a_sm120a"),
            ],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


@cache_once
def _jit_nvfp4_expert_quant_module() -> Module:
    extra_include_paths = _resolve_cutlass_include_paths()
    if not extra_include_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for NVFP4 JIT expert quantization. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )

    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_expert_quant",
            cuda_files=[
                "gemm/nvfp4/nvfp4_expert_quant.cuh",
            ],
            cuda_wrappers=[
                ("scaled_fp4_experts_quant", "scaled_fp4_experts_quant_sm100a"),
                (
                    "silu_and_mul_scaled_fp4_experts_quant",
                    "silu_and_mul_scaled_fp4_experts_quant_sm100a",
                ),
            ],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


@cache_once
def _jit_nvfp4_scaled_mm_module() -> Module:
    extra_include_paths = _resolve_cutlass_include_paths()
    if not extra_include_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for NVFP4 JIT GEMM. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )

    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_scaled_mm",
            cuda_files=[
                "gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh",
                "gemm/nvfp4/nvfp4_scaled_mm_entry.cuh",
            ],
            cuda_wrappers=[("cutlass_scaled_fp4_mm", "cutlass_scaled_fp4_mm")],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


@cache_once
def _jit_nvfp4_blockwise_moe_module() -> Module:
    extra_include_paths = _resolve_cutlass_include_paths()
    if not extra_include_paths:
        raise RuntimeError(
            "Cannot find CUTLASS headers required for NVFP4 JIT MoE grouped GEMM. "
            "Please install flashinfer or deep_gemm with CUTLASS headers."
        )

    with _nvfp4_arch_env():
        return load_jit(
            "nvfp4_blockwise_moe",
            cuda_files=[
                "moe/nvfp4_blockwise_moe.cuh",
            ],
            cuda_wrappers=[
                ("cutlass_fp4_group_mm", "cutlass_fp4_group_mm_sm100a_sm120a")
            ],
            extra_include_paths=extra_include_paths,
            extra_cuda_cflags=_nvfp4_cuda_flags(),
        )


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    module = _jit_nvfp4_scaled_mm_module()
    module.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out


def cutlass_fp4_group_mm(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_blockscale: torch.Tensor,
    b_blockscale: torch.Tensor,
    alphas: torch.Tensor,
    out_dtype: torch.dtype,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    m_topk = a_fp4.shape[0]
    n = b_fp4.shape[1]
    output = torch.empty((m_topk, n), device=a_fp4.device, dtype=out_dtype)
    num_experts = int(params["expert_offsets"].numel())
    device = a_fp4.device

    # Backward compatibility: older callers may not pass scratch tensors.
    a_ptrs = params.get(
        "a_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    b_ptrs = params.get(
        "b_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    out_ptrs = params.get(
        "out_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    a_scales_ptrs = params.get(
        "a_scales_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    b_scales_ptrs = params.get(
        "b_scales_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    alpha_ptrs = params.get(
        "alpha_ptrs", torch.empty((num_experts,), dtype=torch.int64, device=device)
    )
    layout_sfa = params.get(
        "layout_sfa", torch.empty((num_experts, 5), dtype=torch.int64, device=device)
    )
    layout_sfb = params.get(
        "layout_sfb", torch.empty((num_experts, 5), dtype=torch.int64, device=device)
    )

    module = _jit_nvfp4_blockwise_moe_module()
    module.cutlass_fp4_group_mm(
        output,
        a_fp4,
        b_fp4,
        a_blockscale,
        b_blockscale,
        alphas,
        params["ab_strides"],
        params["c_strides"],
        params["problem_sizes"],
        params["expert_offsets"],
        params["blockscale_offsets"],
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        alpha_ptrs,
        layout_sfa,
        layout_sfb,
    )
    return output


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize input tensor to FP4 and return packed FP4 tensor + swizzled scales."""
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    if rounded_n > scale_n:
        output_scale = torch.zeros(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
    else:
        output_scale = torch.empty(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )

    module = _jit_nvfp4_quant_module()
    module.scaled_fp4_quant(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def _shuffle_rows_torch(
    input_tensor: torch.Tensor,
    dst2src_map: torch.Tensor,
    output_tensor_shape: tuple[int, int],
) -> torch.Tensor:
    # Keep compatibility when sgl-kernel is slimmed and shuffle_rows may not be present.
    output = input_tensor.index_select(0, dst2src_map.to(dtype=torch.int64))
    return output.view(output_tensor_shape)


def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
    expert_map: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize packed MoE activations to NVFP4."""
    assert (
        input_tensor.ndim == 2
    ), f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    if expert_map is not None:
        m, k = input_tensor.shape
        output_tensor_shape = (m * topk, k)
        input_tensor = _shuffle_rows_torch(
            input_tensor, expert_map, output_tensor_shape
        )

    m_numtopk, k = input_tensor.shape
    max_tokens_per_expert = int(os.environ.get("MODELOPT_MAX_TOKENS_PER_EXPERT", 65536))
    assert m_numtopk <= max_tokens_per_expert * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT({max_tokens_per_expert})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        " MODELOPT_MAX_TOKENS_PER_EXPERT to set this value."
    )
    scales_k = k // 16
    # output_scales is int32-packed FP8 scales, so second dim is in int32 units.
    padded_k_in_int32 = (scales_k + 3) // 4

    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    if padded_k_in_int32 * 4 > scales_k:
        output_scales = torch.zeros(
            max_tokens_per_expert * topk,
            padded_k_in_int32,
            dtype=torch.int32,
            device=input_tensor.device,
        )
    else:
        output_scales = torch.empty(
            max_tokens_per_expert * topk,
            padded_k_in_int32,
            dtype=torch.int32,
            device=input_tensor.device,
        )

    module = _jit_nvfp4_expert_quant_module()
    module.scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


def scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """Quantize grouped GEMM inputs to FP4 and return logical (m, k//2, l)."""
    device = input_tensor.device
    l, m, k = input_tensor.shape
    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_k_int32 = padded_k // 4
    padded_m = (m + (128 - 1)) // 128 * 128
    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        l, padded_m, padded_k_int32, device=device, dtype=torch.int32
    )

    module = _jit_nvfp4_expert_quant_module()
    module.silu_and_mul_scaled_fp4_experts_quant(
        output.view(l * m, k // 2),
        output_scales.view(l * padded_m, padded_k_int32),
        input_tensor.view(l * m, k),
        input_global_scale,
        mask,
        False,
    )

    output = output.permute(1, 2, 0)
    output_scales = output_scales.view(torch.float8_e4m3fn).view(
        l, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    return output, output_scales


def silu_and_mul_scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """Apply SiLU-and-mul then quantize grouped GEMM inputs to FP4."""
    device = input_tensor.device
    l, m, k_by_2 = input_tensor.shape
    k = k_by_2 // 2
    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_k_int32 = padded_k // 4
    padded_m = (m + (128 - 1)) // 128 * 128
    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        l, padded_m, padded_k_int32, device=device, dtype=torch.int32
    )

    module = _jit_nvfp4_expert_quant_module()
    module.silu_and_mul_scaled_fp4_experts_quant(
        output.view(l * m, k // 2),
        output_scales.view(l * padded_m, padded_k_int32),
        input_tensor.view(l * m, k_by_2),
        input_global_scale,
        mask,
        True,
    )

    output = output.permute(1, 2, 0)
    output_scales = output_scales.view(torch.float8_e4m3fn).view(
        l, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    return output, output_scales


def suggest_nvfp4_global_scale(x: torch.Tensor) -> torch.Tensor:
    """Utility for tests/benchmarks: return global scale used by NVFP4 quantization."""
    tensor_amax = torch.abs(x).max().to(torch.float32)
    return _FLOAT8_E4M3_MAX * _FLOAT4_E2M1_MAX / tensor_amax
