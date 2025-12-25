"""
LP Solver using JIT-compiled IPM kernel.

This module provides Just-In-Time (JIT) compiled interior point method solver
for linear programming problems, similar to DeepGEMM's JIT approach.
"""

import functools
from pathlib import Path
from typing import Optional

import torch

# Import the compiled C++ module
try:
    import lp_solver.lp_solver_cpp as lp_solver_cpp
except ImportError:
    raise ImportError(
        "lp_solver_cpp not found. Make sure sgl-kernel is built with LP solver support."
    )


# Get the resource path where templates and libs are installed
def _get_resource_path() -> str:
    """Get the installation path of lp_solver resources."""
    # The lp_solver_cpp module is in site-packages/lp_solver/
    module_path = Path(lp_solver_cpp.__file__).parent
    return str(module_path)


RESOURCE_PATH = _get_resource_path()


@functools.lru_cache(maxsize=None)
def _verify_resources():
    """Verify that required resources are available."""
    resource_dir = Path(RESOURCE_PATH)

    # Check for templates
    template_dir = resource_dir / "templates"
    if not template_dir.exists():
        raise RuntimeError(f"Template directory not found: {template_dir}")

    if not (template_dir / "ipm.cu").exists():
        raise RuntimeError(f"IPM kernel template not found in {template_dir}")

    # Check for mathdx libraries
    lib_dir = resource_dir / "mathdx" / "lib"
    if not lib_dir.exists():
        raise RuntimeError(f"Library directory not found: {lib_dir}")

    if not (lib_dir / "libcusolverdx.fatbin").exists():
        raise RuntimeError(f"cuSOLVERDx library not found in {lib_dir}")

    # Check for headers
    include_dir = resource_dir / "mathdx" / "include"
    if not include_dir.exists():
        raise RuntimeError(f"Include directory not found: {include_dir}")


def ipm_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    avail_num: torch.Tensor,
    block_dim: int = 128,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Solve interior point method optimization using JIT-compiled CUDA kernel.

    This function uses NVRTC to compile the IPM kernel at runtime with specific
    parameters (NC, NV) for optimal performance. Compiled kernels are cached.

    Args:
        A: Constraint matrix [NC, NV], float32, CUDA tensor
        b: Constraint vector [NC], float32, CUDA tensor
        c: Objective vector [NV], float32, CUDA tensor
        avail_num: Availability counter [1], int32, CUDA tensor
        block_dim: CUDA block dimension (default: 128)
        stream: Optional CUDA stream (uses current stream if None)

    Returns:
        Solution vector [NV], float32, CUDA tensor

    Example:
        >>> import torch
        >>> from sgl_kernel.lp_solver import ipm_solve
        >>>
        >>> NC, NV = 24, 42
        >>> A = torch.randn(NC, NV, device='cuda', dtype=torch.float32)
        >>> b = torch.randn(NC, device='cuda', dtype=torch.float32)
        >>> c = torch.randn(NV, device='cuda', dtype=torch.float32)
        >>> avail_num = torch.zeros(1, device='cuda', dtype=torch.int32)
        >>>
        >>> result = ipm_solve(A, b, c, avail_num)
        >>> print(result.shape)  # [42]
    """
    # Verify resources on first call
    _verify_resources()

    # Validate inputs
    if not A.is_cuda or not b.is_cuda or not c.is_cuda or not avail_num.is_cuda:
        raise ValueError("All tensors must be on CUDA device")

    if A.dtype != torch.float32 or b.dtype != torch.float32 or c.dtype != torch.float32:
        raise ValueError("A, b, c must be float32")

    if avail_num.dtype != torch.int32:
        raise ValueError("avail_num must be int32")

    if A.dim() != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if b.dim() != 1:
        raise ValueError(f"b must be 1D, got shape {b.shape}")

    if c.dim() != 1:
        raise ValueError(f"c must be 1D, got shape {c.shape}")

    if avail_num.numel() != 1:
        raise ValueError("avail_num must have exactly 1 element")

    NC, NV = A.shape
    if b.shape[0] != NC:
        raise ValueError(f"b.shape[0]={b.shape[0]} must equal A.shape[0]={NC}")

    if c.shape[0] != NV:
        raise ValueError(f"c.shape[0]={c.shape[0]} must equal A.shape[1]={NV}")

    # Set stream context if provided
    if stream is not None:
        with torch.cuda.stream(stream):
            solver = lp_solver_cpp.CompiledSolver(RESOURCE_PATH, NC, NV, block_dim)
            return solver.solve(A, b, c, avail_num)
    else:
        solver = lp_solver_cpp.CompiledSolver(RESOURCE_PATH, NC, NV, block_dim)
        return solver.solve(A, b, c, avail_num)


def clear_kernel_cache():
    """
    Clear the JIT kernel cache.

    This forces recompilation of all kernels on next use.
    Useful for development or after updating kernel templates.
    """
    # The cache is in C++, so we need to restart Python to clear it
    # For now, this is just a placeholder
    import warnings

    warnings.warn(
        "Kernel cache is in C++ memory. Restart Python to clear cache.", UserWarning
    )


__all__ = ["ipm_solve", "clear_kernel_cache"]
