# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Host adapter for the BF16 masked grouped GEMM (SM100 and SM90).

Ported from the frozen base-GEMM study (plan section 45); the two containers
are msgspec.Struct per repo convention. ``prepare`` compiles per call — the
provider layer (cutedsl_bf16.py) owns the compile-once cache and per-forward
re-wrapping, because compilation keys only on the config plus layout
STRUCTURE: A/C/masked_m/schedule keep dynamic layouts, so one compiled fn
serves every m_max.
"""

from typing import Any, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import msgspec
import torch
from cutlass.cute.runtime import from_dlpack

# The shipped provider always selects swap_ab; the non-swap orientation is
# kept because it is the planned SM90 port's fallback when the swap_ab
# narrow-N WGMMA path proves unsupported there (plan section 54).
from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.kernel import (
    MaskedGroupedGemmKernel,
    masked_grouped_gemm,
    masked_grouped_gemm_swap_ab,
)


def _kernel_class_for(device: torch.device):
    """SM100+ -> tcgen05 kernel; SM90 -> the WGMMA sibling; else reject.

    The wrappers (`masked_grouped_gemm*`) are pure mode permutations and drive
    either class; only the device kernel differs (plan section 54).
    """
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 10:
        return MaskedGroupedGemmKernel
    if major == 9:
        from sglang.srt.lora.sgl_lora.base_gemm_provider.cutedsl_masked.kernel_sm90 import (
            MaskedGroupedGemmKernelSm90,
        )

        return MaskedGroupedGemmKernelSm90
    raise NotImplementedError(
        f"the masked grouped GEMM needs SM90+; device is sm{major}x"
    )


class MaskedGroupedGemmConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Compile-time kernel configuration (arch picked per device). Hashable."""

    mma_tiler_mn: Tuple[int, int] = (64, 128)
    cluster_shape_mn: Tuple[int, int] = (1, 1)
    use_2cta_instrs: bool = False
    occupancy: int = 1
    mma_inst_tile_k: int = 4
    use_warp_scan: bool = False
    uniform_m: int | None = None
    persistent_clusters: int | None = None
    use_pdl: bool = False
    swap_ab: bool = False
    direct_schedule: bool = False


class PreparedMaskedGroupedGemm(msgspec.Struct, kw_only=True):
    """Compiled callable plus DLPack arguments kept alive by the Torch tensors."""

    compiled_fn: Any
    a_arg: cute.Tensor
    b_arg: cute.Tensor
    c_arg: cute.Tensor
    masked_m_arg: cute.Tensor
    direct_schedule_arg: cute.Tensor
    schedule_tiles_arg: cute.Tensor
    direct_schedule_owner: torch.Tensor
    schedule_tiles_owner: torch.Tensor
    stream: cuda.CUstream

    def launch(self, torch_stream: torch.cuda.Stream | None = None) -> None:
        stream = (
            self.stream
            if torch_stream is None
            else cuda.CUstream(torch_stream.cuda_stream)
        )
        self.compiled_fn(
            self.a_arg,
            self.b_arg,
            self.c_arg,
            self.masked_m_arg,
            self.direct_schedule_arg,
            self.schedule_tiles_arg,
            stream,
        )


def _validate(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    masked_m: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if a.device.type != "cuda" or b.device != a.device or c.device != a.device:
        raise ValueError("A, B, and C must be CUDA tensors on the same device")
    if masked_m.device != a.device:
        raise ValueError("masked_m must be on the same CUDA device as A/B/C")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("A and B must be BF16")
    if c.dtype != torch.bfloat16:
        raise TypeError("C must be BF16")
    if masked_m.dtype != torch.int32:
        raise TypeError("masked_m must be int32")
    if a.ndim != 3 or b.ndim != 3 or c.ndim != 3 or masked_m.ndim != 1:
        raise ValueError("expected A[E,M,K], B[E,N,K], C[E,M,N], masked_m[E]")
    if not a.is_contiguous() or not b.is_contiguous() or not c.is_contiguous():
        raise ValueError("A, B, and C must be contiguous")
    if not masked_m.is_contiguous():
        raise ValueError("masked_m must be contiguous")

    experts, m_max, k = a.shape
    b_experts, n, b_k = b.shape
    if b_experts != experts or b_k != k:
        raise ValueError("B must have shape [A.shape[0], N, A.shape[2]]")
    if c.shape != (experts, m_max, n):
        raise ValueError("C must have shape [E, m_max, N]")
    if masked_m.numel() != experts:
        raise ValueError("masked_m must contain one valid-row count per expert")
    if k % 8 or n % 8:
        raise ValueError("K and N must be multiples of 8 for 16-byte TMA alignment")
    if torch.any(masked_m < 0) or torch.any(masked_m > m_max):
        raise ValueError("masked_m values must be in [0, A.shape[1]]")
    return experts, m_max, n, k


def as_dynamic_cute_tensor(tensor: torch.Tensor, *, leading_dim: int) -> cute.Tensor:
    """Wrap a torch tensor with its shape/strides left DYNAMIC.

    Public because a caller sharing one compiled function across layers must
    wrap that layer's weight exactly the way the compile path wrapped its
    representative one -- a divergent wrapping would silently change the
    argument's MLIR type.
    """
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )


_as_dynamic_cute_tensor = as_dynamic_cute_tensor  # internal callers below


def prepare(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    config: MaskedGroupedGemmConfig = MaskedGroupedGemmConfig(),
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedMaskedGroupedGemm:
    """Validate, compile, and bind a masked grouped GEMM launch.

    Compilation specializes only the kernel configuration. Tensor shapes and
    strides remain dynamic while the contiguous leading dimensions are known.
    """
    experts, m_max, n, k = _validate(a, b, c, masked_m)

    a_arg = _as_dynamic_cute_tensor(a, leading_dim=2)
    b_arg = _as_dynamic_cute_tensor(b, leading_dim=2)
    c_arg = _as_dynamic_cute_tensor(c, leading_dim=2)
    masked_m_arg = _as_dynamic_cute_tensor(masked_m, leading_dim=0)
    if direct_schedule is None:
        direct_schedule = torch.zeros((1,), device=a.device, dtype=torch.int32)
    if schedule_tiles is None:
        schedule_tiles = torch.zeros((1,), device=a.device, dtype=torch.int32)
    if (
        direct_schedule.device != a.device
        or direct_schedule.dtype != torch.int32
        or direct_schedule.ndim != 1
        or not direct_schedule.is_contiguous()
    ):
        raise ValueError("direct_schedule must be contiguous int32 [tiles]")
    if (
        schedule_tiles.device != a.device
        or schedule_tiles.dtype != torch.int32
        or schedule_tiles.shape != (1,)
        or not schedule_tiles.is_contiguous()
    ):
        raise ValueError("schedule_tiles must be contiguous int32 [1]")
    direct_schedule_arg = _as_dynamic_cute_tensor(direct_schedule, leading_dim=0)
    schedule_tiles_arg = _as_dynamic_cute_tensor(schedule_tiles, leading_dim=0)

    gemm = _kernel_class_for(a.device)(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=config.use_2cta_instrs,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        use_tma_store=True,
        mma_inst_tile_k=config.mma_inst_tile_k,
        use_warp_scan=config.use_warp_scan,
        uniform_m=config.uniform_m,
        persistent_clusters=config.persistent_clusters,
        use_pdl=config.use_pdl,
        swap_ab=config.swap_ab,
        use_direct_schedule=config.direct_schedule,
    )
    if config.occupancy not in (1, 2):
        raise ValueError("occupancy must be 1 or 2")
    if config.mma_inst_tile_k not in (2, 4, 8, 16):
        raise ValueError("mma_inst_tile_k must be one of 2, 4, 8, or 16")
    if config.uniform_m is not None and torch.any(masked_m != config.uniform_m):
        raise ValueError("uniform_m specialization does not match masked_m")
    if config.persistent_clusters is not None and config.persistent_clusters <= 0:
        raise ValueError("persistent_clusters must be positive")
    gemm.occupancy = config.occupancy
    problem_shape = (n, m_max, k, experts) if config.swap_ab else (m_max, n, k, experts)
    if not gemm.can_implement(
        problem_shape,
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        "k",
        "k",
        "m" if config.swap_ab else "n",
    ):
        raise testing.CantImplementError(
            f"unsupported config={config} for logical MNKL={problem_shape}"
        )

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    )
    torch_stream = torch.cuda.current_stream(a.device)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    wrapper = masked_grouped_gemm_swap_ab if config.swap_ab else masked_grouped_gemm
    compiled_fn = cute.compile(
        wrapper,
        gemm,
        a_arg,
        b_arg,
        c_arg,
        masked_m_arg,
        direct_schedule_arg,
        schedule_tiles_arg,
        max_active_clusters,
        current_stream,
        lambda x: x,
    )
    return PreparedMaskedGroupedGemm(
        compiled_fn=compiled_fn,
        a_arg=a_arg,
        b_arg=b_arg,
        c_arg=c_arg,
        masked_m_arg=masked_m_arg,
        direct_schedule_arg=direct_schedule_arg,
        schedule_tiles_arg=schedule_tiles_arg,
        direct_schedule_owner=direct_schedule,
        schedule_tiles_owner=schedule_tiles,
        stream=current_stream,
    )
