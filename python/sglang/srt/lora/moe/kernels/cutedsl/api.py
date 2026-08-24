# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Host adapter for the BF16 grouped GEMM: each prepare compiles its row
mode (the provider holds the compile-once cache). A, C, ``group_m``, and the
schedule keep dynamic layouts, so one compiled function serves every m.
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

# The shipped provider always uses swap_ab (tokens on the MMA N axis); the
# non-swap wrapper remains for benchmarking the tokens-on-M orientation.
from sglang.srt.lora.moe.kernels.cutedsl.kernel_sm100 import (
    GroupedGemmKernelSm100,
)


@cute.jit
def grouped_gemm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # physical (expert, m_max, k)
    b: cute.Tensor,  # physical (expert, n, k)
    c: cute.Tensor,  # physical (expert, m_max, n)
    group_m: cute.Tensor,  # physical (expert,)
    direct_schedule: cute.Tensor,
    schedule_tiles: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    # Present A as (M, K, E), B as (N, K, E), C as (M, N, E); the weight
    # stays [E, N, K] in memory, so K is contiguous for the transposed B.
    a_mke = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b_nke = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    c_mne = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        a_mke,
        b_nke,
        c_mne,
        group_m,
        direct_schedule,
        schedule_tiles,
        max_active_clusters,
        stream,
        epilogue_op,
    )


@cute.jit
def grouped_gemm_swap_ab(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # physical routed input (expert, m_max, k)
    b: cute.Tensor,  # physical weight (expert, n_out, k)
    c: cute.Tensor,  # physical output (expert, m_max, n_out)
    group_m: cute.Tensor,
    direct_schedule: cute.Tensor,
    schedule_tiles: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    # W @ X.T, with C exposed as a transposed logical tensor.
    weight_mke = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    token_nke = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    output_mne = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[2, 1, 0]))
    gemm_op(
        weight_mke,
        token_nke,
        output_mne,
        group_m,
        direct_schedule,
        schedule_tiles,
        max_active_clusters,
        stream,
        epilogue_op,
    )


def _kernel_class_for(device: torch.device):
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 10:
        return GroupedGemmKernelSm100
    if major == 9:
        from sglang.srt.lora.moe.kernels.cutedsl.kernel_sm90 import (
            GroupedGemmKernelSm90,
        )

        return GroupedGemmKernelSm90
    raise NotImplementedError(
        f"the masked grouped GEMM needs SM90+; device is sm{major}x"
    )


class GroupedGemmConfig(msgspec.Struct, frozen=True, kw_only=True):
    mma_tiler_mn: Tuple[int, int] = (64, 128)
    cluster_shape_mn: Tuple[int, int] = (1, 1)
    use_2cta_instrs: bool = False
    occupancy: int = 1
    mma_inst_tile_k: int = 4
    persistent_clusters: int | None = None
    # This flag only lets the GEMM release a later kernel. The GEMM itself
    # does not start early against its own producer.
    produce_pdl: bool = False
    swap_ab: bool = False


class PreparedGroupedGemm(msgspec.Struct, kw_only=True):
    """The Torch tensors keep the DLPack arguments alive."""

    compiled_fn: Any
    a_arg: cute.Tensor
    b_arg: cute.Tensor
    c_arg: cute.Tensor
    group_m_arg: cute.Tensor
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
            self.group_m_arg,
            self.direct_schedule_arg,
            self.schedule_tiles_arg,
            stream,
        )


def as_dynamic_cute_tensor(tensor: torch.Tensor, *, leading_dim: int) -> cute.Tensor:
    """Wrap with dynamic shape/strides. Every caller of one compiled function
    must wrap identically: a different wrapping changes the argument's MLIR
    type, and nothing reports the change.
    """
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _compile_prepared(
    *,
    device: torch.device,
    config: GroupedGemmConfig,
    contiguous_segments: bool,
    problem_shape: Tuple[int, int, int, int],
    c_major: str,
    wrapper,
    a_arg: cute.Tensor,
    b_arg: cute.Tensor,
    c_arg: cute.Tensor,
    group_m_arg: cute.Tensor,
    direct_schedule: torch.Tensor | None,
    schedule_tiles: torch.Tensor | None,
) -> PreparedGroupedGemm:
    """Build the kernel, admit the problem, compile, and bind the arguments."""
    if direct_schedule is None:
        direct_schedule = torch.zeros((1,), device=device, dtype=torch.int64)
    if schedule_tiles is None:
        schedule_tiles = torch.zeros((1,), device=device, dtype=torch.int32)
    direct_schedule_arg = as_dynamic_cute_tensor(direct_schedule, leading_dim=0)
    schedule_tiles_arg = as_dynamic_cute_tensor(schedule_tiles, leading_dim=0)

    gemm = _kernel_class_for(device)(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=config.use_2cta_instrs,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        mma_inst_tile_k=config.mma_inst_tile_k,
        persistent_clusters=config.persistent_clusters,
        produce_pdl=config.produce_pdl,
        swap_ab=config.swap_ab,
        contiguous_segments=contiguous_segments,
    )
    gemm.occupancy = config.occupancy
    if not gemm.can_implement(
        problem_shape,
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        "k",
        "k",
        c_major,
    ):
        raise testing.CantImplementError(
            f"unsupported config={config} for logical MNKL={problem_shape}"
        )

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    )
    torch_stream = torch.cuda.current_stream(device)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled_fn = cute.compile(
        wrapper,
        gemm,
        a_arg,
        b_arg,
        c_arg,
        group_m_arg,
        direct_schedule_arg,
        schedule_tiles_arg,
        max_active_clusters,
        current_stream,
        lambda x: x,
    )
    return PreparedGroupedGemm(
        compiled_fn=compiled_fn,
        a_arg=a_arg,
        b_arg=b_arg,
        c_arg=c_arg,
        group_m_arg=group_m_arg,
        direct_schedule_arg=direct_schedule_arg,
        schedule_tiles_arg=schedule_tiles_arg,
        direct_schedule_owner=direct_schedule,
        schedule_tiles_owner=schedule_tiles,
        stream=current_stream,
    )


def prepare_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    config: GroupedGemmConfig = GroupedGemmConfig(),
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    experts, m_max, k = a.shape
    n = b.shape[1]

    a_arg = as_dynamic_cute_tensor(a, leading_dim=2)
    b_arg = as_dynamic_cute_tensor(b, leading_dim=2)
    c_arg = as_dynamic_cute_tensor(c, leading_dim=2)
    masked_m_arg = as_dynamic_cute_tensor(masked_m, leading_dim=0)
    return _compile_prepared(
        device=a.device,
        config=config,
        contiguous_segments=False,
        problem_shape=(
            (n, m_max, k, experts) if config.swap_ab else (m_max, n, k, experts)
        ),
        c_major="m" if config.swap_ab else "n",
        wrapper=grouped_gemm_swap_ab if config.swap_ab else grouped_gemm,
        a_arg=a_arg,
        b_arg=b_arg,
        c_arg=c_arg,
        group_m_arg=masked_m_arg,
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
    )


def prepare_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    seg_offsets: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    """Compile and bind a GEMM over one flat route-major row buffer.

    ``seg_offsets`` rides the ``masked_m`` argument slot; the kernel reads it
    only to add the segment base to a tile index, and the schedule builder
    guarantees the token tile divides the segment alignment.
    """
    m_ceil, k = a.shape
    experts, n, _ = b.shape

    a_arg = as_dynamic_cute_tensor(a.unsqueeze(0), leading_dim=2)
    b_arg = as_dynamic_cute_tensor(b, leading_dim=2)
    c_arg = as_dynamic_cute_tensor(c.unsqueeze(0), leading_dim=2)
    seg_offsets_arg = as_dynamic_cute_tensor(seg_offsets, leading_dim=0)
    return _compile_prepared(
        device=a.device,
        config=config,
        contiguous_segments=True,
        problem_shape=(n, m_ceil, k, experts),
        c_major="m",
        wrapper=grouped_gemm_swap_ab,
        a_arg=a_arg,
        b_arg=b_arg,
        c_arg=c_arg,
        group_m_arg=seg_offsets_arg,
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
    )
