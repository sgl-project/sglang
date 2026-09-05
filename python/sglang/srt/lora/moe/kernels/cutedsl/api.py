# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile grouped GEMMs with dynamic row counts; providers cache the code."""

from typing import Any, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import msgspec
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.srt.lora.moe.kernels.cutedsl.kernel_sm100_bf16 import (
    GroupedGemmKernelSm100Bf16,
)


@cute.jit
def grouped_gemm_bf16_swap_ab(
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
    # W @ X.T puts tokens on N for 8-wide tiles; C uses the transposed layout.
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


def _bf16_kernel_class_for(device: torch.device):
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 10:
        return GroupedGemmKernelSm100Bf16
    if major == 9:
        from sglang.srt.lora.moe.kernels.cutedsl.kernel_sm90_bf16 import (
            GroupedGemmKernelSm90Bf16,
        )

        return GroupedGemmKernelSm90Bf16
    raise NotImplementedError(
        f"the BF16 grouped GEMM needs SM90+; device is sm{major}x"
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


class PreparedGroupedGemm(msgspec.Struct, kw_only=True):
    """Bound arguments retain the tensors backing their DLPack views."""

    compiled_fn: Any
    operand_args: tuple  # the wrapper's tensor operands, in its order
    group_m_arg: cute.Tensor
    direct_schedule_arg: cute.Tensor
    schedule_tiles_arg: cute.Tensor
    direct_schedule_owner: torch.Tensor
    schedule_tiles_owner: torch.Tensor
    stream: cuda.CUstream

    def launch(self) -> None:
        self.compiled_fn(
            *self.operand_args,
            self.group_m_arg,
            self.direct_schedule_arg,
            self.schedule_tiles_arg,
            self.stream,
        )


def as_dynamic_cute_tensor(tensor: torch.Tensor, *, leading_dim: int) -> cute.Tensor:
    """Compile-time and runtime calls must use identical layout wrapping."""
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _compile_prepared(
    *,
    kernel_cls,
    wrapper,
    ab_dtype,
    config: GroupedGemmConfig,
    contiguous_segments: bool,
    problem_shape: Tuple[int, int, int, int],
    operand_args: tuple,
    group_m_arg: cute.Tensor,
    direct_schedule: torch.Tensor | None,
    schedule_tiles: torch.Tensor | None,
    device: torch.device,
) -> PreparedGroupedGemm:
    if direct_schedule is None:
        direct_schedule = torch.zeros((1,), device=device, dtype=torch.int64)
    if schedule_tiles is None:
        schedule_tiles = torch.zeros((1,), device=device, dtype=torch.int32)
    direct_schedule_arg = as_dynamic_cute_tensor(direct_schedule, leading_dim=0)
    schedule_tiles_arg = as_dynamic_cute_tensor(schedule_tiles, leading_dim=0)

    gemm = kernel_cls(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=config.use_2cta_instrs,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        mma_inst_tile_k=config.mma_inst_tile_k,
        persistent_clusters=config.persistent_clusters,
        produce_pdl=config.produce_pdl,
        swap_ab=True,
        contiguous_segments=contiguous_segments,
    )
    gemm.occupancy = config.occupancy
    if not gemm.can_implement(
        problem_shape, ab_dtype, ab_dtype, cutlass.BFloat16, "k", "k", "m"
    ):
        raise testing.CantImplementError(
            f"unsupported {getattr(ab_dtype, '__name__', ab_dtype)} "
            f"config={config} for logical MNKL={problem_shape}"
        )

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        config.cluster_shape_mn[0] * config.cluster_shape_mn[1]
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    compiled_fn = cute.compile(
        wrapper,
        gemm,
        *operand_args,
        group_m_arg,
        direct_schedule_arg,
        schedule_tiles_arg,
        max_active_clusters,
        stream,
        lambda x: x,
    )
    return PreparedGroupedGemm(
        compiled_fn=compiled_fn,
        operand_args=operand_args,
        group_m_arg=group_m_arg,
        direct_schedule_arg=direct_schedule_arg,
        schedule_tiles_arg=schedule_tiles_arg,
        direct_schedule_owner=direct_schedule,
        schedule_tiles_owner=schedule_tiles,
        stream=stream,
    )


def prepare_masked_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    experts, m_max, k = a.shape
    n = b.shape[1]
    return _compile_prepared(
        kernel_cls=_bf16_kernel_class_for(a.device),
        wrapper=grouped_gemm_bf16_swap_ab,
        ab_dtype=cutlass.BFloat16,
        config=config,
        contiguous_segments=False,
        problem_shape=(n, m_max, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a, leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(c, leading_dim=2),
        ),
        group_m_arg=as_dynamic_cute_tensor(masked_m, leading_dim=0),
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
        device=a.device,
    )


def prepare_contiguous_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    seg_offsets: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    """Use seg_offsets in the group_m slot to address flat expert segments."""
    m_ceil, k = a.shape
    experts, n, _ = b.shape
    return _compile_prepared(
        kernel_cls=_bf16_kernel_class_for(a.device),
        wrapper=grouped_gemm_bf16_swap_ab,
        ab_dtype=cutlass.BFloat16,
        config=config,
        contiguous_segments=True,
        problem_shape=(n, m_ceil, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a.unsqueeze(0), leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(c.unsqueeze(0), leading_dim=2),
        ),
        group_m_arg=as_dynamic_cute_tensor(seg_offsets, leading_dim=0),
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
        device=a.device,
    )
