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

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm100_bf16 import (
    GroupedGemmKernelSm100Bf16,
)


@cute.jit
def _grouped_gemm_bf16_swap_ab(
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
    cuda_arch = current_arch().cuda
    major = cuda_arch.major
    if major >= 10:
        return GroupedGemmKernelSm100Bf16
    if major == 9:
        from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm90_bf16 import (
            GroupedGemmKernelSm90Bf16,
        )

        return GroupedGemmKernelSm90Bf16
    raise NotImplementedError(
        f"the BF16 grouped GEMM needs SM90+; device is sm{major}x"
    )


@cute.jit
def _grouped_gemm_fp8_swap_ab(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # physical routed input (expert, m_max, k) e4m3
    b: cute.Tensor,  # physical weight (expert, n_out, k) e4m3
    sf_tokens: cute.Tensor,  # physical (expert, rows, k/128) fp32
    sf_weights: cute.Tensor,  # physical (expert, n/128, k/128) fp32
    c: cute.Tensor,  # physical output (expert, m_max, n_out) bf16
    group_m: cute.Tensor,
    direct_schedule: cute.Tensor,
    schedule_tiles: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    # SFA follows weights (kernel A); SFB follows tokens (kernel B).
    weight_mke = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    token_nke = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    output_mne = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[2, 1, 0]))
    sfa_weights = cute.make_tensor(
        sf_weights.iterator, cute.select(sf_weights.layout, mode=[1, 2, 0])
    )
    sfb_tokens = cute.make_tensor(
        sf_tokens.iterator, cute.select(sf_tokens.layout, mode=[1, 2, 0])
    )
    gemm_op(
        weight_mke,
        token_nke,
        sfa_weights,
        sfb_tokens,
        output_mne,
        group_m,
        direct_schedule,
        schedule_tiles,
        max_active_clusters,
        stream,
        epilogue_op,
    )


def _fp8_kernel_class_for(device: torch.device):
    cuda_arch = current_arch().cuda
    major = cuda_arch.major
    if major >= 10:
        from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm100_fp8 import (
            GroupedGemmKernelSm100Fp8,
        )

        return GroupedGemmKernelSm100Fp8
    if major == 9:
        from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm90_fp8 import (
            GroupedGemmKernelSm90Fp8,
        )

        return GroupedGemmKernelSm90Fp8
    raise NotImplementedError(f"the FP8 grouped GEMM needs SM90+; device is sm{major}x")


@cute.jit
def _blockscaled_operand_mke(
    t: cute.Tensor, ab_dtype: cutlass.Constexpr
) -> cute.Tensor:
    """View a physical (expert, rows, k_store) operand as logical (rows, k, expert).

    FP4 pairs arrive packed in uint8 storage, so the K extent doubles.
    """
    if cutlass.const_expr(ab_dtype.width == 4):
        shape, stride = t.layout.shape, t.layout.stride
        return cute.make_tensor(
            cute.recast_ptr(t.iterator, dtype=ab_dtype),
            cute.make_layout(
                (shape[1], shape[2] * 2, shape[0]),
                stride=(stride[1] * 2, 1, stride[0] * 2),
            ),
        )
    return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 2, 0]))


@cute.jit
def _grouped_gemm_blockscaled_swap_ab(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # physical routed input (expert, m_max, k_store)
    b: cute.Tensor,  # physical weight (expert, n_out, k_store)
    sf_tokens: cute.Tensor,  # flat uint8 scale atoms over (m_max, k, expert)
    sf_weights: cute.Tensor,  # flat uint8 scale atoms over (n_out, k, expert)
    c: cute.Tensor,  # physical output (expert, m_max, n_out) bf16
    alpha: cute.Tensor,  # fp32 (expert, halves) epilogue scales; unread unless use_alpha
    group_m: cute.Tensor,
    direct_schedule: cute.Tensor,
    schedule_tiles: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
    ab_dtype: cutlass.Constexpr = cutlass.Float8E4M3FN,
    sf_dtype: cutlass.Constexpr = cutlass.Float8E8M0FNU,
    use_alpha: cutlass.Constexpr = False,
    alpha_rows_per_half: cutlass.Constexpr = 1,
):
    # SFA follows weights (kernel A); SFB follows tokens (kernel B).
    weight_mke = _blockscaled_operand_mke(b, ab_dtype)
    token_nke = _blockscaled_operand_mke(a, ab_dtype)
    output_mne = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[2, 1, 0]))
    sfa_weights = cute.make_tensor(
        cute.recast_ptr(sf_weights.iterator, dtype=sf_dtype), sf_weights.layout
    )
    sfb_tokens = cute.make_tensor(
        cute.recast_ptr(sf_tokens.iterator, dtype=sf_dtype), sf_tokens.layout
    )
    gemm_op(
        weight_mke,
        token_nke,
        sfa_weights,
        sfb_tokens,
        output_mne,
        alpha,
        group_m,
        direct_schedule,
        schedule_tiles,
        max_active_clusters,
        stream,
        epilogue_op,
        use_alpha,
        alpha_rows_per_half,
    )


def _blockscaled_kernel_class_for(device: torch.device):
    major, _minor = torch.cuda.get_device_capability(device)
    if major == 10:
        from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm100_blockscaled import (
            GroupedGemmKernelSm100BlockScaled,
        )

        return GroupedGemmKernelSm100BlockScaled
    raise NotImplementedError(
        f"the block-scaled grouped GEMM needs the SM100 tcgen05 MMA; device is sm{major}x"
    )


def scale_atom_bytes(rows: int, k: int, sf_vec_size: int, experts: int = 1) -> int:
    """Bytes of the tcgen05 scale-factor atoms covering (rows, k) per expert."""
    row_atoms = -(-rows // 128)
    k_atoms = -(-(k // sf_vec_size) // 4)
    return experts * row_atoms * k_atoms * 512


class GroupedGemmConfig(msgspec.Struct, frozen=True, kw_only=True):
    mma_tiler_mn: Tuple[int, int] = (64, 128)
    cluster_shape_mn: Tuple[int, int] = (1, 1)
    use_2cta_instrs: bool = False
    occupancy: int = 1
    mma_inst_tile_k: int = 4
    persistent_clusters: int | None = None


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
    alpha_owner: torch.Tensor | None = None

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
    kernel_kwargs: dict | None = None,
    wrapper_constexprs: tuple = (),
) -> PreparedGroupedGemm:
    if direct_schedule is None:
        direct_schedule = torch.zeros((1,), device=device, dtype=torch.int64)
    if schedule_tiles is None:
        schedule_tiles = torch.zeros((1,), device=device, dtype=torch.int32)
    direct_schedule_arg = as_dynamic_cute_tensor(direct_schedule, leading_dim=0)
    schedule_tiles_arg = as_dynamic_cute_tensor(schedule_tiles, leading_dim=0)

    kernel_kwargs = dict(kernel_kwargs or {})
    if getattr(kernel_cls, "SUPPORTS_SINGLE_K_PIPELINE", False):
        # Compile the rolling pipeline only for single-K tiles to save registers.
        cta_tile_k = 32 * config.mma_inst_tile_k
        kernel_kwargs["single_k_pipeline"] = problem_shape[2] <= cta_tile_k
    gemm = kernel_cls(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=config.use_2cta_instrs,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
        mma_inst_tile_k=config.mma_inst_tile_k,
        persistent_clusters=config.persistent_clusters,
        swap_ab=True,
        contiguous_segments=contiguous_segments,
        **kernel_kwargs,
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
        *wrapper_constexprs,
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
        wrapper=_grouped_gemm_bf16_swap_ab,
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
        wrapper=_grouped_gemm_bf16_swap_ab,
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


def prepare_masked_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    sf_tokens: torch.Tensor,  # fp32 [E, m_max, K // 128]
    sf_weights: torch.Tensor,  # fp32 [E, N // 128, K // 128] (checkpoint form)
    c: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    assert sf_tokens.dtype == torch.float32 and sf_weights.dtype == torch.float32
    experts, m_max, k = a.shape
    n = b.shape[1]
    return _compile_prepared(
        kernel_cls=_fp8_kernel_class_for(a.device),
        wrapper=_grouped_gemm_fp8_swap_ab,
        ab_dtype=cutlass.Float8E4M3FN,
        config=config,
        contiguous_segments=False,
        problem_shape=(n, m_max, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a, leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(sf_tokens, leading_dim=2),
            as_dynamic_cute_tensor(sf_weights, leading_dim=2),
            as_dynamic_cute_tensor(c, leading_dim=2),
        ),
        group_m_arg=as_dynamic_cute_tensor(masked_m, leading_dim=0),
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
        device=a.device,
    )


def prepare_contiguous_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    sf_tokens: torch.Tensor,  # fp32 [1, M, K // 128]
    sf_weights: torch.Tensor,  # fp32 [E, N // 128, K // 128] (checkpoint form)
    c: torch.Tensor,
    seg_offsets: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    assert sf_tokens.dtype == torch.float32 and sf_weights.dtype == torch.float32
    m_ceil, k = a.shape
    experts, n, _ = b.shape
    return _compile_prepared(
        kernel_cls=_fp8_kernel_class_for(a.device),
        wrapper=_grouped_gemm_fp8_swap_ab,
        ab_dtype=cutlass.Float8E4M3FN,
        config=config,
        contiguous_segments=True,
        problem_shape=(n, m_ceil, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a.unsqueeze(0), leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(sf_tokens, leading_dim=2),
            as_dynamic_cute_tensor(sf_weights, leading_dim=2),
            as_dynamic_cute_tensor(c.unsqueeze(0), leading_dim=2),
        ),
        group_m_arg=as_dynamic_cute_tensor(seg_offsets, leading_dim=0),
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
        device=a.device,
    )


def _blockscaled_dtypes(ab_dtype: torch.dtype, sf_vec_size: int):
    if ab_dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    if ab_dtype == torch.uint8:
        return cutlass.Float4E2M1FN, (
            cutlass.Float8E4M3FN if sf_vec_size == 16 else cutlass.Float8E8M0FNU
        )
    raise ValueError(
        f"block-scaled operands are e4m3 or packed uint8 FP4, got {ab_dtype}"
    )


def _alpha_operand(
    alpha: torch.Tensor | None,
    experts: int,
    device,
    *,
    rows: int,
    rows_per_half: int,
    config: GroupedGemmConfig,
) -> tuple:
    """(alpha operand, its owner, use_alpha) with a one-column dummy when unused.

    The epilogue applies one alpha per output tile, chosen by the tile's first
    row, so a half boundary must sit on a tile edge and every half needs a column.
    """
    if alpha is None:
        alpha = torch.ones((experts, 1), dtype=torch.float32, device=device)
        return as_dynamic_cute_tensor(alpha, leading_dim=1), alpha, False
    assert alpha.dtype == torch.float32 and alpha.shape[0] == experts
    tile_rows = config.mma_tiler_mn[0] // (2 if config.use_2cta_instrs else 1)
    halves = -(-rows // rows_per_half)
    if (halves > 1 and rows_per_half % tile_rows) or halves > alpha.shape[1]:
        raise ValueError(
            f"alpha changes every {rows_per_half} of {rows} output rows, which "
            f"{tile_rows}-row output tiles with {alpha.shape[1]} alpha column(s) "
            "cannot apply"
        )
    return as_dynamic_cute_tensor(alpha, leading_dim=1), alpha, True


def prepare_masked_blockscaled(
    a: torch.Tensor,  # [E, m_max, K] e4m3 or [E, m_max, K // 2] uint8 FP4 pairs
    b: torch.Tensor,  # [E, N, K] or [E, N, K // 2] in the same element format
    sf_tokens: torch.Tensor,  # uint8 atoms, scale_atom_bytes(m_max, K, vec, E)
    sf_weights: torch.Tensor,  # uint8 atoms, scale_atom_bytes(N, K, vec, E)
    c: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    sf_vec_size: int,
    alpha: torch.Tensor | None = None,  # fp32 [E, halves] epilogue scales
    alpha_rows_per_half: int = 1,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    ab_dtype, sf_dtype = _blockscaled_dtypes(a.dtype, sf_vec_size)
    experts, m_max, k_store = a.shape
    n = b.shape[1]
    k = k_store * (8 // ab_dtype.width)
    alpha_arg, alpha_owner, use_alpha = _alpha_operand(
        alpha,
        experts,
        a.device,
        rows=n,
        rows_per_half=alpha_rows_per_half,
        config=config,
    )
    prepared = _compile_prepared(
        kernel_cls=_blockscaled_kernel_class_for(a.device),
        wrapper=_grouped_gemm_blockscaled_swap_ab,
        ab_dtype=ab_dtype,
        config=config,
        contiguous_segments=False,
        problem_shape=(n, m_max, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a, leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(sf_tokens, leading_dim=0),
            as_dynamic_cute_tensor(sf_weights, leading_dim=0),
            as_dynamic_cute_tensor(c, leading_dim=2),
            alpha_arg,
        ),
        group_m_arg=as_dynamic_cute_tensor(masked_m, leading_dim=0),
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
        device=a.device,
        kernel_kwargs=dict(sf_vec_size=sf_vec_size, sf_dtype=sf_dtype),
        wrapper_constexprs=(ab_dtype, sf_dtype, use_alpha, alpha_rows_per_half),
    )
    prepared.alpha_owner = alpha_owner
    return prepared


def prepare_contiguous_blockscaled(
    a: torch.Tensor,  # [M_ceil, K] e4m3 or [M_ceil, K // 2] uint8 FP4 pairs
    b: torch.Tensor,
    sf_tokens: torch.Tensor,  # uint8 atoms, scale_atom_bytes(M_ceil, K, vec)
    sf_weights: torch.Tensor,
    c: torch.Tensor,
    seg_offsets: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    sf_vec_size: int,
    alpha: torch.Tensor | None = None,
    alpha_rows_per_half: int = 1,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    """Use seg_offsets in the group_m slot to address flat expert segments."""
    ab_dtype, sf_dtype = _blockscaled_dtypes(a.dtype, sf_vec_size)
    m_ceil, k_store = a.shape
    experts, n, _ = b.shape
    k = k_store * (8 // ab_dtype.width)
    alpha_arg, alpha_owner, use_alpha = _alpha_operand(
        alpha,
        experts,
        a.device,
        rows=n,
        rows_per_half=alpha_rows_per_half,
        config=config,
    )
    prepared = _compile_prepared(
        kernel_cls=_blockscaled_kernel_class_for(a.device),
        wrapper=_grouped_gemm_blockscaled_swap_ab,
        ab_dtype=ab_dtype,
        config=config,
        contiguous_segments=True,
        problem_shape=(n, m_ceil, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a.unsqueeze(0), leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(sf_tokens, leading_dim=0),
            as_dynamic_cute_tensor(sf_weights, leading_dim=0),
            as_dynamic_cute_tensor(c.unsqueeze(0), leading_dim=2),
            alpha_arg,
        ),
        group_m_arg=as_dynamic_cute_tensor(seg_offsets, leading_dim=0),
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
        device=a.device,
        kernel_kwargs=dict(sf_vec_size=sf_vec_size, sf_dtype=sf_dtype),
        wrapper_constexprs=(ab_dtype, sf_dtype, use_alpha, alpha_rows_per_half),
    )
    prepared.alpha_owner = alpha_owner
    return prepared
