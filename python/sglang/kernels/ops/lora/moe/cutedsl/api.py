# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile grouped GEMMs with dynamic row counts; providers cache the code."""

import importlib
from typing import Any, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import msgspec
import torch
from cutlass.cute.runtime import from_dlpack


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
    stream: cuda.CUstream
    owners: tuple  # the schedule and epilogue tensors behind those views

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


def scale_atom_bytes(rows: int, k: int, sf_vec_size: int, experts: int = 1) -> int:
    """Bytes of the tcgen05 scale-factor atoms covering (rows, k) per expert."""
    row_atoms = -(-rows // 128)
    k_atoms = -(-(k // sf_vec_size) // 4)
    return experts * row_atoms * k_atoms * 512


# Kernel module and class per family and SM major; a module loads on first use.
_KERNEL_CLASSES = {
    "bf16": {
        9: ("kernel_sm90_bf16", "GroupedGemmKernelSm90Bf16"),
        10: ("kernel_sm100_bf16", "GroupedGemmKernelSm100Bf16"),
    },
    "fp8": {
        9: ("kernel_sm90_fp8", "GroupedGemmKernelSm90Fp8"),
        10: ("kernel_sm100_fp8", "GroupedGemmKernelSm100Fp8"),
    },
    "fp8_rowwise": {
        9: ("kernel_sm90_fp8", "GroupedGemmKernelSm90Fp8Rowwise"),
        10: ("kernel_sm100_fp8", "GroupedGemmKernelSm100Fp8Rowwise"),
    },
}


def _kernel_class_for(family: str, device: torch.device):
    major, _minor = torch.cuda.get_device_capability(device)
    kernels = _KERNEL_CLASSES[family]
    if major not in kernels:
        supported = " or ".join(f"SM{arch}0" for arch in kernels)
        raise NotImplementedError(
            f"the {family} grouped GEMM runs on {supported}; device is sm{major}x"
        )
    module, name = kernels[major]
    return getattr(
        importlib.import_module(f"sglang.kernels.ops.lora.moe.cutedsl.{module}"), name
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


@cute.jit
def _grouped_gemm_fp8_swap_ab(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # physical routed input (expert, m_max, k) e4m3
    b: cute.Tensor,  # physical weight (expert, n_out, k) e4m3
    sf_tokens: cute.Tensor,  # fp32: block (expert, rows, k/128) or row (expert, rows)
    sf_weights: cute.Tensor,  # fp32: block (expert, n/128, k/128) or row (expert, n_out)
    c: cute.Tensor,  # physical output (expert, m_max, n_out) bf16
    group_m: cute.Tensor,
    direct_schedule: cute.Tensor,
    schedule_tiles: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    weight_mke = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    token_nke = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    output_mne = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[2, 1, 0]))
    # Kernel A uses weight scales and kernel B uses token scales.
    scale_modes = [1, 2, 0]
    if cutlass.const_expr(cute.rank(sf_weights.layout) == 2):
        scale_modes = [1, 0]
    sfa_weights = cute.make_tensor(
        sf_weights.iterator, cute.select(sf_weights.layout, mode=scale_modes)
    )
    sfb_tokens = cute.make_tensor(
        sf_tokens.iterator, cute.select(sf_tokens.layout, mode=scale_modes)
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
    owners: tuple = (),
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
        stream=stream,
        owners=(direct_schedule, schedule_tiles, *owners),
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
        kernel_cls=_kernel_class_for("bf16", a.device),
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
        kernel_cls=_kernel_class_for("bf16", a.device),
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


def _prepare_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    sf_tokens: torch.Tensor,
    sf_weights: torch.Tensor,
    c: torch.Tensor,
    group: torch.Tensor,
    *,
    rowwise: bool,
    contiguous: bool,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    assert sf_tokens.dtype == torch.float32 and sf_weights.dtype == torch.float32
    if contiguous:
        m, k = a.shape
        experts, n, _ = b.shape
        a, c = a.unsqueeze(0), c.unsqueeze(0)
    else:
        experts, m, k = a.shape
        n = b.shape[1]
    scale_dim = 1 if rowwise else 2
    return _compile_prepared(
        kernel_cls=_kernel_class_for("fp8_rowwise" if rowwise else "fp8", a.device),
        wrapper=_grouped_gemm_fp8_swap_ab,
        ab_dtype=cutlass.Float8E4M3FN,
        config=config,
        contiguous_segments=contiguous,
        problem_shape=(n, m, k, experts),
        operand_args=(
            as_dynamic_cute_tensor(a, leading_dim=2),
            as_dynamic_cute_tensor(b, leading_dim=2),
            as_dynamic_cute_tensor(sf_tokens, leading_dim=scale_dim),
            as_dynamic_cute_tensor(sf_weights, leading_dim=scale_dim),
            as_dynamic_cute_tensor(c, leading_dim=2),
        ),
        group_m_arg=as_dynamic_cute_tensor(group, leading_dim=0),
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
    return _prepare_fp8(
        a,
        b,
        sf_tokens,
        sf_weights,
        c,
        masked_m,
        rowwise=False,
        contiguous=False,
        config=config,
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
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
    return _prepare_fp8(
        a,
        b,
        sf_tokens,
        sf_weights,
        c,
        seg_offsets,
        rowwise=False,
        contiguous=True,
        config=config,
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
    )


def prepare_masked_fp8_rowwise(
    a: torch.Tensor,
    b: torch.Tensor,
    sf_tokens: torch.Tensor,  # fp32 [E, m_max]
    sf_weights: torch.Tensor,  # fp32 [E, N]
    c: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    return _prepare_fp8(
        a,
        b,
        sf_tokens,
        sf_weights,
        c,
        masked_m,
        rowwise=True,
        contiguous=False,
        config=config,
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
    )


def prepare_contiguous_fp8_rowwise(
    a: torch.Tensor,
    b: torch.Tensor,
    sf_tokens: torch.Tensor,  # fp32 [1, M]
    sf_weights: torch.Tensor,  # fp32 [E, N]
    c: torch.Tensor,
    seg_offsets: torch.Tensor,
    *,
    config: GroupedGemmConfig,
    direct_schedule: torch.Tensor | None = None,
    schedule_tiles: torch.Tensor | None = None,
) -> PreparedGroupedGemm:
    return _prepare_fp8(
        a,
        b,
        sf_tokens,
        sf_weights,
        c,
        seg_offsets,
        rowwise=True,
        contiguous=True,
        config=config,
        direct_schedule=direct_schedule,
        schedule_tiles=schedule_tiles,
    )
