# SPDX-License-Identifier: Apache-2.0

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.
import cutlass
import torch
from cutlass import Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from . import dense_blockscaled_gemm_sm120 as _dense
from .compile_utils import compile_block_scaled_gemm
from .cutlass_fp4 import cutlass_fp4_gemm

_CACHE = {}
_make_sfa_layout = _dense.sm120_make_smem_layout_sfa


def _prepare_alpha_for_launch(alpha_tensor, device):
    del device
    return alpha_tensor.reshape(1)


@dsl_user_op
def _make_evict_first_policy(*, loc=None, ip=None) -> Int64:
    """Create the PTX L2 streaming hint consumed by TMA load instructions."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _make_evict_last_policy(*, loc=None, ip=None) -> Int64:
    """Create the PTX L2 persistence hint consumed by TMA load instructions."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _scatter_add_f32(addr: Int64, value, *, loc=None, ip=None):
    """FP32 global reduction used by the accuracy-safe split-K path."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "{ red.relaxed.gpu.global.add.f32 [$0], $1; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _scatter_add_f32x2(addr: Int64, value0, value1, *, loc=None, ip=None):
    """Two adjacent FP32 reductions matching the dense kernel's pair loop."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            value0.ir_value(loc=loc, ip=ip),
            value1.ir_value(loc=loc, ip=ip),
        ],
        "{ red.relaxed.gpu.global.add.f32 [$0], $1; "
        "red.relaxed.gpu.global.add.f32 [$0+4], $2; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_split2_f32_m1(addr: Int64, value, *, loc=None, ip=None):
    """Store one M=1 down-projection partial into its split-owned slice."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 split; .reg .u64 offset; mov.u32 split, %ctaid.y; "
        "mul.wide.u32 offset, split, 20480; add.u64 offset, offset, $0; "
        "st.global.f32 [offset], $1; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_split2_f32x2_m1(addr: Int64, value0, value1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            value0.ir_value(loc=loc, ip=ip),
            value1.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 split; .reg .u64 offset; mov.u32 split, %ctaid.y; "
        "mul.wide.u32 offset, split, 20480; add.u64 offset, offset, $0; "
        "st.global.f32 [offset], $1; st.global.f32 [offset+4], $2; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_split2_f32_m9(addr: Int64, value, *, loc=None, ip=None):
    """Store one M=9 down-projection partial into its split-owned slice."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 split; .reg .u64 offset; mov.u32 split, %ctaid.y; "
        "mul.wide.u32 offset, split, 184320; add.u64 offset, offset, $0; "
        "st.global.f32 [offset], $1; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _store_split2_f32x2_m9(addr: Int64, value0, value1, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            value0.ir_value(loc=loc, ip=ip),
            value1.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 split; .reg .u64 offset; mov.u32 split, %ctaid.y; "
        "mul.wide.u32 offset, split, 184320; add.u64 offset, offset, $0; "
        "st.global.f32 [offset], $1; st.global.f32 [offset+4], $2; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _make_narrow_sfa_layout(tiled_mma, tile_shape_mnk, *args, **kwargs):
    # The SF tile is physically 128 rows even when the live MMA tile is narrower.
    if tile_shape_mnk[0] < 64:
        tile_shape_mnk = (64, tile_shape_mnk[1], tile_shape_mnk[2])
    return _make_sfa_layout(tiled_mma, tile_shape_mnk, *args, **kwargs)


_compute_grid = _dense.DenseGemmKernel._compute_grid


def _compute_direct_grid(
    c,
    tile_shape_mnk,
    max_active_clusters,
    direct_one_m_tile_scheduler,
    split_k_slices,
):
    params, grid = _compute_grid(
        c,
        tile_shape_mnk,
        max_active_clusters,
        direct_one_m_tile_scheduler,
        split_k_slices,
    )
    if direct_one_m_tile_scheduler and split_k_slices == 1:
        grid = (1, 1, params.problem_shape_ntile_mnl[1])
    return params, grid


def decode_fp4_gemm_variant(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    *,
    tile_m: int = 16,
    tile_n: int,
    tile_k: int,
    split_k: int,
    use_direct_scheduler: bool,
    use_prefetch: bool = False,
    enable_pdl: bool = False,
    use_m1_non_tma_c: bool = False,
    split_k_fp32: bool = False,
    split_k_staged: bool = False,
    stream_weight_cache: bool = False,
    stream_weight_scale_cache: bool = False,
    persist_activation_cache: bool = False,
    persist_activation_data_cache: bool = False,
    persist_weight_scale_cache: bool = False,
    k_loop_unroll: int = 2,
    variant_tag: str = "custom",
) -> torch.Tensor:
    if split_k_staged:
        raise ValueError(
            "staged split-K is not part of the integrated production kernel"
        )
    rows, packed_k = input.shape
    columns = weight.shape[1]
    kernel_variant = (
        f"qwen38_{variant_tag}_tm{tile_m}_tn{tile_n}_tk{tile_k}_sk{split_k}"
        f"_direct{int(use_direct_scheduler)}_pf{int(use_prefetch)}"
        f"_m1c{int(use_m1_non_tma_c)}_fp32red{int(split_k_fp32)}"
        f"_staged{int(split_k_staged)}"
        f"_bf{int(stream_weight_cache)}"
        f"_sfbf{int(stream_weight_scale_cache)}"
        f"_ap{int(persist_activation_cache)}"
        f"_adp{int(persist_activation_data_cache)}"
        f"_sfbp{int(persist_weight_scale_cache)}"
        f"_ku{k_loop_unroll}"
        "_tmap1" + (f"_rows{rows}" if split_k_staged else "")
    )
    sf_m = (rows + 127) // 128
    sf_n = (columns + 127) // 128
    sf_k = (packed_k * 2 // 16 + 3) // 4
    if split_k_staged:
        if (
            not split_k_fp32
            or rows not in (1, 9)
            or columns != 5120
            or split_k not in (2, 4, 8, 17)
        ):
            raise ValueError(
                "staged split-K supports FP32 down projections with "
                "M in {1, 9} and split_k in {2, 4, 8, 17}"
            )
        partial_bytes = split_k * rows * columns * 4
        result_bytes = rows * columns * 2
        storage = torch.empty(
            partial_bytes + result_bytes,
            dtype=torch.uint8,
            device=input.device,
        )
        partials = (
            storage[:partial_bytes].view(torch.float32).view(split_k, rows, columns)
        )
        output = partials[0]
        _result = storage[partial_bytes:].view(torch.bfloat16).view(rows, columns)
    else:
        output = torch.empty(
            rows,
            columns,
            device=input.device,
            dtype=torch.float32 if split_k_fp32 else torch.bfloat16,
        )
    cache_key = (
        16,
        (tile_m, tile_n),
        (1, 1),
        False,
        use_prefetch,
        kernel_variant,
        None,
        enable_pdl,
        torch.float32 if split_k_fp32 else torch.bfloat16,
    )
    # These overrides are needed only while compiling the 16-row decode
    # specialization. Restore them immediately so the production large-M
    # FlashInfer dispatcher and the trusted baseline retain their stock code.
    previous_sfa_layout = _dense.sm120_make_smem_layout_sfa
    previous_compute_grid = _dense.DenseGemmKernel._compute_grid
    previous_scatter_add = _dense.scatter_add_bf16
    previous_scatter_add_x2 = _dense.scatter_add_bf16x2
    previous_copy = _dense.cute.copy
    _dense.sm120_make_smem_layout_sfa = _make_narrow_sfa_layout
    _dense.DenseGemmKernel._compute_grid = staticmethod(_compute_direct_grid)
    if split_k_fp32:
        if split_k_staged and rows == 1:
            _dense.scatter_add_bf16 = _store_split2_f32_m1
            _dense.scatter_add_bf16x2 = _store_split2_f32x2_m1
        elif split_k_staged and rows == 9:
            _dense.scatter_add_bf16 = _store_split2_f32_m9
            _dense.scatter_add_bf16x2 = _store_split2_f32x2_m9
        else:
            _dense.scatter_add_bf16 = _scatter_add_f32
            _dense.scatter_add_bf16x2 = _scatter_add_f32x2
    if (
        stream_weight_cache
        or stream_weight_scale_cache
        or persist_activation_cache
        or persist_activation_data_cache
        or persist_weight_scale_cache
    ):
        tma_copy_index = 0

        def _copy_with_operand_cache_policy(atom, src, dst, *args, **kwargs):
            nonlocal tma_copy_index
            # CuTe recursively calls copy while constructing layouts. Only the
            # barrier-backed calls below are runtime TMA loads. The mainloop
            # emits them in A, SFA, B, SFB order.
            if "tma_bar_ptr" not in kwargs:
                return previous_copy(atom, src, dst, *args, **kwargs)
            role = tma_copy_index % 4
            tma_copy_index += 1
            if (role == 2 and stream_weight_cache) or (
                role == 3 and stream_weight_scale_cache
            ):
                kwargs["cache_policy"] = _make_evict_first_policy()
            elif role == 3 and persist_weight_scale_cache:
                kwargs["cache_policy"] = _make_evict_last_policy()
            elif (role in (0, 1) and persist_activation_cache) or (
                role == 0 and persist_activation_data_cache
            ):
                kwargs["cache_policy"] = _make_evict_last_policy()
            return previous_copy(atom, src, dst, *args, **kwargs)

        _dense.cute.copy = _copy_with_operand_cache_policy
    try:
        kernel, _ = compile_block_scaled_gemm(
            _CACHE,
            cache_key,
            lambda: _dense.DenseGemmKernel(
                16,
                (tile_m, tile_n),
                (1, 1),
                tile_k=tile_k,
                use_prefetch=use_prefetch,
                enable_pdl=enable_pdl,
                split_k_slices=split_k,
                # This switch selects the dense kernel's direct atomic
                # epilogue. During FP32 compilation its two helper symbols are
                # temporarily rebound to the FP32 reductions above.
                split_k_atomic_bf16=split_k > 1,
                use_m1_non_tma_c=use_m1_non_tma_c,
                k_loop_unroll=k_loop_unroll,
                # Down and LM-head launch exactly one CTA for every scheduled
                # (split, N-tile). Bypass the persistent scheduler and terminate
                # each CTA after its sole tile.
                direct_one_m_tile_scheduler=use_direct_scheduler,
                single_work_tile_per_cta=use_direct_scheduler,
            ),
            ab_cutlass_dtype=cutlass.Uint8,
            sf_dtype=cutlass.Float8E4M3FN,
            c_cutlass_dtype=(cutlass.Float32 if split_k_fp32 else cutlass.BFloat16),
            ab_assumed_align=32,
            cluster_shape_mn=(1, 1),
            swap_ab=False,
            sf_m=sf_m,
            sf_n=sf_n,
            sf_k=sf_k,
            batch_size=1,
            cache_module_name="qwen38_mm_fp4",
        )
    finally:
        _dense.sm120_make_smem_layout_sfa = previous_sfa_layout
        _dense.DenseGemmKernel._compute_grid = staticmethod(previous_compute_grid)
        _dense.scatter_add_bf16 = previous_scatter_add
        _dense.scatter_add_bf16x2 = previous_scatter_add_x2
        _dense.cute.copy = previous_copy
    if split_k > 1 and not split_k_staged:
        output.zero_()
    kernel(
        input,
        weight.T,
        output,
        sf_m,
        sf_n,
        sf_k,
        input_sf.data_ptr(),
        weight_sf.T.data_ptr(),
        _prepare_alpha_for_launch(alpha, input.device),
    )
    if split_k_staged:
        raise AssertionError("unreachable staged split-K path")
    return output.to(torch.bfloat16) if split_k_fp32 else output


def decode_fp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    rows = input.shape[0]
    reduction_size = input.shape[1] * 2
    columns = weight.shape[1]
    is_down_projection = columns < reduction_size
    is_lm_head = columns == 248320
    use_m1_non_tma_c = rows == 1
    return decode_fp4_gemm_variant(
        input,
        weight,
        input_sf,
        weight_sf,
        alpha,
        # N=64 is the smallest supported SM120 warp-MMA tile. For down it keeps
        # 80 independent output tiles in flight without a split-K reduction.
        tile_n=64,
        tile_k=512,
        # A single CTA accumulates the full K range in FP32 and rounds once to
        # BF16. This removes the staged partial buffer and reduction launch while
        # retaining the accuracy-safe accumulation order.
        split_k=1,
        use_direct_scheduler=is_down_projection or is_lm_head,
        # The producer-prefetch schedule wins the interleaved cold-L2 sweep for
        # the nine-row gate/up shape. It regresses M=1 and does not help the
        # direct-scheduled down/LM paths, so keep the dispatch exact.
        use_prefetch=rows == 9 and not is_down_projection and not is_lm_head,
        use_m1_non_tma_c=use_m1_non_tma_c,
        split_k_fp32=False,
        split_k_staged=False,
        # All recorded decode GEMMs make one pass over their large FP4 weights,
        # while every N tile reuses the much smaller activation operands.
        stream_weight_cache=rows in (1, 9),
        # In end-to-end serving, retaining each layer's 5.6--11 MiB weight-scale
        # tensor evicts attention/SSM state before the next layer. Stream SFB
        # together with the one-pass weights; the original persistent policy is
        # faster only when repeatedly benchmarking one isolated GEMM.
        stream_weight_scale_cache=rows in (1, 9),
        # Preserve both A and SFA; hinting only one of the pair measured neutral.
        persist_activation_cache=rows in (1, 9),
        persist_weight_scale_cache=False,
        variant_tag=(
            "sm120_down"
            if is_down_projection
            else "sm120_lm" if is_lm_head else "sm120_custom"
        )
        + ("_m1c" if use_m1_non_tma_c else ""),
    )


def large_fp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Candidate-owned SM120 paths for prefill-sized GEMMs."""
    if out_dtype is not torch.bfloat16:
        raise ValueError(f"unsupported output dtype: {out_dtype}")
    columns = weight.shape[1]
    if columns == 5120:
        output = torch.empty(
            input.shape[0], columns, device=input.device, dtype=torch.bfloat16
        )
        cutlass_fp4_gemm(
            output,
            input,
            weight.T,
            input_sf,
            weight_sf.T,
            alpha.reshape(1),
        )
        return output
    return decode_fp4_gemm_variant(
        input,
        weight,
        input_sf,
        weight_sf,
        alpha,
        tile_m=128,
        tile_n=128,
        tile_k=128,
        split_k=1,
        use_direct_scheduler=False,
        use_prefetch=columns != 248320,
        enable_pdl=True,
        # Gate/up repeatedly revisits its 11 MiB activation operand while
        # sweeping N. Persist A only: applying the same hint to SFA is neutral,
        # while LM-head regresses under either prefill policy.
        persist_activation_data_cache=columns == 34816,
        # Five-way unrolling trims the large gate/up path without helping the
        # much longer LM-head sweep; keep LM-head on its established KU4.
        k_loop_unroll=5 if columns == 34816 else 4,
        variant_tag="sm120_prefill",
    )
