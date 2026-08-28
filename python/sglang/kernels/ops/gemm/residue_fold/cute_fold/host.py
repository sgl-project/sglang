# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Host entry for the SM103 MExt R1 fold GEMM (CuTeDSL).

Computes D[m_tok, n_w] = fold(A_act[2*m_tok, K] @ W[n_w, K]^T) where the
activation rows are row-pair interleaved [base0, res0, base1, res1, ...]
(sglang.kernels.ops.quantization.residue_nvfp4_quant scaled_fp4_quant_mext_r1 layout_mode=row_pair) and fold sums each
(base, residue) output pair. The kernel writes the transposed compact
D_t[n_w, m_tok]; callers get [m_tok, n_w] back via a transpose-copy.

Compile/call marshaling mirrors flashinfer's _cute_dsl_gemm_fp4_runner
(fake tensors + constexpr baking + TVM-FFI env stream).
"""

from __future__ import annotations

import functools

import torch

_KERNEL_CACHE: dict[tuple, tuple] = {}

# The tactic tables speak `store_mode`; _compile_fold_gemm speaks `mode`.
# Module-level so the warmup can enumerate the same space the call path
# reaches, instead of keeping a second copy that drifts.
STORE_MODE_TO_MODE = {
    "tma": "fold_tma",
}

# Valid sm103 tilers: (128|256, 128|256) (flashinfer sm103 candidates).
DEFAULT_MMA_TILER_MN = (128, 128)
DEFAULT_CLUSTER_SHAPE_MN = (1, 1)
SF_VEC_SIZE = 16


@functools.lru_cache(maxsize=1)
def _cutlass_modules():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernel_sm100_fold import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )
    from .kernel_sm103_fold import (
        Sm103BlockScaledPersistentDenseGemmKernel,
    )

    return (
        cutlass,
        cute,
        make_ptr,
        get_max_active_clusters,
        {
            "sm100": Sm100BlockScaledPersistentDenseGemmKernel,
            "sm103": Sm103BlockScaledPersistentDenseGemmKernel,
        },
    )


def _compile_fold_gemm(
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    out_dtype: torch.dtype,
    mode: str = "fold_tma",
    kernel_arch: str = "sm103",
    ab_stage_override: int | None = None,
    strided_weight: bool = False,
):
    """Compile a fold-GEMM kernel variant:

    - "fold_tma":         N-ext fold, register fold + TMA store
    - "plain_tma":        stock kernel/epilogue baseline (no fold)

    (The former "fold_direct" tiny-m mode was removed: dominated at every
    m_tok after the transposed-staging fold_tma optimization -- see
    fold_vs_plain_latency_results.md.)
    """
    assert mode in ("fold_tma", "plain_tma", "kloop_tma"), mode
    assert kernel_arch in ("sm100", "sm103"), kernel_arch
    assert kernel_arch == "sm103" or mode in ("fold_tma", "plain_tma", "kloop_tma")
    # kloop_tma: wrapped-K-loop -- act (K-concat, 2K) as A, weight as B with a
    # wrapped k coordinate, PLAIN epilogue writing C[m_tok, n_w]. sm100-only,
    # single-alpha.
    assert mode != "kloop_tma" or (
        kernel_arch == "sm100" and not strided_weight
    ), "kloop_tma: sm100, single-alpha"
    # The AB depth knob is sm100-only (the sm103 ctor has no such param).
    assert (
        kernel_arch == "sm100" or ab_stage_override is None
    ), "ab_stage_override is sm100-only"
    key = (
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        mode,
        kernel_arch,
        ab_stage_override,
        strided_weight,
    )
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    cutlass, cute, make_ptr, get_max_active_clusters, kernel_classes = (
        _cutlass_modules()
    )
    kernel_cls = kernel_classes[kernel_arch]
    c_cutlass_dtype = (
        cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    )

    if kernel_arch == "sm100":
        # sm100 kernel always TMA-stores; no use_tma_store knob.
        gemm = kernel_cls(
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            enable_pdl=True,
            fold_pairs_tma=(mode == "fold_tma"),
            ab_stage_override=ab_stage_override,
            k_loop=(mode == "kloop_tma"),
        )
    else:
        gemm = kernel_cls(
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=True,
            fold_pairs_tma=(mode == "fold_tma"),
        )

    sym_m = cute.sym_int()  # n_w (weight rows)
    sym_n = cute.sym_int()  # m2 = 2 * m_tok
    sym_k = cute.sym_int()
    sym_mtok = cute.sym_int()

    # N-ext uses mA=weight and mB=activation. kloop_tma swaps the operands.
    _a_rows, _b_rows = (sym_n, sym_m) if mode == "kloop_tma" else (sym_m, sym_n)
    # `strided_weight` supports an ext-K residue layer WITHOUT materialising a
    # base-K copy: the problem's K is k_base, but the weight rows are k_ext
    # apart because the tensor is a `weight_ext[:, :k_base_packed]` view.
    # The B-side scale needs no special handling: `weight_scale_base` is
    # already the re-swizzled base-K prefix, and sf_k derives from k_base.
    #
    # Only the weight operand gets a dynamic leading dim; the activation is
    # freshly quantised and always compact.
    #
    # A compact tensor keeps its own cache entry (strided_weight=False), so the
    # validated contiguous path compiles to exactly the same binary as before.
    _w_is_a = mode != "kloop_tma"

    def _operand(rows, is_weight):
        if strided_weight and is_weight:
            return cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (rows, sym_k),
                (cute.sym_int(), 1),
                assumed_align=32,
            )
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (rows, sym_k), stride_order=(1, 0), assumed_align=32
        )

    a_fake = _operand(_a_rows, _w_is_a)
    b_fake = _operand(_b_rows, not _w_is_a)
    if mode == "kloop_tma":
        # A is the K-concat activation at 2K -- its own symbolic K, unrelated
        # to B's (the weight's) sym_k. The kernel takes trip count from A and
        # wraps B; runtime shapes enforce the 2x relation.
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (_a_rows, cute.sym_int()),
            stride_order=(1, 0),
            assumed_align=32,
        )
    if mode == "kloop_tma":
        # Plain epilogue, C = [act_rows(m_tok), n_w] row-major -- no fold,
        # no transpose: the operand swap already put tokens on GEMM-M.
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_n, sym_m),
            stride_order=(1, 0),
            assumed_align=16,
        )
    elif mode == "fold_tma":
        # Physical output: row-major D[m_tok, n_w], written through a
        # transposed TMA view.
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_mtok, sym_m),
            stride_order=(1, 0),
            assumed_align=16,
        )
    else:
        # Stock C [n_w, m2] row-major.
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_m, sym_n),
            stride_order=(1, 0),
            assumed_align=16,
        )
    a_sf_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if kernel_arch == "sm100":
        # sm100 wrapper keeps current_stream but has NO fold_pairs param
        # (the fork keys off self.fold_pairs_tma set in the constructor).
        compiled = cute.compile(
            gemm.wrapper,
            a_fake,
            b_fake,
            c_fake,
            1,
            1,
            1,
            1,  # l
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            False,  # swap_ab
            lambda x: x,  # epilogue_op (alpha applied inside the kernel)
            options="--opt-level 2 --enable-tvm-ffi",
        )
    else:
        compiled = cute.compile(
            gemm.wrapper,
            a_fake,
            b_fake,
            c_fake,
            # Int64-typed dynamic args: any concrete int marshals at compile
            # time; the real per-call values are passed at runtime.
            1,
            1,
            1,
            1,  # l
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            False,  # swap_ab
            lambda x: x,  # epilogue_op (alpha applied inside the kernel)
            mode == "fold_tma",  # wrapper fold_pairs
            options="--opt-level 2 --enable-tvm-ffi",
        )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def mext_fold_gemm_sm103(
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN,
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN,
    store_mode: str = "tma",
    kernel_arch: str = "sm103",
    ab_stage_override: int | None = None,
) -> torch.Tensor:
    """Run the fold GEMM. Returns D[m_tok, n_w].

    Args:
        weight_fp4: [n_w, k_base // 2] uint8 packed FP4, K-major. May be a
            `weight_ext[:, :k_base_packed]` VIEW of an ext-K residue weight;
            the row stride is read from the tensor, so no base-K copy is
            needed. Must stay contiguous within a row.
        act_fp4_rowpair: [2 * m_tok, k // 2] uint8 packed FP4, row-pair layout.
        weight_sf: swizzled 128x4 FP8 scale for the k_base prefix
            (quant.vllm_integration layout_utils weight_scale_base).
        act_sf: swizzled 128x4 FP8 scale from scaled_fp4_quant_mext_r1.
        alpha: float32 scalar tensor (global dequant scale product).
    """
    n_w, k_packed = weight_fp4.shape
    m2 = act_fp4_rowpair.shape[0]
    assert m2 % 2 == 0, "activation must be row-pair interleaved (even rows)"
    m_tok = m2 // 2
    k = k_packed * 2

    # ext-K residue: `weight_fp4` may be a `weight_ext[:, :k_base_packed]` view,
    # so its row stride is k_ext_packed while its width is k_base_packed. The
    # shape already gives the problem K (= k_base); the stride is what tells the
    # kernel how far apart the rows sit. Detect it here rather than making the
    # caller pass k_ext -- a narrowed view carries the fact already, and a
    # caller-supplied value could disagree with the tensor silently.
    _w_ld = weight_fp4.stride(0)
    _strided_w = _w_ld != k_packed
    if _strided_w:
        assert weight_fp4.stride(1) == 1, (
            "weight must stay K-contiguous within a row; got stride "
            f"{weight_fp4.stride()}"
        )
        assert (
            _w_ld > k_packed
        ), f"weight row stride {_w_ld} < width {k_packed}: rows would overlap"
        # TMA needs the row starts aligned, and the operands are declared
        # assumed_align=32. A contiguous weight satisfies this via k_packed;
        # a strided one must satisfy it via the EXT width. Every ext layer in
        # the exported checkpoints does (k_ext in {2880,3200,3840,12160,14592}
        # -> packed strides all %32 == 0), but a coarser salient granularity
        # could break it, so refuse loudly instead of corrupting the read.
        assert _w_ld % 32 == 0, (
            f"strided weight row stride {_w_ld} is not 32B-aligned; "
            "materialise a contiguous base-K prefix for this layer"
        )
    if act_fp4_rowpair.stride(0) != k_packed or act_fp4_rowpair.stride(1) != 1:
        # The activation is produced by scaled_fp4_quant_mext_r1 and is always
        # compact; only the weight operand has a dynamic leading dim compiled in.
        raise ValueError(
            f"activation must be compact [{m2}, {k_packed}]; got stride "
            f"{act_fp4_rowpair.stride()}"
        )

    _mode = STORE_MODE_TO_MODE[store_mode]

    compiled = _compile_fold_gemm(
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        mode=_mode,
        kernel_arch=kernel_arch,
        ab_stage_override=ab_stage_override,
        strided_weight=_strided_w,
    )

    alpha_t = alpha.reshape(1).to(torch.float32)
    sf_m = _ceil_div(n_w, 128)  # weight scale-factor tiles
    sf_n = _ceil_div(m2, 128)  # activation scale-factor tiles
    sf_k = _ceil_div(k, SF_VEC_SIZE * 4)

    if store_mode == "tma":
        # N-ext: kernel writes row-major D[m_tok, n_w] via a transposed TMA view.
        d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
        if kernel_arch == "sm100":
            compiled(
                weight_fp4,
                act_fp4_rowpair,
                d_t,
                sf_m,
                sf_n,
                sf_k,
                weight_sf.data_ptr(),
                act_sf.data_ptr(),
                alpha_t,
            )
        else:
            compiled(
                weight_fp4,
                act_fp4_rowpair,
                d_t,
                sf_m,
                sf_n,
                sf_k,
                weight_sf.data_ptr(),
                act_sf.data_ptr(),
                alpha_t,
            )
        return d_t

    raise ValueError(f"unknown store_mode {store_mode!r}")


def kext_kloop_gemm_sm100(
    weight_fp4: torch.Tensor,
    act_fp4_kext: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype=None,
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN,
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN,
) -> torch.Tensor:
    """Wrapped-K-loop GEMM on sm100: D[m_tok, n_w] = act[m_tok, 2K] @ [W|W]^T
    with the weight stored ONCE (its k coordinate wraps in the kernel).

    weight_fp4:   [n_w, k/2] uint8 packed FP4, K-major, contiguous.
    act_fp4_kext: [m_tok, k] uint8 -- the K-concat quant output
                  (`scaled_fp4_quant_mext_r1(layout_mode="concat_k")`): 2K
                  elements per row, base in cols [0, K), residue in [K, 2K).
    act_sf:       SF bytes in canonical (m_tok, 2K) atom order.
    weight_sf:    swizzled SF for (n_w, K).

    Compiles ONCE per (tile, dtype): every dim is symbolic, so serving m is
    a free variable -- no bucket grid needed on this arch.
    """
    import torch

    out_dtype = out_dtype or torch.bfloat16
    n_w, k_packed = weight_fp4.shape
    k = k_packed * 2
    assert (
        weight_fp4.stride(0) == k_packed and weight_fp4.stride(1) == 1
    ), "kloop_tma expects a contiguous weight (mext_r1 weights are)"
    m_tok = act_fp4_kext.shape[0]
    assert act_fp4_kext.shape[1] == k, (
        f"activation must be [rows, {k}] bytes (2K elements), got "
        f"{tuple(act_fp4_kext.shape)}"
    )

    compiled = _compile_fold_gemm(
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn),
        out_dtype,
        mode="kloop_tma",
        kernel_arch="sm100",
    )
    d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
    alpha_t = alpha.reshape(1).to(torch.float32).contiguous()
    # wrapper (sf_m, sf_n, sf_k) = A-side rows/128, B-side rows/128, BASE-K
    # groups (the wrapper doubles the A side under k_loop).
    compiled(
        act_fp4_kext,
        weight_fp4,
        d_t,
        _ceil_div(m_tok, 128),
        _ceil_div(n_w, 128),
        _ceil_div(k // SF_VEC_SIZE, 4),
        act_sf.data_ptr(),
        weight_sf.data_ptr(),
        alpha_t,
    )
    return d_t
