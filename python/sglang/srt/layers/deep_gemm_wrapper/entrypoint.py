import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.utils import is_sail

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm

    # SAIL does not use the TMA scale layout; other platforms keep the existing
    # layout conversion and ownership protection.
    if not is_sail():
        from deep_gemm.utils.layout import (
            get_mn_major_tma_aligned_tensor as _get_mn_major_tma_aligned_tensor,
        )

        def get_mn_major_tma_aligned_tensor(sf: torch.Tensor) -> torch.Tensor:
            """Transform ``sf`` into an MN-major, TMA-aligned layout for DeepGEMM.

            When ``sf`` is already in that layout, sgl-deep-gemm's fast path
            (<= 0.1.4.post1) returns a NON-OWNING ``torch::from_blob`` alias of
            ``sf`` across the TVM-FFI boundary. Callers rebind the result over
            their only reference (``x = get_mn_major_tma_aligned_tensor(x)``),
            which frees the storage while the GEMM still reads through the alias
            -- a use-after-free that surfaces as NaN logits or "pointer resides
            on host memory" during CUDA graph capture once the allocator reuses
            the block. Hand back ``sf`` itself in that case so ownership is
            preserved.
            """
            out = _get_mn_major_tma_aligned_tensor(sf)
            if out.data_ptr() == sf.data_ptr():
                assert out.shape == sf.shape and out.stride() == sf.stride()
                return sf
            return out

    else:

        def get_mn_major_tma_aligned_tensor(hidden_states_scale):
            return hidden_states_scale


_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()


_is_sail = is_sail()

if _is_sail:
    from sglang.srt.layers import deep_gemm_tuner as tuner


# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    configs: Tuple = None,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED
    if _is_sail and lhs[1].shape[-1] == 1 and rhs[1].shape[-1] == 1:
        kernel_type = (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL
        )

    if _is_sail:
        best_config = (
            configs
            if configs is not None
            else tuner.get_deep_gemm_config(
                expected_m, n, k, num_groups=num_groups, dtype="fp8"
            )
        )

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    lhs = _ensure_cuda(lhs)
    rhs = _ensure_cuda(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            fp4_kwargs = {}
            if recipe_a is not None:
                fp4_kwargs["recipe_a"] = recipe_a
            if recipe_b is not None:
                fp4_kwargs["recipe_b"] = recipe_b

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **fp4_kwargs,
                **(
                    (
                        dict(
                            enable_overlap=True,
                            max_block_n=max_block_n,
                            signal=overlap_args.signal,
                        )
                        if not _is_sail  # SAIL names its SBO argument differently
                        else dict(
                            enable_sbo_overlap=True,
                            max_block_n=max_block_n,
                            signal=overlap_args.signal,
                        )
                    )
                    if overlap_args is not None
                    else {}
                ),
                **(
                    dict(
                        configs=best_config,
                    )
                    if _is_sail
                    else {}
                ),
            )


def _ensure_cuda(
    pair: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        pair[0].cuda() if not pair[0].is_cuda else pair[0],
        pair[1].cuda() if not pair[1].is_cuda else pair[1],
    )


def grouped_gemm_nt_bf16_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    configs=None,
):
    num_groups, _, k = a.shape
    _, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    if _is_sail:
        best_config = (
            configs
            if configs is not None
            else tuner.get_deep_gemm_config(
                expected_m, n, k, num_groups=num_groups, dtype="bf16"
            )
        )

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return compile_utils.bf16_grouped_gemm_nt("masked")(
            a,
            b,
            d,
            masked_m,
            expected_m,
            **(
                dict(
                    **(
                        {"enable_sbo_overlap": True}
                        if _is_sail
                        else {"enable_overlap": True}
                    ),
                    max_block_n=max_block_n,
                    signal=overlap_args.signal,
                )
                if overlap_args is not None
                else {}
            ),
            **(
                dict(
                    configs=best_config,
                )
                if _is_sail
                else {}
            ),
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    configs: Tuple = None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG
    if _is_sail and lhs[1].shape[-1] == 1 and rhs[1].shape[-1] == 1:
        kernel_type = (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL
        )

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    fp4_kwargs = {}
    if recipe_a is not None:
        fp4_kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        fp4_kwargs["recipe_b"] = recipe_b

    if _is_sail:
        best_config = (
            configs
            if configs is not None
            else tuner.get_deep_gemm_config(m, n, k, num_groups=num_groups, dtype="fp8")
        )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            lhs,
            rhs,
            out,
            m_indices,
            **fp4_kwargs,
            **({"configs": best_config} if _is_sail else {}),
        )


def grouped_gemm_nt_bf16_contig(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    configs=None,
):
    m, k = a.shape
    num_groups, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    if _is_sail:
        best_config = (
            configs
            if configs is not None
            else tuner.get_deep_gemm_config(
                m, n, k, num_groups=num_groups, dtype="bf16"
            )
        )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        compile_utils.bf16_grouped_gemm_nt("contiguous")(
            a,
            b,
            d,
            m_indices,
            **(
                dict(
                    configs=best_config,
                )
                if _is_sail
                else {}
            ),
        )


def get_contiguous_layout_alignment(expected_m: int, num_groups: int) -> int:
    alignment = (
        deep_gemm.get_m_alignment_for_contiguous_layout()
        if _is_sail
        else deep_gemm.get_mk_alignment_for_contiguous_layout()
    )
    if isinstance(alignment, tuple):
        alignment = alignment[0]

    theoretical_alignment = getattr(
        deep_gemm, "get_theoretical_mk_alignment_for_contiguous_layout", None
    )
    if theoretical_alignment is None:
        return alignment

    per_group_m = (expected_m + num_groups - 1) // num_groups
    try:
        try:
            candidate = theoretical_alignment(per_group_m)
        except TypeError:
            candidate = theoretical_alignment(
                expected_m=expected_m, num_groups=num_groups
            )
    except Exception:
        return alignment
    return candidate if 0 < candidate <= alignment else alignment


@contextmanager
def contiguous_layout_alignment_scope(alignment: Optional[int]):
    if alignment is None:
        yield
        return

    getter = getattr(deep_gemm, "get_mk_alignment_for_contiguous_layout", None)
    setter = getattr(deep_gemm, "set_mk_alignment_for_contiguous_layout", None)
    if getter is None or setter is None:
        yield
        return

    previous = getter()
    if isinstance(previous, tuple):
        previous = previous[0]
    setter(alignment)
    try:
        yield
    finally:
        setter(previous)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    configs: Optional[Tuple] = None,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = (
        compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16_BLOCK32
        if recipe == (1, 32, 32)
        else compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16
    )
    if _is_sail and recipe is None and lhs[1].shape[-1] == rhs[1].shape[-1] == 1:
        kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16_CHANNEL

    if _is_sail:
        best_config = (
            configs
            if configs is not None
            else tuner.get_deep_gemm_config(m, n, k, num_groups=num_groups, dtype="fp8")
        )

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
            **({"recipe": recipe} if recipe is not None else {}),
            **({"configs": best_config} if _is_sail else {}),
        )


def gemm_nt_mxfp8_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    disable_cast = lhs[1].dtype == torch.int and rhs[1].dtype == torch.int

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_fp4_gemm_nt(
            lhs,
            rhs,
            out,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
            disable_ue8m0_cast=disable_cast,
        )


def gemm_nt_bf16bf16f32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
):
    m, k = lhs.shape
    n, _ = rhs.shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_splits: Optional[int],
):
    if x.shape[0] == 0:
        return
    if _is_sail:
        raise NotImplementedError(
            "This SAIL adaptation does not enable the DeepGEMM TF32 MHC contract; "
            "please use the existing TileLang/PyTorch MHC path."
        )
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=num_splits)


def grouped_gemm_nt_f8f8bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
    configs=None,
):
    assert _is_sail, "grouped_gemm_nt_f8f8bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD
    if _is_sail and lhs[1].shape[-1] == 1 and rhs[1].shape[-1] == 1:
        kernel_type = (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL
        )

    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            m, n, k, num_groups=num_groups, nopad=True, dtype="fp8"
        )
    )

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(
            lhs, rhs, out, m_indices, m_rows, configs=best_config
        )


# int8 implementation
def gemm_nt_i8i8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    configs=None,
):
    assert _is_sail, "gemm_nt_i8i8bf16 is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(m, n, k, num_groups=num_groups, dtype="int8")
    )
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_I8I8BF16

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.gemm_int8_int8_bf16_nt(lhs, rhs, out, configs=best_config)


def grouped_gemm_nt_i8i8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    configs=None,
):
    assert _is_sail, (
        "grouped_gemm_nt_i8i8bf16_contig is only supported by SAIL DeepGEMM"
    )

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(m, n, k, num_groups=num_groups, dtype="int8")
    )
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(
            lhs, rhs, out, m_indices, configs=best_config
        )


def grouped_gemm_nt_i8i8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    configs=None,
):
    assert _is_sail, (
        "grouped_gemm_nt_i8i8bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            expected_m, n, k, num_groups=num_groups, dtype="int8"
        )
    )
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            configs=best_config,
            **(
                dict(
                    enable_sbo_overlap=True,
                    max_block_n=max_block_n,
                    signal=overlap_args.signal,
                )
                if overlap_args is not None
                else {}
            ),
        )


def grouped_gemm_nt_i8i8bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
    configs=None,
):
    assert _is_sail, "grouped_gemm_nt_i8i8bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            m, n, k, num_groups=num_groups, nopad=True, dtype="int8"
        )
    )
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_NOPAD

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(
            lhs, rhs, out, m_indices, m_rows, configs=best_config
        )


def grouped_gemm_nt_bf16_nopad(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
    configs=None,
):
    assert _is_sail, "grouped_gemm_nt_bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_NOPAD
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            m, n, k, num_groups=num_groups, nopad=True, dtype="bf16"
        )
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
            lhs, rhs, out, m_indices, m_rows, configs=best_config
        )


# mxfp4 implementation
def gemm_nt_f4f4bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    configs=None,
):
    assert _is_sail, "gemm_nt_f4f4bf16 is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1

    kernel_type = (
        compile_utils.DeepGemmKernelType.GEMM_NT_F4F4BF16
        if bias is None
        else compile_utils.DeepGemmKernelType.GEMM_NT_F4F4BF16_BIAS
    )
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(m, n, k, num_groups=num_groups, dtype="fp4")
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.gemm_fp4_fp4_bf16_nt(lhs, rhs, bias, out, configs=best_config)


def grouped_gemm_nt_f4f4bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    configs=None,
):
    assert _is_sail, (
        "grouped_gemm_nt_f4f4bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED
        if bias is None
        else compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS
    )

    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            expected_m, n, k, num_groups=num_groups, dtype="fp4"
        )
    )

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            return deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
                lhs,
                rhs,
                bias,
                out,
                masked_m,
                expected_m,
                configs=best_config,
                **(
                    dict(
                        enable_sbo_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_f4f4bf16_nopad(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor = None,
    configs=None,
):
    assert _is_sail, "grouped_gemm_nt_f4f4bf16_nopad is only supported by SAIL DeepGEMM"

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape

    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD
        if bias is None
        else compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS
    )
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            m, n, k, num_groups=num_groups, nopad=True, dtype="fp4"
        )
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
            lhs, rhs, bias, out, m_indices, m_rows, configs=best_config
        )


def _get_w4a16_weight_shape(weight: torch.Tensor) -> Tuple[int, int, int]:
    """Return the logical E/N/K; MMA consumes FP4 directly while VALU/INT4 use the
    Marlin layout."""
    if weight.dtype == torch.uint8:
        num_groups, n, packed_k = weight.shape
        return num_groups, n, packed_k * 2
    return weight.shape[0], weight.shape[2] // 2, weight.shape[1] * 16


def grouped_gemm_nt_bf16i4bf16_masked(
    lhs: torch.Tensor,
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    configs=None,
):
    # lhs: shape [e, m, k], dtype bf16
    # VALU/INT4: int32 weights [e, k//16, n*2], scales [e, k//32, n].
    # Scales are uint8 E8M0 (VALU) or BF16 (INT4).
    # MMA: uint8 FP4 weights [e, n, k//2], uint8 tiled scales [e, n//64, k*2].

    assert _is_sail, (
        "grouped_gemm_nt_bf16i4bf16_masked is only supported by SAIL DeepGEMM"
    )

    num_groups, _, k = lhs.shape
    _, n, _ = _get_w4a16_weight_shape(rhs[0])

    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            expected_m, n, k, num_groups=num_groups, dtype="int4"
        )
    )

    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED
        if rhs[0].dtype == torch.uint8
        else compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MASKED
    )

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        deep_gemm.m_grouped_gemm_w4a16_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            best_config,
            **(
                dict(
                    enable_sbo_overlap=True,
                    max_block_n=max_block_n,
                    signal=overlap_args.signal,
                )
                if overlap_args is not None
                else {}
            ),
        )


def grouped_gemm_nt_bf16i4bf16_nopad(
    lhs: torch.Tensor,
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: Optional[torch.Tensor] = None,
    configs=None,
):
    # lhs: shape [m, k], dtype bf16
    # VALU/INT4: int32 weights [e, k//16, n*2], scales [e, k//32, n].
    # Scales are uint8 E8M0 (VALU) or BF16 (INT4).
    # MMA: uint8 FP4 weights [e, n, k//2], uint8 tiled scales [e, n//64, k*2].

    assert _is_sail, (
        "grouped_gemm_nt_bf16i4bf16_nopad is only supported by SAIL DeepGEMM"
    )

    m, k = lhs.shape
    num_groups, n, _ = _get_w4a16_weight_shape(rhs[0])

    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD
        if rhs[0].dtype == torch.uint8
        else compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_NOPAD
    )

    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            m, n, k, num_groups=num_groups, nopad=True, dtype="int4"
        )
    )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_w4a16_nopad(
            lhs,
            rhs,
            out,
            m_indices,
            m_rows,
            best_config,
        )


def _require_sail_fused_api(gemm_name: str) -> None:
    if not _is_sail:
        raise ValueError("gather-read fused MoE is only supported by SAIL DeepGEMM.")
    if not ENABLE_JIT_DEEPGEMM:
        raise RuntimeError("SAIL fused MoE requires DeepGEMM JIT to be enabled.")
    for name in ("moe_align_block_size", gemm_name):
        if not callable(getattr(deep_gemm, name, None)):
            raise RuntimeError(
                f"SAIL fused MoE is missing the DeepGEMM API {name}; please install "
                "a version that includes the fused MoE API refactor (53ea8ff)."
            )


def grouped_gemm_nt_bf16i4bf16_fused(
    lhs: torch.Tensor,
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_w4a16_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs.shape
    num_groups, n, _ = _get_w4a16_weight_shape(rhs[0])
    if rhs[0].dtype == torch.uint8:
        kernel_type = (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED
        )
    elif rhs[1].dtype == torch.uint8:
        kernel_type = (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED
        )
    else:
        kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_w4a16_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_bf16bf16bf16_fused(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_bf16_bf16_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs.shape
    num_groups, n, _ = rhs.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_i8i8bf16_fused(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_int8_int8_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_f8f8bf16_fused(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
    perchannel_quant: bool = False,
):
    gemm_name = "m_grouped_gemm_fp8_fp8_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    # At K=128 the last dim of block scales is also 1, so it cannot be used to
    # infer channel mode.
    kernel_type = (
        compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL
        if perchannel_quant
        else compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED
    )
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def grouped_gemm_nt_f4f4bf16_fused(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_rows: torch.Tensor,
    expert_ids_and_cumsum: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    aligned_num_m_blocks: torch.Tensor,
    configs: Tuple,
):
    gemm_name = "m_grouped_gemm_fp4_fp4_bf16_nt_fused"
    _require_sail_fused_api(gemm_name)
    if lhs[1].dtype != torch.uint16 or not lhs[1].is_contiguous():
        raise ValueError(
            "SAIL FP4 gather-read requires uint16 K-major contiguous activation scales."
        )
    # Consistent with the existing FP4 wrapper, warmup and tuning use the packed
    # byte width for K.
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_fused(
            lhs,
            rhs,
            out,
            m_rows,
            expert_ids_and_cumsum,
            sorted_token_ids,
            aligned_num_m_blocks,
            configs,
        )


def moe_align_block_size(
    A: torch.Tensor,
    B: torch.Tensor,
    topk_ids: torch.Tensor,
    perchannel_quant: bool = False,
    configs=None,
):
    gemm_names = {
        torch.float8_e4m3fn: ("fp8", "m_grouped_gemm_fp8_fp8_bf16_nt_fused"),
        torch.int8: ("int8", "m_grouped_gemm_int8_int8_bf16_nt_fused"),
        torch.bfloat16: ("bf16", "m_grouped_gemm_bf16_bf16_bf16_nt_fused"),
        torch.uint8: ("fp4", "m_grouped_gemm_fp4_fp4_bf16_nt_fused"),
    }
    if A.dtype not in gemm_names:
        raise ValueError(f"SAIL fused MoE does not support activation dtype {A.dtype}.")
    gemm_dtype, gemm_name = gemm_names[A.dtype]
    m, k = A.shape
    num_groups, n, _ = B.shape
    if A.dtype == torch.bfloat16 and B.dtype in (torch.int32, torch.uint8):
        num_groups, n, _ = _get_w4a16_weight_shape(B)
        gemm_dtype, gemm_name = "int4", "m_grouped_gemm_w4a16_fused"
    _require_sail_fused_api(gemm_name)
    best_config = (
        configs
        if configs is not None
        else tuner.get_deep_gemm_config(
            m, n, k, num_groups=num_groups, nopad=True, dtype=gemm_dtype
        )
    )
    return deep_gemm.moe_align_block_size(
        A, B, topk_ids, perchannel_quant=perchannel_quant, config=best_config
    )


def update_deep_gemm_config(gpu_id: int):
    # deep_gemm.set_pdl can initialize CUDA state, so run it only after the
    # scheduler/TP worker has been forked and assigned a GPU.
    if envs.SGLANG_DEEPGEMM_PDL.get() and hasattr(deep_gemm, "set_pdl"):
        deep_gemm.set_pdl(True)

    compile_utils.update_deep_gemm_config(gpu_id)
    if _is_sail:
        tuner.install_signal_handler()


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return
    if not DEEPGEMM_SCALE_UE8M0:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
