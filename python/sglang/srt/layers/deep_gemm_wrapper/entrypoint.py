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
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda, is_sm90_supported, is_sm100_supported

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()


def supports_sm90_mxfp8_fp8_grouped_gemm() -> bool:
    if not ENABLE_JIT_DEEPGEMM:
        return False
    return hasattr(deep_gemm, "m_grouped_mxfp8_fp8_gemm_nt_contiguous") and hasattr(
        deep_gemm, "m_grouped_mxfp8_fp8_gemm_nt_masked"
    )


def is_sm90_mxfp8_deepgemm_enabled() -> bool:
    return (
        is_cuda()
        and is_sm90_supported()
        and not is_sm100_supported()
        and supports_sm90_mxfp8_fp8_grouped_gemm()
    )


def supports_mxfp8_deepgemm() -> bool:
    return DEEPGEMM_BLACKWELL or is_sm90_mxfp8_deepgemm_enabled()


def _ceil_align(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def _sm90_mxfp8_scale_expected_last_dim(
    scale: torch.Tensor,
    k: int,
    recipe: Optional[Tuple[int, int]],
) -> Optional[int]:
    if recipe is None:
        return None
    pack_factor = 4 if scale.dtype in (torch.int, torch.int32) else 1
    return (k + recipe[1] * pack_factor - 1) // (recipe[1] * pack_factor)


def _assert_sm90_mxfp8_scale_matches_recipe(
    scale: torch.Tensor,
    k: int,
    recipe: Optional[Tuple[int, int]],
    *,
    role: str,
) -> None:
    expected_last_dim = _sm90_mxfp8_scale_expected_last_dim(scale, k, recipe)
    if expected_last_dim is None:
        return
    if scale.shape[-1] != expected_last_dim:
        pack_factor = 4 if scale.dtype in (torch.int, torch.int32) else 1
        raise RuntimeError(
            f"SM90 MXFP8 {role} scale shape does not match recipe: "
            f"K={k}, recipe={recipe}, scale_dtype={scale.dtype}, "
            f"scale_shape={tuple(scale.shape)}, pack_factor={pack_factor}, "
            f"expected_last_dim={expected_last_dim}."
        )


def _e8m0_fp32_to_u8(sf: torch.Tensor) -> torch.Tensor:
    sf_i32 = sf.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(sf_i32, 23)
    mant = torch.bitwise_and(sf_i32, 0x7FFFFF)
    round_up = torch.logical_and(
        torch.logical_and(mant > 0, exp != 0xFE),
        ~torch.logical_and(exp == 0, mant <= 0x400000),
    )
    return torch.where(round_up, exp + 1, exp).to(torch.uint8).contiguous()


def _normalize_sm90_mxfp8_pair(
    pair: Tuple[torch.Tensor, torch.Tensor],
    *,
    scale_role: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, sf = pair
    k = x.shape[-1]
    k32 = _ceil_align(k, 32) // 32

    if (
        sf.dtype == torch.uint8
        and sf.shape[-1] == k32
        and x.dtype == torch.float8_e4m3fn
    ):
        return x, sf.contiguous()

    if sf.dtype == torch.int32:
        if x.dtype == torch.float8_e4m3fn:
            return x, sf
        raise RuntimeError(
            f"SM90 MXFP8 {scale_role} scale received packed UE8M0 scales "
            f"but activation dtype is {x.dtype}; expected torch.float8_e4m3fn."
        )

    if (
        sf.dtype == torch.float32
        and sf.shape[-1] == k32
        and x.dtype == torch.float8_e4m3fn
    ):
        return x, _e8m0_fp32_to_u8(sf)

    raise RuntimeError(
        f"SM90 MXFP8 {scale_role} wrapper only performs lossless scale layout adaptation. "
        f"Got activation dtype={x.dtype}, scale dtype={sf.dtype}, "
        f"scale_last_dim={sf.shape[-1]}, expected K/32={k32}."
    )


def _pad_sm90_mxfp8_lhs(
    lhs: Tuple[torch.Tensor, torch.Tensor], expected_m: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, sf = lhs
    if x.shape[-2] >= expected_m:
        return lhs
    padded_x = torch.empty(
        (*x.shape[:-2], expected_m, x.shape[-1]), device=x.device, dtype=x.dtype
    )
    padded_sf = torch.empty(
        (*sf.shape[:-2], expected_m, sf.shape[-1]), device=sf.device, dtype=sf.dtype
    )
    if sf.dtype == torch.int32 and not sf.is_contiguous() and sf.dim() >= 2:
        padded_sf_storage = torch.empty(
            (*sf.shape[:-2], sf.shape[-1], expected_m),
            device=sf.device,
            dtype=sf.dtype,
        )
        padded_sf = padded_sf_storage.transpose(-1, -2)
    padded_x[..., : x.shape[-2], :] = x
    padded_sf[..., : sf.shape[-2], :] = sf
    return padded_x, padded_sf


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
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

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
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def grouped_gemm_nt_mxfp8_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    if not supports_sm90_mxfp8_fp8_grouped_gemm():
        raise RuntimeError(
            "The installed deep_gemm does not expose SM90 MXFP8 grouped GEMM APIs."
        )

    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_MXFP8_F8BF16_MASKED

    lhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(lhs), scale_role="lhs")
    rhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(rhs), scale_role="rhs")
    _assert_sm90_mxfp8_scale_matches_recipe(lhs[1], k, recipe_a, role="lhs")
    _assert_sm90_mxfp8_scale_matches_recipe(rhs[1], k, recipe_b, role="rhs")

    padded_expected_m = _ceil_align(max(lhs[0].shape[-2], expected_m), 128)
    lhs = _pad_sm90_mxfp8_lhs(lhs, padded_expected_m)
    kernel_out = out
    if out.shape[-2] < padded_expected_m:
        kernel_out = torch.empty(
            (*out.shape[:-2], padded_expected_m, out.shape[-1]),
            device=out.device,
            dtype=out.dtype,
        )

    with compile_utils.deep_gemm_execution_hook(
        padded_expected_m, n, k, num_groups, kernel_type
    ):
        ret = deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_masked(
            lhs,
            rhs,
            kernel_out,
            masked_m,
            padded_expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
    if kernel_out is not out:
        out.copy_(kernel_out[..., : out.shape[-2], :])
    return ret


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
):
    num_groups, _, k = a.shape
    _, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_bf16_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            expected_m,
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    fp4_kwargs = {}
    if recipe_a is not None:
        fp4_kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        fp4_kwargs["recipe_b"] = recipe_b

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            lhs, rhs, out, m_indices, **fp4_kwargs
        )


def grouped_gemm_nt_mxfp8_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    if not supports_sm90_mxfp8_fp8_grouped_gemm():
        raise RuntimeError(
            "The installed deep_gemm does not expose SM90 MXFP8 grouped GEMM APIs."
        )

    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_MXFP8_F8BF16_CONTIG

    if m == 0:
        return

    lhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(lhs), scale_role="lhs")
    rhs = _normalize_sm90_mxfp8_pair(_ensure_cuda(rhs), scale_role="rhs")
    _assert_sm90_mxfp8_scale_matches_recipe(lhs[1], k, recipe_a, role="lhs")
    _assert_sm90_mxfp8_scale_matches_recipe(rhs[1], k, recipe_b, role="rhs")

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
            lhs,
            rhs,
            out,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )


def grouped_gemm_nt_bf16_contig(
    a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, m_indices: torch.Tensor
):
    m, k = a.shape
    num_groups, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)


def gemm_nt_f8f8bf16(
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

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
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
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=num_splits)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    # deep_gemm.set_pdl can initialize CUDA state, so run it only after the
    # scheduler/TP worker has been forked and assigned a GPU.
    if envs.SGLANG_DEEPGEMM_PDL.get() and hasattr(deep_gemm, "set_pdl"):
        deep_gemm.set_pdl(True)

    compile_utils.update_deep_gemm_config(gpu_id, server_args)


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
