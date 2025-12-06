import torch

from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():

    @torch.library.custom_op(
        "sglang::flashinfer_mm_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer.gemm import mm_fp4 as flashinfer_mm_fp4_

        return flashinfer_mm_fp4_(
            A, B, A_scale, B_scale, g_scale, dtype, block_size=16, backend=backend
        )

    @torch.library.register_fake(
        "sglang::flashinfer_mm_fp4",
    )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

    if backend == "cutlass":
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        backend=backend,
    )
