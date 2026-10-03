"""Cake quantization kernels through FlashInfer's public API.

Three Cake entries at FlashInfer ``46340689a5ab``, all built for sm_100a /
sm_103a only (exact compute capability 10.0 / 10.3):

* NVFP4 per-token activation quantizer:
  ``flashinfer.quantization.fp4_quantization:nvfp4_quantize(a, a_global_sf,
  per_token_activation=True, backend="cake")`` ->
  ``flashinfer.experimental.cake_nvfp4_per_token.cake_backend:
  nvfp4_quantize_per_token``. BF16/FP16 ``x [M, K]`` contiguous, ``K % 16 == 0``;
  returns ``(fp4 uint8 [M, K/2], sf uint8 [round_up(M,128), round_up(K/16,4)]
  in the swizzled 128x4 layout, per_token_scale f32 [M])``; bitwise equal to the
  CuTe-DSL per-token kernel. ``do_shuffle``, linear/8x4 scale layouts,
  ``expanded_idx_to_permuted_idx`` and ``nvfp4_4over6`` are rejected by
  FlashInfer. Allocates its outputs (prepare outside CUDA-graph capture or use
  FlashInfer's ``prepare_nvfp4_per_token_quantize`` runner, not forwarded here).
* Grouped MXFP8 quantizer:
  ``flashinfer.quantization.fp8_quantization:mxfp8_grouped_quantize(a, mask,
  backend="cake")`` (JIT ``flashinfer.jit.cake_grouped_mxfp8_quantize``).
  BF16/FP16 ``a [B, M, K]`` with ``K % 32 == 0``, int32 ``mask [B]`` of valid
  rows per group (``0 <= mask[i] <= M``, unchecked to stay graph-safe); returns
  ``x_q`` e4m3 logical ``[M, padded_K, B]`` and ``sf`` uint8 logical
  ``[32, 4, padded_M//128, 4, padded_K//128, B]`` (padded to 128), matching the
  FlashInfer masked grouped GEMM conventions. Rows ``>= mask[i]`` are
  unspecified. FlashInfer raises when the generated profile for the dtype is not
  installed (placeholder source); never silently retries another backend.
* Sage-FP8 Q/K/V quantizer for Cake block-sparse Sage attention:
  ``flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake:sage_fp8_quantize_sm100``
  (JIT ``flashinfer.jit.cake_sage_block_sparse_attention``). BF16 BSHD
  ``q [B, Sq, H, 128]``, ``k/v [B, Sk, Hkv, 128]`` -> e4m3 ``(q8, k8, v8)`` plus
  f32 ``q_scale [B, H, Sq]`` (per token), ``k_scale [B, Hkv, ceil(Sk/16)]``
  (per 16-token group), ``v_scale [B, Hkv, 128]`` (per channel); scales are
  bit-exact with ``amax.clamp_min(1e-12) / 448``. Allocates outputs each call.

Not forwarded: the prepared NVFP4 per-token runner / quantize+GEMM chain
(``prepare_nvfp4_per_token_quantize`` / ``prepare_nvfp4_per_token_chain``) and
the per-token GEMM, which belong with the ``gemm`` group's prepared-runner
story.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

ARCHS = BLACKWELL_DATACENTER  # (10, 0), (10, 3)

NVFP4_FI_MODULE = "flashinfer.quantization.fp4_quantization"
NVFP4_FI_BACKEND_MODULE = "flashinfer.experimental.cake_nvfp4_per_token.cake_backend"
NVFP4_SF_VEC = 16

MXFP8_FI_MODULE = "flashinfer.quantization.fp8_quantization"
MXFP8_FI_JIT_MODULE = "flashinfer.jit.cake_grouped_mxfp8_quantize"
MXFP8_BLOCK = 32

SAGE_FI_MODULE = "flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake"
SAGE_FI_JIT_MODULE = "flashinfer.jit.cake_sage_block_sparse_attention"
SAGE_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# NVFP4 per-token activation quantization
# ---------------------------------------------------------------------------


def supports_nvfp4_quantize_per_token(x: torch.Tensor) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Besides module presence, dtype, layout and architecture, this asks
    FlashInfer whether the generated programs for this GPU's SM count are
    registered (``generated_program_available``); that import loads torch and
    the FlashInfer package metadata but compiles nothing.
    """
    import torch

    if not (
        flashinfer_module_available(NVFP4_FI_MODULE, NVFP4_FI_BACKEND_MODULE)
        and cuda_tensor_on(x, ARCHS)
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[1] % NVFP4_SF_VEC == 0
    ):
        return False
    from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
        generated_program_available,
    )

    return bool(generated_program_available(x.device))


def nvfp4_quantize_per_token(
    x: torch.Tensor,
    global_scale_inv,
    *,
    out_scale: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(fp4 [M, K/2], sf swizzled 128x4, scale [M])``.

    ``global_scale_inv`` is the per-token global scale inverse (host float or
    one-element tensor), e.g. ``1 / (448 * 6)``; the per-token scale is
    ``amax(row) * global_scale_inv`` (times ``out_scale`` when given). The
    programs are built with programmatic dependent launch; FlashInfer rejects
    ``enable_pdl=False``. Routed through the public ``nvfp4_quantize`` entry so
    FlashInfer's experimental-backend warning and argument validation apply.
    """
    from flashinfer.quantization.fp4_quantization import SfLayout, nvfp4_quantize

    return nvfp4_quantize(
        x,
        global_scale_inv,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=enable_pdl,
        backend="cake",
        per_token_activation=True,
        out_scale=out_scale,
    )


# ---------------------------------------------------------------------------
# Grouped MXFP8 quantization
# ---------------------------------------------------------------------------


def supports_mxfp8_grouped_quantize(a: torch.Tensor, mask: torch.Tensor) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Also requires the generated profile for ``a.dtype`` to be installed
    (``is_cake_grouped_mxfp8_quantize_available``), which reads the FlashInfer
    source tree without compiling.
    """
    import torch

    if not (
        flashinfer_module_available(MXFP8_FI_MODULE, MXFP8_FI_JIT_MODULE)
        and cuda_tensor_on(a, ARCHS)
        and a.dtype in (torch.bfloat16, torch.float16)
        and a.ndim == 3
        and a.shape[2] % MXFP8_BLOCK == 0
        and mask.is_cuda
        and mask.device == a.device
        and mask.dtype == torch.int32
        and mask.ndim == 1
        and mask.shape[0] == a.shape[0]
    ):
        return False
    from flashinfer.jit.cake_grouped_mxfp8_quantize import (
        is_cake_grouped_mxfp8_quantize_available,
    )

    return bool(is_cake_grouped_mxfp8_quantize_available(a.dtype, a.device))


def mxfp8_grouped_quantize(
    a: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to FlashInfer; returns ``(x_q e4m3 [M, padded_K, B], sf uint8)``.

    ``sf`` has logical shape ``[32, 4, padded_M//128, 4, padded_K//128, B]``;
    both are permuted views grouped by ``B`` (FlashInfer masked grouped GEMM
    convention). Only the first ``mask[i]`` rows of group ``i`` are defined.
    """
    from flashinfer.quantization.fp8_quantization import mxfp8_grouped_quantize

    return mxfp8_grouped_quantize(a, mask, backend="cake")


# ---------------------------------------------------------------------------
# Sage-FP8 Q/K/V quantization
# ---------------------------------------------------------------------------


def supports_sage_fp8_quantize(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    return (
        flashinfer_module_available(SAGE_FI_MODULE, SAGE_FI_JIT_MODULE)
        and cuda_tensor_on(q, ARCHS)
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and q.shape[3] == SAGE_HEAD_DIM
        and k.shape == v.shape
        and k.shape[0] == q.shape[0]
        and k.shape[3] == SAGE_HEAD_DIM
        and k.shape[2] >= 1
        and q.shape[2] % k.shape[2] == 0
        and k.device == q.device
        and v.device == q.device
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    )


def sage_fp8_quantize(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Forward to FlashInfer; returns ``(q8, k8, v8, q_scale, k_scale, v_scale)``.

    Symmetric e4m3 quantization of BF16 BSHD Q/K/V with per-token Q scales,
    per-16-token K scales and per-channel V scales (``amax.clamp_min(1e-12) /
    448``), the input layout of Cake Sage block-sparse attention on SM100.
    """
    from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import sage_fp8_quantize_sm100

    return sage_fp8_quantize_sm100(q, k, v)
