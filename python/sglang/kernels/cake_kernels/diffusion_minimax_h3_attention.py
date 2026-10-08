"""Cake MiniMax-H3 diffusion attention kernels via FlashInfer.

Three FlashInfer families (FlashInfer ``58171ea83f``):

**Packed-varlen attention, SM100a / SM103a** (public entries
``flashinfer.prefill.minimax_h3_varlen_attention`` /
``minimax_h3_varlen_nvfp4_attention``; implementation and prepared runners in
``flashinfer.experimental.minimax_h3_varlen_attention.cake_backend``, JIT in
``...cake_jit``: one generated program per stage shared by both targets and
compiled per exact arch, ``load_cake_minimax_h3_varlen_attention_module(name,
arch)``; routes are keyed ``<variant>__sm_10{0,3}a``). BF16 THD ``q, k, v
[T, H, 128]``, int32 CUDA ``cu_seqlens [B+1]`` (starts at 0, non-decreasing,
ends at ``T``; empty and unaligned segments allowed), non-causal
self-attention with ``H_q == H_kv``, no mask / bias / window / LSE / dropout,
``softmax_scale`` default ``1/sqrt(128)``, contiguous BF16 ``out [T, H, 128]``.

* ``minimax_h3_varlen_attention`` -- BF16 operands **read in place** (FlashInfer
  PR #6039): any token-major view with ``stride(2) == 1``, a head stride that
  is a 16-byte multiple of at least 128 elements, a token stride that is a
  16-byte multiple of at least ``H * head_stride`` and a 16-byte-aligned base
  -- the column chunks of the fused QKV projection ``[T, 3 * H * 128]``
  (strides ``(3 * H * 128, 128, 1)``), the kind slices of the Cake
  pre-attention pack ``[T, H, 3, 128]`` (strides ``(H * 384, 384, 1)``) and
  contiguous tensors; the three may differ in strides. Head-major views are
  rejected. The one-shot form syncs once to read ``cu_seqlens`` unless
  ``cu_seqlens_host`` is given.
* ``minimax_h3_varlen_nvfp4_attention`` -- contiguous THD operands (the
  quantizers read contiguous THD); NVFP4 QK (``pv_mode="fp8"``: E4M3 PV with
  a per-tensor V scale; ``"fp4"``: NVFP4 PV); validated by FlashInfer at
  ``atol=1.0, rtol=0.1``.
* ``prepare_minimax_h3_varlen_attention`` / ``prepare_minimax_h3_varlen_nvfp4_attention``
  -- every allocation at prepare (out, plan tables, split partials, packed
  NVFP4 operands); ``runner.launch()`` / ``runner()`` never allocates or syncs
  and is CUDA-graph capturable. The runner is bound to one ``cu_seqlens`` /
  shape set / tensor-binding set (values may change) -- re-prepare otherwise.

**Dense attention** (``flashinfer.diffusion_ops.cake_minimax_h3_dense_attention``,
JIT ``flashinfer.jit.cake_minimax_h3_dense_attention``): BF16 ``q, k, v, out
[tokens, 7168]`` (56 heads x 128, batch 1), ``1 <= tokens <= 131072``,
non-causal, query pre-scaled by ``bf16(1/sqrt(128)) = 0.08837890625`` with
softmax scale 1.0. SM120 (GB202) is the performance target; the JIT also
builds for cc 9.x / 10.x / 12.x. A per-device zeroed workspace for the
in-kernel tail split-KV merge is lazily allocated and cached inside
FlashInfer: pass ``out`` and launch once before CUDA-graph capture.

**SM120 quantized packed-varlen attention**
(``flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention`` /
``..._sm120_nvfp4_varlen_attention``): same THD contract as above
(``1 <= heads < 32768``), FP8 (per-token Q, per-128-key-block mean-centred K,
E4M3 P and V) or NVFP4 (SageAttention3-style) ``mma.sync`` operands, FP32
softmax, one BF16 rounding of the output. JIT nvcc flags target major version
12 only. The host plan is cached per ``(cu_seqlens, heads, device)`` (pass
``cu_seqlens_host`` to avoid a synchronising ``.tolist()``; most recent 256
plans kept) and the grow-only workspaces are module-internal, one set per
``(device, stream)`` (most recent 8 sets kept): warm up with the exact plan
on the capture stream before CUDA-graph capture.

Not supported here (keep the existing SGLang path): GQA, causal / masked /
windowed attention, LSE output, head_dim != 128, FP8 KV inputs, SM90 for the
varlen families.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from sglang.kernels.cake_kernels._support import (
    SM90,
    SM100,
    SM103,
    SM120,
    SM121,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

HEAD_DIM = 128
NUM_HEADS = 56
WIDTH = NUM_HEADS * HEAD_DIM  # 7168
DENSE_MAX_TOKENS = 131072
DENSE_QUERY_SCALE_BF16 = 0.08837890625
MAX_HEADS = 1 << 15
PV_MODES = ("fp8", "fp4")
# Element multiple that keeps a BF16 stride 16-byte aligned (TMA global
# strides and the BF16 kernel's 16-byte ragged-tail copies).
THD_STRIDE_ALIGN = 16 // 2

FI_VARLEN_MODULE = "flashinfer.experimental.minimax_h3_varlen_attention.cake_backend"
FI_VARLEN_JIT_MODULE = "flashinfer.experimental.minimax_h3_varlen_attention.cake_jit"
VARLEN_ARCHS = (SM100, SM103)

FI_DENSE_MODULE = "flashinfer.diffusion_ops.cake_minimax_h3_dense_attention"
FI_DENSE_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_dense_attention"
DENSE_ARCHS = (SM90, SM100, SM103, SM120, SM121)

FI_SM120_FP8_MODULE = (
    "flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention"
)
FI_SM120_FP8_JIT_MODULE = "flashinfer.jit.cake_minimax_h3_sm120_quant_varlen_attention"
FI_SM120_NVFP4_MODULE = (
    "flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention"
)
FI_SM120_NVFP4_JIT_MODULE = (
    "flashinfer.jit.cake_minimax_h3_sm120_nvfp4_varlen_attention"
)
SM120_ARCHS = (SM120, SM121)

# Backwards-compatible aliases used by tests that only know the family name.
FI_MODULE = FI_VARLEN_MODULE
FI_JIT_MODULE = FI_VARLEN_JIT_MODULE
ARCHS = VARLEN_ARCHS


def _thd_view_ok(t: torch.Tensor, num_heads: int) -> bool:
    """Token-major ``[T, H, 128]`` view the BF16 kernel reads in place
    (FlashInfer ``_check_thd(contiguous=False)``)."""
    row_stride, head_stride, elem_stride = (int(s) for s in t.stride())
    return (
        elem_stride == 1
        and head_stride >= HEAD_DIM
        and head_stride % THD_STRIDE_ALIGN == 0
        and row_stride >= num_heads * head_stride
        and row_stride % THD_STRIDE_ALIGN == 0
        and t.data_ptr() % 16 == 0
    )


def _thd_inputs_ok(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor],
    archs,
    *,
    strided_qkv: bool = False,
) -> bool:
    """``strided_qkv=True`` (BF16 SM100/103 route) admits strided token-major
    q/k/v views; ``False`` requires contiguous operands. ``out`` is always
    contiguous."""
    import torch

    if not (
        cuda_tensor_on(q, archs)
        and q.ndim == 3
        and q.dtype == torch.bfloat16
        and q.shape[2] == HEAD_DIM
        and 1 <= q.shape[1] < MAX_HEADS
    ):
        return False
    num_heads = int(q.shape[1])
    for t in (q, k, v):
        if not (
            t.dtype == torch.bfloat16
            and tuple(t.shape) == tuple(q.shape)
            and t.device == q.device
            and (_thd_view_ok(t, num_heads) if strided_qkv else t.is_contiguous())
        ):
            return False
    if out is not None and not (
        out.dtype == torch.bfloat16
        and out.is_contiguous()
        and tuple(out.shape) == tuple(q.shape)
        and out.device == q.device
    ):
        return False
    return (
        isinstance(cu_seqlens, torch.Tensor)
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.ndim == 1
        and cu_seqlens.numel() >= 2
        and cu_seqlens.device == q.device
    )


def _varlen_route_available(variant: str, device_index: int) -> bool:
    """FlashInfer registers the ``variant`` program for this exact arch."""
    from flashinfer.experimental.minimax_h3_varlen_attention import cake_jit

    from sglang.kernels.cake_kernels._support import device_capability

    arch = {SM100: "sm_100a", SM103: "sm_103a"}.get(device_capability(device_index))
    return arch is not None and cake_jit.route_available(variant, arch)


# ---------------------------------------------------------------------------
# SM100 / SM103 packed-varlen attention (C04 - C07)
# ---------------------------------------------------------------------------


def supports_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer BF16 contract; never raises.

    Strided token-major ``query`` / ``key`` / ``value`` views (fused-QKV column
    chunks, pre-attention pack slices) are admitted and read in place.
    """
    try:
        return (
            flashinfer_module_available(FI_VARLEN_MODULE, FI_VARLEN_JIT_MODULE)
            and _thd_inputs_ok(
                query, key, value, cu_seqlens, out, VARLEN_ARCHS, strided_qkv=True
            )
            and _varlen_route_available("bf16", query.device.index)
        )
    except Exception:
        return False


def minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.prefill.minimax_h3_varlen_attention``; returns BF16 ``out``.

    The operands are read where they live (no THD copies). One host sync to
    read ``cu_seqlens`` unless ``cu_seqlens_host`` is given.
    """
    from flashinfer.prefill import minimax_h3_varlen_attention

    return minimax_h3_varlen_attention(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend="cake",
    )


def supports_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: str = "fp8",
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer NVFP4 contract; never raises."""
    try:
        return (
            pv_mode in PV_MODES
            and flashinfer_module_available(FI_VARLEN_MODULE, FI_VARLEN_JIT_MODULE)
            and _thd_inputs_ok(query, key, value, cu_seqlens, out, VARLEN_ARCHS)
            and _varlen_route_available(
                "nvfp4_fp4pv" if pv_mode == "fp4" else "nvfp4_fp8pv",
                query.device.index,
            )
        )
    except Exception:
        return False


def minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.prefill.minimax_h3_varlen_nvfp4_attention``.

    NVFP4 QK with E4M3 (``pv_mode="fp8"``) or NVFP4 (``"fp4"``) PV; FlashInfer
    tolerance ``atol=1.0, rtol=0.1`` against FP32.
    """
    from flashinfer.prefill import minimax_h3_varlen_nvfp4_attention

    return minimax_h3_varlen_nvfp4_attention(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend="cake",
    )


def prepare_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> Any:
    """Forward to FlashInfer; returns ``MiniMaxH3VarlenAttentionRunner``.

    ``runner.launch()`` (alias ``runner()``) writes ``runner.out`` with no
    allocation and no host sync (CUDA-graph capturable). Admission:
    :func:`supports_minimax_h3_varlen_attention`.
    """
    from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
        prepare_minimax_h3_varlen_attention,
    )

    return prepare_minimax_h3_varlen_attention(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend="cake",
    )


def prepare_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
) -> Any:
    """Forward to FlashInfer; returns ``MiniMaxH3VarlenNVFP4AttentionRunner``.

    ``runner.launch()`` re-quantizes Q/K/V and runs attention (stages also
    callable separately via ``runner.quantize()`` / ``runner.attention()``).
    ``workspace`` must match ``cake_backend.nvfp4_workspace_shapes`` exactly
    when given. Admission: :func:`supports_minimax_h3_varlen_nvfp4_attention`.
    """
    from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
        prepare_minimax_h3_varlen_nvfp4_attention,
    )

    return prepare_minimax_h3_varlen_nvfp4_attention(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        workspace=workspace,
        backend="cake",
    )


# ---------------------------------------------------------------------------
# Dense attention (D03)
# ---------------------------------------------------------------------------


def supports_minimax_h3_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        if not (
            flashinfer_module_available(FI_DENSE_MODULE, FI_DENSE_JIT_MODULE)
            and cuda_tensor_on(q, DENSE_ARCHS)
            and q.ndim == 2
            and 1 <= q.shape[0] <= DENSE_MAX_TOKENS
        ):
            return False
        for t in (k, v) + ((out,) if out is not None else ()):
            if not (
                t.dtype == torch.bfloat16
                and t.is_contiguous()
                and tuple(t.shape) == (q.shape[0], WIDTH)
                and t.device == q.device
            ):
                return False
        return q.dtype == torch.bfloat16 and q.is_contiguous() and q.shape[1] == WIDTH
    except Exception:
        return False


def minimax_h3_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``[tokens, 7168]``.

    The per-device merge workspace is cached inside FlashInfer; supply ``out``
    and launch once before CUDA-graph capture.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_dense_attention import (
        minimax_h3_dense_attention,
    )

    return minimax_h3_dense_attention(q, k, v, out=out)


# ---------------------------------------------------------------------------
# SM120 quantized packed-varlen attention (D13 / D15)
# ---------------------------------------------------------------------------


def supports_minimax_h3_sm120_varlen_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer SM120 FP8 contract; never raises."""
    try:
        return flashinfer_module_available(
            FI_SM120_FP8_MODULE, FI_SM120_FP8_JIT_MODULE
        ) and _thd_inputs_ok(q, k, v, cu_seqlens, out, SM120_ARCHS)
    except Exception:
        return False


def minimax_h3_sm120_varlen_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [tokens, heads, 128]``.

    Plan + grow-only workspaces are cached inside FlashInfer; warm up with the
    exact ``cu_seqlens`` before CUDA-graph capture and pass ``cu_seqlens_host``
    to avoid the ``.tolist()`` sync.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
        minimax_h3_sm120_varlen_attention_fp8,
    )

    return minimax_h3_sm120_varlen_attention_fp8(
        q,
        k,
        v,
        cu_seqlens,
        out,
        cu_seqlens_host=cu_seqlens_host,
        softmax_scale=softmax_scale,
    )


def supports_minimax_h3_sm120_varlen_attention_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer SM120 NVFP4 contract; never raises."""
    try:
        return flashinfer_module_available(
            FI_SM120_NVFP4_MODULE,
            FI_SM120_NVFP4_JIT_MODULE,
            FI_SM120_FP8_MODULE,
        ) and _thd_inputs_ok(q, k, v, cu_seqlens, out, SM120_ARCHS)
    except Exception:
        return False


def minimax_h3_sm120_varlen_attention_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [tokens, heads, 128]``.

    Experimental NVFP4 (SageAttention3 recipe) route; FlashInfer tolerance
    ``atol=1.0, rtol=0.1``. Same plan / workspace caching caveats as the FP8
    entry.
    """
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
        minimax_h3_sm120_varlen_attention_nvfp4,
    )

    return minimax_h3_sm120_varlen_attention_nvfp4(
        q,
        k,
        v,
        cu_seqlens,
        out,
        cu_seqlens_host=cu_seqlens_host,
        softmax_scale=softmax_scale,
    )
