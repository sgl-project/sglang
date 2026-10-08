"""Cake Kimi-K3 AttnRes, MiniMax-H3 varlen attention and NVFP4 attention via FlashInfer.

FlashInfer entries (all at FlashInfer ``46340689a5ab``):

* ``flashinfer.kimi_k3_attn_res.prepare_kimi_k3_attn_res`` /
  ``kimi_k3_attn_res`` (impl ``flashinfer.experimental.cake_kimi_k3_attn_res``;
  SM100 / SM103, 148-SM parts measured): Kimi-K3 attention residual block.
  BF16 only with dense last dim: ``prefix [M, 7168]`` (mutated in place when
  ``delta`` is given), ``delta [M, 7168]`` or ``None``, ``blocks [M, B <= 8,
  7168]`` (the persistent path needs ``B == 8``; rows must not overlap),
  ``norm_weight`` / ``qk_weight`` / ``output_norm_weight [7168]``, contiguous
  ``out [M, 7168]``; ``num_blocks = K in [0, 8]``, ``block_write_idx`` -1 or
  ``< B``, ``eps > 0``. Only registered cells run: ``M in {1, 2, 4, ...,
  16384} x K in {0, 1, 4, 8}``, every ``K`` at ``M in {1, 4096}`` and the
  semantic variants at ``M in {1, 3, 7, 17}``; other ``M`` raise
  ``NotImplementedError`` (query ``generated_program_available(device, M, K)``).
  Tolerance measured at atol 8e-2 / rtol 3e-2 against the FP32 reference
  ``reference_kimi_k3_attn_res``; ``prefix`` / ``blocks`` side effects are
  bit-exact.
* ``flashinfer.prefill.minimax_h3_varlen_attention`` and
  ``flashinfer.experimental.minimax_h3_varlen_attention.cake_backend.prepare_minimax_h3_varlen_attention``
  (SM100 / SM103, separate exact-arch programs): BF16 THD ``q, k, v [T, H,
  128]`` contiguous, int32 CUDA ``cu_seqlens [B + 1]`` (starting at 0,
  nondecreasing, ending at ``T``), BF16 ``out [T, H, 128]``; noncausal
  self-attention with ``Hq == Hkv``, head_dim 128, arbitrary ``T`` / segment
  lengths (empty segments allowed); no GQA / mask / bias / window / LSE /
  dropout; ``softmax_scale`` defaults to ``1 / sqrt(128)``.
* ``flashinfer.prefill.minimax_h3_varlen_nvfp4_attention`` /
  ``prepare_minimax_h3_varlen_nvfp4_attention``: same BF16 THD inputs and
  outputs; ``pv_mode="fp8"`` (default: NVFP4 QK + E4M3 PV with a per-tensor
  ``448 / amax(V)`` scale) or ``"fp4"`` (NVFP4 QK and PV); packed operands
  allocated at prepare or supplied through ``workspace=`` matching
  ``nvfp4_workspace_shapes(heads, PB, pv_mode)``. Validated at atol 1.0 /
  rtol 0.1 against FP32.
* ``flashinfer.prefill.prepare_nvfp4_attention`` (impl
  ``flashinfer.experimental.nvfp4_attention.cake_backend``; SM103 only): BF16
  contiguous ``q, k, v, out [B, H, S, 128]`` with identical shapes, noncausal
  only (``causal=False``), ``S % 512 == 0``; prepare quantizes Q/K/V to E2M1
  with E4M3 block-16 scales (host-side torch) and prepacks the scale tiles;
  the softmax scale is fixed at ``1 / sqrt(128)``; validated at atol 1.0 /
  rtol 0.1 against SDPA.

CUDA graphs: the AttnRes runner's ``launch()`` is allocation-free and
capturable (``plan_route`` picks native / k0_tma / persistent / direct per
``(M, K, enable_pdl)``). MiniMax-H3 one-shot entries read ``cu_seqlens`` to the
host once (one sync) unless ``cu_seqlens_host`` is given; the prepared runners
allocate everything at prepare (plan tables, split partials, packed NVFP4
operands) and ``launch()`` never allocates or syncs; they are bound to one
``cu_seqlens`` / shape set / tensor binding (values may change) -- re-prepare
when segments, shapes or bindings change. The NVFP4 varlen runner
re-quantizes on every ``launch()`` (``quantize()`` / ``attention()`` are also
callable separately). The NVFP4 attention runner quantizes at prepare, so a
new runner is needed whenever input *values* change; graph capture is left to
the caller.

Not supported here (keep the existing SGLang path): AttnRes for unregistered
``M`` cells or non-BF16; MiniMax-H3 varlen with GQA, causal masks or LSE;
NVFP4 attention on SM100 / SM120, causal attention or ``S % 512 != 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from sglang.kernels.cake_kernels.attention_common import (
    SM100,
    SM103,
    archs_in,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_ATTN_RES_MODULE = "flashinfer.kimi_k3_attn_res"
FI_ATTN_RES_BACKEND_MODULE = (
    "flashinfer.experimental.cake_kimi_k3_attn_res.cake_backend"
)
FI_PREFILL_MODULE = "flashinfer.prefill"
FI_MINIMAX_BACKEND_MODULE = (
    "flashinfer.experimental.minimax_h3_varlen_attention.cake_backend"
)
FI_NVFP4_ATTENTION_BACKEND_MODULE = (
    "flashinfer.experimental.nvfp4_attention.cake_backend"
)

ARCHS = (SM100, SM103)
NVFP4_ATTENTION_ARCHS = (SM103,)

ATTN_RES_HIDDEN = 7168
ATTN_RES_MAX_BLOCKS = 8
MINIMAX_HEAD_DIM = 128
MINIMAX_PV_MODES = ("fp8", "fp4")
NVFP4_ATTENTION_HEAD_DIM = 128
NVFP4_ATTENTION_SEQ_MULTIPLE = 512


# --------------------------------------------------------------------------
# Kimi-K3 AttnRes
# --------------------------------------------------------------------------


def supports_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    out: torch.Tensor,
    *,
    num_blocks: int,
    enable_pdl: bool = False,
) -> bool:
    """Admission check; mirrors the input contract and the registered cells.

    ``generated_program_available`` is consulted for the ``(M, K, pdl)`` cell
    (importing FlashInfer); any failure reports ``False``.
    """
    try:
        import torch

        tensors = [prefix, blocks, out] + ([delta] if delta is not None else [])
        if not archs_in(ARCHS, *tensors):
            return False
        if not flashinfer_module_available(
            FI_ATTN_RES_MODULE, FI_ATTN_RES_BACKEND_MODULE
        ):
            return False
        m = int(prefix.shape[0]) if prefix.ndim == 2 else -1
        if m < 1 or blocks.ndim != 3 or out.ndim != 2:
            return False
        if not all(t.dtype == torch.bfloat16 for t in tensors):
            return False
        if int(prefix.shape[1]) != ATTN_RES_HIDDEN or int(prefix.stride(1)) != 1:
            return False
        if tuple(blocks.shape[::2]) != (m, ATTN_RES_HIDDEN) or blocks.stride(2) != 1:
            return False
        if not (0 <= num_blocks <= int(blocks.shape[1]) <= ATTN_RES_MAX_BLOCKS):
            return False
        if tuple(out.shape) != (m, ATTN_RES_HIDDEN) or not out.is_contiguous():
            return False
        if delta is not None and (
            tuple(delta.shape) != (m, ATTN_RES_HIDDEN) or delta.stride(1) != 1
        ):
            return False
        from flashinfer.kimi_k3_attn_res import generated_program_available

        return bool(
            generated_program_available(
                prefix.device, m, num_blocks, enable_pdl=enable_pdl
            )
        )
    except Exception:
        return False


def prepare_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
):
    """Forward to FlashInfer; returns a ``KimiK3AttnResRunner``.

    ``runner.launch()`` returns the bound ``out`` without allocating.
    """
    from flashinfer.kimi_k3_attn_res import prepare_kimi_k3_attn_res

    return prepare_kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
        backend="cake",
    )


def kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Forward to ``flashinfer.kimi_k3_attn_res.kimi_k3_attn_res`` (one-shot)."""
    from flashinfer.kimi_k3_attn_res import kimi_k3_attn_res

    return kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
        backend="cake",
    )


# --------------------------------------------------------------------------
# MiniMax-H3 packed-varlen noncausal attention (BF16 and NVFP4)
# --------------------------------------------------------------------------


def supports_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: Optional[str] = None,
) -> bool:
    """Admission check mirroring ``validate_minimax_h3_varlen_inputs``.

    ``pv_mode=None`` checks the BF16 program; ``"fp8"`` / ``"fp4"`` the NVFP4
    variants (same tensor contract).
    """
    try:
        import torch

        if pv_mode is not None and pv_mode not in MINIMAX_PV_MODES:
            return False
        if not archs_in(ARCHS, query, key, value, cu_seqlens):
            return False
        if not flashinfer_module_available(
            FI_PREFILL_MODULE, FI_MINIMAX_BACKEND_MODULE
        ):
            return False
        return (
            query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and value.dtype == torch.bfloat16
            and query.ndim == 3
            and tuple(key.shape) == tuple(query.shape)
            and tuple(value.shape) == tuple(query.shape)
            and int(query.shape[2]) == MINIMAX_HEAD_DIM
            and query.is_contiguous()
            and key.is_contiguous()
            and value.is_contiguous()
            and cu_seqlens.dtype == torch.int32
            and cu_seqlens.ndim == 1
            and int(cu_seqlens.numel()) >= 2
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
    """Forward to ``flashinfer.prefill.minimax_h3_varlen_attention`` (one-shot)."""
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


def prepare_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
):
    """Forward to the backend prepare; returns ``MiniMaxH3VarlenAttentionRunner``."""
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
    """Forward to ``flashinfer.prefill.minimax_h3_varlen_nvfp4_attention``."""
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
):
    """Forward to the backend prepare; returns ``MiniMaxH3VarlenNVFP4AttentionRunner``."""
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


# --------------------------------------------------------------------------
# NVFP4 dense noncausal attention (SM103)
# --------------------------------------------------------------------------


def supports_nvfp4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    causal: bool = False,
) -> bool:
    """Admission check mirroring the NVFP4 attention contract; never raises."""
    try:
        import torch

        return (
            not causal
            and archs_in(NVFP4_ATTENTION_ARCHS, q, k, v, out)
            and flashinfer_module_available(
                FI_PREFILL_MODULE, FI_NVFP4_ATTENTION_BACKEND_MODULE
            )
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and out.dtype == torch.bfloat16
            and q.ndim == 4
            and tuple(k.shape) == tuple(q.shape)
            and tuple(v.shape) == tuple(q.shape)
            and tuple(out.shape) == tuple(q.shape)
            and int(q.shape[3]) == NVFP4_ATTENTION_HEAD_DIM
            and int(q.shape[2]) % NVFP4_ATTENTION_SEQ_MULTIPLE == 0
            and all(int(s) > 0 for s in q.shape)
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and out.is_contiguous()
        )
    except Exception:
        return False


def prepare_nvfp4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    causal: bool = False,
) -> Any:
    """Forward to ``flashinfer.prefill.prepare_nvfp4_attention(backend="cake")``.

    Returns an ``NVFP4AttentionRunner``; ``runner()`` / ``runner.launch()``
    writes the bound ``out`` without allocating.
    """
    from flashinfer.prefill import prepare_nvfp4_attention

    return prepare_nvfp4_attention(q, k, v, out, causal=causal, backend="cake")
