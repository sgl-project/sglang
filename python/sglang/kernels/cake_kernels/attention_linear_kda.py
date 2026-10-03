"""Cake KDA (Kimi Delta Attention) recurrent decode / prefill and the packed and
fused Kimi-K3 decode kernels via FlashInfer.

FlashInfer entries (contract at ``46340689a5ab``):

* ``flashinfer.kda.recurrent_kda(..., backend="cake")`` -- phase-neutral facade.
  Decode (inventory E1-19): the frozen D128 family, T=1..6. The T=1 route fuses
  the unbounded softplus gate from raw ``g`` / ``A_log`` / ``dt_bias`` for the
  equal-head (HV == H) contract; the T=3 lower-bound route is limited to
  N in {1,2,4,8,16}, H=HV=16; T in {1,2,4,5,6} with precomputed log gates (no
  ``A_log``/``dt_bias``). BF16 state pool ``[N, HV, 128, 128]`` updated in place
  (indexed through ``ssm_state_indices``; the pool may be slot-strided).
  Prefill (E1-20/E1-21): the frozen FlashKDA portfolio -- contiguous BF16
  ``[B, T>1, H, 128]`` q/k/v/g, BF16 beta logits ``[B, T, H]``
  (``beta_is_logit=True``), FP32 ``A_log [H]``, FP32 ``dt_bias [H*128]``,
  ``use_qk_l2norm_in_kernel`` and ``use_gate_in_kernel`` true, ``lower_bound``
  None or finite negative, packed mode B=1 with ``cu_seqlens``, optional int32
  ``ssm_state_indices`` into a BF16 or FP32 state pool (FP32 only when indexed),
  BF16 checkpoints every multiple of 16 tokens with int64
  ``checkpoint_cu_starts``. JIT modules ``flashinfer.jit.cake_kda_decode`` and
  ``flashinfer.jit.cake_kda``.
* ``flashinfer.kda_decode.packed_kda_decode`` (E1-17, ``flashinfer.kda_kernels.
  run_packed_kda_decode(backend="cake")``): serving-native packed Kimi-K3 T=1
  decode locked to H=12, K=V=128, ``scale=1/sqrt(128)``, L2 eps 1e-6,
  ``lower_bound=-5``. JIT module ``flashinfer.jit.cake_kda_packed_t1``.
* ``flashinfer.kda_decode.fused_kda_decode(..., backend="cake",
  state_indices_mode=...)`` (E1-18): fused Kimi conv + recurrent KDA + RMSNorm,
  H in {8, 12, 24, 32, 48, 96}; the host-known ``state_indices_mode`` is
  mandatory for the Cake backend. JIT module
  ``flashinfer.jit.cake_fused_kda_decode``.

All four are built for sm_100a / sm_103a only. They are allocation-free and
CUDA-graph capturable when the caller supplies ``output``; the facade decode
route never inspects index values on the host.

Admission rules absorbed from the superseded sglang PRs:

* #34946 / #34299: the equal-head D128 unbounded-gate decode predicate
  (``HV == H``, ``lower_bound is None``, raw gate with ``A_log``/``dt_bias``),
  the fail-closed prefill gate (``lower_bound`` None or finite negative,
  contiguous q/k/v, FP32 pool only with ``ssm_state_indices``, no FP32 state
  checkpoints), the packed-decode contract (exactly 12 heads x D128,
  ``lower_bound=-5``, int32 contiguous cache indices) and "GQA (HV != H)
  decode falls back". Route telemetry, ``MambaStateIndexContract`` attestation
  and the Cake auto-default were deliberately not carried over.
* #34299 (BBuf review): the Cake routes consume raw BF16 beta logits
  (``beta_is_logit=True``); every non-Cake path must keep
  ``beta.float().sigmoid()`` -- that is the caller's responsibility.

Not supported here (keep the existing SGLang path): ``disable_state_update``
(no frozen-state Cake kernels), ``checkpoint_state_indices``, explicit T=1
``cu_seqlens`` on decode, ``initial_state_source`` / ``initial_state_indices``,
head_dim != 128, GQA decode, the ``packed_t1`` metadata form of the fused decode
(CuTe DSL only), sm_90a / sm_120a / sm_121a.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.kda"
FI_MODULE_DECODE = "flashinfer.kda_decode"
FI_MODULE_PREFILL = "flashinfer.kda_prefill"
FI_JIT_MODULE_DECODE = "flashinfer.jit.cake_kda_decode"
FI_JIT_MODULE_PREFILL = "flashinfer.jit.cake_kda"
FI_JIT_MODULE_PACKED = "flashinfer.jit.cake_kda_packed_t1"
FI_JIT_MODULE_FUSED = "flashinfer.jit.cake_fused_kda_decode"
ARCHS = BLACKWELL_DATACENTER
HEAD_DIM = 128
MAX_SEQUENCES = 65535
PACKED_HEADS = 12
PACKED_LOWER_BOUND = -5.0
FUSED_HEADS = (8, 12, 24, 32, 48, 96)
FUSED_STATE_INDICES_MODES = ("positive_unique", "unique_or_null", "repeated_positive")
T3_LOWER_BOUND_SEQUENCES = (1, 2, 4, 8, 16)
PRECOMPUTED_TOKENS = (1, 2, 4, 5, 6)


def _contiguous(tensor, dtype, device, ndim=None) -> bool:
    return (
        tensor is not None
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
        and (ndim is None or tensor.ndim == ndim)
    )


def _state_pool(tensor, device, num_heads, dtypes) -> bool:
    """Slot-strided ``[N, HV, 128, 128]`` pool with compact inner strides."""
    return (
        tensor is not None
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype in dtypes
        and tensor.ndim == 4
        and tensor.shape[0] > 0
        and tuple(tensor.shape[1:]) == (num_heads, HEAD_DIM, HEAD_DIM)
        and tuple(tensor.stride()[1:]) == (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
        and tensor.stride(0) >= num_heads * HEAD_DIM * HEAD_DIM
        and tensor.data_ptr() % 16 == 0
        and tensor.stride(0) * tensor.element_size() % 16 == 0
    )


def _finite_negative(value: Optional[float]) -> bool:
    return value is None or (math.isfinite(float(value)) and float(value) < 0.0)


def supports_kda_recurrent_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    beta_is_logit: bool = False,
    disable_state_update: bool = False,
    scale: Optional[float] = None,
) -> bool:
    """Admission for ``recurrent_kda(backend="cake")`` decode; never raises.

    Standard decode: ``[B, 1, H, 128]`` inputs without ``cu_seqlens``.
    Spec decode (T = 1 + num_spec_tokens): ``[1, N*T, ...]`` inputs with int32
    ``cu_seqlens [N+1]`` and int32 ``ssm_state_indices [N, T]``.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_MODULE_DECODE, FI_JIT_MODULE_DECODE)
        and cuda_tensor_on(q, ARCHS)
        and not disable_state_update
        and use_qk_l2norm_in_kernel
        and (scale is None or (math.isfinite(float(scale))))
        and q.ndim == 4
        and k.ndim == 4
        and v.ndim == 4
        and g.ndim == 4
        and beta.ndim == 3
    ):
        return False
    device = q.device
    if any(
        t.dtype != torch.bfloat16 or not t.is_cuda or t.device != device
        for t in (q, k, v, g, beta)
    ):
        return False
    batch, tokens, num_heads, head_dim = q.shape
    num_value_heads = v.shape[2]
    if (
        head_dim != HEAD_DIM
        or tuple(k.shape) != (batch, tokens, num_heads, HEAD_DIM)
        or tuple(v.shape) != (batch, tokens, num_value_heads, HEAD_DIM)
        or tuple(g.shape) != (batch, tokens, num_value_heads, HEAD_DIM)
        or tuple(beta.shape) != (batch, tokens, num_value_heads)
        or num_heads <= 0
        or num_value_heads < num_heads
        or num_value_heads % num_heads
        or any(t.stride(-1) != 1 for t in (q, k, v, g, beta))
    ):
        return False

    gated = use_gate_in_kernel and A_log is not None and dt_bias is not None
    if gated and not (
        _contiguous(A_log, torch.float32, device, ndim=1)
        and A_log.numel() == num_value_heads
        and _contiguous(dt_bias, torch.float32, device)
        and dt_bias.numel() == num_value_heads * HEAD_DIM
    ):
        return False
    precomputed = (
        not use_gate_in_kernel
        and lower_bound is None
        and A_log is None
        and dt_bias is None
        and not beta_is_logit
    )

    if num_spec_tokens is None:
        # T=1 standard decode: equal-head unbounded softplus (fused raw gate) or
        # the precomputed-gate route. Explicit cu_seqlens is rejected by Cake.
        if cu_seqlens is not None or tokens != 1 or batch > MAX_SEQUENCES:
            return False
        unbounded = gated and lower_bound is None and num_value_heads == num_heads
        if not (unbounded or precomputed):
            return False
        num_sequences = batch
    else:
        # Spec decode: packed [1, N*T, ...] with cu_seqlens + 2-D indices.
        num_tokens = 1 + int(num_spec_tokens)
        if (
            batch != 1
            or cu_seqlens is None
            or ssm_state_indices is None
            or not _contiguous(cu_seqlens, torch.int32, device, ndim=1)
            or not _contiguous(ssm_state_indices, torch.int32, device, ndim=2)
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        if (
            num_sequences <= 0
            or num_sequences > MAX_SEQUENCES
            or tokens != num_sequences * num_tokens
            or tuple(ssm_state_indices.shape) != (num_sequences, num_tokens)
        ):
            return False
        t3_lower_bound = (
            num_tokens == 3
            and gated
            and lower_bound is not None
            and math.isfinite(float(lower_bound))
            and float(lower_bound) < 0.0
            and not beta_is_logit
            and num_sequences in T3_LOWER_BOUND_SEQUENCES
            and num_heads == 16
            and num_value_heads == 16
        )
        if not (t3_lower_bound or (precomputed and num_tokens in PRECOMPUTED_TOKENS)):
            return False
    if num_spec_tokens is None and ssm_state_indices is not None:
        if not (
            _contiguous(ssm_state_indices, torch.int32, device, ndim=1)
            and ssm_state_indices.numel() == num_sequences
        ):
            return False
    if initial_state is not None and not _state_pool(
        initial_state, device, num_value_heads, (torch.bfloat16,)
    ):
        return False
    return True


def supports_kda_recurrent_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    beta_is_logit: bool = True,
    num_spec_tokens: Optional[int] = None,
    disable_state_update: bool = False,
) -> bool:
    """Admission mirroring FlashInfer's frozen FlashKDA prefill contract."""
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_MODULE_PREFILL, FI_JIT_MODULE_PREFILL)
        and cuda_tensor_on(q, ARCHS)
        and num_spec_tokens is None
        and not disable_state_update
        and use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and _finite_negative(lower_bound)
    ):
        return False
    device = q.device
    if not _contiguous(q, torch.bfloat16, device, ndim=4):
        return False
    batch, tokens, num_heads, head_dim = q.shape
    if batch <= 0 or tokens <= 1 or num_heads <= 0 or head_dim != HEAD_DIM:
        return False
    for tensor in (k, v, g):
        if not _contiguous(tensor, torch.bfloat16, device) or tensor.shape != q.shape:
            return False
    if not (
        beta is not None
        and beta.is_cuda
        and beta.device == device
        and beta.dtype == torch.bfloat16
        and tuple(beta.shape) == (batch, tokens, num_heads)
        and beta.stride(-1) == 1
        and beta.stride(-2) >= num_heads
        and (batch == 1 or beta.stride(0) == tokens * beta.stride(1))
    ):
        return False
    if not (
        _contiguous(A_log, torch.float32, device, ndim=1)
        and A_log.numel() == num_heads
        and _contiguous(dt_bias, torch.float32, device)
        and dt_bias.ndim in (1, 2)
        and dt_bias.numel() == num_heads * HEAD_DIM
        and (dt_bias.ndim == 1 or tuple(dt_bias.shape) == (num_heads, HEAD_DIM))
    ):
        return False
    if cu_seqlens is None:
        num_sequences = batch
    else:
        if (
            batch != 1
            or not cu_seqlens.is_cuda
            or cu_seqlens.device != device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences <= 0 or tokens <= num_sequences:
            return False
    if ssm_state_indices is not None and not (
        initial_state is not None
        and _contiguous(ssm_state_indices, torch.int32, device, ndim=1)
        and ssm_state_indices.numel() == num_sequences
    ):
        return False
    if initial_state is not None:
        if not _state_pool(
            initial_state, device, num_heads, (torch.bfloat16, torch.float32)
        ):
            return False
        # The exported FP32-state portfolio is the indexed state-pool API.
        if initial_state.dtype == torch.float32 and ssm_state_indices is None:
            return False
        if ssm_state_indices is None and initial_state.shape[0] != num_sequences:
            return False
    if (
        checkpoint_every_n_tokens < 0
        or checkpoint_every_n_tokens > 2**31 - 1
        or checkpoint_every_n_tokens % 16
    ):
        return False
    if checkpoint_every_n_tokens:
        if initial_state is not None and initial_state.dtype == torch.float32:
            return False
        if not (
            _contiguous(state_checkpoints, torch.bfloat16, device, ndim=4)
            and tuple(state_checkpoints.shape[1:]) == (num_heads, HEAD_DIM, HEAD_DIM)
            and _contiguous(checkpoint_cu_starts, torch.int64, device, ndim=1)
            and checkpoint_cu_starts.numel() == num_sequences + 1
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    if output is not None and not (
        _contiguous(output, torch.bfloat16, device) and output.shape == q.shape
    ):
        return False
    return True


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    seq_order: Optional[torch.Tensor] = None,
    prefill_workspace=None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    checkpoint_state_indices: Optional[torch.Tensor] = None,
    *,
    disable_state_update: bool = False,
    correction_cache: Optional[torch.Tensor] = None,
    kg_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward to ``flashinfer.kda.recurrent_kda(backend="cake")``.

    Returns ``(output, final_state_or_None)``. With ``ssm_state_indices`` and
    ``output_final_state=True`` the Cake backend returns the whole pool (the
    in-place update is identical to the CuTe path). Prefill graph capture needs
    an eagerly warmed ``prefill_workspace``
    (``flashinfer.RecurrentKDAPrefillWorkspace``) and a caller-owned ``output``.
    """
    from flashinfer.kda import recurrent_kda as fi_recurrent_kda

    return fi_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        beta_is_logit=beta_is_logit,
        seq_order=seq_order,
        prefill_workspace=prefill_workspace,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_state_indices=checkpoint_state_indices,
        disable_state_update=disable_state_update,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
        backend="cake",
    )


def supports_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> bool:
    """Exact-shape lock of the packed Kimi-K3 T=1 decode (H=12, K=V=128)."""
    import torch

    if not (
        flashinfer_module_available(FI_MODULE_DECODE, FI_JIT_MODULE_PACKED)
        and cuda_tensor_on(mixed_qkv, ARCHS)
    ):
        return False
    device = mixed_qkv.device
    width = PACKED_HEADS * HEAD_DIM
    if not (
        mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.ndim == 2
        and mixed_qkv.shape[1] == 3 * width
        and mixed_qkv.stride(1) == 1
    ):
        return False
    batch = mixed_qkv.shape[0]
    if batch <= 0 or batch > MAX_SEQUENCES:
        return False

    def _row_major(tensor, cols):
        return (
            tensor is not None
            and tensor.is_cuda
            and tensor.device == device
            and tensor.dtype == torch.bfloat16
            and tuple(tensor.shape) == (batch, cols)
            and tensor.stride(1) == 1
        )

    return (
        _row_major(raw_gate, width)
        and _row_major(raw_beta, PACKED_HEADS)
        and _contiguous(A_log, torch.float32, device, ndim=1)
        and A_log.numel() == PACKED_HEADS
        and _contiguous(dt_bias, torch.float32, device, ndim=1)
        and dt_bias.numel() == width
        and _state_pool(state, device, PACKED_HEADS, (torch.bfloat16,))
        and _contiguous(state_indices, torch.int32, device, ndim=1)
        and state_indices.numel() == batch
        and (
            output is None
            or (
                _contiguous(output, torch.bfloat16, device)
                and tuple(output.shape) == (batch, 1, PACKED_HEADS, HEAD_DIM)
            )
        )
    )


def packed_kda_decode(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.kda_decode.packed_kda_decode`` (Cake-only).

    ``state`` is updated in place; ``state_indices == -1`` marks inactive
    graph-padding rows (zero output, pool untouched). Returns the BF16
    ``[B, 1, 12, 128]`` output (``output`` itself when supplied).
    """
    from flashinfer.kda_decode import packed_kda_decode as fi_packed_kda_decode

    return fi_packed_kda_decode(
        mixed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output=output,
    )


def supports_kda_fused_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    lower_bound: Optional[float] = -5.0,
    norm_eps: float = 1e-5,
    state_indices_mode: Optional[str] = None,
    output: Optional[torch.Tensor] = None,
) -> bool:
    """Admission for ``fused_kda_decode(backend="cake")``; never raises.

    Mirrors FlashInfer's host validation plus the Cake selector's alignment
    facts (8-B conv_state / output, 16-B BF16 state or 32-B FP32 state). The
    exact variant registry (rows, slots, strides) is FlashInfer's; a call this
    predicate admits can still fail closed with ``RuntimeError`` when no frozen
    variant matches.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE_DECODE, FI_JIT_MODULE_FUSED)
        and cuda_tensor_on(x, ARCHS)
        and state_indices_mode in FUSED_STATE_INDICES_MODES
        and math.isfinite(float(norm_eps))
        and float(norm_eps) >= 0.0
        and _finite_negative(lower_bound)
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] % (3 * HEAD_DIM) == 0
        and x.stride(1) == 1
    ):
        return False
    device = x.device
    num_rows = x.shape[0]
    num_heads = x.shape[1] // (3 * HEAD_DIM)
    hidden = num_heads * HEAD_DIM
    if num_heads not in FUSED_HEADS:
        return False
    if not (
        _contiguous(weight, torch.float32, device)
        and tuple(weight.shape) == (3, 4, hidden)
    ):
        return False
    if not (
        conv_state is not None
        and conv_state.is_cuda
        and conv_state.device == device
        and conv_state.dtype == torch.bfloat16
        and conv_state.ndim == 3
        and tuple(conv_state.shape[1:]) == (3 * hidden, 3)
        and conv_state.stride(0) >= 9 * hidden
        and conv_state.stride(1) == 1
        and conv_state.stride(2) == 3 * hidden
        and conv_state.data_ptr() % 8 == 0
        and conv_state.stride(0) * conv_state.element_size() % 8 == 0
    ):
        return False
    if not (
        _contiguous(raw_gate, torch.bfloat16, device)
        and tuple(raw_gate.shape) == (1, num_rows, num_heads, HEAD_DIM)
        and raw_beta is not None
        and raw_beta.is_cuda
        and raw_beta.device == device
        and raw_beta.dtype == torch.bfloat16
        and tuple(raw_beta.shape) == (1, num_rows, num_heads)
        and raw_beta.stride(2) == 1
        and _contiguous(A_log, torch.float32, device, ndim=1)
        and A_log.numel() == num_heads
        and _contiguous(dt_bias, torch.float32, device, ndim=1)
        and dt_bias.numel() == hidden
        and _contiguous(state_indices, torch.int32, device, ndim=1)
        and state_indices.numel() == num_rows
    ):
        return False
    if not _state_pool(state, device, num_heads, (torch.bfloat16, torch.float32)):
        return False
    state_alignment = 16 if state.dtype == torch.bfloat16 else 32
    if (
        state.data_ptr() % state_alignment
        or state.stride(0) * state.element_size() % state_alignment
        or conv_state.shape[0] != state.shape[0]
    ):
        return False
    gate = output_gate
    if gate is None or not gate.is_cuda or gate.device != device:
        return False
    if gate.ndim == 4:
        if tuple(gate.shape) != (1, num_rows, num_heads, HEAD_DIM):
            return False
        gate = gate[0]
    if not (
        gate.dtype == torch.bfloat16
        and tuple(gate.shape) == (num_rows, num_heads, HEAD_DIM)
        and gate.stride(2) == 1
        and gate.stride(1) == HEAD_DIM
    ):
        return False
    if not (
        _contiguous(norm_weight, torch.float32, device, ndim=1)
        and norm_weight.numel() == HEAD_DIM
    ):
        return False
    if output is not None and not (
        _contiguous(output, torch.bfloat16, device)
        and tuple(output.shape) == (1, num_rows, num_heads, HEAD_DIM)
        and output.data_ptr() % 8 == 0
    ):
        return False
    return True


def fused_kda_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    lower_bound: Optional[float] = -5.0,
    norm_eps: float = 1e-5,
    output: Optional[torch.Tensor] = None,
    *,
    state_indices_mode: Literal[
        "positive_unique", "unique_or_null", "repeated_positive"
    ],
) -> torch.Tensor:
    """Forward to ``flashinfer.kda_decode.fused_kda_decode(backend="cake")``.

    Updates ``conv_state`` and ``state`` in place; returns the BF16
    ``[1, rows, H, 128]`` normalized, gated output. Fails closed (RuntimeError)
    when FlashInfer has no frozen variant for the layout.
    """
    from flashinfer.kda_decode import fused_kda_decode as fi_fused_kda_decode

    return fi_fused_kda_decode(
        x,
        weight,
        conv_state,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_indices,
        state,
        output_gate,
        norm_weight,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        output=output,
        backend="cake",
        state_indices_mode=state_indices_mode,
    )


__all__: Sequence[str] = (
    "supports_kda_recurrent_decode",
    "supports_kda_recurrent_prefill",
    "recurrent_kda",
    "supports_kda_packed_decode",
    "packed_kda_decode",
    "supports_kda_fused_decode",
    "fused_kda_decode",
)
