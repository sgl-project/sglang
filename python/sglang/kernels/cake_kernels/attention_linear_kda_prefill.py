"""Cake prepared KDA prefill export (BF16 / TF32 compute) via FlashInfer.

FlashInfer entries (contract at ``46340689a5ab``; inventory rows E1-23, A55,
A59): ``flashinfer.kda_prefill.prepare_bf16_kda_prefill`` /
``prepare_tf32_kda_prefill`` (``-> flashinfer.cake_kda_tf32_runtime.prepare_fwd
-> FlashKDABlackwellLaunch`` with ``.launch()`` / ``.close()``),
``flashinfer.kda_prefill.KDAPrefillPlanCache`` and
``flashinfer.kda_prefill.kda_prefill_supports_fp32_checkpoints``. JIT module
``flashinfer.jit.cake_kda_tf32``; built for sm_100a / sm_103a only.

Contract: q/k/v/out identical BF16 ``[B, T, H, 128]`` (dense, or strided views
of one packed row with a dense ``[H, 128]`` token payload); ``g`` BF16
``[B, T_storage >= T, H, 128]``; ``beta`` ``[B, T_storage, H]`` as BF16 logits
(``beta_is_logit=True``, the default) or FP32 probabilities; ``A_log`` FP32
``[H]``; ``dt_bias`` FP32 ``[H, 128]`` or ``[H*128]``; ``lower_bound`` None
(unbounded softplus gate) or a finite negative bound (``-5.0`` for Kimi-K3).
External ``initial_state`` / ``final_state`` are FP32 pools ``[N, H, 128, 128]``
(``final_state is initial_state`` for the in-place pool update) indexed by int32
``state_indices``; checkpoints are BF16 rows (or exact FP32 rows when
``kda_prefill_supports_fp32_checkpoints`` reports the carrier for the gate kind)
with int64 ``checkpoint_cu_starts`` (canonical chunk-count prefix sums). Packed
calls use B=1, ``cu_seqlens`` (>= 2 entries) and host ``sequence_lengths``.
``prepare_*`` validates, allocates workspace and compiles on first use;
``launch()`` only submits on the current stream, so graph capture is
caller-owned, but every tensor must stay alive at the same address. A
``KDAPrefillPlanCache`` hit rebinds pointers with one descriptor upload (no
host sync, no data-dependent branch), which is what makes the per-layer
serving call CUDA-graph capturable; a cache miss during capture is a
preparation (compile + allocation) and must happen eagerly first.

Admission rules absorbed from PR #34299: prepared export on the FP32 pool
(``--mamba-ssm-dtype float32``), gate bound None or finite negative, TF32
(``--kda-cake-prefill-precision tf32``) only for bounded-gate models, BF16
checkpoints unless the FP32 carrier is exported, strict fail-closed admission.
BBuf's review record: the bounded-gate FP32-pool prefill carried BF16 state
between 64-token chunks until the FP32 carrier was exported -- check
``kda_prefill_supports_fp32_checkpoints`` before requesting FP32 checkpoints.

Not supported here: training (``requires_grad``), BF16 external state,
``use_qk_l2norm_in_kernel=False``, active (FP32) beta with TF32 and an
unbounded gate, head_dim != 128, sm_90a / sm_120a / sm_121a.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.kda_prefill"
FI_RUNTIME_MODULE = "flashinfer.cake_kda_tf32_runtime"
FI_JIT_MODULE = "flashinfer.jit.cake_kda_tf32"
ARCHS = BLACKWELL_DATACENTER
HEAD_DIM = 128
COMPUTE_DTYPES = ("bf16", "tf32")
CHECKPOINT_GRANULARITY = 16


def _token_tensor(tensor, dtype, device, shape) -> bool:
    """Dense ``[B, T, H, 128]`` or a strided view keeping a dense token payload."""
    return (
        tensor is not None
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and (
            tensor.is_contiguous()
            or (
                tensor.stride(3) == 1
                and tensor.stride(2) == HEAD_DIM
                and tensor.stride(1) % 8 == 0
                and tensor.stride(1) >= shape[2] * HEAD_DIM
                and tensor.stride(0) == shape[1] * tensor.stride(1)
            )
        )
    )


def _fp32_pool(tensor, device, num_heads, float32) -> bool:
    return (
        tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == float32
        and tensor.ndim == 4
        and tensor.shape[0] > 0
        and tuple(tensor.shape[1:]) == (num_heads, HEAD_DIM, HEAD_DIM)
        and tuple(tensor.stride()[1:]) == (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)
        and tensor.stride(0) >= num_heads * HEAD_DIM * HEAD_DIM
    )


def supports_kda_prepared_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = -5.0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[Sequence[int]] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    beta_is_logit: bool = True,
    compute_dtype: str = "bf16",
    fp32_checkpoints_supported: bool = False,
) -> bool:
    """Admission for ``prepare_{bf16,tf32}_kda_prefill``; never raises.

    ``fp32_checkpoints_supported`` is the caller-cached result of
    :func:`kda_prefill_supports_fp32_checkpoints` for this device and gate
    kind; FP32 ``state_checkpoints`` are refused unless it is ``True``.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE, FI_RUNTIME_MODULE, FI_JIT_MODULE)
        and cuda_tensor_on(q, ARCHS)
        and compute_dtype in COMPUTE_DTYPES
        and (
            lower_bound is None
            or (math.isfinite(float(lower_bound)) and float(lower_bound) < 0.0)
        )
        and q.ndim == 4
        and not any(
            t is not None and t.requires_grad
            for t in (q, k, v, g, beta, A_log, dt_bias, initial_state)
        )
    ):
        return False
    device = q.device
    batch, tokens, num_heads, head_dim = q.shape
    if batch <= 0 or tokens <= 0 or num_heads <= 0 or head_dim != HEAD_DIM:
        return False
    shape = (batch, tokens, num_heads, HEAD_DIM)
    for tensor in (q, k, v, out):
        if not _token_tensor(tensor, torch.bfloat16, device, shape):
            return False
    if not (
        g is not None
        and g.is_cuda
        and g.device == device
        and g.dtype == torch.bfloat16
        and g.ndim == 4
        and g.shape[0] == batch
        and g.shape[1] >= tokens
        and tuple(g.shape[2:]) == (num_heads, HEAD_DIM)
        and g.stride(3) == 1
    ):
        return False
    beta_dtype = torch.bfloat16 if beta_is_logit else torch.float32
    if not (
        beta is not None
        and beta.is_cuda
        and beta.device == device
        and beta.dtype == beta_dtype
        and beta.ndim == 3
        and beta.shape[0] == batch
        and beta.shape[1] >= tokens
        and beta.shape[2] == num_heads
        and beta.stride(2) == 1
    ):
        return False
    if not beta_is_logit and compute_dtype == "tf32" and lower_bound is None:
        return False  # active-beta TF32 export requires a bounded gate
    if not (
        A_log.is_cuda
        and A_log.device == device
        and A_log.dtype == torch.float32
        and tuple(A_log.shape) == (num_heads,)
        and A_log.is_contiguous()
        and dt_bias.is_cuda
        and dt_bias.device == device
        and dt_bias.dtype == torch.float32
        and dt_bias.is_contiguous()
        and dt_bias.numel() == num_heads * HEAD_DIM
        and (dt_bias.ndim == 1 or tuple(dt_bias.shape) == (num_heads, HEAD_DIM))
    ):
        return False
    if cu_seqlens is None:
        num_sequences = batch
        if (
            sequence_lengths is not None
            and tuple(sequence_lengths) != (tokens,) * batch
        ):
            return False
    else:
        if (
            batch != 1
            or not cu_seqlens.is_cuda
            or cu_seqlens.device != device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or not cu_seqlens.is_contiguous()
            or sequence_lengths is None
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        lengths = tuple(int(n) for n in sequence_lengths)
        if len(lengths) != num_sequences or any(n < 0 for n in lengths):
            return False
        if sum(lengths) != tokens:
            return False
    for state in (initial_state, final_state):
        if state is not None and not _fp32_pool(
            state, device, num_heads, torch.float32
        ):
            return False
    if state_indices is not None:
        if not (
            state_indices.is_cuda
            and state_indices.device == device
            and state_indices.dtype == torch.int32
            and state_indices.ndim == 1
            and state_indices.numel() == num_sequences
            and state_indices.is_contiguous()
        ):
            return False
    elif initial_state is not None and initial_state.shape[0] < num_sequences:
        return False
    if (
        checkpoint_every_n_tokens < 0
        or checkpoint_every_n_tokens % CHECKPOINT_GRANULARITY
    ):
        return False
    if checkpoint_every_n_tokens:
        if state_checkpoints is None or checkpoint_cu_starts is None:
            return False
        if state_checkpoints.dtype == torch.float32:
            if compute_dtype != "bf16" or not fp32_checkpoints_supported:
                return False
        elif state_checkpoints.dtype != torch.bfloat16:
            return False
        if not (
            state_checkpoints.is_cuda
            and state_checkpoints.device == device
            and state_checkpoints.ndim == 4
            and tuple(state_checkpoints.shape[1:]) == (num_heads, HEAD_DIM, HEAD_DIM)
            and state_checkpoints.is_contiguous()
            and checkpoint_cu_starts.is_cuda
            and checkpoint_cu_starts.device == device
            and checkpoint_cu_starts.dtype == torch.int64
            and checkpoint_cu_starts.ndim == 1
            and checkpoint_cu_starts.numel() == num_sequences + 1
            and checkpoint_cu_starts.is_contiguous()
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    return True


def prepare_bf16_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = -5.0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[Sequence[int]] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    beta_is_logit: bool = True,
    plan_cache=None,
):
    """Forward to ``flashinfer.kda_prefill.prepare_bf16_kda_prefill``.

    Returns the prepared launch (``.launch()`` submits the whole operation on
    the current stream). With ``plan_cache`` the returned object is owned by
    the cache: do not ``close()`` it.
    """
    from flashinfer.kda_prefill import prepare_bf16_kda_prefill as fi_prepare

    return fi_prepare(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        sequence_lengths=sequence_lengths,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_is_logit=beta_is_logit,
        plan_cache=plan_cache,
    )


def prepare_tf32_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = -5.0,
    cu_seqlens: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[Sequence[int]] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    beta_is_logit: bool = True,
):
    """Forward to ``flashinfer.kda_prefill.prepare_tf32_kda_prefill``.

    TF32 compute keeps FP32 external state; checkpoints remain BF16. There is
    no plan-cache form of the TF32 export at the pinned FlashInfer commit.
    """
    from flashinfer.kda_prefill import prepare_tf32_kda_prefill as fi_prepare

    return fi_prepare(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        scale=scale,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        sequence_lengths=sequence_lengths,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        beta_is_logit=beta_is_logit,
    )


def kda_prefill_plan_cache(capacity: int = 64, max_bytes: Optional[int] = None):
    """Construct ``flashinfer.kda_prefill.KDAPrefillPlanCache``.

    Host-side bounded LRU of prepared BF16 launches keyed by structural
    signature; pass it as ``plan_cache=`` to :func:`prepare_bf16_kda_prefill`.
    One cache per stream (it is not safe to share across concurrent streams).
    ``max_bytes`` defaults to ``FLASHINFER_KDA_PLAN_CACHE_MAX_BYTES`` or 8 GiB.
    """
    from flashinfer.kda_prefill import KDAPrefillPlanCache

    return KDAPrefillPlanCache(capacity=capacity, max_bytes=max_bytes)


def kda_prefill_supports_fp32_checkpoints(
    device=None, *, lower_bound: Optional[float] = None
) -> bool:
    """Forward to ``flashinfer.kda_prefill.kda_prefill_supports_fp32_checkpoints``.

    Host-only query of the exported module registry: whether the fused direct
    M128 body keeps an FP32 recurrent carrier for the gate kind selected by
    ``lower_bound`` (None: unbounded softplus; otherwise the bounded gate).
    """
    from flashinfer.kda_prefill import (
        kda_prefill_supports_fp32_checkpoints as fi_supports,
    )

    return bool(fi_supports(device, lower_bound=lower_bound))


__all__: Sequence[str] = (
    "supports_kda_prepared_prefill",
    "prepare_bf16_kda_prefill",
    "prepare_tf32_kda_prefill",
    "kda_prefill_plan_cache",
    "kda_prefill_supports_fp32_checkpoints",
)
