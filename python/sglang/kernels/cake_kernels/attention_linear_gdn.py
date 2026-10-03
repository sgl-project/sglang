"""Cake GDN (Gated Delta Net) prefill and decode via FlashInfer.

FlashInfer entries (contract at ``46340689a5ab``):

* ``flashinfer.gdn_prefill.chunk_gated_delta_rule(..., backend="cake_gdn")``
  (inventory E1-24 non-CP, E1-25 / A49 CP). Rank-3 contiguous
  ``[tokens, heads, 128]`` q/k/v/output, all FP16 or all BF16; GQA
  (``Hk == Hv``, ``Hq % Hk == 0``) or GVA (``Hq == Hk``, ``Hv % Hq == 0``);
  contiguous int32/int64 ``cu_seqlens``; optional FP32 ``g`` (alpha, the
  multiplicative decay in (0, 1]) and ``beta`` (probabilities), both
  ``[tokens, max(Hq, Hv)]``; state pool ``[N, H_sab, 128, 128]`` K-last
  (FP32/BF16 for non-CP, FP32/FP16/BF16 for CP) indexed by ``state_indices``;
  checkpoints every multiple of 64 tokens with ``checkpoint_cu_starts``
  (``sum(floor(len / interval))`` rows for non-CP; CP maps the interval onto
  its chunk length). ``use_cp=True`` selects the four-stage context-parallel
  route ``gdn_kernels.blackwell.cake_gdn_cp_backend.chunk_gated_delta_rule_gdn_cp_sm100``
  (T precompute -> MN precompute -> state fixup -> checkpoint copy -> CP
  prefill); ``use_cp=False`` or ``"auto"`` keeps the manifest-backed non-CP
  route, which requires caller-normalized Q/K (``use_qk_l2norm_in_kernel=False``).
  ``"auto"`` never selects Cake for prefill; ``cake_gdn`` fails closed
  (``CakeGDNUnsupportedError``, a ``NotImplementedError``) when no manifest row
  matches. JIT modules ``flashinfer.jit.cake_gdn`` / ``cake_gdn_cp_backend``.
* ``flashinfer.gdn_kernels.blackwell.cake_gdn_cp_backend.prepare_gdn_cp_prefill``
  (E1-26 / A50): prepared ``GDNCPPrefill`` -- runs the composite once eagerly
  and optionally captures one internal fixed-address graph; ``replay()`` must
  run on the preparation stream.
* ``flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose(...)`` (E1-27):
  BF16 ``[B, T, H, 128]`` q/k, ``[B, T, HV, 128]`` v, BF16 ``[B, T, HV]`` gates
  ``a``/``b``, FP32 ``A_log``/``dt_bias [HV]``, indexed K-last state pool
  ``[pool, HV, 128, 128]`` (BF16 with packed inner strides, or FP32) and int32
  ``[B]`` initial / output indices; optional ``intermediate_states_buffer
  [B, >= T, HV, 128, 128]`` for MTP verify with ``disable_state_update``.
* ``flashinfer.gdn_decode.gated_delta_rule_decode(...)`` (E1-28): K-major FP32
  state ``[B, HV, 128, 128]`` T=1 decode, exact contiguous tensors.

All routes are built for sm_100a / sm_103a only. The CP prefill keeps one
module-level prepared plan keyed by layouts, addresses, scalars and stream;
a cache miss during CUDA-graph capture raises ("warm before capture"). The
non-CP prefill reads ``cu_seqlens`` (and checkpoint starts) to the host once
and caches by (ptr, version, numel); its first resolution must also be eager.

Admission rules absorbed from the superseded sglang PRs:

* #35400: BF16 state pool only for the serving rows, head_dim 128, BF16
  q/k/v/state, FP32 ``A_log``/``dt_bias``, int32 contiguous indices; prefill
  needs ``Hq == Hk`` for the sglang call site and ``checkpoint_every_n % 64 ==
  0``; "the public auto-CP prefill route is never intercepted" (we never pass
  ``backend="auto"`` with ``use_cp``). Raw ``flashinfer.jit`` loading, variant
  selection and grid math were dropped (FlashInfer owns row promotion).
* #35552: ``use_cp`` is forwarded as FlashInfer's own default ("auto"); the CP
  route must equal the non-CP route numerically.
* #40656: verify (MTP) goes through the public pretranspose dispatch with
  FP32 gate parameters; ``backend`` is an explicit argument of the forwarder
  (``"cake_gdn"`` strict, ``"auto"`` lets FlashInfer promote exact rows).

Not supported here (keep the existing SGLang path): FP8 I/O, head_dim != 128,
``gated_delta_rule_mtp`` (no Cake branch), ``chunk_gated_delta_rule2``,
non-indexed pretranspose decode (``state=`` without a pool), sm_90a /
sm_120a / sm_121a.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union

from sglang.kernels.cake_kernels._support import (
    BLACKWELL_DATACENTER,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE_PREFILL = "flashinfer.gdn_prefill"
FI_MODULE_DECODE = "flashinfer.gdn_decode"
FI_MODULE_CP = "flashinfer.gdn_kernels.blackwell.cake_gdn_cp_backend"
FI_JIT_MODULE = "flashinfer.jit.cake_gdn"
FI_JIT_MODULE_CP = "flashinfer.jit.cake_gdn_cp_backend"
ARCHS = BLACKWELL_DATACENTER
HEAD_DIM = 128
CHECKPOINT_BLOCK = 64
STATE_INNER_STRIDES = (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1)


def _contiguous(tensor, dtype, device, ndim=None) -> bool:
    return (
        tensor is not None
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
        and (ndim is None or tensor.ndim == ndim)
    )


def _head_mapping_is_legal(hq: int, hk: int, hv: int) -> bool:
    return (
        hq > 0
        and hk > 0
        and hv > 0
        and ((hq == hk and hv % hq == 0) or (hk == hv and hq % hk == 0))
    )


def supports_gdn_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    *,
    initial_state: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Union[Literal["auto"], bool] = "auto",
    use_qk_l2norm_in_kernel: bool = False,
    output_final_state: bool = False,
    scale: Optional[float] = None,
    cp_chunk_len: Optional[int] = None,
) -> bool:
    """Admission for ``chunk_gated_delta_rule(backend="cake_gdn")``; never raises.

    Structural mirror of FlashInfer's host checks for both the CP
    (``use_cp=True``) and the manifest-backed non-CP route. The non-CP
    manifest (exact head/state/dtype rows) is FlashInfer's; an admitted call
    can still fail closed with ``CakeGDNUnsupportedError``.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE_PREFILL, FI_JIT_MODULE, FI_JIT_MODULE_CP)
        and cuda_tensor_on(q, ARCHS)
        and use_cp in ("auto", True, False)
        and (scale is None or (math.isfinite(float(scale))))
        and q.ndim == 3
        and k.ndim == 3
        and v.ndim == 3
        and q.dtype in (torch.float16, torch.bfloat16)
        and cu_seqlens is not None
    ):
        return False
    device = q.device
    dtype = q.dtype
    total, hq, head_dim = q.shape
    hk, hv = k.shape[1], v.shape[1]
    h_sab = max(hq, hv)
    if not (
        total > 0
        and head_dim == HEAD_DIM
        and _contiguous(q, dtype, device)
        and _contiguous(k, dtype, device)
        and _contiguous(v, dtype, device)
        and tuple(k.shape) == (total, hk, HEAD_DIM)
        and tuple(v.shape) == (total, hv, HEAD_DIM)
        and _head_mapping_is_legal(hq, hk, hv)
    ):
        return False
    if not (
        cu_seqlens.is_cuda
        and cu_seqlens.device == device
        and cu_seqlens.dtype in (torch.int32, torch.int64)
        and cu_seqlens.ndim == 1
        and cu_seqlens.numel() >= 2
        and cu_seqlens.is_contiguous()
    ):
        return False
    num_seqs = cu_seqlens.numel() - 1
    for gate in (g, beta):
        if gate is not None and not (
            _contiguous(gate, torch.float32, device, ndim=2)
            and tuple(gate.shape) == (total, h_sab)
        ):
            return False
    if output is not None and not (
        _contiguous(output, dtype, device, ndim=3)
        and tuple(output.shape) == (total, h_sab, HEAD_DIM)
        and all(output.data_ptr() != t.data_ptr() for t in (q, k, v))
    ):
        return False
    is_cp = use_cp is True
    if not is_cp and use_qk_l2norm_in_kernel:
        return False  # non-CP Cake consumes caller-normalized Q/K
    state_dtypes = (
        (torch.float32, torch.float16, torch.bfloat16)
        if is_cp
        else (torch.float32, torch.bfloat16)
    )
    states = [
        t
        for t in (
            initial_state,
            output_state if output_final_state else None,
            state_checkpoints if checkpoint_every_n_tokens else None,
        )
        if t is not None
    ]
    if states and any(t.dtype != states[0].dtype for t in states):
        return False
    for state in (initial_state, output_state):
        if state is None:
            continue
        if not (
            state.is_cuda
            and state.device == device
            and state.dtype in state_dtypes
            and state.ndim == 4
            and state.shape[0] > 0
            and tuple(state.shape[1:]) == (h_sab, HEAD_DIM, HEAD_DIM)
            and (is_cp or tuple(state.stride()[1:]) == STATE_INNER_STRIDES)
            and all(s > 0 for s in state.stride())
        ):
            return False
    if state_indices is not None:
        if not (
            state_indices.is_cuda
            and state_indices.device == device
            and state_indices.dtype in (torch.int32, torch.int64)
            and state_indices.ndim == 1
            and state_indices.numel() == num_seqs
            and state_indices.is_contiguous()
        ):
            return False
        if output_final_state and output_state is None:
            return False  # pool-indexed final state needs the caller's pool
    elif initial_state is not None and initial_state.shape[0] != num_seqs:
        return False
    if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % CHECKPOINT_BLOCK:
        return False
    if checkpoint_every_n_tokens:
        if not (
            state_checkpoints is not None
            and state_checkpoints.is_cuda
            and state_checkpoints.device == device
            and state_checkpoints.ndim == 4
            and tuple(state_checkpoints.shape[1:]) == (h_sab, HEAD_DIM, HEAD_DIM)
            and state_checkpoints.is_contiguous()
            and (not is_cp or state_checkpoints.dtype == torch.float32)
            and checkpoint_cu_starts is not None
            and checkpoint_cu_starts.is_cuda
            and checkpoint_cu_starts.device == device
            and checkpoint_cu_starts.dtype in (torch.int32, torch.int64)
            and checkpoint_cu_starts.ndim == 1
            and checkpoint_cu_starts.numel() == num_seqs + 1
            and checkpoint_cu_starts.is_contiguous()
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    if cp_chunk_len is not None and (
        not is_cp or cp_chunk_len <= 0 or cp_chunk_len % CHECKPOINT_BLOCK
    ):
        return False
    return True


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Union[Literal["auto"], bool] = "auto",
    state_indices: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    *,
    cp_chunk_len: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Forward to ``flashinfer.gdn_prefill.chunk_gated_delta_rule(backend="cake_gdn")``.

    ``use_cp=True`` requests the Cake CP route; ``False`` / ``"auto"`` the
    non-CP route (Cake never auto-selects CP). Returns ``output`` or
    ``(output, final_state)`` when ``output_final_state``. ``cp_chunk_len``
    maps to FlashInfer's private ``_cp_chunk_len`` (CP only).
    """
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk

    return fi_chunk(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=output,
        output_state=output_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_cp=use_cp,
        state_indices=state_indices,
        _cp_chunk_len=cp_chunk_len,
        backend="cake_gdn",
        max_seqlen=max_seqlen,
    )


def prepare_gdn_cp_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: Optional[int] = None,
    max_seqlen: Optional[int] = None,
    scale: Optional[float] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    capture_graph: bool = True,
):
    """Forward to ``cake_gdn_cp_backend.prepare_gdn_cp_prefill``.

    Runs the four-stage composite once (producing ``output`` / final state)
    and, with ``capture_graph``, captures an internal fixed-address CUDA graph;
    the returned ``GDNCPPrefill`` exposes ``replay()`` (preparation stream
    only when captured) and ``launch_with_bindings(...)``. Use
    :func:`supports_gdn_chunk_gated_delta_rule` with ``use_cp=True``.
    """
    from flashinfer.gdn_kernels.blackwell.cake_gdn_cp_backend import (
        prepare_gdn_cp_prefill as fi_prepare,
    )

    return fi_prepare(
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        initial_state,
        output=output,
        output_state=output_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        _capture_graph=capture_graph,
    )


def supports_gdn_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    scale: Optional[float] = None,
) -> bool:
    """Admission for the Cake pretranspose decode / verify route; never raises.

    Mirrors FlashInfer's structural checks (indexed K-last pool, BF16 I/O,
    FP32 gate parameters, int32 ``[B]`` indices). Which (B, T, H, HV, state
    dtype) rows are promoted is FlashInfer's manifest; ``backend="cake_gdn"``
    fails closed, ``backend="auto"`` falls back to the CuTe path silently.
    """
    import torch

    if not (
        flashinfer_module_available(FI_MODULE_DECODE, FI_JIT_MODULE)
        and cuda_tensor_on(q, ARCHS)
        and (scale is None or math.isfinite(float(scale)))
        and initial_state is not None
        and initial_state_indices is not None
        and q.ndim == 4
        and v.ndim == 4
    ):
        return False
    device = q.device
    batch, seq_len, hq, head_dim = q.shape
    hv, value_dim = v.shape[2], v.shape[3]
    if (
        head_dim != HEAD_DIM
        or value_dim != HEAD_DIM
        or seq_len <= 0
        or batch <= 0
        or hq <= 0
        or hv <= 0
        or hv % hq
        or tuple(k.shape) != tuple(q.shape)
        or tuple(v.shape) != (batch, seq_len, hv, HEAD_DIM)
        or tuple(a.shape) != (batch, seq_len, hv)
        or tuple(b.shape) != (batch, seq_len, hv)
    ):
        return False
    for tensor in (q, k, v, a, b):
        if not (
            tensor.is_cuda
            and tensor.device == device
            and tensor.dtype == torch.bfloat16
            and tensor.stride(-1) == 1
        ):
            return False
    if not (
        _contiguous(A_log, torch.float32, device, ndim=1)
        and A_log.numel() == hv
        and _contiguous(dt_bias, torch.float32, device, ndim=1)
        and dt_bias.numel() == hv
    ):
        return False
    pool = initial_state
    if not (
        pool.is_cuda
        and pool.device == device
        and pool.dtype in (torch.bfloat16, torch.float32)
        and pool.ndim == 4
        and pool.shape[0] > 0
        and tuple(pool.shape[1:]) == (hv, HEAD_DIM, HEAD_DIM)
        and pool.stride(-1) == 1
        and (
            pool.dtype == torch.float32
            or tuple(pool.stride()[1:]) == STATE_INNER_STRIDES
        )
    ):
        return False
    if pool.dtype == torch.float32:
        if seq_len == 1 and not all(
            t.is_contiguous() for t in (q, k, v, pool, A_log, a, dt_bias, b)
        ):
            return False
        if seq_len > 1 and intermediate_states_buffer is None:
            return False
    for indices in (initial_state_indices, output_state_indices):
        if indices is not None and not (
            _contiguous(indices, torch.int32, device, ndim=1)
            and indices.numel() == batch
        ):
            return False
    if seq_len == 1 and (
        intermediate_states_buffer is not None or disable_state_update
    ):
        return False
    if intermediate_states_buffer is not None and not (
        intermediate_states_buffer.is_cuda
        and intermediate_states_buffer.device == device
        and intermediate_states_buffer.dtype == pool.dtype
        and intermediate_states_buffer.is_contiguous()
        and intermediate_states_buffer.ndim == 5
        and intermediate_states_buffer.shape[0] == batch
        and intermediate_states_buffer.shape[1] >= seq_len
        and tuple(intermediate_states_buffer.shape[2:]) == (hv, HEAD_DIM, HEAD_DIM)
    ):
        return False
    if output is not None and not (
        _contiguous(output, torch.bfloat16, device)
        and tuple(output.shape) == (batch, seq_len, hv, HEAD_DIM)
    ):
        return False
    return True


def gated_delta_rule_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    *,
    backend: Literal["cake_gdn", "auto"] = "cake_gdn",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to ``flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose``.

    ``backend="cake_gdn"`` is strict; ``"auto"`` is the PR #40656 pattern
    (FlashInfer promotes exact manifest rows, otherwise CuTe). Returns
    ``(output, state_pool)``; the pool rows at ``output_state_indices`` (or
    ``initial_state_indices``) are updated in place unless
    ``disable_state_update``.
    """
    from flashinfer.gdn_decode import (
        gated_delta_rule_decode_pretranspose as fi_decode_pretranspose,
    )

    return fi_decode_pretranspose(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=disable_state_update,
        backend=backend,
    )


def supports_gdn_decode_nontranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> bool:
    """Admission for the K-major FP32-state T=1 Cake decode; never raises."""
    import torch

    if not (
        flashinfer_module_available(FI_MODULE_DECODE, FI_JIT_MODULE)
        and cuda_tensor_on(q, ARCHS)
        and (scale is None or math.isfinite(float(scale)))
        and q.ndim == 4
        and v.ndim == 4
    ):
        return False
    device = q.device
    batch, seq_len, hq, head_dim = q.shape
    hv = v.shape[2]
    return (
        seq_len == 1
        and head_dim == HEAD_DIM
        and hq > 0
        and hv > 0
        and hv % hq == 0
        and _contiguous(q, torch.bfloat16, device)
        and _contiguous(k, torch.bfloat16, device)
        and tuple(k.shape) == tuple(q.shape)
        and _contiguous(v, torch.bfloat16, device)
        and tuple(v.shape) == (batch, 1, hv, HEAD_DIM)
        and _contiguous(state, torch.float32, device)
        and tuple(state.shape) == (batch, hv, HEAD_DIM, HEAD_DIM)
        and _contiguous(a, torch.bfloat16, device)
        and tuple(a.shape) == (batch, 1, hv)
        and _contiguous(b, torch.bfloat16, device)
        and tuple(b.shape) == (batch, 1, hv)
        and _contiguous(A_log, torch.float32, device, ndim=1)
        and A_log.numel() == hv
        and _contiguous(dt_bias, torch.float32, device, ndim=1)
        and dt_bias.numel() == hv
        and (
            output is None
            or (
                _contiguous(output, torch.bfloat16, device)
                and tuple(output.shape) == (batch, 1, hv, HEAD_DIM)
            )
        )
    )


def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    *,
    backend: Literal["cake_gdn", "auto"] = "cake_gdn",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to ``flashinfer.gdn_decode.gated_delta_rule_decode`` (K-major).

    FP32 ``state [B, HV, 128, 128]`` is updated in place; returns
    ``(output, state)``.
    """
    from flashinfer.gdn_decode import gated_delta_rule_decode as fi_decode

    return fi_decode(
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        scale=scale,
        output=output,
        use_qk_l2norm=use_qk_l2norm,
        backend=backend,
    )


__all__: Sequence[str] = (
    "supports_gdn_chunk_gated_delta_rule",
    "chunk_gated_delta_rule",
    "prepare_gdn_cp_prefill",
    "supports_gdn_decode_pretranspose",
    "gated_delta_rule_decode_pretranspose",
    "supports_gdn_decode_nontranspose",
    "gated_delta_rule_decode",
)
