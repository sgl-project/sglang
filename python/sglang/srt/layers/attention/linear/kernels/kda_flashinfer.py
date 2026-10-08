"""FlashInfer KDA decode/verify wrapper.

Wraps ``flashinfer.kda_decode.recurrent_kda`` (SM100 / Blackwell). FlashInfer has
no KDA prefill kernel, so ``extend`` stays on Triton / CuTe DSL.

Contract with the Triton KDA reference:
  - raw per-K gate ``a`` is activated in-kernel as
    ``-exp(A_log) * softplus(a + dt_bias)``;
  - beta ``b`` is a logit, so this wrapper passes ``sigmoid(b)``;
  - q/k are L2-normalized in-kernel;
  - state layout is ``[N, HV, V, K]`` for committed and speculative state.

Cake route (``SGLANG_CAKE_ROUTES=kda_decode``): the same FlashInfer symbols
with ``backend="cake"`` through ``sglang.kernels.cake_kernels.attention_linear_kda``.
Three call sites can take it, each behind the adapter's ``supports_*`` admission
and falling back to the path below when admission fails:
  - ``FlashInferKDAKernel.decode`` (T=1 recurrent decode, ``[B, 1, H, 128]`` view
    of the ``[1, B, H, 128]`` inputs, 1-D ``ssm_state_indices``, no ``cu_seqlens``;
    equal-head unbounded gate only, so Kimi-Linear-style ``lower_bound=None``);
  - :func:`try_cake_fused_decode` (Kimi-K3 conv + recurrence + gated RMSNorm,
    H=12 / TP8, ``lower_bound=-5``), called from ``KDAAttnBackend.forward_decode``
    ahead of ``kda_fused_decode.covered``;
  - :func:`try_cake_packed_decode` (Kimi-K3 packed T=1 decode, H=12, BF16 pool,
    ``lower_bound=-5``), called after the conv update when the fused handoff is
    not available.
``target_verify`` (T=2..6) never takes Cake: its frozen spec-decode routes need
precomputed log gates or the (T=3, H=16, lower_bound<0) family, neither of
which the in-kernel-gated SGLang verify contract produces.
"""

import logging
import os
from typing import Any, Optional

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

CAKE_ROUTE = "kda_decode"
_CAKE_HEAD_DIM = 128
_CAKE_PACKED_HEADS = 12
_CAKE_PACKED_LOWER_BOUND = -5.0
# SGLang's mamba pools reserve slot 0 for padded rows and mark CUDA-graph
# padding rows with -1 (hybrid_linear_attn_backend); live slots are unique per
# request. That is exactly FlashInfer's ``unique_or_null`` assertion for the
# Cake fused decode (non-positive index -> null row, no state update).
_CAKE_FUSED_STATE_INDICES_MODE = "unique_or_null"
_cake_logged: set = set()


def _cake_kda_decode_enabled() -> bool:
    return cake_route_enabled(CAKE_ROUTE)


def _cake_kda_adapter():
    from sglang.kernels.cake_kernels import attention_linear_kda

    return attention_linear_kda


def _cake_log_once(key: str, msg: str) -> None:
    if key in _cake_logged:
        return
    _cake_logged.add(key)
    logger.info(msg)


def _cake_int32_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.dtype == torch.int32 and indices.is_contiguous():
        return indices
    return indices.to(torch.int32).contiguous()


def try_cake_fused_decode(
    layer: Any,
    fused_static: tuple,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    onorm_gate: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Kimi-K3 fused decode through ``fused_kda_decode(backend="cake")``.

    Takes the engine's ``_k3_fused_decode_args`` stash (per-projection
    transposed fp32 conv weights ``[4, seg]`` x3, conv bias, ``A_log [H]``,
    fp32 o_norm weight, eps) and re-expresses it in the FlashInfer contract:
    conv weight ``[3, 4, H*128]`` (stacked once per layer), conv state as the
    ``[slots, 3*H*128, 3]`` transposed view of the engine's ``[slots, 3, 3*H*128]``
    pool, raw gate ``[1, B, H, 128]``, raw beta ``[1, B, H]``, output gate
    ``[B, H, 128]``. Returns the ``[1, B, H, 128]`` BF16 output or ``None`` when
    the route is off / not admitted (caller keeps its own fused kernel).
    Cake has no conv bias input, so layers with a conv bias are never admitted.
    In-place updates of ``conv_states`` / ``ssm_states`` follow the engine
    kernel's semantics (padding rows leave the pools untouched).
    """
    if not _cake_kda_decode_enabled() or getattr(
        layer, "_cake_fused_decode_disabled", False
    ):
        return None
    if getattr(layer, "bias", None) is not None:
        layer._cake_fused_decode_disabled = True
        _cake_log_once(
            f"fused-bias-{id(layer)}",
            "Cake KDA fused decode skipped: the conv1d has a bias, which the "
            "Cake fused_kda_decode contract does not carry.",
        )
        return None
    rows = int(mixed_qkv.shape[0])
    if ssm_states.ndim != 4 or int(ssm_states.shape[-1]) != _CAKE_HEAD_DIM:
        return None
    num_heads = int(ssm_states.shape[-3])
    seg = num_heads * _CAKE_HEAD_DIM
    if (
        a.ndim != 2
        or tuple(a.shape) != (rows, seg)
        or not a.is_contiguous()
        or b.ndim != 3
        or tuple(b.shape) != (1, rows, num_heads)
        or onorm_gate.ndim != 2
        or tuple(onorm_gate.shape) != (rows, seg)
        or onorm_gate.stride(1) != 1
        or conv_states.ndim != 3
    ):
        return None
    w_q_t, w_k_t, w_v_t, _conv_bias, a_log, onorm_w, onorm_eps = fused_static
    weight = getattr(layer, "_cake_fused_conv_weight", None)
    if weight is None:
        # [3, 4, seg] fp32: projection-major, then the four conv taps.
        weight = torch.stack((w_q_t, w_k_t, w_v_t)).contiguous()
        layer._cake_fused_conv_weight = weight
    conv_state = conv_states.transpose(-1, -2)
    raw_gate = a.view(1, rows, num_heads, _CAKE_HEAD_DIM)
    output_gate = onorm_gate.unflatten(-1, (num_heads, _CAKE_HEAD_DIM))
    indices = _cake_int32_indices(cache_indices)
    dt_bias = layer.dt_bias
    lower_bound = getattr(layer, "lower_bound", None)
    adapter = _cake_kda_adapter()
    admission = getattr(layer, "_cake_fused_admission", None)
    if admission is None:
        admission = layer._cake_fused_admission = {}
    key = (
        rows,
        id(ssm_states),
        ssm_states.dtype,
        ssm_states.stride(0),
        id(conv_states),
        conv_states.stride(0),
        a.dtype,
        b.dtype,
        b.stride(2),
        onorm_gate.dtype,
        onorm_gate.stride(0),
    )
    admitted = admission.get(key)
    if admitted is None:
        admitted = adapter.supports_kda_fused_decode(
            mixed_qkv,
            weight,
            conv_state,
            raw_gate,
            b,
            a_log,
            dt_bias,
            indices,
            ssm_states,
            output_gate,
            onorm_w,
            lower_bound=lower_bound,
            norm_eps=onorm_eps,
            state_indices_mode=_CAKE_FUSED_STATE_INDICES_MODE,
        )
        admission[key] = admitted
        _cake_log_once(
            f"fused-{id(layer)}-{rows}-{admitted}",
            f"Cake KDA fused decode {'taken' if admitted else 'rejected by admission'}: "
            f"rows={rows} heads={num_heads} state={ssm_states.dtype} "
            f"lower_bound={lower_bound}",
        )
    if not admitted:
        return None
    try:
        return adapter.fused_kda_decode(
            mixed_qkv,
            weight,
            conv_state,
            raw_gate,
            b,
            a_log,
            dt_bias,
            indices,
            ssm_states,
            output_gate,
            onorm_w,
            lower_bound=lower_bound,
            norm_eps=onorm_eps,
            state_indices_mode=_CAKE_FUSED_STATE_INDICES_MODE,
        )
    except RuntimeError as exc:
        # FlashInfer fails closed on the host (no frozen variant for this
        # layout) before any launch; keep this layer on the engine kernel.
        layer._cake_fused_decode_disabled = True
        logger.warning(
            "Cake KDA fused decode disabled for layer %s after FlashInfer "
            "rejected the layout (rows=%d, heads=%d): %s",
            getattr(layer, "layer_id", "?"),
            rows,
            num_heads,
            exc,
        )
        return None


def try_cake_packed_decode(
    layer: Any,
    qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Kimi-K3 packed T=1 decode through ``packed_kda_decode`` (Cake-only entry).

    The Cake kernel is locked to H=HV=12, K=V=128, ``lower_bound=-5`` and a
    BF16 state pool, i.e. the K3 TP8 shape with ``--mamba-ssm-dtype bfloat16``.
    ``qkv`` is the post-conv ``[B, 3*1536]`` BF16 activation, ``a`` the raw gate
    ``[B, 1536]``, ``b`` the raw beta logits ``[1, B, 12]`` (or ``[B, 12]``).
    Returns the ``[1, B, 12, 128]`` BF16 output or ``None`` (caller continues
    with its own packed / unpacked decode). The pool is updated in place; rows
    with ``cache_indices == -1`` are inactive.
    """
    if not _cake_kda_decode_enabled():
        return None
    lower_bound = getattr(layer, "lower_bound", None)
    if lower_bound is None or float(lower_bound) != _CAKE_PACKED_LOWER_BOUND:
        return None
    if (
        getattr(layer, "num_v_heads", None) != _CAKE_PACKED_HEADS
        or getattr(layer, "head_v_dim", None) != _CAKE_HEAD_DIM
        or getattr(layer, "head_k_dim", None) != _CAKE_HEAD_DIM
    ):
        return None
    rows = int(qkv.shape[0])
    width = _CAKE_PACKED_HEADS * _CAKE_HEAD_DIM
    if (
        qkv.ndim != 2
        or int(qkv.shape[1]) != 3 * width
        or a.ndim != 2
        or tuple(a.shape) != (rows, width)
        or b.numel() != rows * _CAKE_PACKED_HEADS
    ):
        return None
    raw_beta = b.reshape(rows, _CAKE_PACKED_HEADS)
    a_log = getattr(layer, "_cake_packed_a_log", None)
    if a_log is None:
        a_log = layer.A_log.detach().reshape(-1).float().contiguous()
        layer._cake_packed_a_log = a_log
    indices = _cake_int32_indices(cache_indices)
    adapter = _cake_kda_adapter()
    admission = getattr(layer, "_cake_packed_admission", None)
    if admission is None:
        admission = layer._cake_packed_admission = {}
    key = (
        rows,
        id(ssm_states),
        ssm_states.dtype,
        ssm_states.stride(0),
        qkv.dtype,
        qkv.stride(0),
        a.dtype,
        a.stride(0),
        raw_beta.dtype,
        raw_beta.stride(0),
    )
    admitted = admission.get(key)
    if admitted is None:
        admitted = adapter.supports_kda_packed_decode(
            qkv, a, raw_beta, a_log, layer.dt_bias, ssm_states, indices
        )
        admission[key] = admitted
        _cake_log_once(
            f"packed-{id(layer)}-{rows}-{admitted}",
            f"Cake KDA packed decode {'taken' if admitted else 'rejected by admission'}: "
            f"rows={rows} state={ssm_states.dtype} lower_bound={lower_bound}",
        )
    if not admitted:
        return None
    out = adapter.packed_kda_decode(
        qkv, a, raw_beta, a_log, layer.dt_bias, ssm_states, indices
    )
    return out.view(1, rows, _CAKE_PACKED_HEADS, _CAKE_HEAD_DIM)


# ---------------------------------------------------------------------------
# Lazy import for the FlashInfer KDA kernel
# ---------------------------------------------------------------------------
_flashinfer_kda_available: Optional[bool] = None
_flashinfer_recurrent_kda = None


def _get_flashinfer_kda_kernel():
    """Lazy import for FlashInfer ``recurrent_kda`` (decode + MTP).

    Returns (available, recurrent_kda_fn).
    """
    global _flashinfer_kda_available, _flashinfer_recurrent_kda
    if _flashinfer_kda_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from flashinfer.kda_decode import recurrent_kda

            _flashinfer_recurrent_kda = recurrent_kda
            # recurrent_kda is SM100-only (CuTe DSL, Blackwell).
            _flashinfer_kda_available = (
                is_cuda() and torch.cuda.get_device_capability()[0] >= 10
            )
            if _flashinfer_kda_available:
                logger.info("FlashInfer KDA kernel (recurrent_kda) loaded successfully")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"FlashInfer KDA kernel not available: {e}")
            _flashinfer_kda_available = False
            _flashinfer_recurrent_kda = None
    return _flashinfer_kda_available, _flashinfer_recurrent_kda


def build_fused_accept_indices(
    *,
    slots: torch.Tensor,
    scratch_steps: int,
    draft_token_num: int,
    accept_lens_pool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot-indexed verify indices + accept lengths for fused-accept mode.

    Row n of the returned ``[N, T]`` index tensor addresses the scratch slots of
    the request holding mamba slot ``slots[n]``. A padded row (``slots[n] < 0``)
    yields ONLY negative indices (``-scratch_steps + step`` with
    ``step < scratch_steps``), which recurrent_kda treats as inactive — the
    padding contract must survive any refactor of this arithmetic. The nat
    gather clamps padded slots to row 0 of the pool; their value is never
    consumed (inactive rows). All ops are device-side and capture-safe.
    """
    step = torch.arange(draft_token_num, device=slots.device, dtype=torch.int32)
    ssm_state_indices = (
        slots.to(torch.int32)[:, None] * scratch_steps + step[None, :]
    ).contiguous()  # [N, T]
    num_accepted_tokens = accept_lens_pool.index_select(
        0, slots.clamp(min=0).to(torch.int64)
    )
    return ssm_state_indices, num_accepted_tokens


class FlashInferKDAKernel(LinearAttnKernelBase):
    """FlashInfer KDA kernel: SM100 decode + MTP (target_verify), topk=1.

    Prefill (``extend``) is intentionally not implemented -- FlashInfer ships no
    KDA chunk kernel; the dispatcher keeps prefill on Triton / CuTe DSL.
    """

    def __init__(self):
        available, self._recurrent_kda = _get_flashinfer_kda_kernel()
        if not available or self._recurrent_kda is None:
            raise RuntimeError(
                "FlashInfer KDA kernel (recurrent_kda) is not available. "
                "Requires SM100 (Blackwell) and a FlashInfer build with KDA support."
            )
        # Cache the per-layer constant gate-param prep (A_log/dt_bias reshape+cast),
        # keyed by tensor identity. Layer params are persistent weights so id() is
        # stable; this removes the per-call reshape/float/contiguous work.
        self._gate_cache: dict = {}
        # Cache the constant per-(row-map, batch, T) verify scatter indices
        # (ssm_state_indices), which never change across verify calls.
        self._verify_idx_cache: dict = {}
        # State pools whose stride layout has been validated against the
        # recurrent_kda contract (per-layer views are pool-stable, so id() is
        # a stable key — same lifetime argument as _gate_cache).
        self._state_contract_ok: set = set()
        # Cake T=1 decode admission per (batch, heads, pool) key; the per-call
        # inputs are reshaped views of the same buffers so the predicate's
        # answer is a function of this key.
        self._cake_decode_admission: dict = {}
        logger.info("Using FlashInfer KDA kernel")

    def _check_state_stride_contract(self, ssm_states: torch.Tensor) -> None:
        """One-time (per pool view) check that ``ssm_states`` matches the
        layout ``recurrent_kda`` was compiled for.

        The kernel's state argument is a CuTe fake tensor of shape
        ``[N, HV, V, K]`` with stride ``(sym_int64(divisibility=16), V*K, K, 1)``
        and ``assumed_align=32`` (flashinfer ``kda_kernels/recurrent_kda.py``):
        the slot stride is free — which is what lets the envelope-strided pools
        (unified memory / page-major layout, slot stride = per-slot envelope
        pitch) be passed in and updated IN PLACE on the cu_seqlens path — but
        the inner strides are compiled-in constants and the divisibility /
        alignment are hard assumptions. A pool violating them would mis-address
        state in-kernel without any error; fail loudly here instead.
        """
        key = id(ssm_states)
        if key in self._state_contract_ok:
            return
        if ssm_states.dim() != 4:
            raise ValueError(
                f"recurrent_kda needs a [N, HV, V, K] state pool; got "
                f"shape {tuple(ssm_states.shape)}"
            )
        _, hv, v, k = ssm_states.shape
        if ssm_states.stride()[1:] != (v * k, k, 1):
            raise ValueError(
                "recurrent_kda state inner strides must be compact "
                f"(V*K, K, 1)=({v * k}, {k}, 1); got {ssm_states.stride()[1:]} "
                "(only the slot stride may be non-compact)"
            )
        base_bytes = ssm_states.storage_offset() * ssm_states.element_size()
        if ssm_states.stride(0) % 16 != 0 or base_bytes % 32 != 0:
            raise ValueError(
                "recurrent_kda state pool breaks the compiled stride contract: "
                f"slot stride {ssm_states.stride(0)} elements must be a multiple "
                f"of 16 and the base byte offset {base_bytes} a multiple of 32 "
                "(sym_int64(divisibility=16) / assumed_align=32)"
            )
        self._state_contract_ok.add(key)

    # ---- gate / beta normalization (shared by decode + verify) ----

    def _prep_gate_params(self, A_log: torch.Tensor, dt_bias: torch.Tensor):
        # A_log: [1, 1, H, 1] -> [H] fp32; dt_bias: [H*K] (1D) -> fp32. Cached per
        # layer (constant weights) so this is a dict lookup on the hot path.
        key = (id(A_log), id(dt_bias))
        cached = self._gate_cache.get(key)
        if cached is not None:
            return cached
        A_log_fi = A_log.reshape(-1).float().contiguous()
        dt_bias_fi = (
            dt_bias.reshape(-1).float().contiguous() if dt_bias is not None else None
        )
        self._gate_cache[key] = (A_log_fi, dt_bias_fi)
        return A_log_fi, dt_bias_fi

    @staticmethod
    def _beta_logit_to_prob(b: torch.Tensor) -> torch.Tensor:
        # Triton KDA does beta = sigmoid(b); recurrent_kda wants beta pre-sigmoided.
        # torch.sigmoid computes in fp32 internally, so a single sigmoid on the bf16
        # logit is enough (avoids an explicit fp32 upcast + downcast = 2 extra kernels).
        return torch.sigmoid(b).to(torch.bfloat16)

    # ---- decode ----

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = cache_indices.shape[0]
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # The committed pool goes into the kernel as-is (in-place update); under
        # unified memory / page-major it is an envelope-strided view, which the
        # cu_seqlens path supports — verify the compiled contract once per pool.
        self._check_state_stride_contract(ssm_states)

        # Pack each request as a length-1 sequence ([1, B, ...] + cu_seqlens) so
        # recurrent_kda indexes the committed pool IN-KERNEL via ssm_state_indices.
        # The plain [B, 1, ...] path (no cu_seqlens) instead python-gathers
        # initial_state[indices] and scatters it back with index_put around the
        # kernel (~141us at B=64 in ncu); the cu_seqlens path skips both. q/k/v
        # already arrive as [1, B, H, D] from forward_decode, so the reshape is a
        # no-op view. recurrent_kda's cp.async + shared-mem staging are hardwired to
        # bf16 (2-byte elements) for q/k/v/g/beta and the state, so every input is
        # cast to bf16 -- a no-op for the common bf16 KDA model, a correct downcast
        # otherwise (float16 bits would be reinterpreted as bf16 without the cast).
        query_fi = q.reshape(1, batch_size, num_heads, head_k_dim).to(torch.bfloat16)
        key_fi = k.reshape(1, batch_size, num_heads, head_k_dim).to(torch.bfloat16)
        value_fi = v.reshape(1, batch_size, num_v_heads, head_v_dim).to(torch.bfloat16)
        g_fi = a.reshape(1, batch_size, num_v_heads, head_k_dim).to(torch.bfloat16)
        beta_fi = self._beta_logit_to_prob(b).reshape(1, batch_size, num_v_heads)

        A_log_fi, dt_bias_fi = self._prep_gate_params(A_log, dt_bias)

        if lower_bound is None and _cake_kda_decode_enabled():
            output_cake = self._cake_decode(
                query_fi,
                key_fi,
                value_fi,
                g_fi,
                beta_fi,
                A_log_fi,
                dt_bias_fi,
                ssm_states,
                cache_indices,
            )
            if output_cake is not None:
                return output_cake

        # Gate contract matches the Triton decode path (safe gate when
        # lower_bound set); in-place state update, no rollback for decode.
        output_fi, _ = self._recurrent_kda(
            q=query_fi,
            k=key_fi,
            v=value_fi,
            g=g_fi,
            beta=beta_fi,
            A_log=A_log_fi,
            dt_bias=dt_bias_fi,
            scale=None,
            initial_state=ssm_states,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            cu_seqlens=query_start_loc.to(torch.int32),
            ssm_state_indices=cache_indices.to(torch.int32),
        )

        return output_fi.view(1, batch_size, num_v_heads, head_v_dim)

    def _cake_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """T=1 decode through ``recurrent_kda(backend="cake")``.

        The Cake decode contract is ``[B, 1, H, 128]`` inputs with a 1-D int32
        ``ssm_state_indices`` and no ``cu_seqlens`` (FlashInfer rejects an
        explicit T=1 ``cu_seqlens`` for Cake), so the ``[1, B, H, D]`` inputs are
        re-viewed with the batch outermost -- a free view since the leading
        dimension is 1. The committed pool is updated in place through the
        indices, exactly like the ``cu_seqlens`` CuTe path below. Returns
        ``None`` when the adapter does not admit the call (GQA, non-BF16 pool,
        head_dim != 128, lower_bound set, unsupported device/FlashInfer build).
        """
        batch_size = q.shape[1]
        num_heads, head_dim = q.shape[2], q.shape[3]
        num_v_heads = v.shape[2]
        if head_dim != _CAKE_HEAD_DIM or v.shape[3] != _CAKE_HEAD_DIM:
            return None
        q_c = q.view(batch_size, 1, num_heads, head_dim)
        k_c = k.view(batch_size, 1, num_heads, head_dim)
        v_c = v.view(batch_size, 1, num_v_heads, head_dim)
        g_c = g.view(batch_size, 1, num_v_heads, head_dim)
        beta_c = beta.view(batch_size, 1, num_v_heads)
        indices = _cake_int32_indices(cache_indices)
        key = (
            batch_size,
            num_heads,
            num_v_heads,
            id(ssm_states),
            ssm_states.dtype,
            ssm_states.stride(0),
            q.stride(1),
            v.stride(1),
            g.stride(1),
        )
        admitted = self._cake_decode_admission.get(key)
        adapter = _cake_kda_adapter()
        if admitted is None:
            admitted = adapter.supports_kda_recurrent_decode(
                q_c,
                k_c,
                v_c,
                g_c,
                beta_c,
                ssm_states,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=None,
                ssm_state_indices=indices,
                cu_seqlens=None,
                num_spec_tokens=None,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                beta_is_logit=False,
                disable_state_update=False,
                scale=None,
            )
            self._cake_decode_admission[key] = admitted
            _cake_log_once(
                f"recurrent-{batch_size}-{num_heads}-{num_v_heads}-{admitted}",
                f"Cake KDA recurrent decode {'taken' if admitted else 'rejected by admission'}: "
                f"B={batch_size} H={num_heads} HV={num_v_heads} state={ssm_states.dtype}",
            )
        if not admitted:
            return None
        output, _ = adapter.recurrent_kda(
            q_c,
            k_c,
            v_c,
            g_c,
            beta_c,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=None,
            initial_state=ssm_states,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=None,
            ssm_state_indices=indices,
        )
        return output.view(1, batch_size, num_v_heads, head_dim)

    # ---- target_verify (MTP, topk=1) ----

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        lower_bound: Optional[float] = None,
        fused_accept_state_indices: Optional[torch.Tensor] = None,
        fused_accept_num_accepted: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if retrieve_parent_token is not None:
            raise RuntimeError(
                "FlashInfer KDA verify kernel only supports topk=1 "
                "(retrieve_parent_token must be None)."
            )

        seq_len = q.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = cache_steps  # T = 1 + num_spec_tokens
        num_spec_tokens = draft_token_num - 1
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # Packed [1, N*T, ...] inputs, cu_seqlens = query_start_loc (draft stride).
        # recurrent_kda is bf16-only (see decode), so cast every input to bf16.
        q_fi = q.reshape(1, seq_len, num_heads, head_k_dim).to(torch.bfloat16)
        k_fi = k.reshape(1, seq_len, num_heads, head_k_dim).to(torch.bfloat16)
        v_fi = v.reshape(1, seq_len, num_v_heads, head_v_dim).to(torch.bfloat16)
        g_fi = a.reshape(1, seq_len, num_v_heads, head_k_dim).to(torch.bfloat16)
        beta_fi = self._beta_logit_to_prob(b).reshape(1, seq_len, num_v_heads)

        A_log_fi, dt_bias_fi = self._prep_gate_params(A_log, dt_bias)

        # recurrent_kda indexes a flat state pool. Map each request/step to the
        # matching slot in SGLang's [scratch_row, allocated_step, HV, V, K] buffer.
        scratch = intermediate_states_buffer  # [N_scratch, T, HV, V, K]
        scratch_steps = scratch.shape[1]
        if draft_token_num > scratch_steps:
            raise RuntimeError(
                f"KDA verify needs {draft_token_num} scratch steps, "
                f"but intermediate_ssm only has {scratch_steps}."
            )

        if fused_accept_state_indices is not None:
            # Fused-accept mode: rows are the requests' mamba SLOTS (stable for
            # the request lifetime, unlike batch positions), so last round's
            # checkpoints are addressable this round. The kernel seeds each row
            # from slot[nat - 1] (nat = last round's accept length, gathered
            # from accept_lens_pool; fresh requests were staged with nat = 1 at
            # extend) and overwrites all T slots in place — no committed-pool
            # seed copy here and no SSM commit scatter after verify. Padded
            # graph rows carry slot -1: every derived index stays negative,
            # which recurrent_kda treats as inactive. Both tensors are built
            # once per forward by the backend (see KDAAttnBackend), so the 20
            # KDA layers of a step share one build.
            ssm_state_indices = fused_accept_state_indices
            num_accepted_tokens = fused_accept_num_accepted
        else:
            num_accepted_tokens = None
            base_rows = intermediate_state_indices[:batch_size]
            cache_key = (
                id(intermediate_state_indices),
                batch_size,
                draft_token_num,
                scratch_steps,
            )
            ssm_state_indices = self._verify_idx_cache.get(cache_key)
            if ssm_state_indices is None:
                # The fast seed copy below assumes row n in scratch belongs to
                # request n.
                expected = torch.arange(
                    batch_size, device=base_rows.device, dtype=base_rows.dtype
                )
                if not torch.equal(base_rows, expected):
                    raise RuntimeError(
                        "FlashInfer KDA verify requires an identity intermediate "
                        "row-map (verify_intermediate_state_indices must be arange)."
                    )
                step = torch.arange(draft_token_num, device=q.device, dtype=torch.int32)
                ssm_state_indices = (
                    base_rows.to(torch.int32)[:, None] * scratch_steps + step[None, :]
                ).contiguous()  # [N, T]
                self._verify_idx_cache[cache_key] = ssm_state_indices

            # Seed step 0 from committed state, then recurrent_kda overwrites it
            # with token-0 post-state. Padded graph rows clamp to slot 0; their
            # output is ignored.
            base_state = ssm_states.index_select(
                0, cache_indices[:batch_size].clamp(min=0).to(torch.int64)
            )
            scratch[:batch_size, 0].copy_(base_state)

        # Same storage as scratch, flattened over the allocated step stride.
        state_pool = scratch.view(
            scratch.shape[0] * scratch_steps, num_v_heads, head_v_dim, head_k_dim
        )

        output_fi, _ = self._recurrent_kda(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=g_fi,
            beta=beta_fi,
            A_log=A_log_fi,
            dt_bias=dt_bias_fi,
            scale=None,
            initial_state=state_pool,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            cu_seqlens=query_start_loc.to(torch.int32),
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens=num_accepted_tokens,
        )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)

    # ---- extend (prefill): not provided by FlashInfer ----

    def extend(self, *args, **kwargs):
        raise NotImplementedError(
            "FlashInferKDAKernel has no prefill kernel; keep prefill on Triton / CuTe DSL."
        )
