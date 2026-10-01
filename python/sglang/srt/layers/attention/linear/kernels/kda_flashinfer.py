"""FlashInfer KDA wrapper.

Wraps ``flashinfer.kda_decode.recurrent_kda`` for decode/verify and the public
``flashinfer.kda.recurrent_kda`` facade for CAKE prefill (SM100 / Blackwell).
CAKE decode uses ``flashinfer.packed_kda_decode`` for the exact Kimi-K3 TP8
serving contract.

Contract with the Triton KDA reference:
  - raw per-K gate ``a`` is activated in-kernel as
    ``-exp(A_log) * softplus(a + dt_bias)``;
  - beta ``b`` is a logit, so this wrapper passes ``sigmoid(b)``;
  - q/k are L2-normalized in-kernel;
  - state layout is ``[N, HV, V, K]`` for committed and speculative state.

The optional ``cake`` mode forwards SGLang's post-convolution packed Q/K/V,
raw gate/beta, and indexed state pool directly to the exported CAKE decode
contract. Unsupported shapes and ReplaySSM use the existing Triton packed
path. Prefill consumes raw gate/beta logits, updates the indexed state pool in
place, and can return radix-cache checkpoints without materializing inputs.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.linear.kda_route_telemetry import (
    CAKE_DECODE_EXCEPTION,
    CAKE_PACKED_EXCEPTION,
    CAKE_PREFILL_EXCEPTION,
    PACKED_SELECTOR_EXCEPTION,
    PREFILL_SELECTOR_EXCEPTION,
    TRITON_FALLBACK_EXCEPTION,
    CakePackedDecodeReason,
    CakePrefillReason,
    record_kda_terminal_route,
    stable_kda_exception_detail,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.mem_cache.allocator.mamba import MambaStateIndexContract
from sglang.srt.runtime_context import mamba_cache_chunk_size
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# One-time notice that the TF32 prepared export hands unbounded gates to Triton.
_tf32_unbounded_gate_warned = False

# ---------------------------------------------------------------------------
# Lazy import for the FlashInfer KDA kernel
# ---------------------------------------------------------------------------
_flashinfer_kda_available: Optional[bool] = None
_flashinfer_recurrent_kda = None
_flashinfer_kda_prefill_available: Optional[bool] = None
_flashinfer_recurrent_kda_facade = None
_flashinfer_packed_kda_available: Optional[bool] = None
_flashinfer_packed_kda_decode = None
_cake_packed_decode_route_logged = False

_CAKE_PACKED_NUM_HEADS = 12
_CAKE_PACKED_HEAD_DIM = 128
_CAKE_PACKED_QKV_WIDTH = 3 * _CAKE_PACKED_NUM_HEADS * _CAKE_PACKED_HEAD_DIM
_CAKE_PACKED_GATE_WIDTH = _CAKE_PACKED_NUM_HEADS * _CAKE_PACKED_HEAD_DIM
_CAKE_PACKED_SCALE = _CAKE_PACKED_HEAD_DIM**-0.5
_CAKE_PACKED_LOWER_BOUND = -5.0


@dataclass(frozen=True)
class CakePackedDecodeAdmission:
    """One terminal selector result for a packed CAKE decode attempt."""

    eligible: bool
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class CakePrefillAdmission:
    """One terminal selector result for an ordinary CAKE prefill attempt."""

    eligible: bool
    reason: str
    detail: str = ""


def maybe_build_cake_checkpoint_plan(
    forward_batch: ForwardBatch,
    forward_metadata: ForwardMetadata,
    device: str,
) -> None:
    """Populate native Cake checkpoints for KDA radix-cache tracking.

    Cake row zero is the sequence's initial state. Subsequent rows are states
    after each full checkpoint interval strictly before the sequence end. This
    is the packed indexing already produced by ``_init_track_ssm_indices`` for
    KDA, so only the per-sequence cumulative starts are materialized here.
    """
    if (
        forward_metadata.track_ssm_h_src is None
        or forward_metadata.track_ssm_h_src.numel() == 0
    ):
        return

    checkpoint_every_n_tokens = mamba_cache_chunk_size()
    if checkpoint_every_n_tokens <= 0 or checkpoint_every_n_tokens % 32 != 0:
        raise ValueError(
            "Cake KDA checkpoint interval must be a positive multiple of 32, "
            f"got {checkpoint_every_n_tokens}."
        )
    extend_seq_lens = forward_batch.extend_seq_lens.to(device="cpu", dtype=torch.int64)
    checkpoint_counts = (extend_seq_lens - 1) // checkpoint_every_n_tokens + 1
    checkpoint_cu_starts = torch.zeros(checkpoint_counts.numel() + 1, dtype=torch.int64)
    checkpoint_cu_starts[1:] = torch.cumsum(checkpoint_counts, dim=0)

    if torch.cuda.is_available():
        checkpoint_cu_starts = checkpoint_cu_starts.pin_memory().to(
            device, non_blocking=True
        )
    else:
        # pin_memory() needs a CUDA runtime; CPU-only hosts (unit tests) skip it.
        checkpoint_cu_starts = checkpoint_cu_starts.to(device)
    forward_metadata.state_checkpoint_cu_starts = checkpoint_cu_starts
    forward_metadata.num_state_checkpoints = int(checkpoint_cu_starts[-1])
    forward_metadata.state_checkpoint_every_n_tokens = checkpoint_every_n_tokens


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


def _get_flashinfer_kda_prefill_kernel():
    """Lazy import for the public FlashInfer KDA prefill facade."""
    global _flashinfer_kda_prefill_available, _flashinfer_recurrent_kda_facade
    if _flashinfer_kda_prefill_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from flashinfer.kda import recurrent_kda
            from flashinfer.kda_prefill import (  # noqa: F401
                RecurrentKDAPrefillWorkspace,
            )

            _flashinfer_recurrent_kda_facade = recurrent_kda
            _flashinfer_kda_prefill_available = (
                is_cuda() and torch.cuda.get_device_capability()[0] >= 10
            )
            if _flashinfer_kda_prefill_available:
                logger.info("FlashInfer CAKE KDA prefill kernel loaded successfully")
        except (ImportError, RuntimeError) as e:
            logger.warning("FlashInfer CAKE KDA prefill is not available: %s", e)
            _flashinfer_kda_prefill_available = False
            _flashinfer_recurrent_kda_facade = None
    return _flashinfer_kda_prefill_available, _flashinfer_recurrent_kda_facade


_cuda_device_capability_cache: dict = {}


def _cuda_device_capability(device: torch.device) -> tuple:
    """Per-device cached ``torch.cuda.get_device_capability`` for the hot path."""
    key = device.index
    capability = _cuda_device_capability_cache.get(key)
    if capability is None:
        capability = torch.cuda.get_device_capability(device)
        _cuda_device_capability_cache[key] = capability
    return capability


_flashinfer_prepared_bf16_available: Optional[bool] = None
_flashinfer_prepare_bf16_kda_prefill = None
_flashinfer_prepare_tf32_kda_prefill = None
_CAKE_PREFILL_PRECISIONS = ("bf16", "tf32")


def _cake_prefill_precision() -> str:
    """``bf16`` (default) or ``tf32``: the exported prepared-prefill precision.

    ``SGLANG_KDA_CAKE_PREFILL_PRECISION`` (operator override; also the knob
    for replay scripts and tests that run the kernel without a published
    server-args context) wins over ``--kda-cake-prefill-precision`` from the
    published exec namespace; the default is ``bf16``.  TF32 needs the
    prepared export (no facade route), an FP32 state pool and a bounded gate;
    it trades throughput for accuracy.
    """
    value = os.environ.get("SGLANG_KDA_CAKE_PREFILL_PRECISION")
    if not value:
        try:
            from sglang.srt.runtime_context import get_exec

            value = getattr(get_exec().mamba, "kda_cake_prefill_precision", None)
        except Exception:
            value = None
    if not value:
        value = "bf16"
    value = str(value).strip().lower()
    if value not in _CAKE_PREFILL_PRECISIONS:
        raise ValueError(
            f"kda_cake_prefill_precision must be bf16 or tf32, got {value!r}"
        )
    return value


_flashinfer_kda_prefill_plan_cache_cls = None


def _prepared_qkv_layout_supported(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> bool:
    """True when the prepared BF16 export can read q/k/v without a repack.

    Dense tensors always qualify.  Strided ``split(dim=-1)`` views qualify when
    the per-token ``[H, D]`` payload is dense (``stride(-1) == 1``,
    ``stride(-2) == D``), the three token strides agree and are multiples of 8
    elements, and a leading batch axis (if longer than one) is the plain
    product of tokens and token stride.
    """
    if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
        return True
    tensors = (q, k, v)
    if any(t.ndim != 4 for t in tensors):
        return False
    heads_x_dim = q.shape[2] * q.shape[3]
    token_strides = {t.stride(1) for t in tensors}
    if len(token_strides) != 1:
        return False
    token_stride = token_strides.pop()
    if token_stride < heads_x_dim or token_stride % 8 != 0:
        return False
    for t in tensors:
        if t.stride(3) != 1 or t.stride(2) != t.shape[3]:
            return False
        if t.shape[0] > 1 and t.stride(0) != t.shape[1] * token_stride:
            return False
    return True


# FP32 intermediate-state rows are exported per gate kind: index 0 = unbounded
# softplus gate (lower_bound=None), index 1 = bounded gate.
_flashinfer_kda_prefill_fp32_checkpoints = (False, False)


def _get_flashinfer_prepared_bf16_prefill():
    """Lazy import for the exported BF16 KDA prepared-call prefill API.

    ``flashinfer.prepare_bf16_kda_prefill`` (flashinfer-ai/flashinfer#5278,
    routing #5363, regenerated modules #5370) prepares one complete packed
    prefill on an FP32 external state pool with BF16 checkpoints and submits it
    with ``launch()``. It is the export of the Cake BF16 dispatcher that the
    346-row source/export campaign validated bitwise.
    """
    global _flashinfer_prepared_bf16_available, _flashinfer_prepare_bf16_kda_prefill
    global _flashinfer_prepare_tf32_kda_prefill
    global _flashinfer_kda_prefill_plan_cache_cls
    global _flashinfer_kda_prefill_fp32_checkpoints
    if _flashinfer_prepared_bf16_available is None:
        try:
            from flashinfer import kda_prefill as _kda_prefill_module
            from flashinfer import prepare_bf16_kda_prefill

            _flashinfer_prepare_bf16_kda_prefill = prepare_bf16_kda_prefill
            # Same prepared-call ABI on the TF32 export (BF16 q/k/v/out, FP32
            # state, BF16 checkpoints); selected by --kda-cake-prefill-precision.
            _flashinfer_prepare_tf32_kda_prefill = getattr(
                _kda_prefill_module, "prepare_tf32_kda_prefill", None
            )
            _flashinfer_prepared_bf16_available = True
            # FP32 intermediate states (FP32 chunk carrier + FP32 checkpoint
            # rows) exist only in FlashInfer builds whose module registry
            # exports them for this GPU; older exports write BF16 rows.
            probe = getattr(
                _kda_prefill_module, "kda_prefill_supports_fp32_checkpoints", None
            )
            if callable(probe) and is_cuda():
                device = torch.device("cuda", torch.cuda.current_device())
                _flashinfer_kda_prefill_fp32_checkpoints = (
                    bool(probe(device, lower_bound=None)),
                    bool(probe(device, lower_bound=-1.0)),
                )
            else:
                _flashinfer_kda_prefill_fp32_checkpoints = (False, False)
            try:
                from flashinfer import KDAPrefillPlanCache

                _flashinfer_kda_prefill_plan_cache_cls = KDAPrefillPlanCache
            except ImportError:
                _flashinfer_kda_prefill_plan_cache_cls = None
            logger.info(
                "FlashInfer prepared BF16 KDA prefill export loaded successfully"
            )
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "FlashInfer prepared BF16 KDA prefill export unavailable: %s", exc
            )
            _flashinfer_prepared_bf16_available = False
            _flashinfer_prepare_bf16_kda_prefill = None
    return _flashinfer_prepared_bf16_available, _flashinfer_prepare_bf16_kda_prefill


_KDA_PREFILL_DUMP_DIR = os.environ.get("SGLANG_KDA_PREFILL_DUMP_DIR", "")
_KDA_PREFILL_DUMP_LIMIT = int(os.environ.get("SGLANG_KDA_PREFILL_DUMP_LIMIT", "64"))
_kda_prefill_dump_count = 0


def _maybe_dump_kda_prefill_call(layer_id: int, **tensors) -> None:
    """Persist real prefill activations (opt-in) for offline per-chunk state checks.

    Enabled by SGLANG_KDA_PREFILL_DUMP_DIR; saves at most
    SGLANG_KDA_PREFILL_DUMP_LIMIT calls per process. Tensors are cloned to CPU
    after a device sync, so this is a diagnostic mode only.
    """
    global _kda_prefill_dump_count
    if not _KDA_PREFILL_DUMP_DIR or _kda_prefill_dump_count >= _KDA_PREFILL_DUMP_LIMIT:
        return
    torch.cuda.synchronize()
    os.makedirs(_KDA_PREFILL_DUMP_DIR, exist_ok=True)
    rank = int(os.environ.get("RANK", "-1") or -1)
    if (
        rank < 0
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        rank = torch.distributed.get_rank()
    record = {"layer_id": int(layer_id), "rank": max(rank, 0)}
    for name, value in tensors.items():
        if isinstance(value, torch.Tensor):
            record[name] = value.detach().to("cpu", copy=True)
        else:
            record[name] = value
    path = os.path.join(
        _KDA_PREFILL_DUMP_DIR,
        # TP ranks are separate processes; the pid keeps their files distinct
        # even when no rank information is available.
        f"kda_prefill_rank{record['rank']}_pid{os.getpid()}_"
        f"{_kda_prefill_dump_count:04d}_layer{int(layer_id)}.pt",
    )
    torch.save(record, path)
    _kda_prefill_dump_count += 1


_FACADE_CONTRACT_MESSAGE = "does not support this recurrent_kda prefill contract"


def _is_facade_contract_error(exc: BaseException) -> bool:
    """``recurrent_kda(backend="cake")`` refused the call at its own contract
    check (strided operands, unsupported gate or state layout). That is a
    declared Triton fallback for the layer, not a kernel failure."""
    return isinstance(exc, ValueError) and _FACADE_CONTRACT_MESSAGE in str(exc)


def _cake_prefill_gate_bound_ok(
    lower_bound: Optional[float], allow_unbounded: bool
) -> bool:
    """Bounded gate: finite negative lower bound. The prepared BF16 export also
    serves the unbounded softplus gate (``lower_bound=None``, e.g. Kimi-Linear);
    the TF32 export (bounded-gate modules only) and the facade do not."""
    if lower_bound is None:
        # The exported BF16 schedules decay each 64-token chunk from
        # tile-anchored floored gate prefixes (integer power-of-two anchors per
        # 16-token tile, exact total in FP32), so real Kimi-Linear activations
        # with per-token log2 gates far below -126 keep the recurrent state
        # within the Triton reference error (per-chunk rel L2 ~3e-3).
        return allow_unbounded
    return math.isfinite(float(lower_bound)) and float(lower_bound) < 0.0


_CAKE_DEBUG_CHECKS = os.environ.get("SGLANG_KDA_CAKE_DEBUG_CHECKS", "0") == "1"


def _cake_prefill_api_policy() -> str:
    """``auto`` (default), ``prepared`` or ``facade`` from SGLANG_KDA_CAKE_PREFILL_API.

    ``auto`` selects the prepared BF16 export whenever the recurrent state pool
    is FP32 (the export's external-state contract) and the FlashInfer build
    exposes it; BF16 pools keep the ``recurrent_kda(backend="cake")`` facade.
    """
    policy = os.environ.get("SGLANG_KDA_CAKE_PREFILL_API", "auto").strip().lower()
    if policy not in ("auto", "prepared", "facade"):
        raise ValueError(
            "SGLANG_KDA_CAKE_PREFILL_API must be auto, prepared or facade, "
            f"got {policy!r}"
        )
    return policy


def _get_flashinfer_packed_kda_kernel():
    """Lazy import for the exported CAKE packed-decode facade."""
    global _flashinfer_packed_kda_available, _flashinfer_packed_kda_decode
    if _flashinfer_packed_kda_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from flashinfer import packed_kda_decode
            from flashinfer.jit.cpp_ext import is_cuda_version_at_least

            capability = torch.cuda.get_device_capability() if is_cuda() else None
            _flashinfer_packed_kda_available = bool(
                (capability == (10, 0) and is_cuda_version_at_least("12.8"))
                or (capability == (10, 3) and is_cuda_version_at_least("12.9"))
            )
            _flashinfer_packed_kda_decode = (
                packed_kda_decode if _flashinfer_packed_kda_available else None
            )
            if _flashinfer_packed_kda_available:
                logger.info("FlashInfer CAKE packed KDA decode loaded successfully")
        except (ImportError, RuntimeError) as e:
            logger.warning("FlashInfer CAKE packed KDA decode is not available: %s", e)
            _flashinfer_packed_kda_available = False
            _flashinfer_packed_kda_decode = None
    return _flashinfer_packed_kda_available, _flashinfer_packed_kda_decode


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
    """FlashInfer KDA kernel for SM100 decode, verify, and CAKE prefill.

    ``backend="cute-dsl"`` supports decode and topk=1 target-verify.
    ``backend="cake"`` additionally supports ordinary KDA prefill.
    """

    def __init__(self, backend: str = "cute-dsl"):
        if backend not in ("cute-dsl", "cake"):
            raise ValueError(
                f"FlashInfer KDA backend must be 'cute-dsl' or 'cake', got {backend!r}"
            )
        available, self._recurrent_kda = _get_flashinfer_kda_kernel()
        if not available or self._recurrent_kda is None:
            raise RuntimeError(
                "FlashInfer KDA kernel (recurrent_kda) is not available. "
                "Requires SM100 (Blackwell) and a FlashInfer build with KDA support."
            )
        if backend == "cake":
            capability = torch.cuda.get_device_capability()
            if capability not in ((10, 0), (10, 3)):
                raise RuntimeError(
                    "CAKE KDA requires SM100 or SM103, got compute capability "
                    f"{capability[0]}.{capability[1]}"
                )
            if "backend" not in inspect.signature(self._recurrent_kda).parameters:
                raise RuntimeError(
                    "Installed FlashInfer recurrent_kda does not expose the CAKE "
                    "backend; upgrade FlashInfer."
                )
        self._backend = backend
        self._cake_prefill_precision = (
            _cake_prefill_precision() if backend == "cake" else "bf16"
        )
        # Cache the per-layer constant gate-param prep (A_log/dt_bias reshape+cast)
        # keyed by tensor identity. Layer params are persistent weights, but
        # ``id()`` alone is not a safe key: CPython reuses the id of a collected
        # tensor for a new one, so a caller that hands fresh A_log/dt_bias
        # tensors per call (tests, replays, reloaded weights) would silently get
        # another layer's gate parameters. Every entry therefore carries weak
        # references to the keyed tensors and is only used while they are alive
        # and identical (see ``_prep_gate_params``).
        self._gate_cache: dict = {}
        # Cache the constant per-(row-map, batch, T) verify scatter indices
        # (ssm_state_indices), which never change across verify calls.
        self._verify_idx_cache: dict = {}
        # State pools whose stride layout has been validated against the
        # recurrent_kda contract, keyed by id() and guarded by a weak reference
        # like ``_gate_cache`` (a reused id must re-validate, never skip).
        self._state_contract_ok: dict = {}
        logger.info("Using FlashInfer KDA kernel backend=%s", backend)

    @property
    def cake_prefill_precision(self) -> str:
        """``bf16`` or ``tf32`` prepared export used by Cake prefill."""
        return getattr(self, "_cake_prefill_precision", "bf16")

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
        validated = self._state_contract_ok.get(key)
        if validated is not None and validated() is ssm_states:
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
        self._state_contract_ok[key] = weakref.ref(ssm_states)

    @staticmethod
    def _check_cake_state_contract(
        ssm_states: torch.Tensor,
        *,
        num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
    ) -> None:
        expected_inner = (num_v_heads, head_v_dim, head_k_dim)
        if ssm_states.dtype != torch.bfloat16:
            raise ValueError(
                f"CAKE KDA state pool must be bfloat16, got {ssm_states.dtype}"
            )
        if ssm_states.dim() != 4 or tuple(ssm_states.shape[1:]) != expected_inner:
            raise ValueError(
                "CAKE KDA state pool must have shape [N, HV, V, K] with "
                f"(HV, V, K)={expected_inner}; got {tuple(ssm_states.shape)}"
            )

    @staticmethod
    def _cake_direct_indexed_state_is_supported(
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        *,
        batch_size: int,
    ) -> bool:
        """Return whether the frozen T=1 kernel can index the pool in place.

        The exported direct-state ABI uses an int32 element offset for each
        state slot. Keep the check allocation-free and value-independent so it
        is safe during CUDA graph capture. Index values remain a caller-guaranteed
        contract: ``-1`` is padding, and all active rows are unique and in bounds.
        """
        if (
            not ssm_states.is_cuda
            or not cache_indices.is_cuda
            or ssm_states.device != cache_indices.device
            or ssm_states.dtype != torch.bfloat16
            or cache_indices.dtype != torch.int32
            or cache_indices.ndim != 1
            or cache_indices.numel() != batch_size
            or not cache_indices.is_contiguous()
            or ssm_states.ndim != 4
            or ssm_states.shape[0] <= 0
        ):
            return False

        _, num_value_heads, value_dim, head_dim = ssm_states.shape
        slot_stride = ssm_states.stride(0)
        int32_max = torch.iinfo(torch.int32).max
        return (
            ssm_states.stride()[1:] == (value_dim * head_dim, head_dim, 1)
            and slot_stride >= num_value_heads * value_dim * head_dim
            and slot_stride % 8 == 0
            and slot_stride <= int32_max
            and ssm_states.shape[0] * slot_stride <= int32_max
            and ssm_states.data_ptr() % 16 == 0
        )

    # ---- gate / beta normalization (shared by decode + verify) ----

    def _prep_gate_params(self, A_log: torch.Tensor, dt_bias: torch.Tensor):
        # A_log: [1, 1, H, 1] -> [H] fp32; dt_bias: [H*K] (1D) -> fp32. Cached per
        # layer (constant weights) so this is a dict lookup on the hot path.
        key = (id(A_log), id(dt_bias))
        cached = self._gate_cache.get(key)
        if cached is not None:
            a_ref, d_ref, prepared = cached
            # The ids match; use the entry only if it was built from these very
            # tensors (a reused id after garbage collection must not hit).
            if a_ref() is A_log and (d_ref is None or d_ref() is dt_bias):
                return prepared
        # Model weights arrive as nn.Parameters (requires_grad=True); the
        # exported prepared prefill refuses autograd-tracked inputs, so detach.
        A_log_fi = A_log.detach().reshape(-1).float().contiguous()
        dt_bias_fi = (
            dt_bias.detach().reshape(-1).float().contiguous()
            if dt_bias is not None
            else None
        )
        prepared = (A_log_fi, dt_bias_fi)
        self._gate_cache[key] = (
            weakref.ref(A_log),
            weakref.ref(dt_bias) if dt_bias is not None else None,
            prepared,
        )
        return prepared

    @staticmethod
    def _beta_logit_to_prob(b: torch.Tensor) -> torch.Tensor:
        # Triton KDA does beta = sigmoid(b); recurrent_kda wants beta pre-sigmoided.
        # torch.sigmoid computes in fp32 internally, so a single sigmoid on the bf16
        # logit is enough (avoids an explicit fp32 upcast + downcast = 2 extra kernels).
        return torch.sigmoid(b).to(torch.bfloat16)

    @staticmethod
    def _cake_precompute_gate(
        a: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
        lower_bound: Optional[float],
        batch_size: int,
        num_heads: int,
        num_v_heads: int,
        head_k_dim: int,
    ) -> torch.Tensor:
        """Match SGLang's fused Triton gate transform before BF16 handoff.

        Triton consumes raw BF16 ``a`` but evaluates the transform in FP32.
        CAKE T=1 consumes an already transformed BF16 log-gate, so keep every
        operation in FP32 until the final unavoidable contract conversion.
        """
        if num_v_heads < num_heads or num_v_heads % num_heads != 0:
            raise ValueError(
                f"CAKE KDA requires HV to be a multiple of H, got "
                f"H={num_heads}, HV={num_v_heads}"
            )
        value_heads_per_query = num_v_heads // num_heads
        gate_input = a.reshape(batch_size, num_v_heads, head_k_dim).float()
        if dt_bias is not None:
            dt_bias_by_query = dt_bias.reshape(num_heads, head_k_dim).float()
            dt_bias_by_value = dt_bias_by_query.repeat_interleave(
                value_heads_per_query, dim=0
            )
            gate_input = gate_input + dt_bias_by_value.reshape(
                1, num_v_heads, head_k_dim
            )
        decay = (
            A_log.reshape(num_heads)
            .float()
            .repeat_interleave(value_heads_per_query)
            .reshape(1, num_v_heads, 1)
            .exp()
        )
        if lower_bound is None:
            gate = -decay * torch.nn.functional.softplus(gate_input)
        else:
            gate = float(lower_bound) * torch.sigmoid(decay * gate_input)
        return gate.to(torch.bfloat16).reshape(batch_size, 1, num_v_heads, head_k_dim)

    def _decode_cake(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        lower_bound: Optional[float],
    ) -> torch.Tensor:
        batch_size = cache_indices.shape[0]
        num_heads, head_k_dim = q.shape[2:]
        num_v_heads, head_v_dim = v.shape[2:]

        query_fi = (
            q.reshape(batch_size, 1, num_heads, head_k_dim)
            .to(torch.bfloat16)
            .contiguous()
        )
        key_fi = (
            k.reshape(batch_size, 1, num_heads, head_k_dim)
            .to(torch.bfloat16)
            .contiguous()
        )
        value_fi = (
            v.reshape(batch_size, 1, num_v_heads, head_v_dim)
            .to(torch.bfloat16)
            .contiguous()
        )
        gate_fi = self._cake_precompute_gate(
            a,
            A_log,
            dt_bias,
            lower_bound,
            batch_size,
            num_heads,
            num_v_heads,
            head_k_dim,
        )
        beta_fi = (
            torch.sigmoid(b.float())
            .to(torch.bfloat16)
            .reshape(batch_size, 1, num_v_heads)
        )

        direct_indexed_state = self._cake_direct_indexed_state_is_supported(
            ssm_states,
            cache_indices,
            batch_size=batch_size,
        )
        if direct_indexed_state:
            # Preserve the original int32 indices, including -1 CUDA-graph
            # padding. The frozen direct-state kernel masks padded rows and
            # updates active slots in the caller-owned pool in place.
            state = ssm_states
        else:
            # The current cubin addresses state slots with int32 element
            # offsets. Keep the previous dense adapter for larger envelope-
            # strided page-major/unified pools and other unsupported layouts.
            state_indices = cache_indices.clamp(min=0).to(torch.int64)
            state = ssm_states.index_select(0, state_indices).contiguous()

        output_fi, _ = self._recurrent_kda(
            q=query_fi,
            k=key_fi,
            v=value_fi,
            g=gate_fi,
            beta=beta_fi,
            scale=None,
            initial_state=state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            ssm_state_indices=cache_indices if direct_indexed_state else None,
            backend="cake",
        )
        if not direct_indexed_state:
            ssm_states.index_copy_(0, state_indices, state)
        return output_fi.view(1, batch_size, num_v_heads, head_v_dim)

    @staticmethod
    def _check_cake_fp32_state_contract(
        ssm_states: torch.Tensor,
        *,
        num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
    ) -> None:
        expected_inner = (num_v_heads, head_v_dim, head_k_dim)
        if ssm_states.dtype != torch.float32:
            raise ValueError(
                "prepared BF16 KDA export requires an FP32 state pool, got "
                f"{ssm_states.dtype}"
            )
        if ssm_states.dim() != 4 or tuple(ssm_states.shape[1:]) != expected_inner:
            raise ValueError(
                "prepared BF16 KDA export needs a [N, HV, V, K] state pool with "
                f"(HV, V, K)={expected_inner}; got {tuple(ssm_states.shape)}"
            )
        if ssm_states.stride()[1:] != (head_v_dim * head_k_dim, head_k_dim, 1):
            raise ValueError(
                "prepared BF16 KDA export needs compact per-slot state strides, "
                f"got {tuple(ssm_states.stride())}"
            )

    def _cake_prefill_uses_prepared_export(self, ssm_states: torch.Tensor) -> bool:
        policy = _cake_prefill_api_policy()
        if self.cake_prefill_precision == "tf32":
            # The TF32 export exists only as a prepared call.
            if policy == "facade":
                raise ValueError(
                    "--kda-cake-prefill-precision tf32 requires the prepared export; "
                    "SGLANG_KDA_CAKE_PREFILL_API=facade has no TF32 route"
                )
            available, _ = _get_flashinfer_prepared_bf16_prefill()
            if not available or _flashinfer_prepare_tf32_kda_prefill is None:
                raise RuntimeError(
                    "--kda-cake-prefill-precision tf32 but the installed FlashInfer "
                    "does not export prepare_tf32_kda_prefill"
                )
            if ssm_states.dtype != torch.float32:
                raise ValueError(
                    "--kda-cake-prefill-precision tf32 requires an FP32 state pool "
                    f"(--mamba-ssm-dtype float32); got {ssm_states.dtype}"
                )
            return True
        if policy == "facade":
            return False
        available, _ = _get_flashinfer_prepared_bf16_prefill()
        if policy == "prepared":
            if not available:
                raise RuntimeError(
                    "SGLANG_KDA_CAKE_PREFILL_API=prepared but the installed FlashInfer "
                    "does not export prepare_bf16_kda_prefill"
                )
            return True
        return bool(available) and ssm_states.dtype == torch.float32

    def _extend_cake_prepared_bf16(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc_fi: torch.Tensor,
        extend_seq_lens_cpu,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: Optional[float],
        needs_checkpoints: bool,
        num_state_checkpoints: int,
        state_checkpoint_cu_starts: Optional[torch.Tensor],
        state_checkpoint_every_n_tokens: int,
    ):
        """Run the exported prepared prefill; returns (output, checkpoints, schedule).

        ``--kda-cake-prefill-precision`` selects the BF16 (default) or TF32
        export; both share the prepared-call ABI.
        """
        _get_flashinfer_prepared_bf16_prefill()
        precision = self.cake_prefill_precision
        prepare_kda_prefill = (
            _flashinfer_prepare_tf32_kda_prefill
            if precision == "tf32"
            else _flashinfer_prepare_bf16_kda_prefill
        )
        if prepare_kda_prefill is None:
            raise RuntimeError(
                f"installed FlashInfer does not export the {precision.upper()} "
                "prepared KDA prefill"
            )
        # Upstream forward_extend runs one fused causal conv over the packed qkv
        # row and hands us ``split(dim=-1)`` views, i.e. [1, T, H, D] tensors whose
        # token stride is the full qkv width.  The prepared export reads such
        # views in place when each token's [H, D] payload is dense and q, k and
        # v share one token stride (a multiple of 8 elements); anything else is
        # repacked here.  Cost when the repack fires: one T*H*D BF16 copy per
        # strided operand.
        if not _prepared_qkv_layout_supported(q, k, v):
            if not q.is_contiguous():
                q = q.contiguous()
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()
        num_v_heads, head_v_dim, head_k_dim = v.shape[2], v.shape[3], q.shape[3]
        self._check_cake_fp32_state_contract(
            ssm_states,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
        )
        if extend_seq_lens_cpu is None:
            raise ValueError(
                "prepared BF16 KDA export requires host extend_seq_lens_cpu"
            )
        sequence_lengths = tuple(int(n) for n in extend_seq_lens_cpu)
        if len(sequence_lengths) != query_start_loc_fi.numel() - 1:
            raise ValueError(
                "extend_seq_lens_cpu does not match query_start_loc: "
                f"{len(sequence_lengths)} lengths for "
                f"{query_start_loc_fi.numel() - 1} sequences"
            )
        A_log_fi, dt_bias_fi = self._prep_gate_params(A_log, dt_bias)
        num_heads = q.shape[2]
        A_log_fi = A_log_fi.reshape(num_heads)
        dt_bias_fi = dt_bias_fi.reshape(num_heads, head_k_dim)
        if cache_indices.dtype != torch.int32:
            cache_indices = cache_indices.to(torch.int32)
        out = torch.empty_like(q)
        # Radix-cache resumes from these rows; the FP32 export keeps them at
        # the pool's precision and removes the BF16 -> FP32 conversion below.
        # The TF32 export always writes BF16 checkpoint rows.
        checkpoint_dtype = (
            torch.float32
            if precision == "bf16"
            and _flashinfer_kda_prefill_fp32_checkpoints[
                0 if lower_bound is None else 1
            ]
            else torch.bfloat16
        )
        state_checkpoints = (
            torch.empty(
                (num_state_checkpoints, num_v_heads, head_v_dim, head_k_dim),
                device=q.device,
                dtype=checkpoint_dtype,
            )
            if needs_checkpoints
            else None
        )
        if _CAKE_DEBUG_CHECKS:
            self._debug_check_cake_prepared_inputs(
                q,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc_fi=query_start_loc_fi,
                sequence_lengths=sequence_lengths,
                needs_checkpoints=needs_checkpoints,
                num_state_checkpoints=num_state_checkpoints,
                state_checkpoint_cu_starts=state_checkpoint_cu_starts,
                state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
            )
        # Only the BF16 export takes a plan cache; the TF32 entry point prepares
        # and closes one launch per call.
        plan_cache = self._cake_prefill_plan_cache() if precision == "bf16" else None
        prepare_kwargs = {}
        if plan_cache is not None:
            prepare_kwargs["plan_cache"] = plan_cache
        prepared = prepare_kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log_fi,
            dt_bias=dt_bias_fi,
            out=out,
            initial_state=ssm_states,
            final_state=ssm_states,
            scale=None,
            lower_bound=None if lower_bound is None else float(lower_bound),
            cu_seqlens=query_start_loc_fi,
            sequence_lengths=sequence_lengths,
            state_indices=cache_indices,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=(
                state_checkpoint_cu_starts if needs_checkpoints else None
            ),
            checkpoint_every_n_tokens=(
                int(state_checkpoint_every_n_tokens) if needs_checkpoints else 0
            ),
            beta_is_logit=True,
            **prepare_kwargs,
        )
        if plan_cache is not None:
            # Cached launches are owned by the plan cache and are rebound to
            # the next call's tensors; closing them would drop the workspace.
            prepared.launch()
            return (
                out,
                state_checkpoints,
                str(getattr(prepared, "schedule", f"prepared_{precision}")),
            )
        try:
            prepared.launch()
            schedule = str(getattr(prepared, "schedule", f"prepared_{precision}"))
        finally:
            close = getattr(prepared, "close", None)
            if callable(close):
                close()
        return out, state_checkpoints, schedule

    @staticmethod
    def _debug_check_cake_prepared_inputs(
        q: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc_fi: torch.Tensor,
        sequence_lengths,
        needs_checkpoints: bool,
        num_state_checkpoints: int,
        state_checkpoint_cu_starts: Optional[torch.Tensor],
        state_checkpoint_every_n_tokens: int,
    ) -> None:
        """Host-side bounds/consistency audit of the prepared-prefill inputs.

        Enabled with ``SGLANG_KDA_CAKE_DEBUG_CHECKS=1`` (synchronises the
        stream; diagnostics only).  Raises ``ValueError`` naming the offending
        input instead of letting the kernel fault on it.
        """
        total_tokens = int(q.shape[1])
        num_seqs = len(sequence_lengths)
        pool_rows = int(ssm_states.shape[0])
        ci = cache_indices.detach().to("cpu", dtype=torch.int64)
        qsl = query_start_loc_fi.detach().to("cpu", dtype=torch.int64)
        problems = []
        if ci.numel() != num_seqs:
            problems.append(
                f"cache_indices has {ci.numel()} rows for {num_seqs} sequences"
            )
        if ci.numel():
            lo, hi = int(ci.min()), int(ci.max())
            if lo < 0 or hi >= pool_rows:
                problems.append(
                    f"cache_indices outside [0, {pool_rows}): min={lo} max={hi}"
                )
            if int(torch.unique(ci).numel()) != ci.numel():
                problems.append("duplicate cache_indices in one batch")
        if qsl.numel() != num_seqs + 1 or int(qsl[0]) != 0:
            problems.append(
                f"query_start_loc shape/origin mismatch: {qsl.tolist()[:8]}"
            )
        else:
            diffs = (qsl[1:] - qsl[:-1]).tolist()
            if diffs != [int(n) for n in sequence_lengths]:
                problems.append(
                    f"query_start_loc diffs {diffs[:8]} != extend_seq_lens_cpu {list(sequence_lengths)[:8]}"
                )
            if int(qsl[-1]) != total_tokens:
                problems.append(
                    f"query_start_loc[-1]={int(qsl[-1])} != q tokens {total_tokens}"
                )
        if any(int(n) <= 0 for n in sequence_lengths):
            problems.append(
                f"non-positive sequence length in {list(sequence_lengths)[:8]}"
            )
        ckpt_summary = "ckpt=off"
        if needs_checkpoints:
            every = int(state_checkpoint_every_n_tokens)
            if state_checkpoint_cu_starts is None or every <= 0:
                problems.append("checkpoints requested without cu_starts/interval")
            else:
                cs = state_checkpoint_cu_starts.detach().to("cpu", dtype=torch.int64)
                counts = [(int(n) - 1) // every + 1 for n in sequence_lengths]
                expected = [0]
                for c in counts:
                    expected.append(expected[-1] + c)
                if cs.tolist() != expected:
                    problems.append(
                        f"checkpoint_cu_starts {cs.tolist()[:8]} != expected {expected[:8]}"
                    )
                if int(cs[-1]) != int(num_state_checkpoints):
                    problems.append(
                        f"num_state_checkpoints={num_state_checkpoints} != cu_starts[-1]={int(cs[-1])}"
                    )
                ckpt_summary = f"ckpt=every{every} rows={int(num_state_checkpoints)}"
        summary = (
            f"cake prepared prefill inputs: T={total_tokens} N={num_seqs} "
            f"seq_lens(min/max)={min(sequence_lengths) if num_seqs else 0}/{max(sequence_lengths) if num_seqs else 0} "
            f"cache_idx(min/max)={int(ci.min()) if ci.numel() else -1}/{int(ci.max()) if ci.numel() else -1} "
            f"pool_rows={pool_rows} {ckpt_summary} q_contig={q.is_contiguous()}"
        )
        if problems:
            raise ValueError(summary + " | PROBLEMS: " + "; ".join(problems))
        logger.info(summary)

    def _cake_prefill_plan_cache(self):
        """Per-kernel FlashInfer plan cache shared by every KDA layer.

        Layers of one forward batch share token shape, ``sequence_lengths``
        and checkpoint plan and differ only in tensor addresses (state pool,
        output, checkpoint rows), which the cache rebinds without preparing
        again.  ``SGLANG_KDA_CAKE_PREFILL_PLAN_CACHE=0`` disables it.
        """
        cache = getattr(self, "_kda_prefill_plan_cache", None)
        if cache is not None:
            return cache
        if getattr(self, "_kda_prefill_plan_cache_disabled", False):
            return None
        _get_flashinfer_prepared_bf16_prefill()
        cache_cls = _flashinfer_kda_prefill_plan_cache_cls
        capacity = int(os.environ.get("SGLANG_KDA_CAKE_PREFILL_PLAN_CACHE", "64"))
        if cache_cls is None or capacity <= 0:
            self._kda_prefill_plan_cache_disabled = True
            return None
        self._kda_prefill_plan_cache = cache_cls(capacity)
        return self._kda_prefill_plan_cache

    @staticmethod
    def _cake_prefill_is_supported(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float],
        is_spec_decode: bool,
        return_intermediate_states: bool,
        track_ssm_h_src: Optional[torch.Tensor],
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        allow_unbounded_gate: bool = False,
    ) -> bool:
        """Check the public frozen-prefill contract without a device sync."""
        needs_checkpoints = bool(
            return_intermediate_states
            and track_ssm_h_src is not None
            and track_ssm_h_src.numel() > 0
        )
        if (
            is_spec_decode
            or (return_intermediate_states and track_ssm_h_src is None)
            or (
                needs_checkpoints
                and (
                    state_checkpoint_cu_starts is None
                    or num_state_checkpoints <= 0
                    or state_checkpoint_every_n_tokens <= 0
                    or state_checkpoint_every_n_tokens % 32 != 0
                )
            )
            or not _cake_prefill_gate_bound_ok(lower_bound, allow_unbounded_gate)
        ):
            return False
        if (
            A_log is None
            or dt_bias is None
            or not q.is_cuda
            or q.ndim != 4
            or q.shape[0] != 1
        ):
            return False
        if torch.cuda.is_current_stream_capturing():
            # The public frozen-prefill facade allocates its workspace/output
            # unless the caller owns both buffers. SGLang's backend interface
            # does not expose per-layer capture buffers, so keep explicit
            # prefill CUDA graphs on the allocation-free Triton path.
            return False
        if q.shape[1] <= query_start_loc.numel() - 1:
            # Every packed sequence has T=1, which is decode rather than prefill.
            return False
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            return False
        if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
            return False
        if beta.shape != q.shape[:-1]:
            return False
        return _cuda_device_capability(q.device) in ((10, 0), (10, 3))

    @staticmethod
    def _cake_prefill_admission(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float],
        is_spec_decode: bool,
        return_intermediate_states: bool,
        track_ssm_h_src: Optional[torch.Tensor],
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        allow_unbounded_gate: bool = False,
    ) -> CakePrefillAdmission:
        """Attach stable telemetry reasons without changing admission policy."""
        supported = CakeKDAKernel._cake_prefill_is_supported(
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            lower_bound=lower_bound,
            is_spec_decode=is_spec_decode,
            return_intermediate_states=return_intermediate_states,
            track_ssm_h_src=track_ssm_h_src,
            state_checkpoint_cu_starts=state_checkpoint_cu_starts,
            num_state_checkpoints=num_state_checkpoints,
            state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
            allow_unbounded_gate=allow_unbounded_gate,
        )
        if supported:
            return CakePrefillAdmission(True, CakePrefillReason.ELIGIBLE)
        if is_spec_decode:
            return CakePrefillAdmission(False, CakePrefillReason.SPEC_DECODE)
        if return_intermediate_states and track_ssm_h_src is None:
            return CakePrefillAdmission(
                False,
                CakePrefillReason.INTERIOR_CHECKPOINT,
                "track_ssm_h_src",
            )
        if (
            return_intermediate_states
            and track_ssm_h_src.numel() > 0
            and (
                state_checkpoint_cu_starts is None
                or num_state_checkpoints <= 0
                or state_checkpoint_every_n_tokens <= 0
                or state_checkpoint_every_n_tokens % 32 != 0
            )
        ):
            return CakePrefillAdmission(
                False,
                CakePrefillReason.INTERIOR_CHECKPOINT,
                "state_checkpoint_plan",
            )
        if not _cake_prefill_gate_bound_ok(lower_bound, allow_unbounded_gate):
            return CakePrefillAdmission(
                False, CakePrefillReason.INVALID_LOWER_BOUND, "lower_bound"
            )
        if A_log is None or dt_bias is None:
            detail = "A_log" if A_log is None else "dt_bias"
            return CakePrefillAdmission(
                False, CakePrefillReason.MISSING_GATE_PARAMS, detail
            )
        if not q.is_cuda or q.ndim != 4 or q.shape[0] != 1:
            return CakePrefillAdmission(
                False, CakePrefillReason.UNSUPPORTED_Q_CONTRACT, "q"
            )
        if torch.cuda.is_current_stream_capturing():
            return CakePrefillAdmission(False, CakePrefillReason.CUDA_GRAPH_ALLOCATION)
        if q.shape[1] <= query_start_loc.numel() - 1:
            return CakePrefillAdmission(False, CakePrefillReason.T1_DECODE_SHAPE)
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            return CakePrefillAdmission(
                False, CakePrefillReason.UNSUPPORTED_HEAD_DIM, "q_or_v"
            )
        if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
            detail = next(
                name
                for name, tensor in (("k", k), ("v", v), ("g", g))
                if tensor.shape != q.shape
            )
            return CakePrefillAdmission(False, CakePrefillReason.SHAPE_MISMATCH, detail)
        if beta.shape != q.shape[:-1]:
            return CakePrefillAdmission(False, CakePrefillReason.SHAPE_MISMATCH, "beta")
        if _cuda_device_capability(q.device) not in ((10, 0), (10, 3)):
            return CakePrefillAdmission(
                False, CakePrefillReason.UNSUPPORTED_ARCH, "device_capability"
            )
        return CakePrefillAdmission(
            False, CakePrefillReason.UNSUPPORTED_CONTRACT, "unclassified"
        )

    @staticmethod
    def _extend_triton(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        # Mirror the Triton prefill backend exactly: the model hands that
        # backend ``beta.float().sigmoid()`` (fp32 probabilities), while the
        # Cake route receives the raw bf16 logits. Rounding the probabilities
        # to bf16 here made every fallback prefill differ from
        # ``--linear-attn-prefill-backend triton`` (served Kimi-Linear:
        # mean |Δ input logprob| up to 0.06, greedy divergences).
        return TritonKDAKernel().extend(q, k, v, g, beta.float().sigmoid(), **kwargs)

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        is_spec_decode: bool = False,
        return_intermediate_states: bool = False,
        cake_query_start_loc: Optional[torch.Tensor] = None,
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        layer_id: int,
        **kwargs,
    ) -> torch.Tensor:
        """Run zero-copy CAKE prefill with native state checkpoints."""
        if self._backend != "cake":
            raise NotImplementedError(
                "FlashInfer cute-dsl KDA only supports decode and target_verify"
            )

        fallback_kwargs = dict(
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            is_spec_decode=is_spec_decode,
            return_intermediate_states=return_intermediate_states,
            state_checkpoint_cu_starts=state_checkpoint_cu_starts,
            num_state_checkpoints=num_state_checkpoints,
            state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
            **kwargs,
        )
        track_ssm_h_src = kwargs.get("track_ssm_h_src")
        try:
            use_prepared_export = self._cake_prefill_uses_prepared_export(ssm_states)
            admission = self._cake_prefill_admission(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                lower_bound=lower_bound,
                is_spec_decode=is_spec_decode,
                return_intermediate_states=return_intermediate_states,
                track_ssm_h_src=track_ssm_h_src,
                state_checkpoint_cu_starts=state_checkpoint_cu_starts,
                num_state_checkpoints=num_state_checkpoints,
                state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
                # The prepared BF16 export serves the unbounded softplus gate.
                # The TF32 export ships unbounded-gate modules only for its
                # checkpoint-writing specialisations, so the TF32 route admits
                # bounded gates only; the recurrent_kda facade admits neither.
                allow_unbounded_gate=(
                    use_prepared_export and self.cake_prefill_precision == "bf16"
                ),
            )
        except Exception as exc:
            record_kda_terminal_route(
                mode="prefill",
                layer_id=layer_id,
                eligible=False,
                attempted_cake=False,
                cake_success=False,
                triton_fallback=False,
                fatal=True,
                reason=PREFILL_SELECTOR_EXCEPTION,
                detail=stable_kda_exception_detail(exc),
            )
            raise

        if not admission.eligible:
            global _tf32_unbounded_gate_warned
            if (
                lower_bound is None
                and use_prepared_export
                and self.cake_prefill_precision == "tf32"
                and not _tf32_unbounded_gate_warned
            ):
                _tf32_unbounded_gate_warned = True
                logger.warning(
                    "--kda-cake-prefill-precision tf32: the TF32 prepared export "
                    "serves bounded gates only; layers with an unbounded softplus "
                    "gate (lower_bound=None) run the Triton prefill kernel."
                )
            try:
                output = self._extend_triton(q, k, v, g, beta, **fallback_kwargs)
            except Exception as exc:
                record_kda_terminal_route(
                    mode="prefill",
                    layer_id=layer_id,
                    eligible=False,
                    attempted_cake=False,
                    cake_success=False,
                    triton_fallback=False,
                    fatal=True,
                    reason=TRITON_FALLBACK_EXCEPTION,
                    detail=stable_kda_exception_detail(exc),
                )
                raise
            record_kda_terminal_route(
                mode="prefill",
                layer_id=layer_id,
                eligible=False,
                attempted_cake=False,
                cake_success=False,
                triton_fallback=True,
                fatal=False,
                reason=admission.reason,
                detail=admission.detail,
            )
            return output

        try:
            query_start_loc_fi = (
                cake_query_start_loc
                if cake_query_start_loc is not None
                else query_start_loc.to(torch.int64)
            )
            needs_checkpoints = bool(
                return_intermediate_states
                and track_ssm_h_src is not None
                and track_ssm_h_src.numel() > 0
            )
            if use_prepared_export:
                if _KDA_PREFILL_DUMP_DIR:
                    _maybe_dump_kda_prefill_call(
                        layer_id,
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        beta=beta,
                        A_log=A_log,
                        dt_bias=dt_bias,
                        lower_bound=lower_bound,
                        cu_seqlens=query_start_loc_fi,
                        sequence_lengths=tuple(
                            int(n) for n in (kwargs.get("extend_seq_lens_cpu") or ())
                        ),
                        cache_indices=cache_indices,
                        initial_state=ssm_states[cache_indices.long()],
                        checkpoint_cu_starts=(
                            state_checkpoint_cu_starts if needs_checkpoints else None
                        ),
                        num_state_checkpoints=(
                            int(num_state_checkpoints) if needs_checkpoints else 0
                        ),
                        checkpoint_every_n_tokens=(
                            int(state_checkpoint_every_n_tokens)
                            if needs_checkpoints
                            else 0
                        ),
                    )
                output, state_checkpoints, schedule = self._extend_cake_prepared_bf16(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc_fi=query_start_loc_fi,
                    extend_seq_lens_cpu=kwargs.get("extend_seq_lens_cpu"),
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                    needs_checkpoints=needs_checkpoints,
                    num_state_checkpoints=num_state_checkpoints,
                    state_checkpoint_cu_starts=state_checkpoint_cu_starts,
                    state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
                )
                if return_intermediate_states:
                    h = (
                        (
                            state_checkpoints
                            if state_checkpoints.dtype == ssm_states.dtype
                            else state_checkpoints.to(ssm_states.dtype)
                        ).unsqueeze(0)
                        if state_checkpoints is not None
                        else ssm_states.new_empty((1, 0, *ssm_states.shape[1:]))
                    )
                    result = output, h
                else:
                    result = output
                record_kda_terminal_route(
                    mode="prefill",
                    layer_id=layer_id,
                    eligible=True,
                    attempted_cake=True,
                    cake_success=True,
                    triton_fallback=False,
                    fatal=False,
                    reason=admission.reason,
                    detail=f"prepared_{self.cake_prefill_precision}:{schedule}",
                )
                return result
            self._check_cake_state_contract(
                ssm_states,
                num_v_heads=v.shape[2],
                head_v_dim=v.shape[3],
                head_k_dim=q.shape[3],
            )
            available, recurrent_kda = _get_flashinfer_kda_prefill_kernel()
            if not available or recurrent_kda is None:
                raise RuntimeError(
                    "FlashInfer CAKE KDA prefill is not available. Install a "
                    "FlashInfer build containing the frozen recurrent prefill backend."
                )

            A_log_fi, dt_bias_fi = self._prep_gate_params(A_log, dt_bias)
            state_checkpoints = (
                ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
                if needs_checkpoints
                else None
            )
            # The facade reads q/k/v only as contiguous [1, T, H, D] tensors.
            # Serving hands us ``split(dim=-1)`` views of the fused-projection
            # conv output (token stride = the full qkv width), so repack them
            # here; one T*H*D BF16 copy per strided operand, like the prepared
            # route's repack.
            q_fi = q if q.is_contiguous() else q.contiguous()
            k_fi = k if k.is_contiguous() else k.contiguous()
            v_fi = v if v.is_contiguous() else v.contiguous()
            try:
                recurrent_result = recurrent_kda(
                    q=q_fi,
                    k=k_fi,
                    v=v_fi,
                    g=g,
                    beta=beta,
                    A_log=A_log_fi,
                    dt_bias=dt_bias_fi,
                    scale=None,
                    initial_state=ssm_states,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    lower_bound=lower_bound,
                    cu_seqlens=query_start_loc_fi,
                    ssm_state_indices=cache_indices,
                    beta_is_logit=True,
                    state_checkpoints=state_checkpoints,
                    checkpoint_cu_starts=(
                        state_checkpoint_cu_starts if needs_checkpoints else None
                    ),
                    checkpoint_every_n_tokens=(
                        state_checkpoint_every_n_tokens if needs_checkpoints else 0
                    ),
                    backend="cake",
                )
            except ValueError as exc:
                if not _is_facade_contract_error(exc):
                    raise
                # The facade declined the call before touching the state pool:
                # serve the layer with the Triton prefill kernel and declare it.
                output = self._extend_triton(q, k, v, g, beta, **fallback_kwargs)
                # The refusal is FlashInfer's host-side admission (no kernel
                # was launched), so the schema's declared-fallback shape
                # (eligible=0, attempted_cake=0) applies.
                record_kda_terminal_route(
                    mode="prefill",
                    layer_id=layer_id,
                    eligible=False,
                    attempted_cake=False,
                    cake_success=False,
                    triton_fallback=True,
                    fatal=False,
                    reason=CakePrefillReason.FACADE_CONTRACT,
                    detail=stable_kda_exception_detail(exc),
                )
                return output
            if needs_checkpoints:
                output, final_state, state_checkpoints = recurrent_result
            else:
                output, final_state = recurrent_result
            if final_state is None:
                raise RuntimeError("FlashInfer CAKE prefill did not return final state")
            if return_intermediate_states:
                h = (
                    state_checkpoints.unsqueeze(0)
                    if state_checkpoints is not None
                    else ssm_states.new_empty((1, 0, *ssm_states.shape[1:]))
                )
                result = output, h
            else:
                result = output
        except Exception as exc:
            record_kda_terminal_route(
                mode="prefill",
                layer_id=layer_id,
                eligible=True,
                attempted_cake=True,
                cake_success=False,
                triton_fallback=False,
                fatal=True,
                reason=CAKE_PREFILL_EXCEPTION,
                detail=stable_kda_exception_detail(exc),
            )
            raise
        record_kda_terminal_route(
            mode="prefill",
            layer_id=layer_id,
            eligible=True,
            attempted_cake=True,
            cake_success=True,
            triton_fallback=False,
            fatal=False,
            reason=admission.reason,
            detail=admission.detail,
        )
        return result

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

        if self._backend == "cake":
            # Route telemetry: the bounded-gate (Kimi-K3) decode reaches this
            # plain entry point instead of ``packed_decode``; record the same
            # terminal decision so served parity tests can assert the route.
            layer_id = int(kwargs.get("layer_id", -1))
            if num_v_heads != num_heads:
                # The exported kernel accepts GQA, but its recurrent state has
                # not yet matched SGLang's Triton path at the BF16 promotion
                # tolerance. Keep the production contract to H == HV until the
                # cross-implementation difference is attributed to an oracle.
                from sglang.srt.layers.attention.linear.kernels.kda_triton import (
                    TritonKDAKernel,
                )

                output = TritonKDAKernel().decode(
                    q,
                    k,
                    v,
                    a,
                    b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    lower_bound=lower_bound,
                    **kwargs,
                )
                record_kda_terminal_route(
                    mode="decode",
                    layer_id=layer_id,
                    eligible=False,
                    attempted_cake=False,
                    cake_success=False,
                    triton_fallback=True,
                    fatal=False,
                    reason=CakePackedDecodeReason.GQA_HEADS,
                    detail=f"H={num_heads},HV={num_v_heads}",
                )
                return output
            try:
                self._check_cake_state_contract(
                    ssm_states,
                    num_v_heads=num_v_heads,
                    head_v_dim=head_v_dim,
                    head_k_dim=head_k_dim,
                )
                output = self._decode_cake(
                    q,
                    k,
                    v,
                    a,
                    b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    lower_bound=lower_bound,
                )
            except Exception as exc:
                record_kda_terminal_route(
                    mode="decode",
                    layer_id=layer_id,
                    eligible=True,
                    attempted_cake=True,
                    cake_success=False,
                    triton_fallback=False,
                    fatal=True,
                    reason=CAKE_DECODE_EXCEPTION,
                    detail=stable_kda_exception_detail(exc),
                )
                raise
            record_kda_terminal_route(
                mode="decode",
                layer_id=layer_id,
                eligible=True,
                attempted_cake=True,
                cake_success=True,
                triton_fallback=False,
                fatal=False,
                reason=CakePackedDecodeReason.PLAIN_ELIGIBLE,
                detail=f"H={num_heads},HV={num_v_heads}",
            )
            return output

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
            cached_indices = self._verify_idx_cache.get(cache_key)
            ssm_state_indices = None
            if cached_indices is not None:
                row_map_ref, cached_tensor = cached_indices
                # id() reuse guard: only trust the entry built from this row map.
                if row_map_ref() is intermediate_state_indices:
                    ssm_state_indices = cached_tensor
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
                self._verify_idx_cache[cache_key] = (
                    weakref.ref(intermediate_state_indices),
                    ssm_state_indices,
                )

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


class CakeKDAKernel(FlashInferKDAKernel):
    """Named SGLang backend for FlashInfer's exported CAKE KDA kernels."""

    supports_k3_fused_decode = False
    uses_state_checkpoints = True
    uses_cake_prefill = True
    supports_packed_decode = True
    # The exported packed decode is the Kimi-K3 contract (bounded gate, 12
    # packed heads); decode() takes lower_bound on both the Cake and the
    # Triton fallback path.
    supports_bounded_gate_decode = True
    supports_cake_route_telemetry = True

    def __init__(self):
        super().__init__(backend="cake")
        available, packed_kda_decode = _get_flashinfer_packed_kda_kernel()
        self._packed_kda_decode = packed_kda_decode if available else None
        self._last_packed_decode_admission = CakePackedDecodeAdmission(
            False, CakePackedDecodeReason.KERNEL_UNAVAILABLE
        )

    @staticmethod
    def _cake_reject(reason: str, detail: str = "") -> CakePackedDecodeAdmission:
        return CakePackedDecodeAdmission(False, reason, detail)

    @staticmethod
    def _cake_row_stride_admission(
        name: str, row_stride: int, row_width: int
    ) -> Optional[CakePackedDecodeAdmission]:
        """Validate one frozen-ABI row stride with stable failure ordering."""
        if row_stride == 0:
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.ZERO_ROW_STRIDE, name
            )
        if row_stride < 0:
            # PyTorch tensors cannot currently carry negative strides.  Keep
            # this schema-v1 reason as fail-closed selector armor for foreign
            # tensor-like inputs or a future framework ABI; production CUPTI
            # evidence must label it synthetic/unreachable.
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.NEGATIVE_ROW_STRIDE, name
            )
        if row_stride < row_width:
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.OVERLAPPING_ROW_STRIDE, name
            )
        return None

    @staticmethod
    def _cake_packed_row_admission(
        tensor: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        row_width: int,
    ) -> tuple[Optional[CakePackedDecodeAdmission], Optional[int]]:
        """Admit a row view that can be formed without changing ptr/offset."""
        shape = tuple(tensor.shape)
        allowed_shapes = {
            (batch_size, row_width),
            (batch_size, 1, row_width),
            (1, batch_size, row_width),
        }
        if shape not in allowed_shapes:
            return (
                CakeKDAKernel._cake_reject(
                    CakePackedDecodeReason.UNSUPPORTED_CONTRACT, f"{name}.shape"
                ),
                None,
            )
        try:
            row_view = CakeKDAKernel._cake_packed_row_view(
                tensor, batch_size=batch_size, row_width=row_width
            )
        except RuntimeError:
            return (
                CakeKDAKernel._cake_reject(
                    CakePackedDecodeReason.UNSUPPORTED_CONTRACT, f"{name}.view"
                ),
                None,
            )
        if row_view.stride(1) != 1:
            return (
                CakeKDAKernel._cake_reject(CakePackedDecodeReason.INNER_STRIDE, name),
                None,
            )
        row_stride = row_view.stride(0)
        rejected = CakeKDAKernel._cake_row_stride_admission(name, row_stride, row_width)
        return rejected, row_stride

    @staticmethod
    def _cake_packed_row_view(
        tensor: torch.Tensor, *, batch_size: int, row_width: int
    ) -> torch.Tensor:
        """Form the exact two-dimensional ABI view without canonicalizing B=1."""
        shape = tuple(tensor.shape)
        if shape == (batch_size, row_width):
            row_stride = tensor.stride(0)
        elif shape == (batch_size, 1, row_width) and shape != (
            1,
            batch_size,
            row_width,
        ):
            row_stride = tensor.stride(0)
        elif shape == (1, batch_size, row_width) and shape != (
            batch_size,
            1,
            row_width,
        ):
            row_stride = tensor.stride(1)
        else:
            # B=1 makes the two SGLang singleton layouts shape-identical.
            # Preserve the padded row pitch rather than letting view() replace
            # it with the logical width.
            row_stride = max(tensor.stride(0), tensor.stride(1))
        return tensor.as_strided(
            (batch_size, row_width),
            (row_stride, tensor.stride(-1)),
            storage_offset=tensor.storage_offset(),
        )

    @staticmethod
    def _cake_tensor_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
        """Mirror FlashInfer's positive-stride bounding byte-range check."""
        if tensor.numel() == 0:
            start = int(tensor.data_ptr())
            return start, start
        last_element = sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(tensor.shape, tensor.stride())
        )
        start = int(tensor.data_ptr())
        return start, start + (last_element + 1) * tensor.element_size()

    @staticmethod
    def _cake_tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        lhs_begin, lhs_end = CakeKDAKernel._cake_tensor_byte_range(lhs)
        rhs_begin, rhs_end = CakeKDAKernel._cake_tensor_byte_range(rhs)
        return lhs_begin < rhs_end and rhs_begin < lhs_end

    @staticmethod
    def _cake_packed_decode_cuda_device_reason(
        mixed_qkv: torch.Tensor, tensors: tuple[torch.Tensor, ...]
    ) -> Optional[CakePackedDecodeAdmission]:
        """Require one CUDA device; split out so CPU/mock canaries stay pure."""
        if not mixed_qkv.is_cuda:
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "mixed_qkv.device"
            )
        device = mixed_qkv.device
        if any(not tensor.is_cuda or tensor.device != device for tensor in tensors):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "device_mismatch"
            )
        return None

    @staticmethod
    def _cake_cache_index_admission(
        values, *, batch_size: int, state_slots: int
    ) -> Optional[CakePackedDecodeAdmission]:
        """Validate a host-visible mirror of FlashInfer's index contract."""
        if torch.is_tensor(values):
            if values.device.type != "cpu" or values.numel() != batch_size:
                return CakeKDAKernel._cake_reject(
                    CakePackedDecodeReason.UNSUPPORTED_CONTRACT,
                    "cache_indices_cpu",
                )
            values = values.reshape(-1).tolist()
        else:
            values = list(values)
            if len(values) != batch_size:
                return CakeKDAKernel._cake_reject(
                    CakePackedDecodeReason.UNSUPPORTED_CONTRACT,
                    "cache_indices_cpu",
                )
        active = []
        for value in values:
            value = int(value)
            if value < -1 or value == 0 or value >= state_slots:
                return CakeKDAKernel._cake_reject(
                    CakePackedDecodeReason.CACHE_INDEX_OOB, "cache_indices"
                )
            if value >= 0:
                active.append(value)
        if len(active) != len(set(active)):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.CACHE_INDEX_DUPLICATE, "cache_indices"
            )
        return None

    @staticmethod
    def _cake_cache_index_source_admission(
        cache_indices,
        *,
        cache_indices_cpu,
        cache_index_contract,
        batch_size: int,
        state_slots: int,
    ) -> Optional[CakePackedDecodeAdmission]:
        """Prove CUDA values by allocator provenance or validate a host mirror."""
        host_indices = cache_indices_cpu
        if host_indices is None and cache_indices.device.type == "cpu":
            host_indices = cache_indices
        if host_indices is not None:
            return CakeKDAKernel._cake_cache_index_admission(
                host_indices,
                batch_size=batch_size,
                state_slots=state_slots,
            )
        if not isinstance(
            cache_index_contract, MambaStateIndexContract
        ) or not cache_index_contract.matches(
            cache_indices,
            batch_size=batch_size,
            state_slots=state_slots,
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.CACHE_INDEX_UNVERIFIED,
                "cache_indices",
            )
        return None

    @staticmethod
    def _cake_packed_decode_admission(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float],
        cache_indices_cpu=None,
        cache_index_contract=None,
    ) -> CakePackedDecodeAdmission:
        """Return the exact frozen-ABI selector result without tensor copies.

        CUDA values are admitted only with the frozen provenance attached by
        SGLang's Mamba allocator/metadata path, avoiding a device-to-host
        synchronization. CPU/mock canaries and callers with an existing host
        mirror are checked value-by-value instead.
        """
        batch_size = mixed_qkv.shape[0] if mixed_qkv.ndim == 2 else -1
        state_inner_size = (
            _CAKE_PACKED_NUM_HEADS * _CAKE_PACKED_HEAD_DIM * _CAKE_PACKED_HEAD_DIM
        )
        if not isinstance(scale, (int, float)) or not isinstance(
            lower_bound, (int, float)
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "scalars"
            )
        if not math.isclose(
            scale, _CAKE_PACKED_SCALE, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            lower_bound,
            _CAKE_PACKED_LOWER_BOUND,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "scalars"
            )
        if (
            batch_size <= 0
            or batch_size > 65535
            or tuple(mixed_qkv.shape) != (batch_size, _CAKE_PACKED_QKV_WIDTH)
            or mixed_qkv.dtype != torch.bfloat16
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "mixed_qkv"
            )

        rejected, _ = CakeKDAKernel._cake_packed_row_admission(
            mixed_qkv,
            name="mixed_qkv",
            batch_size=batch_size,
            row_width=_CAKE_PACKED_QKV_WIDTH,
        )
        if rejected is not None:
            return rejected
        if a.dtype != torch.bfloat16:
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "raw_gate.dtype"
            )
        rejected, _ = CakeKDAKernel._cake_packed_row_admission(
            a,
            name="raw_gate",
            batch_size=batch_size,
            row_width=_CAKE_PACKED_GATE_WIDTH,
        )
        if rejected is not None:
            return rejected
        if b.dtype != torch.bfloat16:
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "raw_beta.dtype"
            )
        rejected, _ = CakeKDAKernel._cake_packed_row_admission(
            b,
            name="raw_beta",
            batch_size=batch_size,
            row_width=_CAKE_PACKED_NUM_HEADS,
        )
        if rejected is not None:
            return rejected

        if (
            A_log.dtype != torch.float32
            or dt_bias.dtype != torch.float32
            or not A_log.is_contiguous()
            or not dt_bias.is_contiguous()
            or A_log.numel() != _CAKE_PACKED_NUM_HEADS
            or dt_bias.numel() != _CAKE_PACKED_GATE_WIDTH
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "gate_params"
            )
        if (
            ssm_states.dtype != torch.bfloat16
            or ssm_states.ndim != 4
            or tuple(ssm_states.shape[1:])
            != (
                _CAKE_PACKED_NUM_HEADS,
                _CAKE_PACKED_HEAD_DIM,
                _CAKE_PACKED_HEAD_DIM,
            )
            or ssm_states.shape[0] <= 0
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "state"
            )
        if ssm_states.stride()[1:] != (
            _CAKE_PACKED_HEAD_DIM * _CAKE_PACKED_HEAD_DIM,
            _CAKE_PACKED_HEAD_DIM,
            1,
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.INNER_STRIDE, "state"
            )
        rejected = CakeKDAKernel._cake_row_stride_admission(
            "state", ssm_states.stride(0), state_inner_size
        )
        if rejected is not None:
            return rejected
        if (
            cache_indices.dtype != torch.int32
            or not cache_indices.is_contiguous()
            or tuple(cache_indices.shape) != (batch_size,)
            or num_v_heads != _CAKE_PACKED_NUM_HEADS
            or head_v_dim != _CAKE_PACKED_HEAD_DIM
        ):
            return CakeKDAKernel._cake_reject(
                CakePackedDecodeReason.UNSUPPORTED_CONTRACT, "indices_or_heads"
            )

        rejected = CakeKDAKernel._cake_packed_decode_cuda_device_reason(
            mixed_qkv, (a, b, A_log, dt_bias, ssm_states, cache_indices)
        )
        if rejected is not None:
            return rejected

        for tensor_name, tensor in (
            ("mixed_qkv", mixed_qkv),
            ("raw_gate", a),
            ("raw_beta", b),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
            ("state_indices", cache_indices),
        ):
            if CakeKDAKernel._cake_tensors_overlap(ssm_states, tensor):
                return CakeKDAKernel._cake_reject(
                    CakePackedDecodeReason.STORAGE_ALIAS, f"state:{tensor_name}"
                )

        rejected = CakeKDAKernel._cake_cache_index_source_admission(
            cache_indices,
            cache_indices_cpu=cache_indices_cpu,
            cache_index_contract=cache_index_contract,
            batch_size=batch_size,
            state_slots=ssm_states.shape[0],
        )
        if rejected is not None:
            return rejected
        return CakePackedDecodeAdmission(True, CakePackedDecodeReason.ELIGIBLE)

    @staticmethod
    def _cake_packed_decode_is_supported(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float],
        cache_indices_cpu=None,
        cache_index_contract=None,
    ) -> bool:
        """Compatibility boolean for callers that only need eligibility."""
        return CakeKDAKernel._cake_packed_decode_admission(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            lower_bound=lower_bound,
            cache_indices_cpu=cache_indices_cpu,
            cache_index_contract=cache_index_contract,
        ).eligible

    @staticmethod
    def _packed_decode_triton(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float],
        **kwargs,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.linear.kernels.kda_triton import (
            TritonKDAKernel,
        )

        return TritonKDAKernel().packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            lower_bound=lower_bound,
            **kwargs,
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float] = None,
        cache_indices_cpu=None,
        cache_index_contract=None,
        layer_id: int,
        **kwargs,
    ) -> torch.Tensor:
        """Run CAKE's exact packed decode or explicitly retain Triton semantics."""
        global _cake_packed_decode_route_logged
        try:
            replay_requested = any(
                kwargs.get(name) is not None
                for name in (
                    "replayssm_d",
                    "replayssm_k",
                    "replayssm_g",
                    "replayssm_write_pos",
                    "replayssm_force_flush",
                )
            )
            if replay_requested:
                admission = CakePackedDecodeAdmission(
                    False, CakePackedDecodeReason.REPLAYSSM_REQUESTED
                )
            elif self._packed_kda_decode is None:
                admission = CakePackedDecodeAdmission(
                    False, CakePackedDecodeReason.KERNEL_UNAVAILABLE
                )
            else:
                admission = self._cake_packed_decode_admission(
                    mixed_qkv,
                    a,
                    b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=scale,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    num_v_heads=num_v_heads,
                    head_v_dim=head_v_dim,
                    lower_bound=lower_bound,
                    cache_indices_cpu=cache_indices_cpu,
                    cache_index_contract=cache_index_contract,
                )
        except Exception as exc:
            record_kda_terminal_route(
                mode="decode",
                layer_id=layer_id,
                eligible=False,
                attempted_cake=False,
                cake_success=False,
                triton_fallback=False,
                fatal=True,
                reason=PACKED_SELECTOR_EXCEPTION,
                detail=stable_kda_exception_detail(exc),
            )
            raise
        self._last_packed_decode_admission = admission
        if not admission.eligible:
            try:
                output = self._packed_decode_triton(
                    mixed_qkv,
                    a,
                    b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=scale,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    num_v_heads=num_v_heads,
                    head_v_dim=head_v_dim,
                    lower_bound=lower_bound,
                    **kwargs,
                )
            except Exception as exc:
                record_kda_terminal_route(
                    mode="decode",
                    layer_id=layer_id,
                    eligible=False,
                    attempted_cake=False,
                    cake_success=False,
                    triton_fallback=False,
                    fatal=True,
                    reason=TRITON_FALLBACK_EXCEPTION,
                    detail=stable_kda_exception_detail(exc),
                )
                raise
            record_kda_terminal_route(
                mode="decode",
                layer_id=layer_id,
                eligible=False,
                attempted_cake=False,
                cake_success=False,
                triton_fallback=True,
                fatal=False,
                reason=admission.reason,
                detail=admission.detail,
            )
            return output

        try:
            batch_size = mixed_qkv.shape[0]
            # These are metadata-only views: positive disjoint row strides are
            # forwarded to FlashInfer unchanged, including the production beta
            # stride (144, 1). Neither call changes data_ptr/storage_offset.
            raw_gate = self._cake_packed_row_view(
                a, batch_size=batch_size, row_width=_CAKE_PACKED_GATE_WIDTH
            )
            raw_beta = self._cake_packed_row_view(
                b, batch_size=batch_size, row_width=_CAKE_PACKED_NUM_HEADS
            )
            output = mixed_qkv.new_empty(
                batch_size, 1, _CAKE_PACKED_NUM_HEADS, _CAKE_PACKED_HEAD_DIM
            )
            if not _cake_packed_decode_route_logged:
                logger.info(
                    "FlashInfer CAKE packed KDA decode route active: "
                    "H=12, D=128, direct indexed BF16 state"
                )
                _cake_packed_decode_route_logged = True
            self._packed_kda_decode(
                mixed_qkv=mixed_qkv,
                raw_gate=raw_gate,
                raw_beta=raw_beta,
                A_log=A_log.view(_CAKE_PACKED_NUM_HEADS),
                dt_bias=dt_bias.view(_CAKE_PACKED_GATE_WIDTH),
                state=ssm_states,
                state_indices=cache_indices,
                output=output,
            )
            result = output.transpose(0, 1)
        except Exception as exc:
            record_kda_terminal_route(
                mode="decode",
                layer_id=layer_id,
                eligible=True,
                attempted_cake=True,
                cake_success=False,
                triton_fallback=False,
                fatal=True,
                reason=CAKE_PACKED_EXCEPTION,
                detail=stable_kda_exception_detail(exc),
            )
            raise
        record_kda_terminal_route(
            mode="decode",
            layer_id=layer_id,
            eligible=True,
            attempted_cake=True,
            cake_success=True,
            triton_fallback=False,
            fatal=False,
            reason=admission.reason,
            detail=admission.detail,
            copy_count=0,
            copy_count_source="static_zero_copy_row_view",
        )
        return result
