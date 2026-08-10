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
path. Prefill consumes raw gate/beta logits and retains its gather/scatter
adapter.
"""

import inspect
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

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


class CakePackedDecodeReason:
    """Stable CAKE packed-decode admission reason codes."""

    ELIGIBLE = "eligible"
    KERNEL_UNAVAILABLE = "kernel_unavailable"
    REPLAYSSM_REQUESTED = "replayssm_requested"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    INNER_STRIDE = "inner_stride"
    ZERO_ROW_STRIDE = "zero_row_stride"
    NEGATIVE_ROW_STRIDE = "negative_row_stride"
    OVERLAPPING_ROW_STRIDE = "overlapping_row_stride"
    STORAGE_ALIAS = "storage_alias"
    CACHE_INDEX_OOB = "cache_index_oob"
    CACHE_INDEX_DUPLICATE = "cache_index_duplicate"


@dataclass(frozen=True)
class CakePackedDecodeAdmission:
    """One terminal selector result for a packed CAKE decode attempt."""

    eligible: bool
    reason: str
    detail: str = ""


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
        logger.info("Using FlashInfer KDA kernel backend=%s", backend)

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
    ) -> bool:
        """Check the public frozen-prefill contract without a device sync."""
        needs_interior_snapshot = (
            return_intermediate_states
            and track_ssm_h_src is not None
            and track_ssm_h_src.numel() > 0
        )
        if (
            is_spec_decode
            or needs_interior_snapshot
            or lower_bound is None
            or not math.isfinite(float(lower_bound))
            or float(lower_bound) >= 0.0
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
        return torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))

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

        return TritonKDAKernel().extend(q, k, v, g, beta, **kwargs)

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
        **kwargs,
    ) -> torch.Tensor:
        """Run ordinary CAKE prefill, preserving Triton-only state semantics."""
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
            **kwargs,
        )
        track_ssm_h_src = kwargs.get("track_ssm_h_src")
        if not self._cake_prefill_is_supported(
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
        ):
            return self._extend_triton(q, k, v, g, beta, **fallback_kwargs)

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

        q_fi = q.to(torch.bfloat16).contiguous()
        k_fi = k.to(torch.bfloat16).contiguous()
        v_fi = v.to(torch.bfloat16).contiguous()
        g_fi = g.to(torch.bfloat16).contiguous()
        # SGLang pre-activates beta for ordinary extend. The frozen kernel fuses
        # sigmoid, so reconstruct a logit in FP32 and round only at the handoff.
        beta_fi = torch.logit(beta.float().clamp(1e-7, 1.0 - 1e-7)).to(torch.bfloat16)
        beta_fi = beta_fi.contiguous()
        A_log_fi, dt_bias_fi = self._prep_gate_params(A_log, dt_bias)
        query_start_loc_fi = query_start_loc.to(torch.int64).contiguous()

        state_indices = cache_indices.clamp(min=0).to(torch.int64)
        state_batch = ssm_states.index_select(0, state_indices).contiguous()
        output, final_state = recurrent_kda(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=g_fi,
            beta=beta_fi,
            A_log=A_log_fi,
            dt_bias=dt_bias_fi,
            scale=None,
            initial_state=state_batch,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            cu_seqlens=query_start_loc_fi,
            beta_is_logit=True,
        )
        if final_state is None:
            raise RuntimeError("FlashInfer CAKE prefill did not return final state")
        ssm_states.index_copy_(0, state_indices, final_state)
        if return_intermediate_states:
            # Boundary-aligned track: _track_mamba_state_extend consumes the
            # final state already scattered above. With no interior h source,
            # match NvidiaKDAKernel's empty stand-in while preserving the
            # ordinary CAKE output/final-state path bit for bit.
            h_empty = q.new_empty(
                (1, 0) + tuple(ssm_states.shape[1:]), dtype=torch.float32
            )
            return output, h_empty
        return output

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
            if num_v_heads != num_heads:
                # The exported kernel accepts GQA, but its recurrent state has
                # not yet matched SGLang's Triton path at the BF16 promotion
                # tolerance. Keep the production contract to H == HV until the
                # cross-implementation difference is attributed to an oracle.
                from sglang.srt.layers.attention.linear.kernels.kda_triton import (
                    TritonKDAKernel,
                )

                return TritonKDAKernel().decode(
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
            self._check_cake_state_contract(
                ssm_states,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                head_k_dim=head_k_dim,
            )
            return self._decode_cake(
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

        base_rows = intermediate_state_indices[:batch_size]
        cache_key = (
            id(intermediate_state_indices),
            batch_size,
            draft_token_num,
            scratch_steps,
        )
        ssm_state_indices = self._verify_idx_cache.get(cache_key)
        if ssm_state_indices is None:
            # The fast seed copy below assumes row n in scratch belongs to request n.
            expected = torch.arange(
                batch_size, device=base_rows.device, dtype=base_rows.dtype
            )
            if not torch.equal(base_rows, expected):
                raise RuntimeError(
                    "FlashInfer KDA verify requires an identity intermediate row-map "
                    "(verify_intermediate_state_indices must be arange)."
                )
            step = torch.arange(draft_token_num, device=q.device, dtype=torch.int32)
            ssm_state_indices = (
                base_rows.to(torch.int32)[:, None] * scratch_steps + step[None, :]
            ).contiguous()  # [N, T]
            self._verify_idx_cache[cache_key] = ssm_state_indices

        # Seed step 0 from committed state, then recurrent_kda overwrites it with
        # token-0 post-state. Padded graph rows clamp to slot 0; their output is ignored.
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
        )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)


class CakeKDAKernel(FlashInferKDAKernel):
    """Named SGLang backend for FlashInfer's exported CAKE KDA kernels."""

    supports_k3_fused_decode = False
    supports_packed_decode = True

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
            if value < -1 or value >= state_slots:
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
    ) -> CakePackedDecodeAdmission:
        """Return the exact frozen-ABI selector result without tensor copies.

        FlashInfer intentionally treats CUDA index values as a caller contract
        to avoid a device-to-host synchronization. SGLang applies the value
        checks whenever the tensor is already on CPU (CPU/mock canaries), or
        when the caller supplies an existing CPU mirror.
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

        host_indices = cache_indices_cpu
        if host_indices is None and cache_indices.device.type == "cpu":
            host_indices = cache_indices
        if host_indices is not None:
            rejected = CakeKDAKernel._cake_cache_index_admission(
                host_indices,
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
        **kwargs,
    ) -> torch.Tensor:
        """Run CAKE's exact packed decode or explicitly retain Triton semantics."""
        global _cake_packed_decode_route_logged
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
            )
        self._last_packed_decode_admission = admission
        if not admission.eligible:
            return self._packed_decode_triton(
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
        return output.transpose(0, 1)
