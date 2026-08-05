"""FlashInfer KDA wrapper.

Wraps ``flashinfer.kda_decode.recurrent_kda`` for decode/verify and the public
``flashinfer.kda.recurrent_kda`` facade for CAKE prefill (SM100 / Blackwell).

Contract with the Triton KDA reference:
  - raw per-K gate ``a`` is activated in-kernel as
    ``-exp(A_log) * softplus(a + dt_bias)``;
  - beta ``b`` is a logit, so this wrapper passes ``sigmoid(b)``;
  - q/k are L2-normalized in-kernel;
  - state layout is ``[N, HV, V, K]`` for committed and speculative state.

The optional ``cake`` mode adapts SGLang's raw-gate/indexed-state calls to the
exported CAKE contracts. Decode accepts precomputed BF16 gate/beta tensors;
prefill consumes raw gate/beta logits. Both paths gather active state rows,
launch CAKE, and scatter the updated rows back into SGLang's state pool.
"""

import inspect
import logging
import math
import os
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

        # Slot 0 is SGLang's reserved CUDA-graph padding row. Preserve existing
        # zeros and map any legacy negative marker there as well. Multiple pad
        # rows may race on slot 0 during index_copy_, but that row is never
        # allocated to a request and its value is intentionally disposable.
        state_indices = cache_indices.clamp(min=0).to(torch.int64)
        state_batch = ssm_states.index_select(0, state_indices).contiguous()

        output_fi, _ = self._recurrent_kda(
            q=query_fi,
            k=key_fi,
            v=value_fi,
            g=gate_fi,
            beta=beta_fi,
            scale=None,
            initial_state=state_batch,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            backend="cake",
        )
        ssm_states.index_copy_(0, state_indices, state_batch)
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
    ) -> bool:
        """Check the public frozen-prefill contract without a device sync."""
        if (
            is_spec_decode
            or return_intermediate_states
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

    def __init__(self):
        super().__init__(backend="cake")
