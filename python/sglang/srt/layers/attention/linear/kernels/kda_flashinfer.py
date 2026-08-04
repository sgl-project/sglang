"""FlashInfer KDA decode/verify wrapper.

Wraps ``flashinfer.kda_decode.recurrent_kda`` (SM100 / Blackwell). FlashInfer has
no KDA prefill kernel, so ``extend`` stays on Triton / CuTe DSL.

Contract with the Triton KDA reference:
  - raw per-K gate ``a`` is activated in-kernel as
    ``-exp(A_log) * softplus(a + dt_bias)``;
  - beta ``b`` is a logit, so this wrapper passes ``sigmoid(b)``;
  - q/k are L2-normalized in-kernel;
  - state layout is ``[N, HV, V, K]`` for committed and speculative state.
"""

import logging
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
        logger.info("Using FlashInfer KDA kernel")

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
        **kwargs,
    ) -> torch.Tensor:
        batch_size = cache_indices.shape[0]
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

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

        # Softplus gate (lower_bound=None) to match the Triton KDA decode path;
        # in-place state update into the committed pool (no rollback for decode).
        # query_start_loc is the decode cu_seqlens (one token per request).
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
            lower_bound=None,
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
            lower_bound=None,
            cu_seqlens=query_start_loc.to(torch.int32),
            ssm_state_indices=ssm_state_indices,
            num_spec_tokens=num_spec_tokens,
        )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)

    # ---- extend (prefill): not provided by FlashInfer ----

    def extend(self, *args, **kwargs):
        raise NotImplementedError(
            "FlashInferKDAKernel has no prefill kernel; keep prefill on Triton / CuTe DSL."
        )
