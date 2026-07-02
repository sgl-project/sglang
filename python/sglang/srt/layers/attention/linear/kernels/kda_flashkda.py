from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

# FlashKDA chunk size. Sequences shorter than this fall back to Triton.
_FLASHKDA_CHUNK_SIZE = 64

# FlashKDA's max sequence length, Batches whose longest sequence exceeds this
# fall back to Triton for the whole batch.
_FLASHKDA_MAX_SEQ_LEN = 2048


def _load_flash_kda():
    """Import the optional ``flash_kda`` CUTLASS module."""
    try:
        import flash_kda
    except ImportError as e:
        raise ImportError(
            "The 'flashkda' KDA prefill backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git"
        ) from e
    return flash_kda


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor.
    """
    from sglang.srt.layers.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA (MoonshotAI) fully-fused CUTLASS KDA prefill backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    It is prefill-only, bf16, K == V == 128, HV == H (no GVA), and requires the
    safe (bounded) gate (``lower_bound`` set). The non-safe path and sequences
    outside [chunk_size, max_seq_len] fall back to Triton ``chunk_kda``.
    Requires an SM90+ GPU with the ``flash_kda`` package.
    """

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
        raise NotImplementedError("FlashKDAKernel only supports prefill (extend)")

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
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if self._should_fall_back(
            lower_bound, is_spec_decode, query_start_loc, extend_seq_lens_cpu
        ):
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
            )

        return self._flashkda_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
    ) -> bool:
        """Whether to use the Triton chunk_kda path instead of the fused kernel."""
        # Safe-gate only: the fused kernel does not support the unbounded gate
        # (-exp(A_log)*softplus); those models leave lower_bound unset.
        if lower_bound is None:
            return True
        # FlashKDA writes the committed recurrent state back in place, so it is
        # unsafe for speculative verify / draft-extend forwards (which must stay
        # rollback-able). Those reach this backend through forward_extend, so
        # gate them here rather than relying on the decode/target_verify stubs.
        if is_spec_decode:
            return True
        # Short sequences (< chunk size) and long sequences (> the crossover
        # where Triton's chunked prefill wins) are faster on Triton. Read the
        # per-request lengths from the CPU-side extend_seq_lens to avoid a
        # GPU->CPU sync on every layer; derive from query_start_loc (one sync)
        # only if they are unavailable.
        if extend_seq_lens_cpu is not None:
            if torch.is_tensor(extend_seq_lens_cpu):
                lo = int(extend_seq_lens_cpu.min())
                hi = int(extend_seq_lens_cpu.max())
            else:
                lo = min(extend_seq_lens_cpu)
                hi = max(extend_seq_lens_cpu)
        else:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            lo_t, hi_t = torch.aminmax(seq_lens)
            lo, hi = int(lo_t), int(hi_t)
        return lo < _FLASHKDA_CHUNK_SIZE or hi > _FLASHKDA_MAX_SEQ_LEN

    def _flashkda_extend(
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
    ) -> torch.Tensor:
        flash_kda = _load_flash_kda()

        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # KimiDeltaAttention.forward already applies sigmoid to beta on the
        # prefill path, but flash_kda expects beta LOGITS (it sigmoids
        # internally). Invert back so the kernel recovers the intended value:
        # sigmoid(logit(p)) == p. (triton/cuLA consume the post-sigmoid beta.)
        beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7)).to(torch.bfloat16)
        beta = beta.contiguous()

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float().contiguous()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float().contiguous()

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = ssm_states[cache_indices].contiguous()

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)

        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )

        ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        return out_buf
