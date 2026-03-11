"""FlashInfer-based kernels for GDN (Gated Delta Network) linear attention.

Provides K-last SSM layout support using FlashInfer CUTLASS kernels (SM90+).
The K-last layout stores SSM states as [pool, HV, V, K] instead of V-last
[pool, HV, K, V], enabling more efficient memory access patterns for decode.

Requires ``flashinfer`` with GDN kernel support to be installed, e.g.
``pip install -e ".[cutlass]"`` from the FlashInfer repo.

NOTE: FlashInfer >= 0.6.4 includes a fix (PR#2509) that caches
cudaGetDeviceProperties in the GDN prefill JIT launcher, eliminating ~80ms
of CPU overhead per prefill. Upgrading from 0.6.3 to 0.6.4 recovers ~50%
prefill throughput regression observed with stock FlashInfer.
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import for FlashInfer GDN kernels
# ---------------------------------------------------------------------------
_flashinfer_gdn_available: Optional[bool] = None
_flashinfer_chunk_gated_delta_rule = None
_flashinfer_gated_delta_rule_mtp = None
_flashinfer_gated_delta_rule_decode = None


def _get_flashinfer_gdn_kernels():
    """Lazy import for FlashInfer GDN prefill, decode and verify (MTP) kernels.

    Returns (available, prefill_fn, mtp_fn, decode_fn).
    """
    global _flashinfer_gdn_available, _flashinfer_chunk_gated_delta_rule, _flashinfer_gated_delta_rule_mtp, _flashinfer_gated_delta_rule_decode
    if _flashinfer_gdn_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from flashinfer.gdn_decode import (
                gated_delta_rule_decode_pretranspose,
                gated_delta_rule_mtp,
            )
            from flashinfer.gdn_prefill import chunk_gated_delta_rule

            _flashinfer_chunk_gated_delta_rule = chunk_gated_delta_rule
            _flashinfer_gated_delta_rule_mtp = gated_delta_rule_mtp
            # Use pretranspose (K-last / V-major) decode kernel to match
            # the K-last pool layout [pool, HV, V, K]
            _flashinfer_gated_delta_rule_decode = gated_delta_rule_decode_pretranspose
            # SM90+ required for FlashInfer GDN CUTLASS kernels
            _flashinfer_gdn_available = (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
            )
            if _flashinfer_gdn_available:
                logger.info(
                    "FlashInfer GDN kernels (prefill + decode + MTP) loaded successfully"
                )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"FlashInfer GDN kernels not available: {e}")
            _flashinfer_gdn_available = False
            _flashinfer_gated_delta_rule_decode = None
    return (
        _flashinfer_gdn_available,
        _flashinfer_chunk_gated_delta_rule,
        _flashinfer_gated_delta_rule_mtp,
        _flashinfer_gated_delta_rule_decode,
    )


# ---------------------------------------------------------------------------
# Kernel implementation
# ---------------------------------------------------------------------------


class FlashInferGDNKernel(LinearAttnKernelBase):
    """FlashInfer CUTLASS kernel for GDN with K-last SSM state layout.

    Supports decode (pooled pretranspose), extend (chunked prefill) and
    target_verify (MTP).  Requires SM90+ and FlashInfer with GDN support.
    """

    def __init__(self):
        (
            available,
            self._prefill_fn,
            self._mtp_fn,
            self._decode_fn,
        ) = _get_flashinfer_gdn_kernels()

        if not available:
            raise RuntimeError(
                "FlashInfer GDN kernels are not available. "
                "Requires SM90+ and FlashInfer with GDN kernel support."
            )
        if self._prefill_fn is None:
            raise RuntimeError("FlashInfer GDN prefill kernel is unavailable.")
        if self._mtp_fn is None:
            raise RuntimeError("FlashInfer GDN MTP (verify) kernel is unavailable.")
        if self._decode_fn is None:
            raise RuntimeError("FlashInfer GDN decode kernel is unavailable.")

        logger.info(
            "K-last mode: Using FlashInfer GDN prefill, decode and MTP (verify) kernels"
        )

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
        """K-last decode using FlashInfer pretranspose kernel (stock, no pool indexing).

        TODO: Once FlashInfer PR#2521 is merged and released, switch back
        to pool-indexed decode (passing state_indices directly) to avoid
        the gather/scatter overhead (~7-9% decode regression).
        https://github.com/flashinfer-ai/flashinfer/pull/2521
        """
        batch_size = cache_indices.shape[0]
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        query_fi = q.view(batch_size, 1, num_heads, head_k_dim)
        key_fi = k.view(batch_size, 1, num_heads, head_k_dim)
        value_fi = v.view(batch_size, 1, num_v_heads, head_v_dim)
        a_fi = a.view(batch_size, 1, num_v_heads)
        b_fi = b.view(batch_size, 1, num_v_heads)

        # Gather states from pool
        state_batch = ssm_states[cache_indices]

        output_fi, new_state = self._decode_fn(
            q=query_fi,
            k=key_fi,
            v=value_fi,
            state=state_batch,
            A_log=A_log.detach(),
            a=a_fi,
            dt_bias=dt_bias.detach(),
            b=b_fi,
            scale=None,
            output=None,
            use_qk_l2norm=True,
        )

        # Scatter updated states back to pool
        ssm_states[cache_indices] = new_state

        # [bs, 1, HV, V] -> [1, bs, HV, V]
        return output_fi.view(1, batch_size, num_v_heads, head_v_dim)

    # ---- extend (prefill) ----

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
        **kwargs,
    ) -> tuple:
        """K-last chunked prefill using FlashInfer GDN prefill kernel.

        The FlashInfer kernel natively supports K-last state layout [N, H, V, K].
        q and k are L2-normalized before calling the kernel (the kernel is called
        with ``use_qk_l2norm_in_kernel=False``).
        """
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        # q, k: [1, seq, H, K] -> [seq, H, K]
        # v:    [1, seq, HV, V] -> [seq, HV, V]
        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        q_fi = l2norm_fwd(q[0].contiguous())
        k_fi = l2norm_fwd(k[0].contiguous())
        v_fi = v[0].contiguous()

        # g (alpha) and beta: [1, seq, HV] -> [seq, HV], float32 for FlashInfer
        alpha_fi = torch.exp(g[0].to(torch.float32))
        beta_fi = beta[0].to(torch.float32)

        cu_seqlens_fi = query_start_loc.to(torch.int64)

        # Remap negative padding indices to sentinel slot
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.int64)

        # FlashInfer requires float32 initial state, K-last layout [B, HV, V, K]
        initial_state_fi = ssm_states[ssm_cache_indices].to(torch.float32)

        output_fi, output_state_fi = self._prefill_fn(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha_fi,
            beta=beta_fi,
            scale=None,
            initial_state=initial_state_fi,
            output_final_state=True,
            cu_seqlens=cu_seqlens_fi,
            use_qk_l2norm_in_kernel=False,
        )

        # Write back state to pool
        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            output_state_fi.to(ssm_states.dtype),
        )

        # Output: [seq, HV, V] -> [1, seq, HV, V]
        core_attn_out = output_fi.view(1, total_seq_len, num_v_heads, head_v_dim)

        # Return (output, last_recurrent_state, h) to match Triton kernel interface.
        # h=None since FlashInfer doesn't provide intermediate states
        # (prefix caching for K-last prefill is not supported).
        return core_attn_out, None, None

    # ---- target_verify (MTP) ----

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
        """K-last MTP verify using FlashInfer GDN MTP kernel.

        Only supports topk=1 (retrieve_parent_token must be None).
        """
        if retrieve_parent_token is not None:
            raise RuntimeError(
                "FlashInfer GDN verify kernel only supports topk=1 "
                "(retrieve_parent_token must be None)."
            )

        # Recover batch_size and draft_token_num from g shape
        # g: [1, seq_len, HV] where seq_len = batch_size * draft_token_num
        seq_len = q.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size

        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # Reshape [1, seq, H, D] -> [B, T, H, D] for FlashInfer MTP
        query_mtp = q.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_mtp = k.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_mtp = v.view(batch_size, draft_token_num, num_v_heads, head_v_dim)

        # a, b from g/beta: [1, seq, HV] -> [B, T, HV]
        if a is None or b is None or A_log is None or dt_bias is None:
            raise RuntimeError(
                "FlashInfer GDN MTP kernel requires a_raw, b_raw, A_log, "
                "dt_bias to be passed via kwargs."
            )

        a_mtp = a.view(batch_size, draft_token_num, num_v_heads)
        b_mtp = b.view(batch_size, draft_token_num, num_v_heads)

        output_fi, _ = self._mtp_fn(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            A_log=A_log.detach(),
            a=a_mtp,
            dt_bias=dt_bias.detach(),
            b=b_mtp,
            scale=None,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer,
            disable_state_update=True,
            use_qk_l2norm=True,
        )

        # [B, T, HV, V] -> [1, seq_len, HV, V]
        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
