"""MATE-FlashInfer-based kernels for GDN (Gated Delta Network) linear attention.

MP31 (Pinghu): full support — decode, prefill, MTP.  State dtype: fp32.

Requires mate >= 2.1.0 (MP31).
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
_flashinfer_gated_delta_rule_decode = None


def _get_flashinfer_gdn_kernels():
    """Lazy import for FlashInfer GDN prefill and decode kernels."""
    global _flashinfer_gdn_available
    global _flashinfer_chunk_gated_delta_rule
    global _flashinfer_gated_delta_rule_decode
    if _flashinfer_gdn_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from mate.gdn_decode import gated_delta_rule_decode
            from mate.gdn_prefill import chunk_gated_delta_rule

            _flashinfer_chunk_gated_delta_rule = chunk_gated_delta_rule
            _flashinfer_gated_delta_rule_decode = gated_delta_rule_decode
            _flashinfer_gdn_available = torch.cuda.get_device_capability()[0] >= 3
            if _flashinfer_gdn_available:
                logger.info("FlashInfer GDN kernels loaded successfully")
        except (AttributeError, ImportError, RuntimeError) as e:
            logger.warning(f"FlashInfer GDN kernels not available: {e}")
            _flashinfer_gdn_available = False
            _flashinfer_chunk_gated_delta_rule = None
            _flashinfer_gated_delta_rule_decode = None
    return (
        _flashinfer_gdn_available,
        _flashinfer_chunk_gated_delta_rule,
        _flashinfer_gated_delta_rule_decode,
    )


# ---------------------------------------------------------------------------
# Kernel implementation
# ---------------------------------------------------------------------------


class MusaFlashInferGDNKernel(LinearAttnKernelBase):
    """FlashInfer kernel for GDN with K-last SSM state layout.

    MP31 (Pinghu): decode uses gather/scatter; prefill and MTP verify supported.

    Requires mate >= 2.1.0 (MP31).
    """

    def __init__(self):
        (
            available,
            self._prefill_fn,
            self._decode_fn,
        ) = _get_flashinfer_gdn_kernels()

        if not available:
            raise RuntimeError(
                "FlashInfer GDN kernels are not available. "
                "Requires MP31 and MATE-FlashInfer with GDN kernel support."
            )
        if self._decode_fn is None:
            raise RuntimeError("FlashInfer GDN decode kernel is unavailable.")

        sm_major, sm_minor = torch.cuda.get_device_capability()
        if (sm_major, sm_minor) != (3, 1):
            raise NotImplementedError(
                "FlashInfer GDN decode kernel is only supported on MP31."
            )

        if self._prefill_fn is None:
            raise RuntimeError("FlashInfer GDN prefill kernel is unavailable.")

        self.supports_target_verify = True

        logger.info("Using FlashInfer GDN kernels")

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

        query_fi = q.view(batch_size, 1, num_heads, head_k_dim)
        key_fi = k.view(batch_size, 1, num_heads, head_k_dim)
        value_fi = v.view(batch_size, 1, num_v_heads, head_v_dim)
        a_fi = a.view(batch_size, 1, num_v_heads)
        b_fi = b.view(batch_size, 1, num_v_heads)

        output_fi, _ = self._decode_fn(
            q=query_fi,
            k=key_fi,
            v=value_fi,
            state=ssm_states,
            A_log=A_log.detach().float(),
            a=a_fi,
            dt_bias=dt_bias.detach().float(),
            b=b_fi,
            state_indices=cache_indices,
            use_qk_l2norm=True,
            disable_state_update=False,
        )

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
        # MP31: chunked prefill using FlashInfer GDN prefill kernel.
        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # Remap negative padding indices to sentinel slot
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.int64)

        initial_state_fi = ssm_states[ssm_cache_indices]

        output_fi, output_state_fi = self._prefill_fn(
            q=q.squeeze(0),
            k=k.squeeze(0),
            v=v.squeeze(0),
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            scale=None,
            initial_state=initial_state_fi,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            is_log_space=True,
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
        # h=None since FlashInfer doesn't provide intermediate states.
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
        # MP31: MTP verify using FlashInfer gated_delta_rule_mtp kernel.
        if retrieve_parent_token is not None:
            raise RuntimeError(
                "FlashInfer GDN verify kernel only supports topk=1 "
                "(retrieve_parent_token must be None)."
            )

        seq_len = q.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size

        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        query_mtp = q.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_mtp = k.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_mtp = v.view(batch_size, draft_token_num, num_v_heads, head_v_dim)

        if a is None or b is None or A_log is None or dt_bias is None:
            raise RuntimeError(
                "FlashInfer GDN MTP kernel requires a, b, A_log, dt_bias."
            )

        a_mtp = a.view(batch_size, draft_token_num, num_v_heads)
        b_mtp = b.view(batch_size, draft_token_num, num_v_heads)

        output_fi, _ = self._decode_fn(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            state=ssm_states,
            state_indices=cache_indices,
            A_log=A_log.detach().float(),
            a=a_mtp,
            dt_bias=dt_bias.detach().float(),
            b=b_mtp,
            scale=None,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer,
            disable_state_update=True,
            use_qk_l2norm=True,
        )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
