"""Aiter-based kernels for GDN (Gated Delta Network) linear attention.

Uses aiter (AMD AI Tensor Engine for ROCm) kernels when SGLANG_USE_AITER=1 on HIP.
Requires aiter package with GDN/FLA kernel support.
"""

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_aiter_fused_sigmoid_gating_delta_rule_update = None
_aiter_chunk_gated_delta_rule = None
_aiter_available = False


def _load_aiter_gdn_kernels():
    """Lazy import aiter GDN kernels. Returns (decode_available, extend_available)."""
    global _aiter_fused_sigmoid_gating_delta_rule_update
    global _aiter_chunk_gated_delta_rule
    global _aiter_available

    if _aiter_available or not is_hip():
        return _aiter_fused_sigmoid_gating_delta_rule_update is not None, (
            _aiter_chunk_gated_delta_rule is not None
        )
    try:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
        from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule
        _aiter_fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update
        _aiter_chunk_gated_delta_rule = chunk_gated_delta_rule
    except ImportError:
        raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")

    _aiter_available = True
    return _aiter_fused_sigmoid_gating_delta_rule_update is not None, (
        _aiter_chunk_gated_delta_rule is not None
    )


class AiterGDNKernel(LinearAttnKernelBase):
    """Aiter-based kernel for GDN (Gated Delta Network) linear attention.

    Uses aiter kernels for decode and extend on HIP.
    Use --linear-attn-*-backend triton for extend/target_verify if aiter lacks support.
    """

    def __init__(self):
        decode_ok, extend_ok = _load_aiter_gdn_kernels()
        if not decode_ok:
            raise RuntimeError(
                "Aiter GDN decode kernel is not available. "
                "Requires HIP, SGLANG_USE_AITER=1, and aiter package with "
                "fused_sigmoid_gating_delta_rule_update support."
            )
        self._decode_fn = _aiter_fused_sigmoid_gating_delta_rule_update
        self._extend_fn = _aiter_chunk_gated_delta_rule
        self._extend_available = extend_ok

        if not extend_ok:
            logger.warning(
                "Aiter GDN extend kernel not available. "
                "Use --linear-attn-prefill-backend triton for prefill."
            )
        logger.info(
            f"Aiter GDN: decode=aiter, extend={'aiter' if extend_ok else 'unavailable'}"
        )

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
        # Ensure contiguous tensors and int32 indices to avoid ROCm assertion/crash.
        cu_seqlens = (
            query_start_loc.to(torch.int32).contiguous()
            if query_start_loc is not None
            else None
        )
        idx = (
            cache_indices.to(torch.int32).contiguous()
            if cache_indices is not None
            else None
        )
        return self._decode_fn(
            A_log=A_log.contiguous(),
            dt_bias=dt_bias.contiguous(),
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            initial_state_source=ssm_states.contiguous(),
            initial_state_indices=idx,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

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
        if not self._extend_available or self._extend_fn is None:
            raise NotImplementedError(
                "Aiter GDN extend kernel is not available. "
                "Use --linear-attn-prefill-backend triton for prefill."
            )
        
        # aiter chunk_gated_delta_rule: no head_first, no initial_state_indices.
        # For continuous batching, pass initial_state=ssm_states[cache_indices].
        # Ensure contiguous tensors and int32 cu_seqlens to avoid ROCm assertion/crash.
        recurrent_state = (
            ssm_states[cache_indices].contiguous()
            if cache_indices is not None
            else ssm_states.contiguous()
        )
        cu_seqlens = (
            query_start_loc.to(torch.int32).contiguous()
            if query_start_loc is not None
            else None
        )
        o, final_state = self._extend_fn(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            initial_state=recurrent_state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # aiter may return [N, K, V] but sglang expects [N, H, K, V]. Reshape to match.
        num_seqs = cache_indices.shape[0] if cache_indices is not None else 1
        target_shape = (num_seqs,) + tuple(ssm_states.shape[1:])
        if final_state.shape != target_shape:
            if len(final_state.shape) == 3:
                # [N, K, V] -> [N, H, K, V] by broadcasting head dim
                n, k, v = final_state.shape
                h = ssm_states.shape[1]
                final_state = final_state.unsqueeze(1).expand(n, h, k, v)
            # Slice to match cache_indices if aiter returned more sequences
            if final_state.shape[0] != num_seqs:
                final_state = final_state[:num_seqs]
        # gdn_backend expects (core_attn_out, last_recurrent_state, h)
        return o, final_state, final_state

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
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Aiter GDN does not support target_verify (speculative decoding). "
            "Use --linear-attn-decode-backend triton and --linear-attn-prefill-backend triton "
            "for speculative decoding."
        )
