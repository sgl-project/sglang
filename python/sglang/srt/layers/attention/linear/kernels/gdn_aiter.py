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
        from aiter import fused_sigmoid_gating_delta_rule_update as _aiter_decode

        _aiter_fused_sigmoid_gating_delta_rule_update = _aiter_decode
    except ImportError:
        try:
            from aiter.fla import fused_sigmoid_gating_delta_rule_update as _aiter_decode

            _aiter_fused_sigmoid_gating_delta_rule_update = _aiter_decode
        except ImportError:
            pass

    try:
        from aiter import chunk_gated_delta_rule as _aiter_extend

        _aiter_chunk_gated_delta_rule = _aiter_extend
    except ImportError:
        try:
            from aiter.fla import chunk_gated_delta_rule as _aiter_extend

            _aiter_chunk_gated_delta_rule = _aiter_extend
        except ImportError:
            pass

    _aiter_available = True
    return _aiter_fused_sigmoid_gating_delta_rule_update is not None, (
        _aiter_chunk_gated_delta_rule is not None
    )


class AiterGDNKernel(LinearAttnKernelBase):
    """Aiter-based kernel for GDN (Gated Delta Network) linear attention.

    Uses aiter kernels for decode and extend when available on HIP.
    Falls back to Triton for extend/target_verify if aiter only provides decode.
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
            from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
                TritonGDNKernel,
            )

            self._triton_fallback = TritonGDNKernel()
            logger.info(
                "Aiter GDN: decode=aiter, extend=triton (aiter extend not available)"
            )
        else:
            self._triton_fallback = None
            logger.info("Aiter GDN: decode=aiter, extend=aiter")

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
        return self._decode_fn(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
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
        if self._extend_available and self._extend_fn is not None:
            from sglang.srt.utils import is_cpu, is_npu

            recurrent_state = ssm_states
            recurrent_state_indices_args = {"initial_state_indices": cache_indices}
            if is_npu() or is_cpu():
                recurrent_state = ssm_states[cache_indices]
                recurrent_state_indices_args = {}
            # chunk_gated_delta_rule returns (o, last_recurrent_state, h)
            return self._extend_fn(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
                **recurrent_state_indices_args,
            )
        return self._triton_fallback.extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

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
        # target_verify requires Triton (intermediate_states_buffer, etc.)
        if self._triton_fallback is not None:
            return self._triton_fallback.target_verify(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )
        raise NotImplementedError(
            "AiterGDNKernel target_verify requires Triton fallback. "
            "Aiter extend kernel was available but target_verify is not supported."
        )
