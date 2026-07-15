import logging
from typing import Optional

import torch

from sglang.jit_kernel.cutedsl_kda import cutedsl_fused_sigmoid_gating_kda_update
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)


def _is_blackwell() -> bool:
    """True iff running on SM100+ (Blackwell), where the chunk prefill kernels run."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


class CuteDSLKDAKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for KDA.

    Decode: ``cutedsl_fused_sigmoid_gating_kda_update`` (SM90+).
    Extend (prefill): SM100 chunk pipeline ``chunk_kda_cutedsl`` (SM100+ only,
    ``head_k_dim`` must be 128). On SM90 the prefill path is unsupported; callers
    query :attr:`supports_prefill` and fall back to Triton.
    """

    def __init__(self):
        self.supports_prefill = _is_blackwell()
        self._extend_fn: Optional[callable] = None
        self._l2norm_fn: Optional[callable] = None

    def _ensure_extend_loaded(self, head_k_dim: int) -> None:
        if self._extend_fn is not None:
            return
        if not self.supports_prefill:
            major = (
                torch.cuda.get_device_capability()[0]
                if torch.cuda.is_available()
                else -1
            )
            raise RuntimeError(
                f"CuTe DSL KDA prefill requires SM100+ (Blackwell); got SM{major}."
            )
        if head_k_dim != 128:
            raise RuntimeError(
                f"CuTe DSL KDA prefill requires head_k_dim=128, got {head_k_dim}."
            )
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
        from sglang.kernels.ops.attention.linear.kda_blackwell import (
            chunk_kda_cutedsl,
        )

        self._extend_fn = chunk_kda_cutedsl
        self._l2norm_fn = l2norm_fwd
        logger.info("Using CuTe DSL KDA prefill (Blackwell)")

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
        return cutedsl_fused_sigmoid_gating_kda_update(
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
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        head_k_dim = k.shape[-1]
        self._ensure_extend_loaded(head_k_dim)

        # [1, T, HV, D] -> [T, HV, D]; L2-norm Q/K outside the kernel.
        q_n = self._l2norm_fn(q[0].contiguous()).to(torch.bfloat16)
        k_n = self._l2norm_fn(k[0].contiguous()).to(torch.bfloat16)
        v_in = v[0].contiguous().to(torch.bfloat16)
        # Trim g/beta to q's real token count: the [:real_num_tokens] slice in
        # unified_linear_attention_with_output narrows their batch dim (a no-op),
        # not tokens, so padded rows survive and break the kernel's shape check.
        num_tokens = q_n.shape[0]
        g_in = g[0][:num_tokens]  # raw forget gate; activated inside chunk_kda_cutedsl
        beta_in = beta[0][:num_tokens].to(torch.float32)
        cu_seqlens = query_start_loc.to(torch.int32)

        # Pool gather: remap padding (-1) to the last (sentinel) slot. State is
        # [slots, HV, V, K] == cutedsl [V,K] layout, no transpose needed.
        ssm_cache_indices = torch.where(
            cache_indices >= 0, cache_indices, ssm_states.shape[0] - 1
        ).to(torch.long)
        initial_state = ssm_states[ssm_cache_indices].contiguous()

        o, final_state = self._extend_fn(
            q_n,
            k_n,
            v_in,
            g_in,
            beta_in,
            initial_state,
            cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )

        ssm_states.index_copy_(0, ssm_cache_indices, final_state.to(ssm_states.dtype))
        # Match chunk_kda's output layout [1, T, HV, V].
        return o.unsqueeze(0)

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLKDAKernel does not support target_verify")
