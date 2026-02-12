import torch

from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class CuteDSLGDNKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for GDN decode (CUDA only)."""

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
        return cutedsl_fused_sigmoid_gating_delta_rule_update(
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

    def extend(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")
