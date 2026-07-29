from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu

if not is_cpu():
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.kda import chunk_kda


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()

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
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: feed the conv-1d output ``mixed_qkv``
        straight into a single fused Triton kernel that does Q/K/V extraction,
        gate/beta computation, l2-norm, and the recurrent state update.

        Returns output tensor of shape [1, B, HV, V] to match the existing
        decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # a may come in as [B, HV, K] (or [B, 1, HV*K]); b may come in as
        # [B, 1, HV]. Flatten both to the 2D shapes the kernel expects.
        if a.dim() != 2:
            a = a.reshape(B, -1)
        if b.dim() != 2:
            b = b.reshape(B, -1)
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        # [B, 1, HV, V] -> [1, B, HV, V] view to match existing decode layout.
        return out.transpose(0, 1)

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
        return fused_sigmoid_gating_delta_rule_update(
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
            is_kda=True,
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
