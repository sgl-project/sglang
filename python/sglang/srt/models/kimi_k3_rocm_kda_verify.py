"""Kimi-K3 KDA target verify on AITER's fused FlyDSL kernel (ROCm gfx950).

The KDA backend calls this through its ``KDAFusedVerify`` hook under
``--linear-attn-verify-backend aiter``; this module unpacks the Kimi-K3 layer
state the tensor-only kernel adapter takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention import kda_fused_verify_aiter_hip

if TYPE_CHECKING:
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention


class KimiK3HipFusedVerify:
    """Dense chain verify of one Kimi-K3 KDA layer whose f_b the model defers:
    f_b, convolution, gate, recurrence and gated RMSNorm in one kernel."""

    def __init__(
        self,
        *,
        f_b_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        a_log: torch.Tensor,
        num_draft_tokens: int,
    ):
        self.f_b_weight = f_b_weight
        self.norm_weight = norm_weight
        self.norm_eps = norm_eps
        self.a_log = a_log
        # The warmed draft-token count is the only one the kernel may run.
        self.num_draft_tokens = num_draft_tokens

    def forward(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        intermediate_state_cache: Optional[torch.Tensor],
        intermediate_conv_window_cache: torch.Tensor,
        num_draft_tokens: int,
        replay_ring: Optional[tuple[torch.Tensor, ...]],
        unsupported: Optional[str],
    ) -> Optional[torch.Tensor]:
        if unsupported is not None:
            kda_fused_verify_aiter_hip.warn_rejected_once(unsupported)
            return None
        tokens = mixed_qkv.shape[0]
        batch = tokens // num_draft_tokens
        kwargs = dict(
            f_a=a[0],
            mixed_qkv=mixed_qkv,
            raw_beta=b,
            output_gate=layer._k3_onorm_gate[:tokens].view(
                -1, layer.num_v_heads, layer.head_v_dim
            ),
            conv_state=conv_states.transpose(-1, -2),
            state=ssm_states,
            cache_indices=cache_indices[:batch],
            intermediate_indices=intermediate_state_indices[:batch],
            intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
            intermediate_states=(
                intermediate_state_cache if replay_ring is None else None
            ),
            replay_ring=replay_ring,
            num_draft_tokens=num_draft_tokens,
        )
        if not kda_fused_verify_aiter_hip.covered(
            **kwargs, armed_draft_tokens=self.num_draft_tokens
        ):
            kda_fused_verify_aiter_hip.warn_rejected_once(
                f"tokens={tokens}, draft_tokens={num_draft_tokens} "
                f"(armed {self.num_draft_tokens}), "
                f"conv_state={conv_states.dtype}, "
                f"cache_indices={cache_indices.dtype}, "
                f"inter_indices={intermediate_state_indices.dtype}, "
                f"state={tuple(ssm_states.shape)}/{ssm_states.dtype}"
            )
            return None
        core_attn_out = kda_fused_verify_aiter_hip.run(
            **kwargs,
            f_b_weight=self.f_b_weight,
            conv_weight=layer.conv_weights,
            A_log=self.a_log,
            dt_bias=layer.dt_bias,
            lower_bound=float(layer.lower_bound),
            norm_weight=self.norm_weight,
            norm_eps=self.norm_eps,
        )
        layer._k3_onorm_consumed = True
        return core_attn_out

    def unfused_gate(self, a: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

        gate = kimi_k3_tiny_gemm(a[0], self.f_b_weight.flatten(0, 1))
        return gate.unflatten(-1, self.f_b_weight.shape[:2]).unsqueeze(0)
