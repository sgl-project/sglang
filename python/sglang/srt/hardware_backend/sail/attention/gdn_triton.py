"""SAIL GDN Triton-backend slots served by the PLA operators.

GDN and KDA share the PLA recurrent operators (no safe gate, no fused conv
step), so the wrappers in ``sail.kernels.kda`` serve both. What PLA does not
accept -- the ReplaySSM ring write, a non-fp32 state pool -- keeps the
community Triton kernel.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.hardware_backend.sail.kernels import kda as sail_kda
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


class SailTritonGDNKernel(TritonGDNKernel):
    """Triton GDN kernel slots served by PLA."""

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
        # The ReplaySSM buffered decode writes per-layer rings; that layout is
        # Triton-only.
        if not sail_kda.pla_enabled() or kwargs.get("replayssm_d") is not None:
            return super().packed_decode(
                mixed_qkv,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                **kwargs,
            )
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)
        out, _ = sail_kda.packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
        )
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
        if not sail_kda.recurrent_update_supported(False, ssm_states, None):
            return super().decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )
        return sail_kda.recurrent_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=sail_kda.PLA_SOFT_BETA,
            softplus_threshold=sail_kda.PLA_SOFT_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
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
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if not sail_kda.recurrent_update_supported(
            bool(kwargs.get("cache_ring", False)),
            ssm_states,
            intermediate_states_buffer,
        ):
            return super().target_verify(
                A_log,
                dt_bias,
                q,
                k,
                v,
                a,
                b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token,
                **kwargs,
            )
        return sail_kda.recurrent_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=sail_kda.PLA_SOFT_BETA,
            softplus_threshold=sail_kda.PLA_SOFT_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
