"""KDA chain spec-verify on gfx950 in one HIP kernel (kda_verify_gfx950)."""

from typing import Optional

import torch

from sglang.kernels.ops.attention.linear.kda_verify_gfx950 import (
    kda_verify_gfx950,
    supports_kda_verify_gfx950,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.utils import print_info_once


def kda_chain_verify_gfx950(
    layer: RadixLinearAttention,
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    draft_token_num: int,
    conv_states: torch.Tensor,
    conv_state_indices: torch.Tensor,
    ssm_states: torch.Tensor,
    intermediate_state_cache: Optional[torch.Tensor],
    intermediate_conv_window_cache: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_next_token: Optional[torch.Tensor],
    retrieve_next_sibling: Optional[torch.Tensor],
    retrieve_parent_token: Optional[torch.Tensor],
    ring_kwargs: dict,
) -> Optional[torch.Tensor]:
    """Return chain verify output, or None for an unsupported shape, dtype or layout.

    ``ring_kwargs`` selects ReplaySSM; without it, write per-step snapshots.
    """
    replayssm_rawv = ring_kwargs.get("replayssm_rawv")
    if (
        retrieve_next_token is not None
        or retrieve_next_sibling is not None
        or retrieve_parent_token is not None
        or layer.bias is not None
        or not isinstance(layer.conv_weights, torch.Tensor)
        # the kernel reads these as fp32 / int32 without checking
        or layer.A_log.dtype != torch.float32
        or layer.dt_bias.dtype != torch.float32
        or conv_state_indices.dtype != torch.int32
        or intermediate_state_indices.dtype != torch.int32
        or (
            replayssm_rawv is not None
            and not KDAAttnBackend._replayssm_ring_ok(
                layer=layer,
                draft_token_num=draft_token_num,
                mixed_qkv=mixed_qkv,
                replayssm_rawv=replayssm_rawv,
                replayssm_rawk=ring_kwargs.get("replayssm_rawk"),
                replayssm_g=ring_kwargs.get("replayssm_g"),
                replayssm_beta=ring_kwargs.get("replayssm_beta"),
            )
        )
        or not supports_kda_verify_gfx950(
            num_q_heads=layer.num_q_heads,
            num_v_heads=layer.num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            T=draft_token_num,
            mixed_qkv=mixed_qkv,
            conv_weight=layer.conv_weights,
            conv_state=conv_states,
            a=a,
            b=b,
            ssm_states=ssm_states,
            replayssm_rawv=replayssm_rawv,
            intermediate_states_buffer=intermediate_state_cache,
            intermediate_conv_window=intermediate_conv_window_cache,
        )
    ):
        return None
    print_info_once(f"KDA chain verify: gfx950 HIP kernel, {draft_token_num} tokens")
    return kda_verify_gfx950(
        mixed_qkv=mixed_qkv,
        conv_weight=layer.conv_weights,
        conv_state=conv_states.transpose(-1, -2),
        intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
        intermediate_state_indices=intermediate_state_indices,
        a=a,
        b=b,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        ssm_states=ssm_states,
        cache_indices=conv_state_indices,
        intermediate_states_buffer=intermediate_state_cache,
        scale=layer.head_k_dim**-0.5,
        T=draft_token_num,
        num_q_heads=layer.num_q_heads,
        num_v_heads=layer.num_v_heads,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
        lower_bound=layer.lower_bound,
        **ring_kwargs,
    )
