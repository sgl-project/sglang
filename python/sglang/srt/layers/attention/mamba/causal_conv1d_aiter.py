# Copyright (c) 2024, SGLang Team.
# Wrapper for aiter.causal_conv1d_update, following aiter/op_tests/test_causal_conv1d.py.
# Aiter does not support speculative decoding; use --linear-attn-*-backend triton for that.

from typing import Optional

import torch

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    **kwargs,
) -> torch.Tensor:
    """
    Wrapper for aiter.causal_conv1d_update.
    Aiter does not support speculative decoding; use --linear-attn-*-backend triton.
    """
    if (
        intermediate_conv_window is not None
        or retrieve_next_token is not None
        or retrieve_next_sibling is not None
        or num_accepted_tokens is not None
    ):
        raise NotImplementedError(
            "Aiter causal_conv1d_update does not support speculative decoding "
            "(intermediate_conv_window, retrieve_next_token, etc.). "
            "Use --linear-attn-decode-backend triton and --linear-attn-prefill-backend triton "
            "for speculative decoding."
        )

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )

    import aiter

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    out = torch.empty_like(x)

    use_silu = activation in ["silu", "swish"]

    aiter.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        out,
        use_silu,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out

causal_conv1d_fn = causal_conv1d_update