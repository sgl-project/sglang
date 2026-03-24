# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aiter-based causal_conv1d for ROCm/HIP.

Uses aiter.causal_conv1d_update when SGLANG_USE_AITER=1 on HIP.
For prefill (causal_conv1d_fn), falls back to Triton.
"""

from typing import List, Optional, Union

import torch

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn as causal_conv1d_fn_triton,
    PAD_SLOT_ID,
)

# Lazy import for decode path
_aiter_causal_conv1d_update = None


def _load_aiter():
    global _aiter_causal_conv1d_update
    if _aiter_causal_conv1d_update is None:
        import aiter

        _aiter_causal_conv1d_update = aiter.causal_conv1d_update
    return _aiter_causal_conv1d_update


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
) -> torch.Tensor:
    """Aiter wrapper for causal_conv1d_update. Raises for speculative decoding."""
    if (
        intermediate_conv_window is not None
        or retrieve_next_token is not None
        or retrieve_next_sibling is not None
        or retrieve_parent_token is not None
        or num_accepted_tokens is not None
    ):
        raise NotImplementedError(
            "Aiter causal_conv1d_update does not support speculative decoding. "
            "Use Triton causal_conv1d for target_verify."
        )

    aiter_fn = _load_aiter()

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    use_silu = activation in ["silu", "swish"]

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    out = torch.zeros_like(x)
    bias_t = (
        bias.to(dtype=x.dtype) if bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    weight_t = weight.to(dtype=x.dtype) if weight.dtype != x.dtype else weight
    cache_seqlens_t = (
        cache_seqlens.to(torch.int32)
        if cache_seqlens is not None
        else torch.empty(0, dtype=torch.int32, device=x.device)
    )
    conv_state_indices_t = (
        conv_state_indices.to(torch.int32).contiguous()
        if conv_state_indices is not None
        else torch.empty(0, dtype=torch.int32, device=x.device)
    )

    aiter_fn(
        x,
        conv_state,
        weight_t,
        bias_t,
        out,
        use_silu,
        cache_seqlens_t,
        conv_state_indices_t,
        pad_slot_id,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    **kwargs,
) -> torch.Tensor:
    """Prefill path: use Triton (aiter has no prefill equivalent)."""
    return causal_conv1d_fn_triton(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
        validate_data=validate_data,
        **kwargs,
    )
