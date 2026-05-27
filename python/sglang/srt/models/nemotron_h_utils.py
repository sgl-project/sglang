"""DP-attention helpers for the Nemotron-H model.

Kept out of the modeling file (``nemotron_h.py``) to keep that file focused on
module definitions. These helpers deal with the token padding that DP-attention
introduces for collective alignment, and with building the per-layer
``LayerCommunicator``.
"""

from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MOE
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

ATTN_LAYERS = (MAMBA, ATTENTION)


def _is_sparse_layer(layer_type: str) -> bool:
    return layer_type == MOE


def _is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def _get_real_num_tokens(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> int:
    """Number of real (non DP-padding) rows in ``hidden_states``."""
    real_tokens = hidden_states.shape[0]
    num_token_non_padded_cpu = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if num_token_non_padded_cpu is not None:
        real_tokens = min(real_tokens, int(num_token_non_padded_cpu))
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_mixed()
        and forward_batch.extend_seq_lens_cpu is not None
    ):
        real_tokens = min(real_tokens, int(sum(forward_batch.extend_seq_lens_cpu)))
    return real_tokens


def _zero_dp_padding_rows(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor] = None,
):
    """Zero the DP-padding rows of ``hidden_states`` (and ``residual``) in place.

    Always returns the ``(hidden_states, residual)`` pair (``residual`` may be
    None); callers that only need the hidden states can ignore the second value.
    """
    real_tokens = _get_real_num_tokens(hidden_states, forward_batch)
    if real_tokens < hidden_states.shape[0]:
        hidden_states[real_tokens:].zero_()
        if residual is not None and residual.shape[0] == hidden_states.shape[0]:
            residual[real_tokens:].zero_()
    return hidden_states, residual


def _zero_dp_global_padding_rows(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Zero the per-DP-group padding rows after a global (full-TP) gather."""
    actual_tokens = getattr(forward_batch, "original_global_num_tokens_cpu", None)
    padded_tokens = getattr(forward_batch, "global_num_tokens_cpu", None)
    if actual_tokens is None or padded_tokens is None:
        return _zero_dp_padding_rows(hidden_states, forward_batch)[0]

    offset = 0
    for actual, padded in zip(actual_tokens, padded_tokens):
        actual = min(int(actual), int(padded))
        padded = int(padded)
        if offset + actual < hidden_states.shape[0]:
            end = min(offset + padded, hidden_states.shape[0])
            hidden_states[offset + actual : end].zero_()
        offset += padded
        if offset >= hidden_states.shape[0]:
            break
    return hidden_states


def _pad_to_original_num_tokens(
    output: torch.Tensor, original_num_tokens: int
) -> torch.Tensor:
    if output.shape[0] == original_num_tokens:
        return output
    padded = output.new_zeros((original_num_tokens, *output.shape[1:]))
    padded[: output.shape[0]] = output
    return padded


def _build_layer_scatter_modes() -> LayerScatterModes:
    # Nemotron-H uses attention/mamba as local-DP, attn-TP-partial producers.
    # The following MLP/MoE layer is the only place that gathers/reduces that
    # partial into the full TP layout before scattering back to the local DP slice.
    return LayerScatterModes(
        layer_input_mode=ScatterMode.TP_ATTN_FULL,
        attn_mode=ScatterMode.TP_ATTN_FULL,
        mlp_mode=ScatterMode.FULL,
        middle_residual_mode=ScatterMode.TP_ATTN_FULL,
        layer_output_mode=ScatterMode.TP_ATTN_FULL,
    )


def _make_layer_communicator(
    layer_norm: RMSNorm, *, for_attn: bool
) -> LayerCommunicator:
    return LayerCommunicator(
        layer_scatter_modes=_build_layer_scatter_modes(),
        input_layernorm=layer_norm if for_attn else nn.Identity(),
        post_attention_layernorm=nn.Identity() if for_attn else layer_norm,
        # Keep residual+RMSNorm at the same per-token boundary as normal TP
        # execution. The DP gather should move already-normalized tokens into the
        # global MLP/MoE TP layout, not change the residual-add boundary.
        force_layernorm_after_dp_gather=False,
    )
