"""DP-attention helpers for the Nemotron-H model."""

import torch
from torch import nn

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    apply_flashinfer_allreduce_fusion,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

ATTN_LAYERS = (MAMBA, ATTENTION)


def is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def get_real_num_tokens(
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


def pad_to_original_num_tokens(
    output: torch.Tensor, original_num_tokens: int
) -> torch.Tensor:
    if output.shape[0] == original_num_tokens:
        return output
    padded = output.new_empty((original_num_tokens, *output.shape[1:]))
    padded[: output.shape[0]] = output
    return padded


def _build_layer_scatter_modes(is_sparse: bool = False) -> LayerScatterModes:
    scatter_mlp = is_sparse and not get_moe_a2a_backend().is_none()
    mlp_mode = ScatterMode.SCATTERED if scatter_mlp else ScatterMode.FULL
    middle_residual_mode = (
        ScatterMode.SCATTERED if scatter_mlp else ScatterMode.TP_ATTN_FULL
    )
    return LayerScatterModes(
        layer_input_mode=ScatterMode.TP_ATTN_FULL,
        attn_mode=ScatterMode.TP_ATTN_FULL,
        mlp_mode=mlp_mode,
        middle_residual_mode=middle_residual_mode,
        layer_output_mode=ScatterMode.TP_ATTN_FULL,
    )


def make_layer_communicator(
    layer_norm: RMSNorm,
    *,
    for_attn: bool,
    allow_reduce_scatter: bool = False,
    is_sparse: bool = False,
    is_last_layer: bool = False,
) -> LayerCommunicator:
    return LayerCommunicator(
        layer_scatter_modes=_build_layer_scatter_modes(is_sparse),
        input_layernorm=layer_norm if for_attn else nn.Identity(),
        post_attention_layernorm=nn.Identity() if for_attn else layer_norm,
        force_layernorm_before_dp_gather=True,
        allow_reduce_scatter=allow_reduce_scatter,
        is_last_layer=is_last_layer,
    )


def input_norm_maybe_fuse_allreduce(
    norm: RMSNorm,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual is not None and getattr(
        hidden_states, "_sglang_needs_allreduce_fusion", False
    ):
        if apply_flashinfer_allreduce_fusion(hidden_states.shape[0]) and hasattr(
            norm, "forward_with_allreduce_fusion"
        ):
            return norm.forward_with_allreduce_fusion(
                hidden_states, residual, use_attn_tp_group=False
            )
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return norm(hidden_states, residual)

    if residual is None:
        residual = hidden_states
        hidden_states = norm(hidden_states)
        return hidden_states, residual
    return norm(hidden_states, residual)
