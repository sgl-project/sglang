"""Prefix-track copies of the eager GDN extend.

Each copy runs as one copy kernel (``sglang.kernels.ops.mamba.gdn_extend_state_copies``)
where its predicate holds and as the index ops otherwise. The CUDA-graph captured
extend tracks through ``gdn_prefill_track``, which reads per-row modes over a fixed
row bound; the eager extend has explicit index lists instead.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.mamba.gdn_extend_state_copies import (
    track_conv_rows,
    track_conv_rows_ok,
    track_state_rows,
    track_state_rows_ok,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata


def track_conv_windows(
    *, conv_states: torch.Tensor, mixed_qkv: torch.Tensor, metadata: "ForwardMetadata"
) -> None:
    """``conv_states[dst] = mixed_qkv[tokens].transpose(1, 2)`` for the track slots."""
    tokens = metadata.track_conv_indices
    dst = metadata.conv_states_mask_indices
    if track_conv_rows_ok(conv=conv_states, x=mixed_qkv, tokens=tokens):
        track_conv_rows(conv=conv_states, x=mixed_qkv, tokens=tokens, dst=dst)
    else:
        conv_states[dst] = mixed_qkv[tokens].transpose(1, 2)


def track_ssm_states_in_one_launch(
    *,
    ssm_states: torch.Tensor,
    h: Optional[torch.Tensor],
    h_track_buf: Optional[torch.Tensor],
    metadata: "ForwardMetadata",
) -> bool:
    """Make the copies of ``_track_mamba_state_extend`` with one copy kernel. Returns
    False, having copied nothing, when the index ops must make them."""
    recompute = metadata.track_ssm_recompute_dst
    if not track_state_rows_ok(
        states=ssm_states,
        h=h,
        h_rows=metadata.track_ssm_h_src.numel(),
        snapshot=h_track_buf is not None,
        recompute_rows=0 if recompute is None else recompute.numel(),
    ):
        return False
    track_state_rows(
        states=ssm_states,
        h=h,
        h_src=metadata.track_ssm_h_src,
        h_dst=metadata.track_ssm_h_dst,
        final_src=metadata.track_ssm_final_src,
        final_dst=metadata.track_ssm_final_dst,
    )
    return True
