"""Ascend collectives and LSE merge for MLA decode context parallelism."""

from __future__ import annotations

import torch

from sglang.srt.layers.dcp.comm import dcp_a2a_lse_reduce
from sglang.srt.runtime_context import get_parallel


def mask_empty_mla_dcp_shards_npu(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    local_seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent an empty local KV shard as the online-softmax identity.

    Graph padding rows deliberately have a zero local sequence length.  FIA is
    still invoked for the fixed graph shape, so its undefined empty-row output
    must not participate in the cross-rank LSE merge.
    """
    num_rows = partial_output.shape[0]
    num_reqs = local_seq_lens.numel()
    if num_reqs == 0:
        return partial_output, partial_lse
    if num_rows % num_reqs != 0:
        raise ValueError(
            "DCP attention rows must be divisible by local sequence lengths: "
            f"rows={num_rows}, requests={num_reqs}."
        )
    rows_per_req = num_rows // num_reqs
    valid = (local_seq_lens > 0).repeat_interleave(rows_per_req).view(-1, 1, 1)
    return (
        torch.where(valid, partial_output, torch.zeros_like(partial_output)),
        torch.where(
            valid,
            partial_lse,
            torch.full_like(partial_lse, float("-inf")),
        ),
    )


def all_gather_mla_decode_q_npu(
    q_nope: torch.Tensor, q_rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather TP-local query heads across the DCP group with HCCL."""
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return q_nope, q_rope

    nope_dim = q_nope.shape[-1]
    combined = torch.cat([q_nope, q_rope], dim=-1).contiguous()
    combined = parallel.dcp_group.all_gather(combined, dim=1)
    q_nope, q_rope = combined.split([nope_dim, q_rope.shape[-1]], dim=-1)
    return q_nope.contiguous(), q_rope.contiguous()


def merge_mla_dcp_output_npu(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Compactly exchange and merge DCP partial attention states.

    ``partial_output`` is ``[B, H_local * N, D]``.  AllToAll routes the same
    ``H_local`` head slice from every KV-shard rank to its owner.  The local
    Triton combine kernel then reads BF16/FP16 output and FP32 LSE directly from
    the packed receive buffer and returns ``[B, H_local, D]``.  This reuses the
    common DCP A2A path, avoids expanding every output element to FP32 for HCCL,
    and does not materialize an unpacked intermediate tensor.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return partial_output

    dcp_size = parallel.dcp_size
    batch_size, total_heads, head_dim = partial_output.shape
    if total_heads % dcp_size != 0:
        raise ValueError(
            f"DCP query heads ({total_heads}) must be divisible by "
            f"dcp_size ({dcp_size})."
        )
    local_heads = total_heads // dcp_size
    if batch_size == 0:
        # All ranks in a DCP group share the same DP batch.  An idle DSpark DP
        # group therefore has no peer communication to perform; avoid both the
        # ambiguous empty LSE reshape and a zero-sized HCCL all-to-all.
        return partial_output.new_empty((0, local_heads, head_dim))

    partial_lse = (
        partial_lse.reshape(batch_size, total_heads, -1)[..., 0]
        .to(torch.float32)
        .contiguous()
    )
    return dcp_a2a_lse_reduce(
        partial_output,
        partial_lse,
        parallel.dcp_group,
        is_lse_base_on_e=True,
        comm_backend="a2a",
    )
