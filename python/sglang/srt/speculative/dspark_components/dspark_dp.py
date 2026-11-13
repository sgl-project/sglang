"""Opt-in DSpark budget synchronization and packed attention-DP inputs.

The worker selects and validates the supported execution path. These helpers
do not select models or devices; cross-DP tier selection stays in the planner.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyHooks

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def make_dspark_verify_hooks(*, coordinate_dp_verify: bool) -> DSparkVerifyHooks:
    return DSparkVerifyHooks(
        sync_budget=sync_verify_budget,
        prepare_forward_batch=(
            prepare_compact_dp_forward_batch if coordinate_dp_verify else None
        ),
        dynamic_tiers_with_dp=True,
        coordinate_dp_verify=coordinate_dp_verify,
    )


def sync_verify_budget(budget: Optional[int]) -> Optional[int]:
    # Only ranks owning the SAME requests share a budget. None (a missing
    # confidence snapshot) must also be broadcast to keep their decisions equal.
    return get_dp_tp_group().broadcast_object(budget, src=0)


def prepare_compact_dp_forward_batch(forward_batch: "ForwardBatch") -> None:
    """Publish coordinated execution widths, independently of request counts.

    graph_num_tokens includes sink padding and is equal across participating
    DP groups. verify_lens retains each request's actual verification length.
    A uniform fallback has no ragged layout and keeps the original DP metadata.
    """
    layout = getattr(forward_batch.spec_info, "ragged_verify_layout", None)
    if layout is None:
        return
    tokens = layout.graph_num_tokens
    forward_batch.dp_verify_uses_packed_tokens = True
    counts = [tokens] * get_parallel().num_dp_ranks
    forward_batch.global_num_tokens_cpu = counts
    forward_batch.global_num_tokens_for_logprob_cpu = counts.copy()
    for name in ("global_num_tokens_gpu", "global_num_tokens_for_logprob_gpu"):
        tensor = getattr(forward_batch, name)
        if tensor is None or tensor.numel() != len(counts):
            tensor = torch.empty(
                len(counts), dtype=torch.int64, device=forward_batch.input_ids.device
            )
            setattr(forward_batch, name, tensor)
        tensor.fill_(tokens)
    # Here "non_padded" counts rows presented to the model/communication,
    # including the layout's sink rows, before any further runner padding.
    # It is NOT sum(verify_lens) or a count of accepted/output tokens. Masking
    # these rows out would change the coordinated execution shape. Acceptance
    # and KV commit still select real request rows through the ragged layout.
    if forward_batch.global_num_token_non_padded is not None:
        forward_batch.global_num_token_non_padded.fill_(tokens)
    forward_batch.global_num_token_non_padded_cpu = tokens
