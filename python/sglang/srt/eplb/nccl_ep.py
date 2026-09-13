"""Order fixed-topology expert relocation against NCCL EP Graph submissions."""

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def expert_update_session(old, new):
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_graph import (
        get_nccl_ep_graph_resources,
    )

    # Validate before moving any weights. Captured routing and GEMMs retain
    # these allocations; replacing their layouts requires a new model/EP setup.
    for field in (
        "ep_size",
        "num_layers",
        "num_logical_experts",
        "num_physical_experts",
    ):
        if getattr(old, field) != getattr(new, field):
            raise ValueError(f"NCCL EP EPLB requires fixed topology: {field} changed")
    for field in (
        "physical_to_logical_map",
        "physical_to_logical_map_cpu",
        "logical_to_all_physical_map",
        "logical_to_all_physical_map_cpu",
        "logical_to_all_physical_map_num_valid",
        "logical_to_rank_dispatch_physical_map",
    ):
        before, after = getattr(old, field), getattr(new, field)
        if (before is None) != (after is None) or (
            before is not None
            and (before.shape, before.dtype, before.device)
            != (after.shape, after.dtype, after.device)
        ):
            raise ValueError(f"NCCL EP EPLB requires stable routing tensors: {field}")

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("NCCL EP EPLB must run between forward passes")
    owner = get_nccl_ep_graph_resources()
    session = owner.submission_session("eplb") if owner is not None else nullcontext()
    with session:
        if owner is not None:
            if owner.borrower is not None or owner.capturing:
                raise RuntimeError("NCCL EP EPLB requires a completed EP transaction")
            if owner.state is not None and (
                old.num_physical_experts != owner.state.num_experts
                or old.num_local_physical_experts != owner.state.num_local_experts
            ):
                raise ValueError("NCCL EP EPLB topology does not match the Graph group")
        # submission_session orders the preceding Graph stream before these
        # writes, and the next replay after them. ExpertLocationUpdater copies
        # weights first and publishes mappings in place on this same stream.
        yield
