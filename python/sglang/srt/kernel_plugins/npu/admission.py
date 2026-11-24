"""Agree on distributed execution before entering any TP model collective."""


def all_admit(local):
    import torch
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    # Install rejects a draft group that is not this TP group. The vote uses
    # that same group so every rank that entered the call participates.
    vote = torch.tensor(int(local), dtype=torch.int32, device="cpu")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
    return bool(vote.item())
