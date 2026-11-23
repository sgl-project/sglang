"""Agree on distributed execution before entering any TP model collective."""


def all_admit(local):
    import torch
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    vote = torch.tensor(int(local), dtype=torch.int32, device="cpu")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
    return bool(vote.item())
