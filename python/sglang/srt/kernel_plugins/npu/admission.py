"""Agree on distributed execution before entering any TP model collective."""


def all_admit(local, *, outstanding_chain=False):
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    vote = torch.tensor(
        -1 if outstanding_chain else int(local), dtype=torch.int32, device="cpu"
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=get_parallel().tp_group.cpu_group)
    decision = vote.item()
    if decision < 0:
        raise RuntimeError("Previous NextN chain has not been consumed")
    return bool(decision)


def candidate_present(local, valid=True):
    import torch
    import torch.distributed as dist
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group.cpu_group
    vote = torch.tensor(
        int(local) if valid else dist.get_world_size(group) + 1, dtype=torch.int32
    )
    dist.all_reduce(vote, group=group)
    count = int(vote.item())
    if count not in (0, dist.get_world_size(group)):
        raise RuntimeError("Ascend candidate ownership diverged across TP ranks")
    return count != 0
