"""Global token layout shared by the benchmark and correctness check.

Only TP=world, EP=1 (replicated inputs) and TP=world, EP=world
(token-sharded inputs with MoE TP=1) are supported.
"""

import hashlib

import torch
import torch.distributed as dist


def token_counts(global_tokens, world):
    if global_tokens < 1 or world < 1:
        raise ValueError("global_tokens and world must be positive")
    q, r = divmod(global_tokens, world)
    return [q + (rank < r) for rank in range(world)]


def tensor_hash(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def make_input(global_tokens, hidden_size, backend, seed, device):
    """Broadcast one global input, then slice it for EP, outside measurement."""
    rank, world = dist.get_rank(), dist.get_world_size()
    shape = (global_tokens, hidden_size)
    if rank == 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        full = (torch.randn(shape, generator=g) * 0.5).to(torch.bfloat16).to(device)
    else:
        full = torch.empty(shape, dtype=torch.bfloat16, device=device)
    dist.broadcast(full, src=0)
    input_hash = tensor_hash(full)
    counts = token_counts(global_tokens, world)
    if backend == "deepep":
        start = sum(counts[:rank])
        local = full[start : start + counts[rank]].clone()
    elif backend == "none":
        local = full
        counts = [global_tokens] * world
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return local, counts, input_hash


def gather_outputs(local, counts, backend):
    """Gather every rank, padding only for the collective, after the forward.

    Empty input ranks still participate. EP padding never enters the MoE block.
    For TP, keep every replica so a faulty nonzero rank cannot evade comparison.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    if len(counts) != world or local.shape[0] != counts[rank]:
        raise ValueError("output rows do not match the declared input layout")
    padded = local.new_zeros((max(counts), local.shape[1]))
    padded[: local.shape[0]].copy_(local)
    outputs = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(outputs, padded)
    pieces = [out[:count].cpu() for out, count in zip(outputs, counts)]
    if backend == "deepep":
        return torch.cat(pieces, dim=0), []
    if backend == "none":
        return pieces[0], pieces
    raise ValueError(f"unsupported backend: {backend}")
