"""Small deterministic expert layouts for local and native EPLB validation."""

from types import SimpleNamespace

import torch


def metadata(mapping, *, ep_size=1, rank=0, device="cuda", num_logical=None):
    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    physical = torch.tensor(mapping, dtype=torch.int64, device=device)
    logical = num_logical or (max(max(row) for row in mapping) + 1)
    inverse = torch.full(
        (len(mapping), logical, len(mapping[0])), -1, dtype=torch.int64
    )
    for layer, row in enumerate(mapping):
        for expert in range(logical):
            slots = [slot for slot, value in enumerate(row) if value == expert]
            inverse[layer, expert, : len(slots)] = torch.tensor(slots)
    return ExpertLocationMetadata._init_raw(
        server_args=SimpleNamespace(
            ep_dispatch_algorithm="static", ep_join_mode=None, nnodes=1, ep_size=ep_size
        ),
        ep_size=ep_size,
        physical_to_logical_map=physical,
        logical_to_all_physical_map=inverse.to(device),
        moe_ep_rank=rank,
    )


def tensor_addresses(meta, weights):
    return [
        value.data_ptr()
        for value in vars(meta).values()
        if isinstance(value, torch.Tensor)
    ] + [value.data_ptr() for layer in weights.values() for value in layer]


def received_counts(mapping, routes):
    """CPU routing oracle for every physical slot, including redundant slots."""
    counts = torch.zeros(len(mapping), len(mapping[0]), dtype=torch.int32)
    for rank, logical_ids in enumerate(routes):
        placement = metadata(mapping, ep_size=len(routes), rank=rank, device="cpu")
        for layer in range(len(mapping)):
            physical = placement.logical_to_rank_dispatch_physical_map[layer][
                logical_ids.clamp_min(0)
            ]
            counts[layer].scatter_add_(
                0, physical.flatten(), (logical_ids >= 0).int().flatten()
            )
    return counts
