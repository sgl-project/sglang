"""Platform-independent DWDP layer discovery and expert parameter gathering."""

import torch
import torch.distributed as dist


def collect_moe_layers(model):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    decoder = model.model if hasattr(model, "model") else model
    moe_layers = []
    for layer_idx, layer in enumerate(decoder.layers):
        experts = next((m for m in layer.modules() if isinstance(m, FusedMoE)), None)
        if experts is not None:
            moe_layers.append((layer_idx, experts))
    return moe_layers


def validate_moe_layers(moe_layers, dwdp_size, model):
    if not moe_layers:
        raise RuntimeError(
            f"DWDP is enabled but no FusedMoE layers were found in "
            f"{type(model).__name__}"
        )
    expert_counts = {e.num_global_routed_experts for _, e in moe_layers}
    if len(expert_counts) != 1:
        raise RuntimeError(
            f"DWDP requires a uniform routed expert count across MoE layers, "
            f"got {sorted(expert_counts)}"
        )
    num_routed = expert_counts.pop()
    if num_routed % dwdp_size != 0:
        raise ValueError(
            f"DWDP requires num_routed_experts ({num_routed}) to be divisible "
            f"by dwdp_size ({dwdp_size})"
        )
    return num_routed


def gather_per_expert_tensors(experts, local_experts, num_total, dwdp_size, group):
    for name, data in experts.named_per_expert_tensors(local_experts):
        shards = [torch.empty_like(data) for _ in range(dwdp_size)]
        dist.all_gather(shards, data, group=group.device_group)
        yield name, torch.cat(shards, dim=0)[:num_total].contiguous()
