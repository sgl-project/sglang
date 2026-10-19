# SPDX-License-Identifier: Apache-2.0
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


def gather_per_expert_tensors(experts, local_experts, num_total, dwdp_size, group):
    for name, data in experts.named_per_expert_tensors(local_experts):
        shards = [torch.empty_like(data) for _ in range(dwdp_size)]
        dist.all_gather(shards, data, group=group.device_group)
        yield name, torch.cat(shards, dim=0)[:num_total].contiguous()
