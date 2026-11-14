# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""TP16 B1/B2 MoE front: SM placement, early router/latent-down staging, MMA router
and shared gate/up, FP32 routing, latent multicast/gather and shared SiTU/down rows."""

import functools

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from sglang.kernels.jit.kimi_k3.tensor_adapter import to_cute
from sglang.kernels.ops.communication.kimi_k3_specialized.front_tp16 import (
    FrontPlacementProbe,
)


@functools.cache
def _compile_probe(device, cluster, length):
    output = make_fake_tensor(Int32, (length,), (1,), assumed_align=16)
    with torch.accelerator.device_index(device):
        probe = FrontPlacementProbe(cluster)
        return cute.compile(probe, output, Uint64(20000), make_fake_stream())


def plan_designated(groups, *, engine_clusters=16, cluster_size=6):
    """Largest-capacity-first plan of the routed-expert clusters over the SM groups."""
    capacities = [len(group) // cluster_size for group in groups]
    if sum(capacities) < engine_clusters:
        raise ValueError("SM probe found insufficient routed-engine cluster capacity")
    order = sorted(range(len(groups)), key=lambda i: (-capacities[i], -len(groups[i])))
    counts = [0] * len(groups)
    remaining = engine_clusters
    for i in order:
        counts[i] = min(capacities[i], remaining)
        remaining -= counts[i]
    keep = set()
    for group, count in zip(groups, counts):
        quota = len(group) - cluster_size * count
        chosen: list[int] = []
        for sm in sorted(group):
            if sm % 2 == 0 and sm + 1 in group and len(chosen) + 2 <= quota:
                chosen.extend((sm, sm + 1))
        for sm in sorted(group):
            if len(chosen) < quota and sm not in chosen:
                chosen.append(sm)
        keep.update(chosen)
    return keep, counts


def initialize_designated(table: torch.Tensor):
    """Probe SM groups and fill the model-owned int32[nsm+16] table; returns CPU audit
    data. Probe storage is freed before capture. Probes: 1- and 3-wave grids of
    2..16-CTA clusters, 220 KiB SMEM (one CTA per SM), a 20 us global-timer spin."""
    device = table.device
    nsm = torch.cuda.get_device_properties(device).multi_processor_count
    parent = list(range(4096))
    seen = set()
    launches = []

    def find(sm):
        while parent[sm] != sm:
            parent[sm] = parent[parent[sm]]
            sm = parent[sm]
        return sm

    with torch.accelerator.device_index(device.index):
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        for cluster in range(2, 17):
            for waves in (1, 3):
                clusters = max(1, (nsm // cluster) * waves)
                output = table.new_full((2 * cluster * clusters,), -1)
                try:
                    compiled = _compile_probe(device.index, cluster, output.numel())
                    compiled(to_cute(output), Uint64(20000), stream)
                except RuntimeError as e:
                    launches.append(dict(cluster=cluster, waves=waves, error=str(e)))
                    break
                torch.accelerator.synchronize(device)
                members: dict[int, list[int]] = {}
                for sm, group in output.view(-1, 2).cpu().tolist():
                    if sm >= 0:
                        members.setdefault(group, []).append(sm)
                        seen.add(sm)
                for group in members.values():
                    for sm in group[1:]:
                        parent[find(sm)] = find(group[0])
                launches.append(dict(cluster=cluster, waves=waves, clusters=clusters))
    if len(seen) != nsm:
        raise RuntimeError(f"Placement probe saw {len(seen)} of {nsm} SMs")
    group_map: dict[int, list[int]] = {}
    for sm in sorted(seen):
        group_map.setdefault(find(sm), []).append(sm)
    groups = sorted(group_map.values(), key=lambda group: group[0])
    keep, counts = plan_designated(groups)
    values = [int(sm in keep) for sm in range(nsm + 16)]
    table.copy_(torch.tensor(values, dtype=torch.int32, device=device))
    return dict(
        groups=groups,
        engine_clusters=counts,
        designated=sorted(keep),
        launches=launches,
    )
