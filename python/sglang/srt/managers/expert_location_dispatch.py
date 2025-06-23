# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from sglang.srt.managers.expert_location import get_global_expert_location_metadata
from sglang.srt.managers.schedule_batch import global_server_args_dict


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random", "workload_based"]
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(cls, layer_id: int):
        ep_dispatch_algorithm = global_server_args_dict["ep_dispatch_algorithm"]
        expert_location_metadata = get_global_expert_location_metadata()

        if ep_dispatch_algorithm is None:
            return None

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if (info is not None) and (info.ep_dispatch_algorithm == "fake"):
        router_logits = torch.randn_like(router_logits)
        if correction_bias is not None:
            correction_bias = torch.zeros_like(correction_bias)
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm == "workload_based":
        return _topk_ids_logical_to_physical_workload_heuristic(topk_ids, info)
    if info.ep_dispatch_algorithm in ["dynamic", "fake"]:
        return _topk_ids_logical_to_physical_dynamic(topk_ids, info)
    raise NotImplementedError(f"Unknown algorithm {info.ep_dispatch_algorithm}")


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_dynamic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids

def _topk_ids_logical_to_physical_workload_heuristic(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    device = topk_ids.device
    E = info.partial_logical_to_all_physical_map.size(0)
    flat_logical = topk_ids.flatten().to(torch.int64)
    N = flat_logical.numel()

    workload = torch.bincount(flat_logical, minlength=E).tolist()
    phys_ids = info.partial_logical_to_all_physical_map
    num_valid = info.partial_logical_to_all_physical_map_num_valid

    num_physical = int(info.num_physical_experts)
    G = global_server_args_dict["tp_size"]
    num_red = global_server_args_dict["ep_num_redundant_experts"]
    M = (num_physical + num_red) // G

    gpu_fixed_workload = [0] * G
    expert_gpu_map = [[] for _ in range(E)]
    single_copy_experts, multi_copy_experts = [], []

    for e in range(E):
        valid_p = phys_ids[e, :num_valid[e]].tolist()
        gpu_ids = [p // M for p in valid_p]
        expert_gpu_map[e] = gpu_ids
        if len(gpu_ids) == 1:
            gpu_fixed_workload[gpu_ids[0]] += workload[e]
            single_copy_experts.append(e)
        else:
            multi_copy_experts.append(e)

    total_token_demand = sum(workload[e] for e in multi_copy_experts)
    total_fixed = sum(gpu_fixed_workload)
    target_loads = [(total_token_demand + total_fixed) / G] * G
    candidate_window = [max(1e-5, target_loads[g] - gpu_fixed_workload[g]) for g in range(G)]

    lookup_phys_id = torch.full((E, G), -1, dtype=torch.int64, device=device)
    for e in range(E):
        for p in phys_ids[e, :num_valid[e]]:
            g = p.item() // M
            lookup_phys_id[e, g] = p

    sorted_e, token_pos = torch.sort(flat_logical)
    seg_start = torch.cat([
        torch.tensor([0], device=device),
        torch.nonzero(sorted_e[1:] != sorted_e[:-1]).flatten() + 1,
        torch.tensor([sorted_e.numel()], device=device)
    ])
    token_pos_per_e = [
        token_pos[seg_start[i]:seg_start[i + 1]]
        for i in range(seg_start.numel() - 1)
    ]
    unique_e_ids = sorted_e[seg_start[:-1]]
    e_to_index = {e.item(): i for i, e in enumerate(unique_e_ids)}

    token_chunks, phys_chunks = [], []
    y = torch.zeros((E, G), dtype=torch.int32)

    for e in multi_copy_experts:
        if workload[e] == 0 or e not in e_to_index:
            continue

        gpus = expert_gpu_map[e]
        weights = [candidate_window[g] for g in gpus]
        total_w = sum(weights)
        ratios = [w / total_w if total_w > 0 else 1.0 / len(gpus) for w in weights]

        total_tokens = workload[e]
        tokens_per_gpu = [int(total_tokens * r) for r in ratios]
        while sum(tokens_per_gpu) < total_tokens:
            tokens_per_gpu[torch.tensor(ratios).argmax().item()] += 1

        tok_indices = token_pos_per_e[e_to_index[e]]
        cursor = 0
        for i, g in enumerate(gpus):
            n_tok = tokens_per_gpu[i]
            if n_tok == 0:
                continue
            y[e, g] = n_tok
            idxs = tok_indices[cursor:cursor + n_tok]
            cursor += n_tok
            phys_id = lookup_phys_id[e, g].item()
            phys_tensor = torch.full((n_tok,), phys_id, dtype=torch.int64, device=device)
            token_chunks.append(idxs)
            phys_chunks.append(phys_tensor)

    for e in single_copy_experts:
        if workload[e] == 0 or e not in e_to_index:
            continue
        g = expert_gpu_map[e][0]
        tok_indices = token_pos_per_e[e_to_index[e]]
        y[e, g] = len(tok_indices)
        phys_id = lookup_phys_id[e, g].item()
        phys_tensor = torch.full((len(tok_indices),), phys_id, dtype=torch.int64, device=device)
        token_chunks.append(tok_indices)
        phys_chunks.append(phys_tensor)

    flat_physical = torch.empty(N, dtype=torch.int64, device=device)
    if token_chunks:
        all_tokens = torch.cat(token_chunks)
        all_phys = torch.cat(phys_chunks)
        flat_physical[all_tokens] = all_phys

    return flat_physical.view_as(topk_ids)