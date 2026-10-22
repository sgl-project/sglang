"""Two-rank smoke for the EvenMoE EPLB/LPLB/Waterfill pipeline in SGLang.

Run one process per GPU with ``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR`` and
``MASTER_PORT`` set. This test uses real NCCL and real EvenMoE Triton policies,
with a small explicit context fixture for SGLang-owned resources.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from evenmoe import MoELoadBalancer
from evenmoe.adapters.sglang import (
    commit_placement,
    to_load_balancer_kwargs,
    to_placement_request,
)

from sglang.srt.eplb import moe_load_balancer_glue as glue
from sglang.srt.layers.moe.topk import StandardTopKOutput


class _EPGroup:
    @property
    def world_size(self):
        return dist.get_world_size()

    @property
    def rank_in_group(self):
        return dist.get_rank()

    def all_reduce(self, value):
        dist.all_reduce(value)
        return value


class _Recorder:
    def __init__(self):
        self.last_ids = None

    def on_select_experts(self, topk_ids):
        self.last_ids = topk_ids


class _Metadata:
    num_logical_experts = 2
    num_physical_experts = 4
    num_local_physical_experts = 2
    ep_size = 2

    def __init__(self, device, rank):
        self.physical_to_logical_map = torch.tensor(
            [[0, 1, 0, 1]], dtype=torch.int32, device=device
        )
        self.logical_to_all_physical_map = torch.tensor(
            [[[0, 2], [1, 3]]], dtype=torch.int32, device=device
        )
        self.logical_to_all_physical_map_num_valid = torch.tensor(
            [[2, 2]], dtype=torch.int32, device=device
        )
        self.logical_to_rank_dispatch_physical_map = torch.tensor(
            [[rank * 2, rank * 2 + 1]], dtype=torch.int32, device=device
        )


def _run_l1_eplb(mlb, device):
    request = to_placement_request(
        torch.tensor([[12, 4]], dtype=torch.int32, device=device),
        num_physical_experts=4,
        num_local_physical_experts=2,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )
    plan = mlb.plan_placement(request)
    assert plan.physical_to_logical_map.shape == (1, 4)
    assert plan.logical_to_all_physical_map.shape[:2] == (1, 2)


def _run_l2_mode(mlb, metadata, recorder, rank, mode, device, context):
    if mode == "combined_idle" and rank != 0:
        logical_ids = torch.empty((0, 2), dtype=torch.int32, device=device)
    elif rank == 0:
        logical_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device=device)
    else:
        logical_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.int32, device=device)
    num_tokens = logical_ids.shape[0]
    with patch.object(glue, "get_context", return_value=context):
        output = glue.route_topk_with_mlb(
            moe_load_balancer=mlb,
            layer_id=0,
            topk_output=StandardTopKOutput(
                topk_weights=torch.ones(
                    (num_tokens, 2), dtype=torch.float32, device=device
                ),
                topk_ids=logical_ids,
                router_logits=torch.zeros(
                    (num_tokens, 2), dtype=torch.float32, device=device
                ),
            ),
            num_tokens=num_tokens,
            num_token_non_padded=torch.tensor(
                num_tokens, dtype=torch.int32, device=device
            ),
            routed_scaling_factor=1.0,
        )

    expected_width = 3 if mlb.routing_capabilities.routes_shared_expert else 2
    assert output.topk_ids.shape == (num_tokens, expected_width)
    assert recorder.last_ids.shape == (num_tokens, 2)
    assert torch.all((recorder.last_ids >= 0) & (recorder.last_ids < 4))
    assert torch.equal(
        metadata.physical_to_logical_map[0, recorder.last_ids.long()], logical_ids
    )


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)

    recorder = _Recorder()
    metadata = _Metadata(device, rank)
    ep_group = _EPGroup()
    context = SimpleNamespace(
        parallel=SimpleNamespace(
            moe_ep_rank=rank,
            moe_ep_group=ep_group,
            ep_size=2,
            elastic_ep_initial_size=None,
        ),
        resources=SimpleNamespace(
            mlb_model_info={"num_logical_experts": 2, "num_groups": None},
            expert_location_metadata=metadata,
            expert_distribution_recorder=recorder,
            experts_capturer=None,
        ),
    )

    _run_l1_eplb(MoELoadBalancer(), device)
    modes = {
        "lplb": "lplb",
        "waterfill": "waterfill",
        "waterfill_dynamic": "waterfill_dynamic",
        "combined": "lplb+waterfill",
        "combined_idle": "lplb+waterfill",
    }
    for mode, algorithm in modes.items():
        context.config_bag = lambda name: SimpleNamespace(
            moe=SimpleNamespace(
                moe_load_balancer_algorithm=algorithm, ep_num_redundant_experts=2
            )
        )
        mlb = MoELoadBalancer.from_algorithm(**to_load_balancer_kwargs(context))
        commit_placement(mlb, context, [0])
        _run_l2_mode(mlb, metadata, recorder, rank, mode, device, context)
        replacement = _Metadata(device, rank)
        replacement.physical_to_logical_map = replacement.physical_to_logical_map.flip(
            -1
        )
        replacement.logical_to_all_physical_map = (
            replacement.logical_to_all_physical_map.flip(1)
        )
        replacement.logical_to_rank_dispatch_physical_map = (
            replacement.logical_to_rank_dispatch_physical_map.flip(-1)
        )
        context.resources.expert_location_metadata = replacement
        commit_placement(mlb, context, [0])
        _run_l2_mode(mlb, replacement, recorder, rank, mode, device, context)
        context.resources.expert_location_metadata = metadata
        dist.barrier()

    if rank == 0:
        print("PASS: EPLB, LPLB, Waterfill, and combined L2 pipeline")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
