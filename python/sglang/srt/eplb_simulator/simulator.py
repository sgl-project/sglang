from dataclasses import dataclass

import torch
from sglang.srt.eplb_simulator.configs import MyServerArgs, MY_MODEL_CONFIG_FOR_EXPERT_LOCATION
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution import (
    compute_gpu_physical_count,
    compute_utilization_rate,
)
from sglang.srt.managers.expert_location import ExpertLocationMetadata


def simulate_execution_given_logical_count_of_batch(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
):
    TODO

    gpu_physical_count_of_batch = compute_gpu_physical_count(
        physical_count_of_whatever=physical_count_of_batch,
        num_gpu=server_args.tp_size,
    )

    utilization_rate = compute_utilization_rate(
        gpu_physical_count_of_batch=gpu_physical_count_of_batch,
    )

    # NOTE: first 3 layers are dense layers, thus those parts are not meaningful
    mean_utilization_rate = torch.mean(utilization_rate).item()

    return dict(
        gpu_physical_count_of_batch=gpu_physical_count_of_batch,
        utilization_rate=utilization_rate,
        mean_utilization_rate=mean_utilization_rate,
        num_simulated_batches=logical_count_of_batch.shape[0],
    )


def compute_physical_count_of_batch(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
    model_config_for_expert_location=MY_MODEL_CONFIG_FOR_EXPERT_LOCATION,
):
    if server_args.enable_expert_location_by_eplb:
        num_physical_expert = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )

        # TODO
        eplb_input_logical_count = einops.einsum(
            logical_count_of_seq,
            "num_seq num_layer num_expert -> num_layer num_expert",
        )

        expert_location_metadata = MyExpertLocationMetadata.init_by_eplb(
            server_args,
            logical_count=eplb_input_logical_count,
            num_physical_experts=num_physical_expert,
        )
        # print(f"hi {expert_location_metadata=}")
        physical_count_of_batch = simulate_logical_to_physical_by_random_dispatching(
            logical_count_of_whatever=logical_count_of_batch,
            logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map,
            num_physical_expert=num_physical_expert,
        )
        # print(f"hi {physical_count_of_batch=}")
    else:
        physical_count_of_batch = logical_count_of_batch


def simulate_logical_to_physical_by_random_dispatching(
    logical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_logical_expert)
    logical_to_all_physical_map: torch.Tensor,  # (num_layer, num_logical_experts, X)
    num_physical_expert: int,
):
    """output: (..., num_layer, num_physical_expert)"""
    *prefix_dims, num_layer, num_logical_expert = logical_count_of_whatever.shape

    physical_count_of_whatever = torch.zeros(
        (*prefix_dims, num_layer, num_physical_expert),
        dtype=torch.float32,
    )

    for layer_id in range(num_layer):
        for logical_expert_id in range(num_logical_expert):
            all_physical_expert_ids = (
                ExpertLocationMetadata.logical_to_all_physical_raw(
                    logical_to_all_physical_map, layer_id, logical_expert_id
                )
            )
            for physical_expert_id in all_physical_expert_ids:
                physical_count_of_whatever[
                :, layer_id, physical_expert_id
                ] += logical_count_of_whatever[:, layer_id, logical_expert_id] / len(
                    all_physical_expert_ids
                )

    return physical_count_of_whatever


@dataclass
class MyExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)

    @staticmethod
    def init_by_eplb(
        server_args: MyServerArgs,
        logical_count: torch.Tensor,
        num_physical_experts: int,
    ):
        model_config_for_expert_location = MY_MODEL_CONFIG_FOR_EXPERT_LOCATION

        physical_to_logical_map, logical_to_all_physical_map, _ = (
            deepseek_eplb.rebalance_experts(
                weight=logical_count,
                num_replicas=num_physical_experts,
                num_groups=model_config_for_expert_location.num_groups,
                num_nodes=server_args.nnodes,
                num_gpus=server_args.tp_size,
                hack_shuffle=server_args.deepseek_eplb_hack_shuffle,
            )
        )

        return MyExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )
