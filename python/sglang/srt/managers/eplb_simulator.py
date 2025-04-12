# TODO where to put this file?
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import einops
import torch
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_location import ExpertLocationMetadata, ModelConfigForExpertLocation
from tqdm.auto import tqdm


@dataclass
class MyServerArgs:
    chunked_prefill_size: int
    ep_num_redundant_experts: int
    nnodes: int
    tp_size: int
    enable_expert_location_by_eplb: bool


@dataclass
class MyExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)

    @staticmethod
    def init_by_eplb(server_args: MyServerArgs, logical_count: torch.Tensor, num_physical_experts: int):
        model_config_for_expert_location = _MY_MODEL_CONFIG_FOR_EXPERT_LOCATION

        physical_to_logical_map, logical_to_all_physical_map, _ = (
            deepseek_eplb.rebalance_experts(
                weight=logical_count,
                num_replicas=num_physical_experts,
                num_groups=model_config_for_expert_location.num_groups,
                num_nodes=server_args.nnodes,
                num_gpus=server_args.tp_size,
            )
        )

        return MyExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )


# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
_MY_MODEL_CONFIG_FOR_EXPERT_LOCATION = ModelConfigForExpertLocation(
    num_layers=61,
    num_logical_experts=256,
    num_groups=8,
)


def read_logical_count_of_seq(dir_data: Path):
    physical_count_of_forward_pass_id_and_rank = defaultdict(lambda: defaultdict())
    for path in tqdm(list(dir_data.glob("*.pt"))):
        for record in torch.load(path, weights_only=True):
            assert physical_count_of_forward_pass_id_and_rank[record["forward_pass_id"]].get(record["rank"]) is None
            physical_count_of_forward_pass_id_and_rank[record["forward_pass_id"]][record["rank"]] = record[
                "physical_count"]
    print(len(physical_count_of_forward_pass_id_and_rank))

    items = []
    for forward_pass_id, physical_count_of_rank in sorted(physical_count_of_forward_pass_id_and_rank.items()):
        physical_count_of_rank_tensor = torch.cat([
            physical_count
            for rank, physical_count
            in sorted(physical_count_of_rank.items())
        ], dim=-1)
        items.append(physical_count_of_rank_tensor)

    logical_count_of_seq = torch.stack(items)
    print(f"{logical_count_of_seq.shape=}")

    return logical_count_of_seq


def scan_combinations(
        logical_count_of_seq: torch.Tensor,
):
    server_args_list = [
        *[
            MyServerArgs(
                chunked_prefill_size=8192 * 32,
                ep_num_redundant_experts=32,
                nnodes=4,
                tp_size=32,
                enable_expert_location_by_eplb=enable_expert_location_by_eplb,
            )
            for enable_expert_location_by_eplb in [False, True]
        ]
    ]

    for server_args in server_args_list:
        mean_utilization_rate = simulate_execution(logical_count_of_seq=logical_count_of_seq, server_args=server_args)
        print(f"{server_args=} {mean_utilization_rate=:.2f}")


def simulate_execution(
        logical_count_of_seq: torch.Tensor,
        server_args: MyServerArgs,
):
    model_config_for_expert_location = _MY_MODEL_CONFIG_FOR_EXPERT_LOCATION
    num_physical_expert = model_config_for_expert_location.num_logical_experts + server_args.ep_num_redundant_experts

    logical_count_of_batch = simulate_batching(
        logical_count_of_seq=logical_count_of_seq,
        chunked_prefill_size=server_args.chunked_prefill_size,
    )

    if server_args.enable_expert_location_by_eplb:
        expert_location_metadata = MyExpertLocationMetadata.init_by_eplb(
            server_args,
            logical_count=einops.einsum(logical_count_of_seq, "num_seq num_layer num_expert -> num_layer num_expert"),
            num_physical_experts=num_physical_expert,
        )
        physical_count_of_batch = simulate_logical_to_physical(
            logical_count_of_whatever=logical_count_of_batch,
            logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map,
            num_physical_expert=num_physical_expert,
        )
    else:
        physical_count_of_batch = logical_count_of_batch

    gpu_physical_count_of_batch = compute_gpu_physical_count_of_batch(
        physical_count_of_batch=physical_count_of_batch,
        num_gpu=server_args.tp_size,
    )

    utilization_rate = compute_utilization_rate(
        gpu_physical_count_of_batch=gpu_physical_count_of_batch,
    )

    # NOTE: first 3 layers are MLP, thus those parts are not meaningful
    mean_utilization_rate = torch.mean(utilization_rate).item()

    return mean_utilization_rate


def simulate_batching(
        logical_count_of_seq: torch.Tensor,  # (num_seq, num_layer, num_logical_expert)
        chunked_prefill_size: int,
) -> torch.Tensor:
    """output: (num_batch, num_layer, num_logical_expert)"""
    tensor_chunks = chunker(
        logical_count_of_seq,
        state_reducer=lambda count, tensor: count + compute_num_token(tensor).item(),
        should_chunk=lambda count: count > chunked_prefill_size,
    )
    return torch.stack([torch.stack(tensor_chunk).sum(dim=0) for tensor_chunk in tensor_chunks])


def simulate_logical_to_physical(
        logical_count_of_whatever: torch.Tensor,  # (*, num_layer, num_logical_expert)
        logical_to_all_physical_map: torch.Tensor,  # (num_layer, num_logical_experts, X)
        num_physical_expert: int,
):
    """output: (*, num_layer, num_physical_expert)"""
    num_whatever, num_layer, num_logical_expert = logical_count_of_whatever.shape

    physical_count_of_whatever = torch.zeros(
        (num_whatever, num_layer, num_physical_expert),
        dtype=logical_to_all_physical_map.dtype,
    )

    for layer_id in range(num_layer):
        for logical_expert_id in range(num_logical_expert):
            all_physical_expert_ids = ExpertLocationMetadata.logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )
            for physical_expert_id in all_physical_expert_ids:
                physical_count_of_whatever[:, layer_id, physical_expert_id] += \
                    logical_count_of_whatever[:, layer_id, logical_expert_id] / len(all_physical_expert_ids)

    return physical_count_of_whatever


def compute_gpu_physical_count_of_batch(
        physical_count_of_batch: torch.Tensor,  # (num_batch, num_layer, num_physical_expert)
        num_gpu: int,
):
    """output: gpu_physical_count_of_batch (num_batch, num_layer, num_gpu)"""
    return einops.reduce(
        physical_count_of_batch,
        "num_batch num_layer (num_gpu num_expert_per_gpu) -> num_batch num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
        gpu_physical_count_of_batch: torch.Tensor,  # (num_batch, num_layer, num_gpu)
):
    """output: utilization_rate (num_batch, num_layer)"""
    max_gpu_physical_count = einops.reduce(gpu_physical_count_of_batch,
                                           "num_batch num_layer num_gpu -> num_batch num_layer", 'max')
    avg_gpu_physical_count = einops.reduce(gpu_physical_count_of_batch,
                                           "num_batch num_layer num_gpu -> num_batch num_layer", 'mean')
    return avg_gpu_physical_count / max_gpu_physical_count


def compute_num_token(whatever_with_num_layer_and_num_expert: torch.Tensor):
    return whatever_with_num_layer_and_num_expert[..., -1, :].sum(dim=-1)


def chunker(objects, state_reducer, should_chunk):
    state = 0
    outputs = []
    for obj in objects:
        outputs.append(obj)
        state = state_reducer(state, obj)
        if should_chunk(state):
            yield outputs
            outputs = []
            state = 0
    if len(outputs) > 0:
        yield outputs
