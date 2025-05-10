import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import einops
import polars as pl
import torch
from tqdm.auto import tqdm

from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_ = compute_utilization_rate, compute_gpu_physical_count


# ------------------------------------------- TODO refactor below ---------------------------------------------


def read_physical_count_of_forward_pass_id_and_rank(dir_data: Path):
    physical_count_of_forward_pass_id_and_rank = defaultdict(lambda: defaultdict())
    for path in tqdm(list(dir_data.glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        last_physical_to_logical_map = data_pack["last_physical_to_logical_map"]
        for record in data_pack["records"]:
            assert (
                physical_count_of_forward_pass_id_and_rank[
                    record["forward_pass_id"]
                ].get(record["rank"])
                is None
            )
            physical_count_of_forward_pass_id_and_rank[record["forward_pass_id"]][
                record["rank"]
            ] = record["physical_count"]
    # print(len(physical_count_of_forward_pass_id_and_rank))
    return physical_count_of_forward_pass_id_and_rank, last_physical_to_logical_map


def read_physical_count_of_forward_pass(dir_data: Path):
    physical_count_of_forward_pass_id_and_rank, last_physical_to_logical_map = (
        read_physical_count_of_forward_pass_id_and_rank(dir_data)
    )

    items = []
    for forward_pass_id, physical_count_of_rank in sorted(
        physical_count_of_forward_pass_id_and_rank.items()
    ):
        physical_count_of_rank_tensor = torch.stack(
            [
                physical_count
                for rank, physical_count in sorted(physical_count_of_rank.items())
            ]
        ).sum(dim=0)
        items.append(physical_count_of_rank_tensor)

    physical_count_of_forward_pass = torch.stack(items)
    print(f"{physical_count_of_forward_pass.shape=}")

    return physical_count_of_forward_pass, last_physical_to_logical_map


def scan_combinations(
    logical_count_of_seq: torch.Tensor,
    override_eplb_input_logical_count: Optional[torch.Tensor] = None,
):
    num_gpu_per_node = 8
    server_args_list = [
        *[
            MyServerArgs(
                num_tokens_in_batch_overall=num_tokens_in_batch_per_gpu
                * num_gpu_per_node
                * nnodes,
                ep_num_redundant_experts=ep_num_redundant_experts,
                nnodes=nnodes,
                tp_size=num_gpu_per_node * nnodes,
                enable_expert_location_by_eplb=enable_expert_location_by_eplb,
                init_expert_location=init_expert_location,
            )
            # for init_expert_location in ["/host_home/temp_sglang_server2local/1744461420780309768.json", None]
            for init_expert_location in ["from_variable"]
            # decode
            # for ep_num_redundant_experts in [0, 32]
            # for nnodes in [
            #     1,
            #     2,
            #     4,
            #     *([8] if ep_num_redundant_experts == 0 else []),
            #     *([9] if ep_num_redundant_experts == 32 else []),
            # ]
            # for num_tokens_in_batch_per_gpu in [64, 128]
            # prefill
            for ep_num_redundant_experts in [0, 32]
            for nnodes in [4]
            for num_tokens_in_batch_per_gpu in [8192]
            # for ep_num_redundant_experts in [0, 32, 64]
            # for nnodes in [1, 2, 4]
            # for num_tokens_in_batch_per_gpu in [1024, 4096, 8192, 16384]
            for enable_expert_location_by_eplb in [
                *([False] if ep_num_redundant_experts == 0 else []),
                True,
            ]
        ]
    ]

    rows = []
    for server_args in server_args_list:
        print()
        info = simulate_execution(
            logical_count_of_seq=logical_count_of_seq,
            server_args=server_args,
            override_eplb_input_logical_count=override_eplb_input_logical_count,
        )
        print(f"{server_args=} {info=}")
        rows.append(dict(**dataclasses.asdict(server_args), **info))

    df = pl.DataFrame(rows)
    return df


def analyze_actual_utilization_rate(dir_data: Path, num_gpu: int):
    physical_count_of_forward_pass = read_physical_count_of_forward_pass(dir_data)
    gpu_physical_count_of_forward_pass = compute_gpu_physical_count(
        physical_count_of_whatever=physical_count_of_forward_pass,
        num_gpu=num_gpu,
    )
    print(f"{gpu_physical_count_of_forward_pass.shape=}")
    utilization_rate = compute_utilization_rate(gpu_physical_count_of_forward_pass)
    print(f"{utilization_rate.shape=}")
    print(f"{torch.mean(utilization_rate, dim=0)=}")
    print(f"{torch.mean(utilization_rate[:, 3:]).item()=}")
    print(dir_data, torch.mean(utilization_rate).item())


def simulate_execution(
    logical_count_of_seq: torch.Tensor,
    server_args: MyServerArgs,
    override_eplb_input_logical_count: Optional[torch.Tensor] = None,
):
    model_config_for_expert_location = _MY_MODEL_CONFIG_FOR_EXPERT_LOCATION

    logical_count_of_batch = simulate_batching(
        logical_count_of_seq=logical_count_of_seq,
        num_tokens_in_batch_overall=server_args.num_tokens_in_batch_overall,
    )
    print(f"{logical_count_of_batch.shape=}")


def simulate_batching(
    logical_count_of_seq: torch.Tensor,  # (num_seq, num_layer, num_logical_expert)
    num_tokens_in_batch_overall: int,
) -> torch.Tensor:
    """output: (num_batch, num_layer, num_logical_expert)"""
    tensor_chunks = chunker(
        logical_count_of_seq,
        state_reducer=lambda count, tensor: count + compute_num_token(tensor).item(),
        should_chunk=lambda count: count >= num_tokens_in_batch_overall,
    )
    return torch.stack(
        [torch.stack(tensor_chunk).sum(dim=0) for tensor_chunk in tensor_chunks]
    )


def compute_num_token(whatever_with_num_layer_and_num_expert: torch.Tensor):
    num_token_mul_num_experts = whatever_with_num_layer_and_num_expert[..., -1, :].sum(
        dim=-1
    )
    return num_token_mul_num_experts / _MY_MODEL_CONFIG_NUM_EXPERTS_PER_TOK


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
