import math
from dataclasses import dataclass
from typing import List, Union, Literal

import einops
import polars as pl
import torch
from sglang.srt.eplb_simulator.configs import MyServerArgs, MY_MODEL_CONFIG_FOR_EXPERT_LOCATION
from sglang.srt.eplb_simulator.reader import ExpertDistributionModeDetailPerTokenAndBenchServingPack
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution import (
    compute_gpu_physical_count,
    compute_utilization_rate,
)
from sglang.srt.managers.expert_location import ExpertLocationMetadata

_Phase = Union[Literal["prefill", "decode"]]


def simulate_execution_given_pack(
    pack: ExpertDistributionModeDetailPerTokenAndBenchServingPack,
    phase: str,
    server_args: MyServerArgs,
    assert_physical_equal_logical_expert: bool,
):
    num_physical_expert = _compute_num_physical_experts(server_args)

    token_indices_of_batch = _simulate_scheduled_pack_indices_given_seq_metadata(
        pack.df_metadata,
        phase=phase,
        num_tokens_in_batch_overall=server_args.num_tokens_in_batch_overall,
    )

    vanilla_physical_count_of_batch = torch.stack([
        compute_global_physical_count_from_topk_ids(
            topk_ids=pack.topk_ids[token_indices_of_batch[i], :, :],
            num_physical_expert=num_physical_expert,
        )
        for i in range(len(token_indices_of_batch))
    ])

    assert assert_physical_equal_logical_expert
    logical_count_of_batch = vanilla_physical_count_of_batch

    return _simulate_execution_given_logical_count_of_batch(logical_count_of_batch=logical_count_of_batch,
                                                            server_args=server_args)


def _simulate_scheduled_pack_indices_given_seq_metadata(
    df_metadata: pl.DataFrame,
    phase: _Phase,
    num_tokens_in_batch_overall: int,
) -> List[torch.Tensor]:
    """
    :return: `output[i]` denotes all pack indices that will be used in i-th step
    """
    if phase == "prefill":
        all_pack_indices = torch.tensor([
            x
            for row in df_metadata.iter_rows(named=True)
            for x in range(row["pack_input_except_history_start_index"], row["pack_output_start_index"])
        ])
        num_chunks = math.ceil(len(all_pack_indices) / num_tokens_in_batch_overall)
        return torch.chunk(all_pack_indices, num_chunks)

    if phase == "decode":
        # loose bound
        num_steps_upper_bound = int(df_metadata["end_index"].max() / num_tokens_in_batch_overall * 2.5)

        pack_indices_of_step = torch.full((num_steps_upper_bound, num_tokens_in_batch_overall), -1, dtype=torch.int32)
        curr_lens = torch.zeros((num_tokens_in_batch_overall,), dtype=torch.int32)

        for row in df_metadata.iter_rows(named=True):
            chosen_location = torch.argmin(curr_lens).item()
            output_values = list(range(row["pack_output_start_index"], row["end_index"]))
            output_start = curr_lens[chosen_location]

            pack_indices_of_step[output_start: output_start + len(output_values), chosen_location] = output_values
            curr_lens[chosen_location] += len(output_values)

        return [x[x != -1] for x in pack_indices_of_step[:torch.max(curr_lens)]]

    raise NotImplementedError


def _simulate_execution_given_logical_count_of_batch(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
):
    physical_count_of_batch = _simulate_eplb_physical_count_of_batch(
        logical_count_of_batch=logical_count_of_batch,
        server_args=server_args,
    )

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


def _simulate_eplb_physical_count_of_batch(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
):
    num_batches, _, _ = logical_count_of_batch.shape
    num_physical_expert = _compute_num_physical_experts(server_args)

    if server_args.enable_expert_location_by_eplb:
        expert_location_metadata_arr = _simulate_expert_location_metadata_arr(
            logical_count_of_batch=logical_count_of_batch,
            server_args=server_args,
            num_physical_expert=num_physical_expert,
        )
        outputs = [
            _simulate_logical_to_physical_by_random_dispatching(
                logical_count_of_whatever=logical_count_of_batch[batch_index, :, :],
                logical_to_all_physical_map=expert_location_metadata_arr[batch_index].logical_to_all_physical_map,
                num_physical_expert=num_physical_expert,
            )
            for batch_index in range(num_batches)
        ]
        return torch.stack(outputs)
    else:
        return logical_count_of_batch


def _simulate_expert_location_metadata_arr(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
    num_physical_expert: int,
) -> List["MyExpertLocationMetadata"]:
    num_batches, _, _ = logical_count_of_batch.shape
    return [
        MyExpertLocationMetadata.init_by_eplb(
            server_args,
            logical_count=einops.einsum(
                logical_count_of_batch[max(0, batch_index - server_args.eplb_history_num_batch):batch_index, :, :],
                "num_interest_batches num_layer num_expert -> num_layer num_expert",
            ),
            num_physical_experts=num_physical_expert,
        )
        for batch_index in range(num_batches)
    ]


def _simulate_logical_to_physical_by_random_dispatching(
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


def compute_global_physical_count_from_topk_ids(
    topk_ids: torch.Tensor,  # (num_tokens, num_layers, num_topk)
    num_physical_expert: int,
):
    """
    :return: global_physical_count - (num_layers, num_physical_experts)
    """
    topk_ids_flattened = einops.rearrange(topk_ids,
                                          'num_tokens num_layers num_topk -> num_layers (num_tokens num_topk)')
    return torch.stack([
        torch.bincount(x, minlength=num_physical_expert)
        for x in topk_ids_flattened
    ])


def _compute_num_physical_experts(
    server_args: MyServerArgs,
    model_config_for_expert_location=MY_MODEL_CONFIG_FOR_EXPERT_LOCATION,
):
    return (
        model_config_for_expert_location.num_logical_experts
        + server_args.ep_num_redundant_experts
    )


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
