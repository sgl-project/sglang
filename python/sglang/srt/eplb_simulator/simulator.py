import math
from dataclasses import dataclass
from typing import List, Literal, Union

import einops
import polars as pl
import torch
from tqdm.auto import tqdm, trange

from sglang.srt.eplb_simulator.configs import (
    MY_MODEL_CONFIG_FOR_EXPERT_LOCATION,
    MyServerArgs,
)
from sglang.srt.eplb_simulator.reader import (
    ExpertDistributionModeDetailPerTokenAndBenchServingPack,
)
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution import (
    compute_gpu_physical_count,
    compute_utilization_rate,
)
from sglang.srt.managers.expert_location import ExpertLocationMetadata

_Phase = Union[Literal["prefill", "decode"]]


def simulate_execution_given_pack(
    pack: ExpertDistributionModeDetailPerTokenAndBenchServingPack,
    server_args: MyServerArgs,
    assert_vanilla_physical_equal_logical_expert: bool,
    model_config_for_expert_location=MY_MODEL_CONFIG_FOR_EXPERT_LOCATION,
):
    with torch.device("cuda"):
        token_indices_of_batch = _simulate_scheduled_pack_indices_given_seq_metadata(
            pack.df_metadata,
            phase=server_args.phase,
            num_tokens_in_batch_overall=server_args.num_tokens_in_batch_overall,
            decode_max_left_padding=server_args.decode_max_left_padding,
        )

        assert assert_vanilla_physical_equal_logical_expert
        logical_count_of_batch = torch.stack(
            [
                compute_global_physical_count_from_topk_ids(
                    topk_ids=pack.topk_ids[token_indices_of_batch[i], :, :],
                    num_physical_expert=model_config_for_expert_location.num_logical_experts,
                )
                for i in trange(
                    len(token_indices_of_batch), desc="vanilla_physical_count_of_batch"
                )
            ]
        )

        simulation_output = _simulate_execution_given_logical_count_of_batch(
            logical_count_of_batch=logical_count_of_batch,
            server_args=server_args,
        )

        simulation_output = dict(
            logical_count_of_batch=logical_count_of_batch,
            num_tokens_of_batch=torch.tensor(
                [len(x) for x in token_indices_of_batch], dtype=torch.int32
            ),
            **simulation_output,
        )

        num_batch, _ = simulation_output["utilization_rate"].shape
        simulation_output["df_step"] = pl.DataFrame(
            dict(
                step=list(range(num_batch)),
                utilization_rate=einops.reduce(
                    simulation_output["utilization_rate"],
                    "num_batch num_layer -> num_batch",
                    "mean",
                ).tolist(),
                num_tokens_of_batch=simulation_output["num_tokens_of_batch"].tolist(),
            )
        )

        return simulation_output


def _simulate_scheduled_pack_indices_given_seq_metadata(
    df_metadata: pl.DataFrame,
    phase: _Phase,
    num_tokens_in_batch_overall: int,
    decode_max_left_padding: int,
) -> List[torch.Tensor]:
    """
    :return: `output[i]` denotes all pack indices that will be used in i-th step
    """
    if phase == "prefill":
        all_pack_indices = torch.tensor(
            [
                x
                for row in df_metadata.iter_rows(named=True)
                for x in range(
                    row["pack_input_except_history_start_index"],
                    row["pack_output_start_index"],
                )
            ]
        )
        num_chunks = math.ceil(len(all_pack_indices) / num_tokens_in_batch_overall)
        return torch.chunk(all_pack_indices, num_chunks)

    if phase == "decode":
        # loose bound
        num_steps_upper_bound = max(
            1_000_000,
            int(
                df_metadata["pack_end_index"].max() // num_tokens_in_batch_overall * 1.5
            ),
        )

        pack_indices_of_step = torch.full(
            (num_steps_upper_bound, num_tokens_in_batch_overall),
            fill_value=-1,
            dtype=torch.int32,
        )
        curr_lens = torch.randint(
            0,
            decode_max_left_padding + 1,
            (num_tokens_in_batch_overall,),
            dtype=torch.int32,
        )

        for row in tqdm(df_metadata.iter_rows(named=True), total=len(df_metadata)):
            chosen_location = torch.argmin(curr_lens).item()
            output_values = list(
                range(row["pack_output_start_index"], row["pack_end_index"])
            )
            output_start = curr_lens[chosen_location]
            output_end = output_start + len(output_values)
            assert (
                output_end <= num_steps_upper_bound
            ), f"{num_steps_upper_bound=} {output_end=}"

            pack_indices_of_step[output_start:output_end, chosen_location] = (
                torch.tensor(output_values, dtype=torch.int32)
            )
            curr_lens[chosen_location] += len(output_values)

        return [x[x != -1] for x in pack_indices_of_step[: torch.max(curr_lens)]]

    raise NotImplementedError


def _simulate_execution_given_logical_count_of_batch(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
):
    balanced_physical_count_of_batch, expert_location_metadata_arr = (
        _simulate_eplb_physical_count_of_batch(
            logical_count_of_batch=logical_count_of_batch,
            server_args=server_args,
        )
    )

    gpu_physical_count_of_batch = compute_gpu_physical_count(
        physical_count_of_whatever=balanced_physical_count_of_batch,
        num_gpu=server_args.tp_size,
    )

    utilization_rate = compute_utilization_rate(
        gpu_physical_count_of_batch=gpu_physical_count_of_batch,
    )

    # NOTE: first 3 layers are dense layers, thus those parts are not meaningful
    mean_utilization_rate = torch.mean(utilization_rate).item()

    return dict(
        balanced_physical_count_of_batch=balanced_physical_count_of_batch,
        gpu_physical_count_of_batch=gpu_physical_count_of_batch,
        utilization_rate=utilization_rate,
        mean_utilization_rate=mean_utilization_rate,
        num_simulated_batches=logical_count_of_batch.shape[0],
        expert_location_metadata_arr=expert_location_metadata_arr,
    )


def _simulate_eplb_physical_count_of_batch(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
):
    num_batches, _, _ = logical_count_of_batch.shape
    num_physical_expert = _compute_num_physical_experts(server_args)

    if (expert_location_mode := server_args.expert_location_mode) is not None:
        expert_location_metadata_arr = _simulate_expert_location_metadata_arr(
            logical_count_of_batch=logical_count_of_batch,
            server_args=server_args,
            num_physical_expert=num_physical_expert,
            expert_location_mode=expert_location_mode,
        )
        outputs = [
            _simulate_logical_to_physical_by_random_dispatching(
                logical_count=logical_count_of_batch[batch_index, :, :],
                logical_to_all_physical_map=expert_location_metadata_arr[
                    batch_index
                ].logical_to_all_physical_map,
                num_physical_expert=num_physical_expert,
            )
            for batch_index in trange(num_batches)
        ]
        return torch.stack(outputs), expert_location_metadata_arr
    else:
        return logical_count_of_batch, None


def _simulate_expert_location_metadata_arr(
    logical_count_of_batch: torch.Tensor,
    server_args: MyServerArgs,
    num_physical_expert: int,
    expert_location_mode: str,
) -> List["MyExpertLocationMetadata"]:
    num_batches, num_layer, num_logical_expert = logical_count_of_batch.shape

    if expert_location_mode == "previous_chunk":
        chunk_size = server_args.eplb_rebalance_num_iterations
        num_chunks = math.ceil(num_batches / chunk_size)

        output_chunks = [
            MyExpertLocationMetadata.init_by_eplb(
                server_args,
                # NOTE first chunk has no statistics
                logical_count=torch.zeros((1, num_layer, num_logical_expert)),
                num_physical_experts=num_physical_expert,
            )
        ] + [
            MyExpertLocationMetadata.init_by_eplb(
                server_args,
                logical_count=logical_count_of_batch[
                    (chunk_index - 1) * chunk_size : chunk_index * chunk_size,
                    :,
                    :,
                ],
                num_physical_experts=num_physical_expert,
            )
            for chunk_index in trange(
                1, num_chunks, desc="Expert location init by eplb"
            )
        ]

        return [
            output_chunks[batch_index // chunk_size]
            for batch_index in range(num_batches)
        ]

    elif expert_location_mode == "global_average":
        output = MyExpertLocationMetadata.init_by_eplb(
            server_args,
            logical_count=logical_count_of_batch,
            num_physical_experts=num_physical_expert,
        )
        return [output for _ in range(num_batches)]

    raise NotImplementedError


def _simulate_logical_to_physical_by_random_dispatching(
    logical_count: torch.Tensor,  # (num_layer, num_logical_expert)
    logical_to_all_physical_map: torch.Tensor,  # (num_layer, num_logical_experts, X)
    num_physical_expert: int,
):
    """output: (num_layer, num_physical_expert)"""
    num_layer, num_logical_expert = logical_count.shape
    _, _, x_dim = logical_to_all_physical_map.shape

    num_physical_expert_per_logical_expert = einops.einsum(
        logical_to_all_physical_map != -1,
        "num_layer num_logical_experts X -> num_layer num_logical_experts",
    )
    assert torch.all(num_physical_expert_per_logical_expert >= 1)
    logical_count_amortized = logical_count / num_physical_expert_per_logical_expert
    logical_count_repeated = einops.repeat(
        logical_count_amortized,
        "num_layer num_logical_count -> num_layer num_logical_count x_dim",
        x_dim=x_dim,
    )

    # change `-1` to `num_physical_expert` (a dummy location that is not used)
    logical_to_all_physical_map_noneg1 = logical_to_all_physical_map.masked_fill(
        logical_to_all_physical_map == -1, num_physical_expert
    )

    physical_count_of_whatever = torch.zeros(
        (num_layer, num_physical_expert + 1),
        dtype=torch.float32,
    )

    rearrange_expr = (
        "num_layer num_logical_count x_dim -> num_layer (num_logical_count x_dim)"
    )
    physical_count_of_whatever.scatter_add_(
        dim=1,
        index=einops.rearrange(
            logical_to_all_physical_map_noneg1, rearrange_expr
        ).long(),
        src=einops.rearrange(logical_count_repeated, rearrange_expr),
    )

    return physical_count_of_whatever[:, :-1]


def compute_global_physical_count_from_topk_ids(
    topk_ids: torch.Tensor,  # (num_tokens, num_layers, num_topk)
    num_physical_expert: int,
):
    """
    :return: global_physical_count - (num_layers, num_physical_experts)
    """
    topk_ids_flattened = einops.rearrange(
        topk_ids, "num_tokens num_layers num_topk -> num_layers (num_tokens num_topk)"
    )
    num_layers, _ = topk_ids_flattened.shape

    topk_ids_flattened_noneg1 = topk_ids_flattened.masked_fill(
        topk_ids_flattened == -1, num_physical_expert
    )

    global_physical_count = torch.zeros(
        (num_layers, num_physical_expert + 1), dtype=torch.int64
    )
    global_physical_count.scatter_add_(
        dim=1,
        index=topk_ids_flattened_noneg1.long(),
        src=torch.full_like(topk_ids_flattened, 1, dtype=torch.int64),
    )
    return global_physical_count[:, :-1]


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

        num_local_physical_experts = num_physical_experts // server_args.tp_size
        physical_to_logical_map, logical_to_all_physical_map, _ = (
            deepseek_eplb.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_physical_experts,
                num_groups=model_config_for_expert_location.num_groups,
                num_nodes=server_args.nnodes,
                phase=server_args.phase,
            )
        )

        device = logical_count.device
        physical_to_logical_map = physical_to_logical_map.to(device)
        logical_to_all_physical_map = logical_to_all_physical_map.to(device)

        return MyExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )
