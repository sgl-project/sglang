import torch
from sglang.srt.eplb_simulator.configs import MyServerArgs
from sglang.srt.managers.expert_distribution import (
    compute_gpu_physical_count,
    compute_utilization_rate,
)


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
        utilization_rate=utilization_rate,
        mean_utilization_rate=mean_utilization_rate,
        num_simulated_batches=logical_count_of_batch.shape[0],
    )
