import pytest
import torch
from sgl_kernel import balance_topk_ids


@pytest.mark.parametrize("num_gpus", [8, 16, 32])
@pytest.mark.parametrize(
    "num_logical_experts, num_physical_experts",
    [(32, 32), (32, 64), (32, 128), (256, 512), (256, 1024)],
)
@pytest.mark.parametrize(
    "input_shape", [(1, 1), (11,), (1024, 8), (377,), (875, 5), (14000, 8)]
)
def test_balance_topk_ids(
    num_gpus, num_logical_experts, num_physical_experts, input_shape
):
    assert num_logical_experts >= num_gpus
    assert num_logical_experts % num_gpus == 0
    assert num_physical_experts % num_logical_experts == 0

    topk_ids = torch.randint(
        low=0,
        high=num_logical_experts,
        size=input_shape,
        device="cuda:0",
        dtype=torch.int32,
    )

    max_workload_after_balance = torch.empty([1], dtype=torch.int32, device="cuda:0")
    new_topk_ids = torch.empty_like(topk_ids)
    gpu_workloads_balance_mapping = torch.empty(
        [num_physical_experts // num_logical_experts, num_gpus],
        dtype=torch.int32,
        device="cuda:0",
    )
    balance_topk_ids(
        topk_ids,
        num_gpus,
        num_logical_experts,
        num_physical_experts,
        max_workload_after_balance,
        gpu_workloads_balance_mapping,
        new_topk_ids,
    )

    logical_expert_workloads = torch.bincount(
        topk_ids.flatten(), minlength=num_logical_experts
    )
    origin_gpu_workload = logical_expert_workloads.view(num_gpus, -1).sum(1)
    torch.testing.assert_close(
        gpu_workloads_balance_mapping.sum(0), origin_gpu_workload
    )

    num_copy = num_physical_experts // num_logical_experts
    expectd_new_gpu_workload = torch.zeros(
        [num_gpus], dtype=torch.int32, device="cuda:0"
    )
    for i in range(num_copy):
        workload = gpu_workloads_balance_mapping[i]
        workload = torch.roll(workload, shifts=-i)
        expectd_new_gpu_workload += workload

    new_gpu_workload = (
        torch.bincount(new_topk_ids.flatten(), minlength=num_physical_experts)
        .view(num_gpus, -1)
        .sum(1)
        .to(torch.int32)
    )

    assert (
        new_gpu_workload.max() <= max_workload_after_balance.item()
    ), f"{new_gpu_workload.max()=}, {max_workload_after_balance.item()=}"

    torch.testing.assert_close(new_gpu_workload, expectd_new_gpu_workload)

    physical_to_logical_map = []
    num_physical_experts_each_gpu = num_physical_experts // num_gpus
    num_logical_experts_each_gpu = num_logical_experts // num_gpus
    for i in range(0, num_gpus):
        start_idx = i * num_logical_experts_each_gpu
        for logical_expert_id in range(
            start_idx, start_idx + num_physical_experts_each_gpu
        ):
            physical_to_logical_map.append(logical_expert_id % num_logical_experts)

    physical_to_logical_map = torch.tensor(
        physical_to_logical_map, dtype=torch.int32, device="cuda:0"
    )
    expected_old_topk_ids = physical_to_logical_map[new_topk_ids]
    torch.testing.assert_close(topk_ids, expected_old_topk_ids)


if __name__ == "__main__":
    pytest.main([__file__])
