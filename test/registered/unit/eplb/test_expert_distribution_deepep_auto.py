from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.eplb.expert_distribution as expert_distribution
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _metadata():
    return SimpleNamespace(
        num_layers=2,
        num_physical_experts=4,
        num_local_physical_experts=2,
    )


def _make_auto_gatherer():
    with (
        get_context().override_server_args(
            expert_distribution_recorder_mode="stat",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            elastic_ep_backend=None,
        ),
        patch.object(expert_distribution, "get_device", return_value="cpu"),
    ):
        return expert_distribution._SinglePassGatherer.init_new(_metadata(), rank=1)


def test_deepep_auto_counts_selected_experts_for_extend():
    gatherer = _make_auto_gatherer()

    gatherer.reset()
    with patch.object(expert_distribution, "get_is_extend_in_batch", return_value=True):
        gatherer.on_select_experts(0, torch.tensor([[0, 3], [3, -1]]))
    result = gatherer.collect()["global_physical_count"]

    assert torch.equal(result, torch.tensor([[1, 0, 0, 2], [0, 0, 0, 0]]))


def test_deepep_auto_counts_selected_experts_for_decode():
    gatherer = _make_auto_gatherer()

    gatherer.reset()
    with patch.object(
        expert_distribution, "get_is_extend_in_batch", return_value=False
    ):
        gatherer.on_select_experts(0, torch.tensor([[0, 3]]))
    gatherer.on_deepep_dispatch_low_latency(0, torch.tensor([2, 4]))
    result = gatherer.collect()["global_physical_count"]

    # rank=1 owns physical experts 2 and 3. The selected TopK ids above must
    # not be counted again for a low-latency pass.
    assert torch.equal(result, torch.tensor([[0, 0, 2, 4], [0, 0, 0, 0]]))


def test_deepep_auto_keeps_normal_and_low_latency_paths_separate():
    gatherer = _make_auto_gatherer()

    gatherer.reset()
    with patch.object(expert_distribution, "get_is_extend_in_batch", return_value=True):
        gatherer.on_select_experts(0, torch.tensor([[0, 3]]))
    with patch.object(
        expert_distribution, "get_is_extend_in_batch", return_value=False
    ):
        gatherer.on_select_experts(1, torch.tensor([[0, 3]]))
    gatherer.on_deepep_dispatch_low_latency(1, torch.tensor([5, 7]))

    result = gatherer.collect()["global_physical_count"]
    assert torch.equal(result, torch.tensor([[1, 0, 0, 1], [0, 0, 5, 7]]))
