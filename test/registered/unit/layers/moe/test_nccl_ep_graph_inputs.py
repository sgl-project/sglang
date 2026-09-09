"""Independent expectations shared with the two-rank NCCL EP experiments."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires a GPU")
def test_non_ep_graph_replays_changed_data_at_fixed_addresses():
    from nccl_ep_test.local_graph import exercise

    result = exercise()
    assert result["ep_tested"] is False
    assert result["checked"] == 180


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires a GPU")
def test_bucket_graphs_reuse_maximum_capacity_storage_serially():
    from nccl_ep_test.local_graph import exercise

    result = exercise()
    views = result["shared_buffer_views"]
    assert views["8"] == views["16"] == views["32"]
    assert result["checked"] == 180


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires a GPU")
def test_sglang_runner_updates_and_pads_the_actual_graph_inputs():
    from nccl_ep_test.runner_inputs import exercise_inputs

    result = exercise_inputs()
    assert result["selected_buckets"] == [8, 16, 8]
    assert result["valid_rows"] == [5, 0, 5]
    assert result["recapture_generations"] == 1
    assert result["ep_tested"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires a GPU")
@pytest.mark.parametrize("routing_dtype", [torch.int32, torch.int64])
def test_real_sglang_topk_mask_and_fp8_scales_change_on_graph_replay(routing_dtype):
    from nccl_ep_test.sglang_non_ep import exercise_topk_fp8

    result = exercise_topk_fp8(routing_dtype=routing_dtype)
    assert result["valid_rows"] == [5, 0, 5]
    assert result["scales_applied"] is True
    assert result["ep_tested"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Real CUDA routing mask required"
)
def test_server_runner_router_matches_literal_two_rank_fixtures():
    from nccl_ep_test.runner_inputs import input_batch
    from nccl_ep_test.sglang_graph import runner_fixture, runner_routing

    values = ([1, 2, 4, 8, 1], [8, 4, 2, 1, 8])
    for valid in ((5, 5), (0, 5), (2, 3)):
        fixture = runner_fixture(8, values, valid)
        for rank in range(2):
            batch = input_batch(values[rank] + [0, 0, 0], valid_rows=valid[rank])
            x, ids, weights = runner_routing(batch, rank)
            torch.testing.assert_close(x.cpu(), fixture.tokens[rank], rtol=0, atol=0)
            torch.testing.assert_close(
                ids.cpu().long(), fixture.expert_ids[rank], rtol=0, atol=0
            )
            torch.testing.assert_close(
                weights.cpu(), fixture.weights[rank], rtol=0, atol=0
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
