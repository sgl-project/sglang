"""Independent expectations shared with the two-rank NCCL EP experiments."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.oracle import RoutingBatch, expected_combine

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_weighted_experts_restore_each_source_token():
    batch = RoutingBatch(
        tokens=(torch.tensor([[1.0, 2.0], [4.0, 8.0]]), torch.tensor([[2.0, 4.0]])),
        expert_ids=(torch.tensor([[0, 2], [1, -1]]), torch.tensor([[3, 0]])),
        weights=(
            torch.tensor([[0.25, 0.75], [1.0, 0.0]]),
            torch.tensor([[0.75, 0.25]]),
        ),
        num_experts=4,
    )
    outputs = expected_combine(batch)
    torch.testing.assert_close(
        outputs[0], torch.tensor([[2.5, 5.0], [8.0, 16.0]]), rtol=0, atol=0
    )
    torch.testing.assert_close(outputs[1], torch.tensor([[6.5, 13.0]]), rtol=0, atol=0)


def test_received_tokens_allow_reordering_and_ignore_padding():
    from nccl_ep_test.oracle import assert_dispatch_matches

    batch = RoutingBatch(
        tokens=(torch.tensor([[1.0, 2.0], [4.0, 8.0]]), torch.tensor([[2.0, 4.0]])),
        expert_ids=(torch.tensor([[0, 2], [1, -1]]), torch.tensor([[3, 0]])),
        weights=(
            torch.tensor([[0.25, 0.75], [1.0, 0.0]]),
            torch.tensor([[0.75, 0.25]]),
        ),
        num_experts=4,
    )
    received = torch.full((2, 4, 2), float("nan"))
    received[0, :2] = torch.tensor([[2.0, 4.0], [1.0, 2.0]])
    received[1, 0] = torch.tensor([4.0, 8.0])
    assert_dispatch_matches(batch, 0, received, torch.tensor([2, 1]))


def test_empty_effective_rank_still_has_fixed_capacity_and_zero_output():
    from nccl_ep_test.oracle import make_fixture

    batch = make_fixture(8, case="empty_rank", hidden=8)
    assert batch.tokens[0].shape == batch.tokens[1].shape == (8, 8)
    assert (batch.expert_ids[0] == -1).all()
    assert (batch.expert_ids[1] >= 0).all()
    torch.testing.assert_close(
        expected_combine(batch)[0], torch.zeros(8, 8), rtol=0, atol=0
    )


@pytest.mark.parametrize("corruption", ["token_order", "routing", "double_count"])
def test_combine_comparison_rejects_wrong_answers(corruption):
    from nccl_ep_test.oracle import assert_combine_matches, make_fixture

    batch = make_fixture(8, hidden=8)
    if corruption == "token_order":
        actual = expected_combine(batch)[0].roll(1, dims=0)
    elif corruption == "routing":
        wrong = RoutingBatch(
            batch.tokens,
            tuple((ids + 1) % 4 for ids in batch.expert_ids),
            batch.weights,
            4,
        )
        actual = expected_combine(wrong)[0]
    else:
        actual = expected_combine(batch)[0] * 2
    with pytest.raises(AssertionError):
        assert_combine_matches(batch, 0, actual)


def test_duplicate_probe_checks_receive_capacity_before_launch():
    from nccl_ep_test.oracle import make_fixture, validate_capacity

    batch = make_fixture(8, case="duplicates", hidden=8)
    validate_capacity(batch, 32)
    with pytest.raises(ValueError, match="Expert 0.*capacity"):
        validate_capacity(batch, 8)


def test_scaled_fp8_payload_requires_scales():
    from nccl_ep_test.oracle import dequantize_fp8

    source = torch.tensor([1.0, 2.0, 4.0, 8.0]).repeat(32).view(1, 1, 128)
    scales = torch.tensor([[[8.0 / 448.0]]])
    quantized = (source / scales).to(torch.float8_e4m3fn)
    actual = dequantize_fp8(quantized, scales)
    torch.testing.assert_close(actual, source.to(torch.bfloat16), rtol=0, atol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(quantized.float(), source, rtol=0, atol=0)


def test_dispatch_comparison_checks_tail_hidden_elements_and_multiplicity():
    from nccl_ep_test.oracle import assert_dispatch_matches

    batch = RoutingBatch(
        (torch.ones(1, 128), torch.full((1, 128), 2.0)),
        (torch.tensor([[0, 0]]), torch.tensor([[2, -1]])),
        (torch.tensor([[0.25, 0.75]]), torch.tensor([[1.0, 0.0]])),
        4,
    )
    received = torch.zeros(2, 2, 128)
    received[0] = 1
    assert_dispatch_matches(batch, 0, received, torch.tensor([2, 0]))
    received[0, 1, -1] = 2
    with pytest.raises(AssertionError, match="contents/multiplicity"):
        assert_dispatch_matches(batch, 0, received, torch.tensor([2, 0]))


def test_true_zero_length_rank_is_distinct_from_masked_padding():
    from nccl_ep_test.oracle import expected_dispatch, make_fixture, validate_capacity

    full = make_fixture(8, hidden=8)
    batch = RoutingBatch(
        (full.tokens[0][:0], full.tokens[1]),
        (full.expert_ids[0][:0], full.expert_ids[1]),
        (full.weights[0][:0], full.weights[1]),
        4,
    )
    validate_capacity(batch, 8)
    assert expected_combine(batch)[0].shape == (0, 8)
    assert sum(len(rows) for rows in expected_dispatch(batch)) == 16


def test_padding_oracle_does_not_need_zero_weights_to_mask_a_token():
    from nccl_ep_test.oracle import make_fixture

    batch = make_fixture(8, case="empty_rank", hidden=8)
    assert (batch.weights[0] > 0).all()
    torch.testing.assert_close(
        expected_combine(batch)[0], torch.zeros(8, 8), rtol=0, atol=0
    )


def test_dynamic_fixtures_separately_expose_stale_tokens_routes_and_weights():
    from nccl_ep_test.oracle import make_fixture

    # Rank 0, row 0 is [1,4], routed to experts [0,2] with [.25,.75].
    # Changing all three together can hide individual stale-data errors.
    expected = {
        "tokens": [5.0, 20.0],
        "routing": [3.5, 14.0],
        "weights": [1.5, 6.0],
    }
    for change, row in expected.items():
        batch = make_fixture(8, step=1, hidden=2, change=change)
        torch.testing.assert_close(
            expected_combine(batch)[0][0], torch.tensor(row), rtol=0, atol=0
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
