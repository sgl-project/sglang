# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from sglang.srt.layers.mova import (
    RoutedValueExperts,
    _prepare_mova_moe_config,
    mova_router_topk,
    routed_linear,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_router_bias_changes_selection_but_not_mixture_weights():
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float32)

    weights, selected = mova_router_topk(
        logits,
        bias,
        score_func="sigmoid",
        top_k=1,
        scaling_factor=2.5,
    )

    assert selected.tolist() == [[2]]
    torch.testing.assert_close(weights, torch.sigmoid(logits[:, 2:3]) * 2.5)


def test_router_renormalizes_before_scaling():
    logits = torch.tensor([[2.0, 1.0, -2.0]], dtype=torch.float32)
    weights, selected = mova_router_topk(
        logits,
        None,
        score_func="sigmoid",
        top_k=2,
        scaling_factor=2.5,
    )

    scores = torch.sigmoid(logits)
    expected_ids = torch.topk(scores, 2, dim=-1).indices
    expected = torch.gather(scores, -1, expected_ids)
    expected = expected / expected.sum(-1, keepdim=True) * 2.5
    torch.testing.assert_close(selected, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected)


def test_softmax_router_bias_is_selection_only():
    logits = torch.tensor([[3.0, 2.0, -4.0]], dtype=torch.float32)
    bias = torch.tensor([0.0, 0.0, 20.0], dtype=torch.float32)

    weights, selected = mova_router_topk(
        logits,
        bias,
        score_func="softmax",
        top_k=1,
        scaling_factor=1.0,
    )

    unbiased_scores = torch.softmax(logits, dim=-1)
    assert selected.tolist() == [[2]]
    torch.testing.assert_close(weights, unbiased_scores[:, 2:3])


def test_routed_linear_cpu_matches_independent_expected_math():
    hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    expert_weights = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    selected = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])

    expected_rows = []
    for token, expert_ids, mixture in zip(hidden, selected, weights):
        projections = torch.stack(
            [
                torch.nn.functional.silu(expert_weights[expert_id] @ token)
                for expert_id in expert_ids.tolist()
            ]
        )
        expected_rows.append((projections * mixture[:, None]).sum(dim=0))

    expected = torch.stack(expected_rows)
    actual = routed_linear(hidden, expert_weights, weights, selected)
    torch.testing.assert_close(actual, expected)


def test_value_expert_loader_shards_output_dimension():
    experts = RoutedValueExperts(
        num_experts=2,
        input_size=3,
        output_size=4,
        tp_rank=1,
        tp_size=2,
    )
    packed = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    experts.weight_loader(experts.weight, packed)
    torch.testing.assert_close(experts.weight, packed[:, 2:])

    single = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 100
    experts.weight_loader(experts.weight, single, loaded_shard_id=0)
    torch.testing.assert_close(experts.weight[0], single[2:])


def test_mova_moe_config_drops_unsupported_tma_without_mutating_source():
    source = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "USE_TMA": True,
    }

    prepared = _prepare_mova_moe_config(source)

    assert "USE_TMA" not in prepared
    assert source["USE_TMA"] is True


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
