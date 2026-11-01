# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.moe_token_dispatcher import (
    FixedCapacityMoeTokenDispatcher,
    create_moe_token_dispatcher,
)

_MOD = "sglang.multimodal_gen.runtime.layers.moe_token_dispatcher"


def _identity_a2a():
    return patch.object(
        FixedCapacityMoeTokenDispatcher, "_all_to_all", lambda self, tensor: tensor
    )


def _dispatcher(ep_size: int = 2, num_local_experts: int = 2):
    return FixedCapacityMoeTokenDispatcher(
        process_group=object(), ep_size=ep_size, num_local_experts=num_local_experts
    )


def test_ep4_dispatch_and_combine():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    ids = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])
    dispatcher = _dispatcher(ep_size=4, num_local_experts=2)

    with _identity_a2a():
        dispatched = dispatcher.dispatch(
            hidden_states=hidden,
            topk_weights=weights,
            global_expert_ids=ids,
        )
        combined = dispatcher.combine(
            expert_output=torch.tensor(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                    [6.0, 60.0],
                    [7.0, 70.0],
                    [8.0, 80.0],
                ]
            ),
            dispatched=dispatched,
        )

    assert torch.equal(dispatched.hidden_states, hidden.repeat(4, 1))
    assert torch.equal(dispatched.topk_weights, weights.repeat(4, 1))
    assert torch.equal(
        dispatched.local_expert_ids,
        torch.tensor(
            [
                [0, -1, -1, -1],
                [1, -1, -1, -1],
                [-1, 0, -1, -1],
                [-1, 1, -1, -1],
                [-1, -1, 0, -1],
                [-1, -1, 1, -1],
                [-1, -1, -1, 0],
                [-1, -1, -1, 1],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        combined,
        torch.tensor([[16.0, 160.0], [20.0, 200.0]]),
    )


@pytest.mark.parametrize(
    "hidden_rows, weights_shape, ids_shape, match",
    [
        (3, (3, 1), (3, 2), "same shape"),
        (3, (3,), (3,), "two-dimensional"),
        (4, (3, 2), (3, 2), "match the routed token count"),
    ],
)
def test_dispatch_rejects_malformed_routing(
    hidden_rows, weights_shape, ids_shape, match
):
    with pytest.raises(ValueError, match=match):
        _dispatcher().dispatch(
            hidden_states=torch.zeros(hidden_rows, 2),
            topk_weights=torch.zeros(weights_shape),
            global_expert_ids=torch.zeros(ids_shape, dtype=torch.long),
        )


def test_create_dispatcher_returns_none_below_ep2():
    assert create_moe_token_dispatcher(ep_size=1, num_local_experts=8) is None


def test_create_dispatcher_rejects_uninitialized_sp():
    with patch(f"{_MOD}.sequence_parallel_is_initialized", return_value=False):
        with pytest.raises(RuntimeError, match="reuses the SP group"):
            create_moe_token_dispatcher(ep_size=2, num_local_experts=4)
