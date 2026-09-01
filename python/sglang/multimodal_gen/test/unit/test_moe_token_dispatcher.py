# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.moe_token_dispatcher import (
    FixedCapacityMoeDispatchedTokens,
    FixedCapacityMoeTokenDispatcher,
    create_moe_token_dispatcher,
)

_MOD = "sglang.multimodal_gen.runtime.layers.moe_token_dispatcher"


# Identity isolates routing and reshape logic from the collective.
def _identity_a2a():
    return patch.object(
        FixedCapacityMoeTokenDispatcher, "_all_to_all", lambda self, tensor: tensor
    )


def _dispatcher(ep_size: int = 2, num_local_experts: int = 2):
    return FixedCapacityMoeTokenDispatcher(
        process_group=object(), ep_size=ep_size, num_local_experts=num_local_experts
    )


def test_dispatch_masks_non_local_routes_and_repeats_payload():
    # 4 experts across ep_size=2: rank 0 owns {0,1}, rank 1 owns {2,3}.
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]])
    ids = torch.tensor([[0, 3], [1, 2], [2, 0]])

    with _identity_a2a():
        out = _dispatcher().dispatch(
            hidden_states=hidden, topk_weights=weights, global_expert_ids=ids
        )

    assert torch.equal(out.hidden_states, hidden.repeat(2, 1))
    assert torch.equal(out.topk_weights, weights.repeat(2, 1))
    # Local ids per destination block; routes owned by the other rank become -1.
    expected_ids = torch.tensor(
        [
            [0, -1],
            [1, -1],
            [-1, 0],  # destination rank 0 (experts 0,1)
            [-1, 1],
            [-1, 0],
            [0, -1],  # destination rank 1 (experts 2,3 -> local 0,1)
        ],
        dtype=torch.int32,
    )
    assert torch.equal(out.local_expert_ids, expected_ids)
    assert out.num_source_tokens == 3


def test_combine_reshapes_and_sums_across_ranks():
    dispatched = FixedCapacityMoeDispatchedTokens(
        hidden_states=torch.empty(0),
        topk_weights=torch.empty(0),
        local_expert_ids=torch.empty(0),
        num_source_tokens=3,
    )
    expert_output = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [30.0, 30.0],
        ]
    )

    with _identity_a2a():
        out = _dispatcher().combine(expert_output=expert_output, dispatched=dispatched)

    assert torch.equal(out, torch.tensor([[11.0, 11.0], [22.0, 22.0], [33.0, 33.0]]))


def test_ep4_dispatch_and_combine():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    )
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


def test_dispatch_rejects_mismatched_ids_and_weights():
    with pytest.raises(ValueError, match="same shape"):
        _dispatcher().dispatch(
            hidden_states=torch.zeros(3, 2),
            topk_weights=torch.zeros(3, 1),
            global_expert_ids=torch.zeros(3, 2, dtype=torch.long),
        )


def test_dispatch_rejects_one_dimensional_routing():
    with pytest.raises(ValueError, match="two-dimensional"):
        _dispatcher().dispatch(
            hidden_states=torch.zeros(3, 2),
            topk_weights=torch.zeros(3),
            global_expert_ids=torch.zeros(3, dtype=torch.long),
        )


def test_dispatch_rejects_hidden_state_row_mismatch():
    with pytest.raises(ValueError, match="match the routed token count"):
        _dispatcher().dispatch(
            hidden_states=torch.zeros(4, 2),
            topk_weights=torch.zeros(3, 2),
            global_expert_ids=torch.zeros(3, 2, dtype=torch.long),
        )


def test_create_dispatcher_returns_none_below_ep2():
    assert create_moe_token_dispatcher(ep_size=1, num_local_experts=8) is None


def test_create_dispatcher_rejects_uninitialized_sp():
    with patch(f"{_MOD}.sequence_parallel_is_initialized", return_value=False):
        with pytest.raises(RuntimeError, match="reuses the SP group"):
            create_moe_token_dispatcher(ep_size=2, num_local_experts=4)


def test_create_dispatcher_rejects_sp_size_mismatch():
    fake_group = SimpleNamespace(world_size=4, device_group=object())
    with (
        patch(f"{_MOD}.sequence_parallel_is_initialized", return_value=True),
        patch(f"{_MOD}.get_sp_group", return_value=fake_group),
    ):
        with pytest.raises(RuntimeError, match="SP world size"):
            create_moe_token_dispatcher(ep_size=2, num_local_experts=4)
