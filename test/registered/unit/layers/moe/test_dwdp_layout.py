from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    compute_peer_ranges,
    lookup_owner,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


@pytest.mark.parametrize("size", [2, 4, 8])
def test_equal_routed_partition_layout(size):
    layout = DwdpExpertLayout(256, size, size - 1)
    local = 256 // size
    assert layout.num_experts_per_worker == local
    assert layout.local_expert_start == 256 - local
    assert layout.local_expert_end == 256
    assert layout.peer_ranges == [
        (rank * local, (rank + 1) * local) for rank in range(size)
    ]


def test_fused_shared_experts_extend_only_last_partition():
    layouts = [
        DwdpExpertLayout(
            256,
            4,
            rank,
            num_fused_shared_experts=1,
        )
        for rank in range(4)
    ]
    assert [
        layout.local_expert_end - layout.local_expert_start for layout in layouts
    ] == [
        64,
        64,
        64,
        65,
    ]
    assert layouts[0].peer_ranges == [
        (0, 64),
        (64, 128),
        (128, 192),
        (192, 257),
    ]
    assert lookup_owner(256, layouts[0].peer_ranges) == 3


def test_lookup_owner_rejects_uncovered_expert():
    ranges = compute_peer_ranges(
        dwdp_size=2,
        num_experts_per_worker=2,
        num_prefetch_experts=2,
        num_experts_total=4,
    )
    with pytest.raises(ValueError, match="not owned"):
        lookup_owner(4, ranges)


def test_fused_moe_dwdp_topology_round_trip():
    layer = FusedMoE.__new__(FusedMoE)
    mapping = object()
    mask = object()
    layer.moe_ep_size = 4
    layer.moe_ep_rank = 2
    layer._num_global_routed = 16
    layer._num_local_routed = 4
    layer.num_experts = 5
    layer.num_local_experts = 5
    layer.moe_runner_config = SimpleNamespace(num_local_experts=5)
    layer.dispatcher = SimpleNamespace(
        moe_ep_size=4,
        moe_ep_rank=2,
        num_local_experts=5,
        num_local_routed_experts=4,
        local_expert_mapping=mapping,
        expert_mask_gpu=mask,
    )
    layer._dwdp_bound = False
    layer._dwdp_original_topology = None

    layer.bind_dwdp_partitioned_weights()
    assert layer.moe_ep_size == 1
    assert layer._num_local_routed == 16

    layer.unbind_dwdp_weights()
    assert layer.moe_ep_size == 4
    assert layer.moe_ep_rank == 2
    assert layer._num_local_routed == 4
    assert layer.dispatcher.local_expert_mapping is mapping
    assert layer.dispatcher.expert_mask_gpu is mask
    assert layer._dwdp_bound is False
