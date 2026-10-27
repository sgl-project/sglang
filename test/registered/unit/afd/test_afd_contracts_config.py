"""Essential topology, S2 partition and fail-closed argument contracts."""

import pytest

from sglang.srt.afd import config, contracts
from sglang.srt.afd.model_adapters import base as adapter
from sglang.test.afd.config_fixtures import _server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "case", ["missing", "order", "wire-rank", "world-rank", "duplicate-lane", "width"]
)
def test_topology_negative_matrix_fails_before_startup(case):
    from msgspec.structs import replace

    lanes = 2 if case == "duplicate-lane" else 1
    endpoints = list(contracts.AFDPairedTopology.paired(lanes=lanes).endpoints)
    if case == "missing":
        endpoints = endpoints[1:]
    elif case == "order":
        endpoints.reverse()
    elif case == "wire-rank":
        endpoints = [replace(e, transport_rank=1 - e.transport_rank) for e in endpoints]
    elif case == "world-rank":
        endpoints[1] = replace(endpoints[1], coordination_rank=0)
    elif case == "duplicate-lane":
        endpoints[1] = replace(endpoints[1], ordinal=0)
    else:
        endpoints = contracts.AFDPairedTopology.paired(lanes=2).endpoints
    topology = contracts.AFDPairedTopology(endpoints=tuple(endpoints), lanes=lanes)
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_PAIRED_LAYOUT_REQUIRED"):
        topology.validate()


@pytest.mark.parametrize("lanes", [0, -1, True, 1.0, "2"])
def test_topology_rejects_non_positive_lane_counts(lanes):
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_LANE_COUNT_INVALID"):
        contracts.AFDPairedTopology.paired(lanes=lanes)


@pytest.mark.parametrize(
    "ffn_lanes,attention_lanes",
    [(1, 1), (2, 2), (4, 4), (8, 8), (4, 8), (4, 12), (4, 16), (2, 8), (1, 4)],
)
def test_lane_group_fanin_routes_k_attention_lanes_into_one_ffn_rank(
    ffn_lanes, attention_lanes
):
    group = attention_lanes // ffn_lanes
    topology = contracts.AFDPairedTopology.paired(
        lanes=ffn_lanes,
        attention_lanes=attention_lanes,
    )
    topology.validate()
    assert (topology.ffn_size, topology.attention_size) == (ffn_lanes, attention_lanes)
    assert topology.expects_dp_attention(role=contracts.AFDRole.FFN) is False
    assert topology.expects_dp_attention(role=contracts.AFDRole.ATTENTION) is (
        attention_lanes > 1
    )
    assert topology.lanes_per_ffn == group
    assert topology.coordination_world_size == ffn_lanes + attention_lanes
    assert topology.pair_world_size == 1 + group
    assert topology.expected_parallelism(role=contracts.AFDRole.ATTENTION) == (
        attention_lanes,
        attention_lanes,
        1,
        1,
    )
    assert topology.expected_parallelism(role=contracts.AFDRole.FFN) == (
        ffn_lanes,
        1,
        ffn_lanes,
        1,
    )
    for ordinal in range(attention_lanes):
        endpoint = topology.local(role=contracts.AFDRole.ATTENTION, ordinal=ordinal)
        assert endpoint.transport_rank == 1 + ordinal % group
        assert endpoint.coordination_rank == ffn_lanes + ordinal
        assert (
            topology.group_ordinal(role=contracts.AFDRole.ATTENTION, ordinal=ordinal)
            == ordinal // group
        )
        assert topology.peers(role=contracts.AFDRole.ATTENTION, ordinal=ordinal) == (
            topology.local(role=contracts.AFDRole.FFN, ordinal=ordinal // group),
        )
    for ordinal in range(ffn_lanes):
        endpoint = topology.local(role=contracts.AFDRole.FFN, ordinal=ordinal)
        assert (endpoint.transport_rank, endpoint.coordination_rank) == (0, ordinal)
        assert (
            topology.group_ordinal(role=contracts.AFDRole.FFN, ordinal=ordinal)
            == ordinal
        )
        assert topology.attention_lane_group(ffn_ordinal=ordinal) == tuple(
            range(ordinal * group, (ordinal + 1) * group)
        )
        peers = topology.peers(role=contracts.AFDRole.FFN, ordinal=ordinal)
        assert peers == tuple(
            topology.local(role=contracts.AFDRole.ATTENTION, ordinal=lane)
            for lane in topology.attention_lane_group(ffn_ordinal=ordinal)
        )
        assert tuple(peer.transport_rank for peer in peers) == tuple(
            range(1, group + 1)
        )
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_ROLE_IDENTITY_INVALID"):
        topology.local(role=contracts.AFDRole.FFN, ordinal=ffn_lanes)
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_ROLE_IDENTITY_INVALID"):
        topology.local(role=contracts.AFDRole.ATTENTION, ordinal=attention_lanes)


@pytest.mark.parametrize(
    "batch_size,stages,expected",
    [(8, 2, ((0, 4, 0, 4), (4, 8, 4, 8))), (5, 2, ((0, 3, 0, 4), (3, 5, 4, 10)))],
)
def test_s2_request_and_token_ranges_are_contiguous(batch_size, stages, expected):
    token_lengths = (1,) * batch_size if batch_size == 8 else (1, 2, 1, 2, 4)
    actual = adapter._stage_ranges(
        batch_size=batch_size,
        token_lengths=token_lengths,
        stages=stages,
    )
    assert actual == expected
    assert tuple(item[0] for item in actual) == (0,) + tuple(
        item[1] for item in actual[:-1]
    )
    assert tuple(item[2] for item in actual) == (0,) + tuple(
        item[3] for item in actual[:-1]
    )


@pytest.mark.parametrize(
    "stages,expected",
    [(2, ((0, 1, 0, 8), (1, 1, 8, 8)))],
)
def test_single_request_stage_ranges_keep_empty_tail_stages(stages, expected):
    assert (
        adapter._stage_ranges(
            batch_size=1,
            token_lengths=(8,),
            stages=stages,
        )
        == expected
    )


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"tp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"dp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"ep_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"pp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
        ({"nnodes": 2}, "AFD_TOPOLOGY_ROLE_NODE_SPLIT_UNSUPPORTED"),
        ({"enable_two_batch_overlap": True}, "AFD_TBO_UNSUPPORTED"),
        (
            {"enable_single_batch_overlap": True},
            "AFD_SINGLE_BATCH_OVERLAP_UNSUPPORTED",
        ),
        ({"moe_a2a_backend": "deepep"}, "AFD_INTERNAL_COLLECTIVE_UNSUPPORTED"),
        ({"speculative_algorithm": "EAGLE"}, "AFD_MTP_SPECULATIVE_UNSUPPORTED"),
        ({"attn_dp_size": 2}, "AFD_TOPOLOGY_PARALLELISM_MISMATCH"),
    ],
)
def test_startup_exclusions_fail_closed(overrides, code):
    with pytest.raises(contracts.AFDError, match=code):
        config.validate_afd_server_args(_server_args(**overrides))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
