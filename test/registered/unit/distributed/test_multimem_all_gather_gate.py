from types import SimpleNamespace

import pytest

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("nnodes", "same_node", "one_clique", "enabled"),
    [
        (1, True, None, True),
        (2, True, None, True),
        (2, False, True, True),
        (2, False, False, False),
    ],
)
def test_cross_node_gate(monkeypatch, nnodes, same_node, one_clique, enabled):
    world_size = 8
    tp_group = SimpleNamespace(world_size=world_size, cpu_group=object())
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        lambda group, source_rank: [same_node] * world_size,
    )
    monkeypatch.setattr(
        triton_symm_mem_ag, "get_parallel", lambda: SimpleNamespace(nnodes=nnodes)
    )

    def is_one_clique(cpu_group):
        if same_node:
            pytest.fail("a group on one node needs no fabric clique query")
        assert cpu_group is tp_group.cpu_group
        return one_clique

    monkeypatch.setattr(triton_symm_mem_ag, "_is_one_nvlink_clique", is_one_clique)

    gatherer = triton_symm_mem_ag.MultimemAllGatherer(max_tokens=8)

    assert (gatherer._state is not None) is enabled
