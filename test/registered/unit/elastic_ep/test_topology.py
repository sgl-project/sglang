from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.elastic_ep.topology import (
    collapse_physical_rank_status,
    derive_attn_tp_size,
    physical_ep_rank_to_dp_rank,
    physical_ep_size_to_dp_size,
)
from sglang.srt.managers.io_struct import ScaleElasticEPReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_attention_topology_mapping_and_complete_boundaries():
    attn_tp_size = derive_attn_tp_size(tp_size=4, dp_size=2, attn_cp_size=1)
    assert attn_tp_size == 2
    attn_replica_size = attn_tp_size
    assert physical_ep_size_to_dp_size(6, attn_replica_size) == 3
    mapping = [
        physical_ep_rank_to_dp_rank(rank, attn_replica_size) for rank in range(6)
    ]
    assert mapping == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="must be divisible"):
        physical_ep_size_to_dp_size(5, attn_replica_size)
    assert physical_ep_size_to_dp_size(5, 1) == 5


def test_incomplete_replica_is_rejected_before_scale_request():
    parallel = SimpleNamespace(max_ep_size=8, attn_tp_size=2, attn_cp_size=1)
    with (
        patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel),
        patch.object(ElasticEPStateManager, "get_effective_ep_size", return_value=4),
        patch.object(ElasticEPStateManager, "request_scale") as request_scale,
    ):
        result = Scheduler.handle_scale_elastic_ep(
            SimpleNamespace(), ScaleElasticEPReqInput(new_ep_size=5)
        )

    assert not result.success
    assert "incomplete attention replica" in result.message
    request_scale.assert_not_called()


def test_physical_health_uses_all_members_rule():
    assert collapse_physical_rank_status(
        [True, True, True, True, True, False], attn_replica_size=2
    ) == [True, True, False]
    assert collapse_physical_rank_status([True, False, True], attn_replica_size=1) == [
        True,
        False,
        True,
    ]
