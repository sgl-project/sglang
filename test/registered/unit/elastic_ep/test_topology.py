from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from sglang.srt.arg_groups.parallel_hook import handle_elastic_ep
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    ScaleCohort,
    register_scale_cohort,
    validate_scale_cohort_topology,
)
from sglang.srt.elastic_ep.topology import (
    attn_replica_size,
    collapse_physical_rank_status,
    physical_ep_rank_to_dp_rank,
    physical_ep_size_to_dp_size,
)
from sglang.srt.managers.io_struct import ScaleElasticEPReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_attention_topology_mapping_and_complete_boundaries():
    parallel = SimpleNamespace(attn_tp_size=2, attn_cp_size=1)
    with patch("sglang.srt.elastic_ep.topology.get_parallel", return_value=parallel):
        replica_size = attn_replica_size()
    assert replica_size == 2
    assert physical_ep_size_to_dp_size(6, replica_size) == 3
    mapping = [physical_ep_rank_to_dp_rank(rank, replica_size) for rank in range(6)]
    assert mapping == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="must be divisible"):
        physical_ep_size_to_dp_size(5, replica_size)
    assert physical_ep_size_to_dp_size(5, 1) == 5


def test_scale_cohort_records_and_validates_attention_topology():
    store = MagicMock()
    joining_parallel = SimpleNamespace(attn_tp_size=2, attn_cp_size=1)
    with (
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_parallel",
            return_value=joining_parallel,
        ),
    ):
        register_scale_cohort(
            rank_offset=4,
            target_ep_size=6,
            cuda_graph_enabled=True,
        )

    key, payload = store.set.call_args.args
    assert key == "elastic_ep/scale_cohort/4"
    cohort = msgspec.json.decode(payload, type=ScaleCohort)
    assert cohort == ScaleCohort(
        target_ep_size=6,
        cuda_graph_enabled=True,
        attn_tp_size=2,
        attn_cp_size=1,
    )

    with patch(
        "sglang.srt.elastic_ep.elastic_ep.get_parallel",
        return_value=joining_parallel,
    ):
        validate_scale_cohort_topology(cohort)

    primary_parallel = SimpleNamespace(attn_tp_size=1, attn_cp_size=1)
    with (
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_parallel",
            return_value=primary_parallel,
        ),
        pytest.raises(ValueError, match="same attention topology"),
    ):
        validate_scale_cohort_topology(cohort)


def test_attention_tp_scale_requires_moe_dense_tp_one():
    cfg = SimpleNamespace(
        elastic_ep_backend="mooncake",
        enable_eplb=False,
        pp_size=1,
        mooncake_ib_device=None,
        ep_join_mode=None,
        ep_join_rank_offset=0,
        max_ep_size=8,
        tp_size=4,
        elastic_ep_initial_size=4,
        dp_size=2,
        moe_dense_tp_size=None,
    )
    resolved = SimpleNamespace(attn_cp_size=1)
    with (
        patch("sglang.srt.arg_groups.parallel_hook.resolving_view", return_value=cfg),
        patch(
            "sglang.srt.arg_groups.parallel_hook.resolved_view",
            return_value=resolved,
        ),
        patch("sglang.srt.arg_groups.parallel_hook.declare_resolution"),
        patch(
            "sglang.srt.arg_groups.validation_hook.validate_ib_devices",
            side_effect=lambda value: value,
        ),
        pytest.raises(AssertionError, match="moe-dense-tp-size 1"),
    ):
        handle_elastic_ep(SimpleNamespace())


def test_incomplete_replica_is_rejected_before_scale_request():
    parallel = SimpleNamespace(max_world_size=8, attn_tp_size=2, attn_cp_size=1)
    with (
        patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel),
        patch("sglang.srt.elastic_ep.topology.get_parallel", return_value=parallel),
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
