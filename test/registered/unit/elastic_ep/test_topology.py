import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch

from sglang.srt.arg_groups.parallel_hook import handle_elastic_ep
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    RecoveryOperation,
    ScaleCohort,
    get_membership_observation,
    get_recovery_operation_for_slot,
    observe_recovery_membership,
    poll_recovery_control_plane,
    pre_dispatch_recovery_fence,
    register_scale_cohort,
    validate_post_forward_recovery_generation,
    validate_scale_cohort_topology,
)
from sglang.srt.elastic_ep.runtime_topology import (
    MembershipObservation,
    RuntimeTopology,
    publish_membership_observation,
)
from sglang.srt.elastic_ep.topology import (
    attn_replica_size,
    collapse_physical_rank_status,
    physical_ep_rank_to_dp_rank,
    physical_ep_size_to_dp_size,
)
from sglang.srt.managers.io_struct import (
    ObserveElasticEPMembershipReqInput,
    ScaleElasticEPReqInput,
)
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
        attn_dp_size=2,
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


class _MembershipStore:
    def __init__(self):
        self.values = {}

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]

    def set(self, key, value):
        self.values[key] = value

    def compare_set(self, key, expected, value):
        current = self.values.get(key, expected)
        if current == expected:
            self.values[key] = value
        return self.values[key]


def test_membership_observation_accepts_replay_and_rejects_stale_aba_epochs():
    store = _MembershipStore()
    topology = RuntimeTopology(
        runtime_instance_id="runtime/1",
        initial_ep_size=2,
        allocation_width=1,
        effective_ep_size=2,
        max_committed_ep_size=2,
        topology_generation=4,
    )
    observation = MembershipObservation(
        runtime_instance_id="runtime/1",
        topology_generation=4,
        rank=1,
        failed_allocation_incarnation="allocation/7",
        membership_epoch=9,
    )
    with (
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_runtime_topology",
            return_value=topology,
        ),
    ):
        observation_key = publish_membership_observation(observation)
        assert publish_membership_observation(observation) == observation_key
        with pytest.raises(ValueError, match="stale|ABA"):
            publish_membership_observation(replace(observation, membership_epoch=8))
        replacement = replace(
            observation,
            failed_allocation_incarnation="allocation/8",
            membership_epoch=10,
        )
        replacement_key = publish_membership_observation(replacement)
        assert get_membership_observation() == replacement
        assert replacement_key != observation_key
        with pytest.raises(ValueError, match="stale|ABA"):
            publish_membership_observation(replace(replacement, membership_epoch=9))
        with pytest.raises(ValueError, match="stale|ABA"):
            publish_membership_observation(
                replace(
                    observation,
                    membership_epoch=10,
                )
            )
        assert get_membership_observation() == replacement
        with pytest.raises(ValueError, match="stale|ABA"):
            publish_membership_observation(observation)
        with pytest.raises(ValueError, match="stale|ABA"):
            publish_membership_observation(
                replace(
                    observation,
                    failed_allocation_incarnation="allocation/7",
                    membership_epoch=11,
                )
            )


def test_expired_membership_observation_does_not_publish_state():
    store = _MembershipStore()
    topology = RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=2,
        allocation_width=1,
        effective_ep_size=2,
        max_committed_ep_size=2,
        topology_generation=4,
    )
    observation = MembershipObservation(
        runtime_instance_id="runtime-1",
        topology_generation=4,
        rank=1,
        failed_allocation_incarnation="allocation-7",
        membership_epoch=9,
    )
    with (
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_runtime_topology",
            return_value=topology,
        ),
        pytest.raises(TimeoutError, match="no state was changed"),
    ):
        publish_membership_observation(
            observation, deadline_unix_seconds=time.time() - 1
        )

    assert store.values == {}


def test_membership_publication_serializes_duplicate_interleavings():
    store = _MembershipStore()
    topology = RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=2,
        allocation_width=1,
        effective_ep_size=2,
        max_committed_ep_size=2,
    )
    observation = MembershipObservation(
        runtime_instance_id="runtime-1",
        topology_generation=0,
        rank=1,
        failed_allocation_incarnation="allocation-a",
        membership_epoch=1,
    )
    with (
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_runtime_topology",
            return_value=topology,
        ),
    ):

        def publish_once():
            try:
                return ("accepted", publish_membership_observation(observation))
            except ValueError as exc:
                return ("rejected", str(exc))

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [
                future.result()
                for future in (
                    executor.submit(publish_once),
                    executor.submit(publish_once),
                )
            ]

    assert [result[0] for result in results].count("accepted") == 2
    assert results[0][1] == results[1][1]


def test_membership_endpoint_only_publishes_until_scheduler_safe_boundary():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._send_membership_observation_progress = MagicMock()
    request = ObserveElasticEPMembershipReqInput(
        runtime_instance_id="runtime-1",
        topology_generation=0,
        rank=1,
        failed_allocation_incarnation="allocation-a",
        membership_epoch=1,
        submission_id="submission-1",
    )
    with (
        patch(
            "sglang.srt.elastic_ep.elastic_ep.observe_recovery_membership",
            return_value="observation-key",
        ),
        patch("sglang.srt.elastic_ep.elastic_ep.poll_recovery_control_plane") as poll,
        patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(dp_rank=0),
        ),
    ):
        result = scheduler.handle_observe_elastic_ep_membership(request)

    assert result.accepted
    assert result.observation_key == "observation-key"
    assert result.submission_id == "submission-1"
    assert result.responder_rank == 0
    assert scheduler._send_membership_observation_progress.call_count == 3
    poll.assert_not_called()


def test_membership_admission_precedes_recovery_and_legacy_stays_unfenced():
    store = _MembershipStore()
    topology = RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=2,
        allocation_width=1,
        effective_ep_size=2,
        max_committed_ep_size=2,
    )
    state = ElasticEPState(
        active_ranks=torch.tensor([1, 0]),
        last_active_ranks=torch.tensor([1, 0]),
        active_ranks_cpu=torch.tensor([1, 0]),
        effective_ep_size=2,
    )
    with (
        patch.object(ElasticEPStateManager, "_instance", state),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(elastic_ep_recovery_policy="fixed_slot")
            ),
        ),
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_global_tcp_store",
            return_value=store,
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_runtime_topology",
            return_value=topology,
        ),
        patch(
            "sglang.srt.elastic_ep.runtime_topology.get_runtime_topology",
            return_value=topology,
        ),
    ):
        assert state.recovery_phase == "idle"
        fence = pre_dispatch_recovery_fence()
        assert fence is not None
        assert fence.topology_generation == 0
        observe_recovery_membership(
            runtime_instance_id="runtime-1",
            topology_generation=0,
            rank=1,
            failed_allocation_incarnation="inc-1",
            membership_epoch=1,
        )
        assert state.recovery_phase == "awaiting_recovery"
        assert state.active_ranks_cpu[1].item() == 0
        observe_recovery_membership(
            runtime_instance_id="runtime-1",
            topology_generation=0,
            rank=1,
            failed_allocation_incarnation="inc-1",
            membership_epoch=1,
        )
        with pytest.raises(ValueError, match="stale|ABA"):
            observe_recovery_membership(
                runtime_instance_id="runtime-1",
                topology_generation=0,
                rank=1,
                failed_allocation_incarnation="inc-1",
                membership_epoch=0,
            )
        assert not poll_recovery_control_plane()
        assert state.recovery_phase == "awaiting_recovery"
        assert state.active_ranks_cpu[1].item() == 0
        observe_recovery_membership(
            runtime_instance_id="runtime-1",
            topology_generation=0,
            rank=1,
            failed_allocation_incarnation="inc-1",
            membership_epoch=1,
        )
        with pytest.raises(ValueError, match="awaiting"):
            observe_recovery_membership(
                runtime_instance_id="runtime-1",
                topology_generation=0,
                rank=0,
                failed_allocation_incarnation="inc-1",
                membership_epoch=2,
            )
        assert ElasticEPStateManager.request_recovery(
            RecoveryOperation(
                runtime_instance_id="runtime-1",
                topology_generation=0,
                operation_id="recover-1",
                allocation_id="alloc-1",
                rank_offset=1,
                failed_allocation_incarnation="inc-1",
            )
        )
        assert state.recovery_phase == "restoring"
        assert (
            get_recovery_operation_for_slot(
                "runtime-1", 0, 1, "alloc-1"
            ).failed_allocation_incarnation
            == "inc-1"
        )
        assert (
            get_recovery_operation_for_slot(
                "runtime-1", 0, 1, "alloc-1", "other-incarnation"
            )
            is None
        )
        state.recovery_phase = "warming_up"
        state.pending_since = time.monotonic()
        with pytest.raises(RuntimeError, match="user forward"):
            pre_dispatch_recovery_fence()
        fence = pre_dispatch_recovery_fence(internal_warmup=True)
        assert fence.topology_generation == 0
        state.pending_since = time.monotonic() - 1000
        with patch(
            "sglang.srt.elastic_ep.elastic_ep.get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(
                    elastic_ep_recovery_policy="fixed_slot",
                    elastic_ep_recovery_timeout=1,
                )
            ),
        ):
            assert poll_recovery_control_plane()
        assert state.recovery_phase == "failed"
        assert state.runtime_health == "recovery_unsupported"

    with patch(
        "sglang.srt.elastic_ep.elastic_ep.get_exec",
        return_value=SimpleNamespace(moe=SimpleNamespace()),
    ):
        assert not pre_dispatch_recovery_fence()


def test_idle_forward_fence_rejects_membership_observed_mid_forward():
    topology = RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=2,
        allocation_width=1,
        effective_ep_size=2,
        max_committed_ep_size=2,
    )
    state = ElasticEPState(
        active_ranks=torch.tensor([1, 1]),
        last_active_ranks=torch.tensor([1, 1]),
        active_ranks_cpu=torch.tensor([1, 1]),
        effective_ep_size=2,
    )
    observation = MembershipObservation(
        runtime_instance_id="runtime-1",
        topology_generation=0,
        rank=1,
        failed_allocation_incarnation="inc-1",
        membership_epoch=1,
    )
    with (
        patch.object(ElasticEPStateManager, "_instance", state),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(elastic_ep_recovery_policy="fixed_slot")
            ),
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_runtime_topology",
            return_value=topology,
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_membership_observation",
            side_effect=[None, None, observation],
        ),
    ):
        fence = pre_dispatch_recovery_fence()
        assert fence is not None
        assert fence.membership_observation is None
        with pytest.raises(RuntimeError, match="membership changed"):
            validate_post_forward_recovery_generation(fence)


def test_ready_recovery_allows_user_forward_and_fences_membership_change():
    topology = RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=2,
        allocation_width=1,
        effective_ep_size=2,
        max_committed_ep_size=2,
    )
    state = ElasticEPState(
        active_ranks=torch.tensor([1, 1]),
        last_active_ranks=torch.tensor([1, 1]),
        active_ranks_cpu=torch.tensor([1, 1]),
        effective_ep_size=2,
        recovery_phase="ready",
    )
    observation = MembershipObservation(
        runtime_instance_id="runtime-1",
        topology_generation=0,
        rank=1,
        failed_allocation_incarnation="inc-1",
        membership_epoch=1,
    )
    with (
        patch.object(ElasticEPStateManager, "_instance", state),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(elastic_ep_recovery_policy="fixed_slot")
            ),
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_runtime_topology",
            return_value=topology,
        ),
        patch(
            "sglang.srt.elastic_ep.elastic_ep.get_membership_observation",
            return_value=observation,
        ),
    ):
        fence = pre_dispatch_recovery_fence()
        assert fence.topology_generation == 0
        validate_post_forward_recovery_generation(fence)
        with patch(
            "sglang.srt.elastic_ep.elastic_ep.get_membership_observation",
            return_value=replace(observation, membership_epoch=2),
        ):
            with pytest.raises(RuntimeError, match="membership changed"):
                validate_post_forward_recovery_generation(fence)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
