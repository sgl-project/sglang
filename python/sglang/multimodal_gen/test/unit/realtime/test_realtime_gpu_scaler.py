# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.entrypoints.realtime_gpu_scaler import (
    CapacityRoleSnapshot,
    CapacityScaler,
    CoordinatorCapacityClient,
    ScalingPolicy,
    KubernetesScaleClient,
)


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"status":"ok"}'


def test_scaler_patches_only_the_named_deployment_scale_subresource(tmp_path):
    token = tmp_path / "token"
    token.write_text("service-account-token")
    calls = []

    def open_request(request, **kwargs):
        calls.append((request, kwargs))
        return _Response()

    client = KubernetesScaleClient(
        host="kubernetes.default.svc",
        port=443,
        namespace="minwm-realtime",
        token_path=token,
        ca_path=Path("/missing-test-ca"),
        opener=open_request,
        ssl_context=object(),
    )
    client.scale("minwm-async-vae", 3)

    request, kwargs = calls[0]
    assert request.full_url.endswith(
        "/apis/apps/v1/namespaces/minwm-realtime/"
        "deployments/minwm-async-vae/scale"
    )
    assert request.method == "PATCH"
    assert request.headers["Authorization"] == "Bearer service-account-token"
    assert json.loads(request.data) == {"spec": {"replicas": 3}}
    assert kwargs["timeout"] == 10


def test_scaler_uses_the_statefulset_scale_subresource_for_denoiser(tmp_path):
    token = tmp_path / "token"
    token.write_text("service-account-token")
    calls = []

    client = KubernetesScaleClient(
        host="kubernetes.default.svc",
        port=443,
        namespace="minwm-realtime",
        token_path=token,
        ca_path=Path("/missing-test-ca"),
        opener=lambda request, **kwargs: calls.append((request, kwargs)) or _Response(),
        ssl_context=object(),
    )
    client.scale("minwm-async-denoiser", 8)

    assert calls[0][0].full_url.endswith(
        "/apis/apps/v1/namespaces/minwm-realtime/"
        "statefulsets/minwm-async-denoiser/scale"
    )


def test_scaler_reads_the_current_scale_before_reconciling(tmp_path):
    token = tmp_path / "token"
    token.write_text("token")
    calls = []

    class ScaleResponse(_Response):
        def read(self):
            return b'{"spec":{"replicas":3}}'

    client = KubernetesScaleClient(
        host="kubernetes.default.svc",
        port=443,
        namespace="minwm-realtime",
        token_path=token,
        ca_path=Path("/missing-test-ca"),
        opener=lambda request, **kwargs: calls.append((request, kwargs))
        or ScaleResponse(),
        ssl_context=object(),
    )

    assert client.get_scale("minwm-async-vae") == 3
    assert calls[0][0].method == "GET"


@pytest.mark.parametrize("replicas", [-1, 9])
def test_scaler_enforces_the_production_gpu_replica_bound(tmp_path, replicas):
    token = tmp_path / "token"
    token.write_text("token")
    client = KubernetesScaleClient(
        host="kubernetes.default.svc",
        port=443,
        namespace="minwm-realtime",
        token_path=token,
        ca_path=Path("/missing-test-ca"),
        opener=lambda *_args, **_kwargs: _Response(),
        ssl_context=object(),
    )

    with pytest.raises(ValueError, match="between 0 and 8"):
        client.scale("minwm-async-denoiser", replicas)


def test_scaling_policy_adds_capacity_when_waiters_have_no_free_slot():
    policy = ScalingPolicy(
        minimum_replicas=0,
        maximum_replicas=8,
        sessions_per_replica=4,
        idle_observations_before_scale_down=3,
    )

    decision = policy.recommend(
        "denoiser",
        current_replicas=1,
        snapshot=CapacityRoleSnapshot(
            waiting_sessions=3,
            active_sessions=4,
            queued_sessions=0,
            free_slots=0,
            draining_workers=0,
        ),
    )

    assert decision.target_replicas == 2
    assert decision.reason == "waiting-capacity"


def test_scaling_policy_never_scales_down_with_active_or_queued_sessions():
    policy = ScalingPolicy(
        minimum_replicas=0,
        maximum_replicas=8,
        sessions_per_replica=4,
        idle_observations_before_scale_down=2,
    )

    for _ in range(4):
        decision = policy.recommend(
            "vae",
            current_replicas=2,
            snapshot=CapacityRoleSnapshot(
                waiting_sessions=0,
                active_sessions=1,
                queued_sessions=1,
                free_slots=7,
                draining_workers=0,
            ),
        )

    assert decision.target_replicas == 2
    assert decision.reason == "sessions-active"


def test_scaling_policy_requires_consecutive_idle_observations_and_scales_to_zero():
    policy = ScalingPolicy(
        minimum_replicas=0,
        maximum_replicas=8,
        sessions_per_replica=4,
        idle_observations_before_scale_down=3,
    )
    idle = CapacityRoleSnapshot(
        waiting_sessions=0,
        active_sessions=0,
        queued_sessions=0,
        free_slots=4,
        draining_workers=0,
    )

    assert policy.recommend("denoiser", 1, idle).target_replicas == 1
    assert policy.recommend("denoiser", 1, idle).target_replicas == 1
    decision = policy.recommend("denoiser", 1, idle)

    assert decision.target_replicas == 0
    assert decision.reason == "sustained-idle"


def test_scaling_policy_resets_idle_window_after_new_work_arrives():
    policy = ScalingPolicy(
        minimum_replicas=0,
        maximum_replicas=8,
        sessions_per_replica=4,
        idle_observations_before_scale_down=2,
    )
    idle = CapacityRoleSnapshot(0, 0, 0, 4, 0)
    busy = CapacityRoleSnapshot(1, 0, 0, 4, 0)

    assert policy.recommend("vae", 1, idle).target_replicas == 1
    assert policy.recommend("vae", 1, busy).target_replicas == 2
    assert policy.recommend("vae", 2, idle).target_replicas == 2


def test_scaling_policy_preserves_scheduled_scale_to_zero_until_work_arrives():
    policy = ScalingPolicy(
        minimum_replicas=1,
        maximum_replicas=8,
        sessions_per_replica=4,
        idle_observations_before_scale_down=2,
    )
    idle = CapacityRoleSnapshot(0, 0, 0, 0, 0)
    waiting = CapacityRoleSnapshot(1, 0, 0, 0, 0)

    assert policy.recommend("denoiser", 0, idle).target_replicas == 0
    assert policy.recommend("denoiser", 0, waiting).target_replicas == 1


def test_capacity_scaler_reconciles_each_role_from_a_shared_snapshot():
    class Kubernetes:
        def __init__(self):
            self.replicas = {
                "minwm-async-denoiser": 1,
                "minwm-async-vae": 1,
            }
            self.scales = []

        def get_scale(self, workload):
            return self.replicas[workload]

        def scale(self, workload, replicas):
            self.scales.append((workload, replicas))
            self.replicas[workload] = replicas

    kubernetes = Kubernetes()
    policy = ScalingPolicy(
        minimum_replicas=0,
        maximum_replicas=8,
        sessions_per_replica=4,
        idle_observations_before_scale_down=3,
    )
    scaler = CapacityScaler(
        kubernetes=kubernetes,
        policy_by_role={"denoiser": policy, "vae": policy},
    )

    decisions = scaler.reconcile(
        {
            "denoiser": CapacityRoleSnapshot(2, 4, 0, 0, 0),
            "vae": CapacityRoleSnapshot(0, 2, 0, 2, 0),
        }
    )

    assert decisions["denoiser"].target_replicas == 2
    assert decisions["vae"].target_replicas == 1
    assert kubernetes.scales == [("minwm-async-denoiser", 2)]


def test_capacity_client_parses_the_shared_coordinator_snapshot():
    calls = []

    class CapacityResponse(_Response):
        def read(self):
            return json.dumps(
                {
                    "observed_at": "2026-08-06T00:00:00Z",
                    "roles": {
                        "denoiser": {
                            "waiting_sessions": 2,
                            "active_sessions": 4,
                            "queued_sessions": 1,
                            "free_slots": 0,
                            "draining_workers": 0,
                        },
                        "vae": {
                            "waiting_sessions": 0,
                            "active_sessions": 4,
                            "queued_sessions": 0,
                            "free_slots": 12,
                            "draining_workers": 0,
                        },
                    },
                }
            ).encode()

    client = CoordinatorCapacityClient(
        "http://minwm-realtime-coordinator:18081",
        opener=lambda request, **kwargs: calls.append((request, kwargs))
        or CapacityResponse(),
    )

    snapshots = client.fetch()

    assert snapshots["denoiser"] == CapacityRoleSnapshot(2, 4, 1, 0, 0)
    assert snapshots["vae"] == CapacityRoleSnapshot(0, 4, 0, 12, 0)
    assert calls[0][0].full_url.endswith("/v1/capacity")
    assert calls[0][1]["timeout"] == 5
