from pathlib import Path
from typing import Optional, get_type_hints

import pytest

from scripts.playground.disaggregation import pd_flip_controller as controller_module
from scripts.playground.disaggregation.pd_flip_monitor import SampleCounts


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / "scripts/playground/disaggregation/pd_flip_docker/env.example"
RUN_CONTROLLER = (
    REPO_ROOT / "scripts/playground/disaggregation/pd_flip_docker/run_controller.sh"
)


def metric(name, **capacity):
    return controller_module.NodeMetrics(
        name=name,
        worker_url=f"http://{name}",
        router_worker_id=name,
        worker_role="decode",
        raw_status={"success": True, "status": capacity},
    )


def make_controller(*, first_migration_ratio=0.5):
    config = controller_module.PDClusterConfig(
        router_url="http://router",
        nodes=[controller_module.PDNode("d0", "http://d0", "d0")],
        first_migration_ratio=first_migration_ratio,
    )
    return controller_module.PDFlipController(config, client=object())


def test_progressive_session_prefix_is_unique_by_default_and_explicitly_controllable():
    source, target = metric("source"), metric("target")
    first = make_controller()._progressive_session_prefix(source, target)
    second = make_controller()._progressive_session_prefix(source, target)
    assert first != second
    assert first.startswith("pd-flip-source-to-target-")
    config = controller_module.PDClusterConfig(
        router_url="http://router",
        nodes=[controller_module.PDNode("d0", "http://d0", "d0")],
        session_id_prefix="controlled",
    )
    controlled = controller_module.PDFlipController(config, client=object())
    assert controlled._progressive_session_prefix(source, target) == "controlled"


def test_controller_uses_status_capacity_and_halves_ratio():
    controller = make_controller(first_migration_ratio=0.75)
    source = metric(
        "d0",
        running_requests=[
            {"rid": "r0", "kv_committed_len": 100},
            {"rid": "r1", "kv_committed_len": 100},
            {"rid": "r2", "kv_committed_len": 100},
            {"rid": "r3", "kv_committed_len": 100},
        ],
    )
    target = metric(
        "d1",
        free_request_slots=1,
        available_kv_tokens=150,
        reserved_decode_tokens_per_req=16,
    )

    selection = controller._select_progressive_first_batch(source, target)

    assert selection.selected_rids == ("r0",)
    assert selection.effective_ratio == 0.1875
    assert selection.required_kv_tokens == 116


def test_controller_preserves_running_order_and_reserves_decode_capacity():
    controller = make_controller(first_migration_ratio=0.5)
    source = metric(
        "d0",
        running_requests=[
            {"rid": 17, "kv_committed_len": "100"},
            {"rid": "r1", "kv_committed_len": 40},
        ],
    )
    target = metric(
        "d1",
        free_request_slots=1,
        available_kv_tokens=115,
        reserved_decode_tokens_per_req=16,
    )

    assert controller._select_progressive_first_batch(source, target) is None


@pytest.mark.parametrize(
    "bad_entry",
    [
        None,
        {},
        {"rid": None, "kv_committed_len": 100},
        {"rid": "bad"},
        {"rid": "bad", "kv_committed_len": None},
        {"rid": "bad", "kv_committed_len": "not-an-int"},
        {"rid": "bad", "kv_committed_len": -1},
    ],
)
def test_controller_rejects_entire_prefix_when_running_metadata_is_invalid(
    bad_entry,
):
    controller = make_controller()
    source = metric(
        "d0",
        running_requests=[
            bad_entry,
            {"rid": "later-valid", "kv_committed_len": 1},
        ],
    )
    target = metric(
        "d1",
        free_request_slots=2,
        available_kv_tokens=1000,
        reserved_decode_tokens_per_req=0,
    )

    assert controller._select_progressive_first_batch(source, target) is None


def test_controller_returns_none_for_empty_running_prefix():
    controller = make_controller()
    source = metric("d0", running_requests=[])
    target = metric(
        "d1",
        free_request_slots=1,
        available_kv_tokens=1000,
        reserved_decode_tokens_per_req=0,
    )

    assert controller._select_progressive_first_batch(source, target) is None


def test_progressive_config_defaults_and_from_dict_values():
    defaults = controller_module.PDClusterConfig.from_dict(
        {
            "router_url": "http://router",
            "nodes": [{"name": "d0", "worker_url": "http://d0"}],
        }
    )
    assert defaults.first_migration_ratio == 0.5
    assert defaults.observation_seconds == 10.0
    assert defaults.slo_threshold == 0.9
    assert defaults.min_prefill_slo_samples == 20
    assert defaults.min_decode_slo_samples == 20
    assert defaults.session_journal_path == "pd_flip_session.json"

    configured = controller_module.PDClusterConfig.from_dict(
        {
            "router_url": "http://router",
            "nodes": [{"name": "d0", "worker_url": "http://d0"}],
            "first_migration_ratio": "0.75",
            "observation_seconds": "12.5",
            "slo_threshold": "0.95",
            "min_prefill_slo_samples": "30",
            "min_decode_slo_samples": "40",
            "session_journal_path": "state/session.json",
        }
    )
    assert configured.first_migration_ratio == 0.75
    assert configured.observation_seconds == 12.5
    assert configured.slo_threshold == 0.95
    assert configured.min_prefill_slo_samples == 30
    assert configured.min_decode_slo_samples == 40
    assert configured.session_journal_path == "state/session.json"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_migration_ratio", 0),
        ("first_migration_ratio", 1),
        ("observation_seconds", -0.1),
        ("slo_threshold", -0.1),
        ("slo_threshold", 1.1),
        ("min_prefill_slo_samples", 0),
        ("min_decode_slo_samples", 0),
    ],
)
def test_progressive_config_from_dict_rejects_invalid_policy_values(field, value):
    data = {
        "router_url": "http://router",
        "nodes": [{"name": "d0", "worker_url": "http://d0"}],
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        controller_module.PDClusterConfig.from_dict(data)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_progressive_config_accepts_slo_threshold_boundaries(threshold):
    config = controller_module.PDClusterConfig(
        router_url="http://router",
        nodes=[controller_module.PDNode("d0", "http://d0", "d0")],
        first_migration_ratio=0.5,
        observation_seconds=0.0,
        slo_threshold=threshold,
        min_prefill_slo_samples=1,
        min_decode_slo_samples=1,
    )

    assert config.slo_threshold == threshold


def test_progressive_cli_values_reach_config_from_args():
    args = controller_module.build_arg_parser().parse_args(
        [
            "--router-url",
            "http://router",
            "--node",
            "name=d0,worker_url=http://d0",
            "--first-migration-ratio",
            "0.625",
            "--observation-seconds",
            "15",
            "--slo-threshold",
            "0.92",
            "--min-prefill-slo-samples",
            "24",
            "--min-decode-slo-samples",
            "28",
            "--session-journal-path",
            "state/cli-session.json",
            "--session-id-prefix",
            "controlled-case",
            "metrics",
        ]
    )

    config = controller_module.config_from_args(args)

    assert config.first_migration_ratio == 0.625
    assert config.observation_seconds == 15.0
    assert config.slo_threshold == 0.92
    assert config.min_prefill_slo_samples == 24
    assert config.min_decode_slo_samples == 28
    assert config.session_journal_path == "state/cli-session.json"
    assert config.session_id_prefix == "controlled-case"


def test_progressive_cli_config_rejects_invalid_policy_values():
    args = controller_module.build_arg_parser().parse_args(
        [
            "--router-url",
            "http://router",
            "--node",
            "name=d0,worker_url=http://d0",
            "--first-migration-ratio",
            "1",
            "metrics",
        ]
    )

    with pytest.raises(ValueError, match="first_migration_ratio"):
        controller_module.config_from_args(args)


def test_docker_environment_passes_progressive_policy_cli_values():
    env_text = ENV_EXAMPLE.read_text(encoding="utf-8")
    script_text = RUN_CONTROLLER.read_text(encoding="utf-8")
    expected = {
        "PD_FLIP_FIRST_MIGRATION_RATIO": ("--first-migration-ratio", "0.5"),
        "PD_FLIP_OBSERVATION_SECONDS": ("--observation-seconds", "10"),
        "PD_FLIP_SLO_THRESHOLD": ("--slo-threshold", "0.9"),
        "PD_FLIP_MIN_PREFILL_SLO_SAMPLES": (
            "--min-prefill-slo-samples",
            "20",
        ),
        "PD_FLIP_MIN_DECODE_SLO_SAMPLES": (
            "--min-decode-slo-samples",
            "20",
        ),
    }
    for variable, (option, default) in expected.items():
        assert f"{variable}=" in env_text
        assert f'{option} "${{{variable}:-{default}}}"' in script_text


def test_progressive_policy_symbols_are_the_production_helpers():
    assert controller_module.ProgressiveDecision.START.value == "start"
    assert (
        controller_module.evaluate_slo_decision(14, 20, 19, 20, 0.9, 20, 20)
        is controller_module.ProgressiveDecision.START
    )
    assert (
        controller_module.select_first_batch.__module__ == "pd_flip_progressive_policy"
    )
    assert (
        get_type_hints(
            controller_module.PDFlipController._select_progressive_first_batch
        )["return"]
        == Optional[controller_module.RatioSelection]
    )


class ProgressiveScenarioClient:
    def __init__(
        self,
        *,
        running_rids=None,
        waiting_rids=None,
        fail_path=None,
        fail_session_id=None,
        source_start_response=None,
        delta_pending_once=True,
    ):
        self.running_rids = list(running_rids or ["r0", "r1"])
        self.waiting_rids = list(waiting_rids or [])
        self.fail_path = fail_path
        self.fail_session_id = fail_session_id
        self.source_start_response = source_start_response
        self.delta_pending_once = delta_pending_once
        self.steps = []
        self.posts = []
        self.source_starts = []
        self.delta_attempts = {}
        self.sessions = {}
        self.source_role = "decode"

    def get_json(self, base_url, path):
        if path == "/pd_flip/router/workers":
            return {
                "workers": [
                    {"worker_id": "source", "role": "decode", "draining": False},
                    {"worker_id": "target", "role": "decode", "draining": False},
                ]
            }
        if path == "/pd_flip/runtime_role/status":
            if base_url == "http://source":
                if self.source_role == "prefill":
                    self.steps.append("wait_source_prefill_loop")
                return {
                    "success": True,
                    "role": self.source_role,
                    "status": {
                        "role": self.source_role,
                        "active_event_loop_role": self.source_role,
                        "is_idle": not self.running_rids,
                        "running_requests": [
                            {"rid": rid, "kv_committed_len": 8}
                            for rid in self.running_rids
                        ],
                    },
                }
            return {
                "success": True,
                "role": "decode",
                "status": {
                    "role": "decode",
                    "is_idle": True,
                    "free_request_slots": 8,
                    "available_kv_tokens": 1024,
                    "reserved_decode_tokens_per_req": 1,
                    "running_requests": [],
                },
            }
        if path == "/v1/loads?include=all":
            return {
                "loads": [
                    {
                        "num_running_reqs": (
                            len(self.running_rids) if base_url == "http://source" else 0
                        ),
                        "num_waiting_reqs": (
                            len(self.waiting_rids) if base_url == "http://source" else 0
                        ),
                        "num_total_tokens": 16 if base_url == "http://source" else 0,
                    }
                ]
            }
        if path == "/pd_flip/migration/status":
            return {
                "success": True,
                "status": {"state": "transferred", "pending_reqs": 0, "failed_reqs": 0},
            }
        raise AssertionError((base_url, path))

    def post_json(self, base_url, path, payload):
        self.posts.append((base_url, path, dict(payload)))
        if path == self.fail_path and (
            self.fail_session_id is None
            or payload.get("session_id") == self.fail_session_id
        ):
            return {"success": False, "message": f"forced failure at {path}"}
        if path == "/pd_flip/router/worker/drain":
            self.steps.append(
                "router_drain_source"
                if payload["draining"]
                else "router_undrain_source"
            )
            return {"success": True}
        if path == "/pd_flip/router/worker/role":
            self.steps.append("refresh_router_source_role")
            return {"success": True}
        if path == "/pd_flip/runtime_role/admission":
            self.steps.append(
                "pause_source_admission"
                if payload["paused"]
                else "resume_source_admission"
            )
            return {"success": True}
        if path == "/pd_flip/runtime_role/set":
            self.steps.append("set_source_runtime_role")
            self.source_role = payload["role"]
            return {"success": True}
        if path == "/pd_flip/migration/source/start":
            self.steps.append("source_start")
            self.source_starts.append(dict(payload))
            if self.source_start_response is not None:
                return self.source_start_response
            rids = list(payload["rids"])
            if payload["include_waiting"]:
                rids.extend(self.waiting_rids)
            manifests = [{"rid": rid} for rid in rids]
            self.sessions[payload["session_id"]] = manifests
            return {"success": True, "manifests": manifests}
        if path == "/pd_flip/migration/target/prepare":
            self.steps.append("target_prepare")
            return {"success": True}
        if path == "/pd_flip/migration/source/delta":
            session_id = payload["session_id"]
            attempts = self.delta_attempts.get(session_id, 0) + 1
            self.delta_attempts[session_id] = attempts
            self.steps.append("source_delta")
            if self.delta_pending_once and attempts == 1:
                return {
                    "success": False,
                    "message": (
                        "source batch quiesce pending; retry delta after quiesce"
                    ),
                    "manifests": [],
                }
            return {"success": True, "manifests": self.sessions[session_id]}
        if path == "/pd_flip/migration/target/delta/prepare":
            self.steps.append("target_delta_prepare")
            return {"success": True}
        if path == "/pd_flip/migration/target/commit":
            self.steps.append("target_commit")
            return {"success": True}
        if path == "/pd_flip/migration/source/finish":
            self.steps.append("source_finish")
            released = set(payload["released_rids"])
            self.running_rids = [
                rid for rid in self.running_rids if rid not in released
            ]
            self.waiting_rids = [
                rid for rid in self.waiting_rids if rid not in released
            ]
            return {"success": True}
        if path == "/pd_flip/migration/target/activate":
            self.steps.append("target_activate")
            return {"success": True}
        if path in (
            "/pd_flip/migration/target/abort",
            "/pd_flip/migration/abort",
        ):
            self.steps.append("abort")
            return {"success": True}
        raise AssertionError((base_url, path, payload))


class ProgressiveScenarioMonitor:
    def __init__(self, observation):
        self.observation = observation
        self.reset_calls = 0
        self.collect_calls = 0
        self.events = []

    def collect_cluster(self, nodes):
        list(nodes)
        self.collect_calls += 1
        self.events.append("collect")
        counts = (14, 20, 19, 20) if self.reset_calls == 0 else self.observation
        prefill_good, prefill_total, decode_good, decode_total = counts
        return controller_module.ClusterSLOSnapshot(
            timestamp=float(self.collect_calls),
            prefill_nodes=1,
            decode_nodes=2,
            prefill_slo_attainment=prefill_good / prefill_total,
            decode_slo_attainment=decode_good / decode_total,
            nodes=[],
            prefill_counts=SampleCounts(prefill_good, prefill_total),
            decode_counts=SampleCounts(decode_good, decode_total),
        )

    def reset_window(self):
        self.reset_calls += 1
        self.events.append("reset")


def progressive_scenario(
    observation,
    *,
    client=None,
    observation_seconds=0.0,
    migration_poll_interval_seconds=0.0,
):
    client = client or ProgressiveScenarioClient()
    config = controller_module.PDClusterConfig(
        router_url="http://router",
        nodes=[
            controller_module.PDNode("source", "http://source", "source"),
            controller_module.PDNode("target", "http://target", "target"),
        ],
        migration_poll_interval_seconds=migration_poll_interval_seconds,
        observation_seconds=observation_seconds,
        post_migration_idle_timeout_seconds=0.0,
        min_prefill_slo_samples=20,
        min_decode_slo_samples=20,
        session_id_prefix="pd-flip-source-to-target",
    )
    controller = controller_module.PDFlipController(config, client)
    return controller, client, ProgressiveScenarioMonitor(observation)


def test_progressive_flow_recovers_after_first_batch():
    controller, client, monitor = progressive_scenario((18, 20, 19, 20))

    result = controller.monitor_progressive(monitor, iterations=1)

    assert result.success
    assert [item["state"] for item in result.state_trace] == [
        "safe",
        "selecting",
        "first_migrating",
        "observing",
        "recovering",
        "safe",
    ]
    assert client.steps.count("set_source_runtime_role") == 0
    assert client.source_starts == [
        {
            "session_id": "pd-flip-source-to-target-first",
            "target_url": "http://target",
            "rids": ["r0"],
            "include_waiting": False,
        }
    ]
    assert client.delta_attempts == {"pd-flip-source-to-target-first": 2}
    assert monitor.reset_calls == 1


def test_progressive_flow_commits_when_prefill_stays_risky():
    controller, client, monitor = progressive_scenario((14, 20, 19, 20))

    result = controller.monitor_progressive(monitor, iterations=1)

    assert result.success
    assert "second_migrating" in [item["state"] for item in result.state_trace]
    assert client.source_starts[0]["rids"] == ["r0"]
    assert client.source_starts[0]["include_waiting"] is False
    assert client.source_starts[1]["rids"] == ["r1"]
    assert client.source_starts[1]["include_waiting"] is True
    assert client.steps[-5:] == [
        "set_source_runtime_role",
        "wait_source_prefill_loop",
        "refresh_router_source_role",
        "resume_source_admission",
        "router_undrain_source",
    ]


def test_progressive_final_batch_uses_all_returned_running_and_waiting_rids():
    client = ProgressiveScenarioClient(waiting_rids=["w0"])
    controller, client, monitor = progressive_scenario((14, 20, 19, 20), client=client)

    result = controller.monitor_progressive(monitor, iterations=1)

    assert result.success
    final_session = "pd-flip-source-to-target-final"
    payloads = [
        payload
        for _, path, payload in client.posts
        if path
        in {
            "/pd_flip/migration/source/delta",
            "/pd_flip/migration/target/commit",
            "/pd_flip/migration/source/finish",
            "/pd_flip/migration/target/activate",
        }
        and payload["session_id"] == final_session
    ]
    assert payloads
    assert all(
        payload.get("rids", payload.get("released_rids")) == ["r1", "w0"]
        for payload in payloads
    )


def test_progressive_commit_skips_empty_final_source_start():
    client = ProgressiveScenarioClient(running_rids=["only"])
    controller, client, monitor = progressive_scenario((14, 20, 19, 20), client=client)

    result = controller.monitor_progressive(monitor, iterations=1)

    assert result.success
    assert len(client.source_starts) == 1
    assert client.source_starts[0]["rids"] == ["only"]


def test_progressive_failure_before_source_finish_aborts_both_sides():
    client = ProgressiveScenarioClient(fail_path="/pd_flip/migration/target/commit")
    controller, client, monitor = progressive_scenario((18, 20, 19, 20), client=client)

    result = controller.monitor_progressive(monitor, iterations=1)

    assert not result.success
    abort_records = [
        record for record in result.actions if record.step == "abort_decode_migration"
    ]
    assert [record.target for record in abort_records] == ["target", "source"]
    assert "source_finish" not in client.steps
    assert client.steps[-2:] == ["resume_source_admission", "router_undrain_source"]


def test_progressive_failure_after_source_finish_does_not_abort_ownership():
    client = ProgressiveScenarioClient(fail_path="/pd_flip/migration/target/activate")
    controller, client, monitor = progressive_scenario((18, 20, 19, 20), client=client)

    result = controller.monitor_progressive(monitor, iterations=1)

    assert not result.success
    assert "source_finish" in client.steps
    assert not [
        record for record in result.actions if record.step == "abort_decode_migration"
    ]
    assert client.running_rids == ["r1"]
    assert result.state_trace[-1]["reason"] == "post_finish_error"


def test_progressive_final_batch_activate_failure_reports_post_finish_phase():
    client = ProgressiveScenarioClient(
        fail_path="/pd_flip/migration/target/activate",
        fail_session_id="pd-flip-source-to-target-final",
    )
    controller, client, monitor = progressive_scenario((14, 20, 19, 20), client=client)

    result = controller.monitor_progressive(monitor, iterations=1)

    assert not result.success
    assert len(client.source_starts) == 2
    assert result.state_trace[-1]["reason"] == "post_finish_error"
    assert not [
        record for record in result.actions if record.step == "abort_decode_migration"
    ]


@pytest.mark.parametrize(
    "response",
    [
        {"success": True, "manifests": [{"rid": "r0"}, {"rid": "r0"}]},
        {"success": True, "manifests": [{}]},
        {"success": True, "manifests": ["not-a-manifest"]},
        {"success": True, "manifests": [{"rid": "r0"}, {"rid": "extra"}]},
        {"success": True, "manifests": "not-a-list"},
        ["not-a-response-item"],
    ],
)
def test_atomic_batch_rejects_malformed_source_start_manifests(response):
    client = ProgressiveScenarioClient(source_start_response=response)
    controller, _, _ = progressive_scenario((18, 20, 19, 20), client=client)
    source = metric("source", running_requests=[])
    target = metric("target", running_requests=[])
    records = []

    with pytest.raises(RuntimeError, match="invalid source start response manifests"):
        controller._execute_atomic_batch(
            source,
            target,
            "strict-source-start",
            ("r0",),
            False,
            records=records,
        )

    assert "target_prepare" not in client.steps
    assert [
        record.target for record in records if record.step == "abort_decode_migration"
    ] == [
        "target",
        "source",
    ]


def test_delta_pending_accepts_only_exact_task7_response():
    exact = {
        "success": False,
        "message": "source batch quiesce pending; retry delta after quiesce",
        "manifests": [],
    }

    assert controller_module._delta_quiesce_pending(exact)


@pytest.mark.parametrize(
    "response",
    [
        {"success": False, "message": "quiesce pending", "manifests": []},
        {
            "success": True,
            "message": "source batch quiesce pending; retry delta after quiesce",
            "manifests": [],
        },
        [
            {
                "success": False,
                "message": "source batch quiesce pending; retry delta after quiesce",
                "manifests": [],
            },
            {"success": True, "message": "ok", "manifests": []},
        ],
        {
            "success": False,
            "message": "source batch quiesce pending; retry delta after quiesce",
        },
        {
            "success": False,
            "message": "source batch quiesce pending; retry delta after quiesce",
            "manifests": (),
        },
    ],
)
def test_delta_pending_rejects_ambiguous_or_malformed_responses(response):
    assert not controller_module._delta_quiesce_pending(response)


@pytest.mark.parametrize(
    ("poll_interval", "expected_sleeps"),
    [(1.0, [1.0, 1.0, 0.5]), (0.0, [2.5])],
)
def test_progressive_observation_uses_exact_fresh_deadline_without_busy_loop(
    monkeypatch, poll_interval, expected_sleeps
):
    class FakeClock:
        def __init__(self):
            self.now = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            assert seconds > 0, "observation must not busy-loop with zero sleeps"
            self.sleeps.append(seconds)
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(controller_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(controller_module.time, "sleep", clock.sleep)
    client = ProgressiveScenarioClient(delta_pending_once=False)
    controller, _, monitor = progressive_scenario(
        (18, 20, 19, 20),
        client=client,
        observation_seconds=2.5,
        migration_poll_interval_seconds=poll_interval,
    )

    result = controller.monitor_progressive(monitor, iterations=1)

    assert result.success
    assert monitor.events[0:3] == ["collect", "reset", "collect"]
    assert clock.now == 2.5
    assert clock.sleeps == expected_sleeps
