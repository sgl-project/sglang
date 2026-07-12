import ast
import json
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.playground.disaggregation import pd_flip_controller as controller_module


def load_scheduler_methods():
    scheduler_path = (
        Path(__file__).resolve().parents[2] / "python/sglang/srt/managers/scheduler.py"
    )
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    scheduler_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    names = {
        "start_pd_flip_migration_source",
        "prepare_pd_flip_migration_target",
        "prepare_pd_flip_migration_target_delta",
        "commit_pd_flip_migration_target",
        "activate_pd_flip_migration_target",
        "get_pd_flip_migration_status",
        "finish_pd_flip_migration_source",
        "abort_pd_flip_migration",
        "_pd_flip_migration_status_dict",
        "_pd_flip_migration_request_measurements",
        "_pd_flip_delta_manifest_signature",
        "_pd_flip_record_sender_metric",
        "_pd_flip_mark_source_delta_applied",
        "_pd_flip_apply_delta_manifest_to_target",
        "_pd_flip_preserve_base_measurement",
        "_pd_flip_migration_timing_debug",
        "_pd_flip_migration_index_debug",
        "_pd_flip_json_safe_timing",
        "_pd_flip_can_rollover_session",
        "_pd_flip_archive_rollover_session",
    }
    methods = [
        node
        for node in scheduler_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            ast.ClassDef(
                name="Scheduler",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)

    class Output(SimpleNamespace):
        pass

    class ForbiddenMode:
        class Value:
            value = "decode"

        DECODE = Value()

        @staticmethod
        def to_engine_type(_):
            return "decode"

    namespace = {
        "PDFlipMigrationReqOutput": Output,
        "DisaggregationMode": ForbiddenMode,
        "json": json,
        "array": array,
        "time": __import__("time"),
    }
    exec(compile(extracted, str(scheduler_path), "exec"), namespace)
    return namespace["Scheduler"]


Scheduler = load_scheduler_methods()
Req = SimpleNamespace


class ReconciliationClient:
    def __init__(self, source_state, target_state):
        self.states = {
            "http://source": source_state,
            "http://target": target_state,
        }
        self.steps = []

    def get_json(self, base_url, path):
        assert path == "/pd_flip/migration/status"
        states = self.states[base_url]
        if not isinstance(states, list):
            states = [states]
        return [
            {
                "success": True,
                "status": {"session_id": "s", "state": state},
            }
            for state in states
        ]

    def post_json(self, base_url, path, payload):
        assert payload["session_id"] == "s"
        step = {
            ("http://target", "/pd_flip/migration/target/activate"): "activate_target",
            ("http://target", "/pd_flip/migration/target/abort"): "abort_target",
            ("http://source", "/pd_flip/migration/abort"): "abort_source",
        }[(base_url, path)]
        self.steps.append(step)
        if step == "activate_target":
            self.states[base_url] = "active"
        elif step == "abort_target":
            self.states[base_url] = "target_aborted"
        else:
            self.states[base_url] = "source_aborted"
        return {"success": True, "status": {"state": self.states[base_url]}}


def make_controller(tmp_path, source_state, target_state):
    journal_path = tmp_path / "state" / "session.json"
    config = controller_module.PDClusterConfig(
        router_url="http://router",
        nodes=[
            controller_module.PDNode("source", "http://source", "source"),
            controller_module.PDNode("target", "http://target", "target"),
        ],
        session_journal_path=str(journal_path),
    )
    client = ReconciliationClient(source_state, target_state)
    controller = controller_module.PDFlipController(config, client)
    controller.session_journal.write(
        {
            "session_id": "s",
            "source_name": "source",
            "source_url": "http://source",
            "target_name": "target",
            "target_url": "http://target",
            "batch_rids": ["r0", "r1"],
            "phase": "source_finish_complete",
            "source_finished": True,
        }
    )
    return controller, client


def test_journal_uses_atomic_replace_and_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "session.json"
    journal = controller_module.PDFlipSessionJournal(path)
    replacements = []
    real_replace = controller_module.os.replace
    monkeypatch.setattr(
        controller_module.os,
        "replace",
        lambda source, target: (
            replacements.append((Path(source), Path(target))),
            real_replace(source, target),
        )[1],
    )

    journal.write({"session_id": "s", "phase": "intent"})

    assert journal.read() == {"phase": "intent", "session_id": "s"}
    assert replacements == [(path.with_suffix(".json.tmp"), path)]
    assert not path.with_suffix(".json.tmp").exists()


def test_load_config_resolves_default_journal_beside_config(tmp_path):
    config_path = tmp_path / "config" / "cluster.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "router_url": "http://router",
                "nodes": [{"name": "source", "worker_url": "http://source"}],
            }
        ),
        encoding="utf-8",
    )

    config = controller_module.load_config(str(config_path))

    assert (
        Path(config.session_journal_path) == config_path.parent / "pd_flip_session.json"
    )


def test_reconcile_activates_ready_target_after_source_finished(tmp_path):
    controller, client = make_controller(
        tmp_path, "source_released", "ready_to_activate"
    )

    result = controller.reconcile_session("s")

    assert result.success
    assert client.steps == ["activate_target"]
    assert controller.session_journal.read()["phase"] == "target_active"


def test_reconcile_activates_only_when_every_dp_rank_is_ready(tmp_path):
    controller, client = make_controller(
        tmp_path,
        ["source_released", "source_released"],
        ["ready_to_activate", "ready_to_activate"],
    )

    assert controller.reconcile_session("s").success
    assert client.steps == ["activate_target"]


def test_reconcile_mixed_dp_target_states_require_operator(tmp_path):
    controller, client = make_controller(
        tmp_path,
        ["source_released", "source_released"],
        ["active", "ready_to_activate"],
    )

    result = controller.reconcile_session("s")

    assert not result.success
    assert "operator recovery" in result.message
    assert client.steps == []


def test_reconcile_does_not_repeat_active_session(tmp_path):
    controller, client = make_controller(tmp_path, "source_released", "active")

    assert controller.reconcile_session("s").success
    assert controller.reconcile_session("s").success
    assert client.steps == []


@pytest.mark.parametrize(
    "target_state",
    [
        "target_prepared",
        "target_transferred_held",
        "target_delta_started",
        "target_delta_transferred",
        "ready_to_activate",
        "target_failed",
    ],
)
def test_reconcile_aborts_both_sides_before_source_release(tmp_path, target_state):
    controller, client = make_controller(tmp_path, "source_started", target_state)

    result = controller.reconcile_session("s")

    assert result.success
    assert client.steps == ["abort_target", "abort_source"]
    assert controller.session_journal.read()["phase"] == "aborted"


@pytest.mark.parametrize(
    "response",
    [
        [],
        [None],
        [{"success": False, "status": {"session_id": "s", "state": "active"}}],
        [{"success": True}],
        [{"success": True, "status": []}],
        [{"success": True, "status": {"session_id": "other", "state": "active"}}],
    ],
)
def test_strict_status_aggregation_rejects_invalid_rank_response(response):
    with pytest.raises(ValueError):
        controller_module._strict_migration_statuses(response, "s")


@pytest.mark.parametrize("content", ["{broken", "[]", "null"])
def test_reconcile_corrupt_journal_requires_operator_without_actions(tmp_path, content):
    controller, client = make_controller(tmp_path, "source_released", "active")
    controller.session_journal.path.write_text(content, encoding="utf-8")

    result = controller.reconcile_session("s")

    assert not result.success
    assert "operator recovery" in result.message
    assert client.steps == []


def test_reconcile_requires_operator_for_ambiguous_ownership(tmp_path):
    controller, client = make_controller(tmp_path, "source_released", "target_aborted")

    result = controller.reconcile_session("s")

    assert not result.success
    assert "operator recovery" in result.message
    assert client.steps == []


def test_reconcile_does_not_repeat_aborted_session(tmp_path):
    controller, client = make_controller(tmp_path, "source_aborted", "target_aborted")

    assert controller.reconcile_session("s").success
    assert controller.reconcile_session("s").success
    assert client.steps == []


def test_reconcile_records_incomplete_abort_for_operator_recovery(tmp_path):
    controller, client = make_controller(tmp_path, "source_started", "target_prepared")
    original_post = client.post_json

    def fail_source_abort(base_url, path, payload):
        if base_url == "http://source":
            return {"success": False, "message": "source abort failed"}
        return original_post(base_url, path, payload)

    client.post_json = fail_source_abort

    result = controller.reconcile_session("s")

    assert not result.success
    assert "operator recovery" in result.message
    assert controller.session_journal.read()["phase"] == "abort_incomplete"


def test_reconcile_rejects_conflicting_session_without_requests(tmp_path):
    controller, client = make_controller(tmp_path, "source_released", "active")

    result = controller.reconcile_session("other")

    assert not result.success
    assert client.steps == []
    assert controller.session_journal.read()["session_id"] == "s"


def worker_with_session(role, state):
    worker = Scheduler.__new__(Scheduler)
    worker.pd_flip_migration_session = {
        "session_id": "s",
        "role": role,
        "state": state,
        "manifests": [{"rid": "r0"}],
        "source_entries": {},
        "target_entries": {},
        "pending_reqs": 0,
        "transferred_reqs": 1,
        "released_reqs": 1 if state in {"source_released", "active"} else 0,
        "failed_reqs": 0,
        "held_reqs": 0,
    }
    return worker


def test_worker_repeated_source_start_returns_existing_session_without_scan():
    worker = worker_with_session("source", "source_released")

    output = Scheduler.start_pd_flip_migration_source(
        worker, Req(session_id="s", rids=["r0"], target_url=None, include_waiting=False)
    )

    assert output.success
    assert output.status["state"] == "source_released"
    assert output.manifests == [{"rid": "r0"}]


def test_worker_conflicting_source_start_does_not_replace_session():
    worker = worker_with_session("source", "source_started")

    output = Scheduler.start_pd_flip_migration_source(
        worker,
        Req(session_id="other", rids=["r0"], target_url=None, include_waiting=False),
    )

    assert not output.success
    assert worker.pd_flip_migration_session["session_id"] == "s"


def test_worker_released_source_rolls_over_to_second_session():
    worker = worker_with_session("source", "source_released")
    worker.disaggregation_mode = SimpleNamespace(value="decode")
    worker.running_batch = SimpleNamespace(reqs=[])
    worker.waiting_queue = []
    worker._pd_flip_select_source_batch = lambda *_args, **_kwargs: []
    worker._pd_flip_start_source_entries = lambda *_args: ({}, "")

    output = Scheduler.start_pd_flip_migration_source(
        worker,
        Req(session_id="s2", rids=[], target_url="http://target", include_waiting=False),
    )

    assert output.success
    assert worker.pd_flip_migration_session["session_id"] == "s2"
    assert worker.pd_flip_migration_session_archive[-1]["session_id"] == "s"
    assert output.status["session_archive"][-1]["session_id"] == "s"


def test_worker_active_target_rolls_over_without_releasing_active_request():
    active_request = object()
    worker = worker_with_session("target", "active")
    worker.pd_flip_migration_session["target_entries"] = {
        "r0": {
            "phase": "active",
            "held": False,
            "request_adopted": True,
            "final_owner": "target",
        }
    }
    worker.disaggregation_mode = SimpleNamespace(value="decode")
    worker.waiting_queue = [active_request]
    worker._pd_flip_prepare_target_entries = lambda *_args: ({}, "")

    output = Scheduler.prepare_pd_flip_migration_target(
        worker,
        Req(
            session_id="s2",
            source_url="http://source",
            manifests=[],
            adopt_on_success=False,
            prepare_only=True,
            adopt_on_commit=False,
        ),
    )

    assert output.success
    assert worker.pd_flip_migration_session["session_id"] == "s2"
    assert worker.waiting_queue == [active_request]


def test_worker_rollover_predicate_defaults_missing_entry_fields_to_unsafe():
    source = worker_with_session("source", "source_released").pd_flip_migration_session
    source["source_entries"] = {"r0": {}}
    target = worker_with_session("target", "active").pd_flip_migration_session
    target["target_entries"] = {"r0": {"phase": "active"}}

    assert not Scheduler._pd_flip_can_rollover_session(source, "source")
    assert not Scheduler._pd_flip_can_rollover_session(target, "target")


def test_worker_repeated_target_prepare_returns_existing_session_without_allocating():
    worker = worker_with_session("target", "ready_to_activate")

    output = Scheduler.prepare_pd_flip_migration_target(
        worker,
        Req(
            session_id="s",
            source_url="http://source",
            manifests=[{"rid": "r0"}],
            adopt_on_success=False,
            prepare_only=True,
            adopt_on_commit=False,
        ),
    )

    assert output.success
    assert output.status["state"] == "ready_to_activate"


def test_worker_repeated_target_delta_prepare_is_signature_idempotent():
    worker = worker_with_session("target", "target_delta_transferred")
    manifests = [
        {
            "rid": "r0",
            "pd_flip_session_id": "s",
            "delta_from_len": 7,
            "kv_committed_len": 9,
        }
    ]
    worker.pd_flip_migration_session.update(
        delta_manifests=manifests,
        target_delta_manifest_signature=Scheduler._pd_flip_delta_manifest_signature(
            manifests
        ),
    )
    worker._pd_flip_target_pump_transfer = lambda _: pytest.fail("must not pump")

    output = Scheduler.prepare_pd_flip_migration_target_delta(
        worker,
        Req(session_id="s", source_url="http://source", manifests=manifests),
    )

    assert output.success
    assert output.manifests == manifests


@pytest.mark.parametrize("session_id", [None, "other"])
def test_worker_target_delta_prepare_rejects_missing_or_conflicting_session(session_id):
    worker = worker_with_session("target", "target_transferred_held")
    worker._pd_flip_target_pump_transfer = lambda _: pytest.fail("must not pump")

    output = Scheduler.prepare_pd_flip_migration_target_delta(
        worker,
        Req(session_id=session_id, source_url="http://source", manifests=[]),
    )

    assert not output.success
    assert worker.pd_flip_migration_session["state"] == "target_transferred_held"


def test_worker_target_delta_prepare_rejects_different_signature_without_side_effects():
    worker = worker_with_session("target", "target_delta_transferred")
    previous = [{"rid": "r0", "kv_committed_len": 9}]
    worker.pd_flip_migration_session.update(
        delta_manifests=previous,
        target_delta_manifest_signature=Scheduler._pd_flip_delta_manifest_signature(
            previous
        ),
    )
    worker._pd_flip_target_pump_transfer = lambda _: pytest.fail("must not pump")

    output = Scheduler.prepare_pd_flip_migration_target_delta(
        worker,
        Req(
            session_id="s",
            source_url="http://source",
            manifests=[{"rid": "r0", "kv_committed_len": 10}],
        ),
    )

    assert not output.success
    assert worker.pd_flip_migration_session["delta_manifests"] == previous


def test_worker_status_rejects_conflicting_session_without_pump():
    worker = worker_with_session("source", "source_started")
    worker._pd_flip_source_pump_transfer = lambda _: pytest.fail("must not pump")

    output = Scheduler.get_pd_flip_migration_status(worker, Req(session_id="other"))

    assert not output.success
    assert worker.pd_flip_migration_session["state"] == "source_started"


def test_worker_repeated_target_commit_is_terminal_noop():
    worker = worker_with_session("target", "ready_to_activate")
    worker._pd_flip_target_pump_transfer = lambda _: pytest.fail("must not pump")

    output = Scheduler.commit_pd_flip_migration_target(
        worker, Req(session_id="s", rids=["r0"])
    )

    assert output.success
    assert output.status["state"] == "ready_to_activate"


def test_worker_repeated_target_activate_does_not_requeue():
    worker = worker_with_session("target", "active")
    worker.waiting_queue = ["sentinel"]

    output = Scheduler.activate_pd_flip_migration_target(
        worker, Req(session_id="s", rids=["r0"])
    )

    assert output.success
    assert worker.waiting_queue == ["sentinel"]


def test_worker_repeated_source_finish_does_not_release_again():
    worker = worker_with_session("source", "source_released")
    worker._pd_flip_source_pump_transfer = lambda _: pytest.fail("must not pump")

    output = Scheduler.finish_pd_flip_migration_source(
        worker, Req(session_id="s", released_rids=["r0"])
    )

    assert output.success
    assert output.status["state"] == "source_released"


def test_worker_repeated_abort_is_noop_and_conflict_fails():
    worker = worker_with_session("source", "source_aborted")
    worker._pd_flip_abort_source_session = lambda *_: pytest.fail("must not abort")

    repeated = Scheduler.abort_pd_flip_migration(worker, Req(session_id="s", reason=""))
    conflicting = Scheduler.abort_pd_flip_migration(
        worker, Req(session_id="other", reason="")
    )

    assert repeated.success
    assert not conflicting.success
    assert worker.pd_flip_migration_session["session_id"] == "s"


def test_worker_status_exports_real_request_measurement_fields():
    worker = worker_with_session("target", "active")
    worker.pd_flip_migration_session.update(
        timing_debug={"commit_received_mono": 14.0},
        target_entries={
            "r0": {
                "phase": "active",
                "manifest": {
                    "origin_input_ids": [1, 2, 3, 4],
                    "kv_committed_len": 7,
                    "last_emitted_output_seq": 5,
                    "pd_flip_source_queue": "running",
                },
                "mooncake_hit_len": 3,
                "committed_len": 9,
                "stitch_mode": "partial_prefix_stitch",
                "mooncake_bytes": 101,
                "source_transfer_bytes": 202,
                "delta_transfer_bytes": 33,
                "source_transfer_duration_s": 0.2,
                "delta_transfer_duration_s": 0.4,
                "target_hicache_prefix_match_s": 0.1,
                "target_hicache_restore_tokens": 2,
                "final_owner": "target",
                "timing_debug": {
                    "source_send_s": 0.2,
                    "source_delta_sent_mono": 11.0,
                    "source_delta_transferred_mono": 11.4,
                    "target_held_mono": 12.0,
                    "source_waiting_frozen_mono": 10.0,
                    "target_adopted_mono": 15.0,
                    "target_hicache_restore_duration_s": 0.6,
                },
            }
        },
    )

    status = worker._pd_flip_migration_status_dict()
    row = status["request_measurements"][0]

    assert row == {
        "rid": "r0",
        "p_tokens": 4,
        "h_tokens": 3,
        "c0_tokens": 7,
        "c1_tokens": 9,
        "stitch_mode": "partial_prefix_stitch",
        "mooncake_bytes": 101,
        "mooncake_bytes_available": True,
        "source_bytes": 202,
        "delta_bytes": 33,
        "mooncake_restore_tokens": 2,
        "mooncake_duration_seconds": 0.6,
        "source_duration_seconds": 0.2,
        "delta_duration_seconds": pytest.approx(0.4),
        "held_at_mono": 12.0,
        "freeze_at_mono": 10.0,
        "commit_at_mono": 14.0,
        "activate_at_mono": 15.0,
        "source_queue": "running",
        "final_owner": "target",
        "output_boundary": 5,
        "rollback_reason": None,
    }


def test_worker_status_measurement_missing_fields_are_stable():
    worker = worker_with_session("source", "source_started")
    worker.pd_flip_migration_session["source_entries"] = {"r0": {}}

    row = worker._pd_flip_migration_status_dict()["request_measurements"][0]

    assert row["p_tokens"] == 0
    assert row["c0_tokens"] == 0
    assert row["mooncake_bytes"] is None
    assert row["mooncake_bytes_available"] is False
    assert row["delta_duration_seconds"] is None
    assert row["final_owner"] is None


@pytest.mark.parametrize(
    ("segment", "metric_latency", "expected_duration"),
    [("source", 0.25, 0.25), ("delta", None, 0.4)],
)
def test_worker_records_exact_sender_transfer_metric(
    segment, metric_latency, expected_duration
):
    started_key = f"{segment}_transfer_started_mono"
    completed_key = f"{segment}_transfer_completed_mono"
    entry = {"timing_debug": {started_key: 10.0, completed_key: 10.4}}
    sender = Req(
        get_transfer_metric=lambda: Req(
            transfer_total_bytes=4096,
            transfer_latency_s=metric_latency,
        )
    )

    Scheduler._pd_flip_record_sender_metric(entry, sender, segment)

    assert entry[f"{segment}_transfer_bytes"] == 4096
    assert entry[f"{segment}_transfer_duration_s"] == pytest.approx(
        expected_duration
    )


def test_worker_sender_without_metric_api_keeps_bytes_unknown():
    entry = {
        "timing_debug": {
            "source_transfer_started_mono": 10.0,
            "source_transfer_completed_mono": 10.5,
        }
    }

    Scheduler._pd_flip_record_sender_metric(entry, Req(), "source")

    assert entry["source_transfer_bytes"] is None
    assert entry["source_transfer_duration_s"] == 0.5


def test_source_delta_measurement_preserves_base_boundary_across_generations():
    worker = Scheduler()
    entry = {
        "committed_len": 4,
        "manifest": {
            "rid": "r0",
            "origin_input_ids": [1, 2, 3],
            "kv_committed_len": 4,
            "pd_flip_source_queue": "running",
        },
    }

    worker._pd_flip_mark_source_delta_applied(
        entry,
        {
            "rid": "r0",
            "origin_input_ids": [1, 2, 3],
            "kv_committed_len": 6,
            "delta_from_len": 4,
            "delta_noop": True,
        },
    )
    worker._pd_flip_mark_source_delta_applied(
        entry,
        {
            "rid": "r0",
            "origin_input_ids": [1, 2, 3],
            "kv_committed_len": 8,
            "delta_from_len": 6,
            "delta_noop": True,
        },
    )

    row = worker._pd_flip_migration_request_measurements(
        {"source_entries": {"r0": entry}}
    )[0]
    assert row["p_tokens"] == 3
    assert row["c0_tokens"] == 4
    assert row["c1_tokens"] == 8


def test_target_delta_measurement_preserves_base_boundary_across_generations():
    worker = Scheduler()
    worker._pd_flip_note_timing = lambda *args, **kwargs: None
    req = Req(
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        kv_committed_len=4,
        kv_allocated_len=4,
    )
    entry = {
        "decode_req": Req(req=req),
        "committed_len": 4,
        "manifest": {
            "rid": "r0",
            "origin_input_ids": [1, 2, 3],
            "kv_committed_len": 4,
            "pd_flip_source_queue": "running",
        },
    }

    for from_len, to_len in [(4, 6), (6, 8)]:
        worker._pd_flip_apply_delta_manifest_to_target(
            entry,
            {
                "rid": "r0",
                "origin_input_ids": [1, 2, 3],
                "output_ids": list(range(to_len - 3 + 1)),
                "kv_committed_len": to_len,
                "delta_from_len": from_len,
            },
        )

    row = worker._pd_flip_migration_request_measurements(
        {"target_entries": {"r0": entry}}
    )[0]
    assert row["p_tokens"] == 3
    assert row["c0_tokens"] == 4
    assert row["c1_tokens"] == 8


def test_controller_progressive_observability_uses_raw_policy_and_slo_counts(tmp_path):
    controller, _ = make_controller(tmp_path, "source_started", "target_prepared")
    snapshot = Req(
        prefill_counts=Req(good=14, total=20),
        decode_counts=Req(good=19, total=20),
    )
    selection = Req(configured_ratio=0.5, effective_ratio=0.25, fallback_count=1)

    fields = controller._progressive_observability_fields(snapshot, selection)

    assert fields == {
        "configured_ratio": 0.5,
        "effective_ratio": 0.25,
        "capacity_fallback_count": 1,
        "prefill_slo_good": 14,
        "prefill_slo_total": 20,
        "decode_slo_good": 19,
        "decode_slo_total": 20,
    }


def legacy_two_phase_scenario(
    tmp_path, *, transfer_result="complete", fail_source_abort=False
):
    journal_path = tmp_path / "legacy-session.json"
    config = controller_module.PDClusterConfig(
        router_url="http://router",
        nodes=[
            controller_module.PDNode("source", "http://source", "source"),
            controller_module.PDNode("target", "http://target", "target"),
        ],
        session_journal_path=str(journal_path),
    )
    controller = controller_module.PDFlipController(config, client=object())
    source = controller_module.NodeMetrics(
        "source", "http://source", "source", worker_role="decode"
    )
    target = controller_module.NodeMetrics(
        "target", "http://target", "target", worker_role="decode"
    )
    observed_phases = []

    controller.collect_metrics = lambda: [source, target]
    controller._post_router = lambda *args, **kwargs: {"success": True}
    controller._observe_source_quiesce_for_duration = lambda *args, **kwargs: None
    controller._wait_two_phase_migration_or_recovery = (
        lambda **kwargs: transfer_result
    )
    controller._sync_two_phase_delta_before_commit = lambda **kwargs: None
    controller._assert_source_idle_after_migration = lambda *args, **kwargs: None
    controller._cleanup_source_after_failure = lambda *args, **kwargs: None

    def post_worker(records, step, node, path, payload):
        record = controller.session_journal.read()
        observed_phases.append((path, record and record.get("phase")))
        if fail_source_abort and path == "/pd_flip/migration/abort":
            raise RuntimeError("source abort failed")
        if path == "/pd_flip/migration/source/start":
            return {
                "success": True,
                "manifests": [{"rid": "r0"}, {"rid": "waiting-r1"}],
            }
        return {"success": True}

    controller._post_worker = post_worker
    monitor = Req(
        collect_cluster=lambda _: Req(
            prefill_slo_attainment=0.0, decode_slo_attainment=1.0
        )
    )
    return controller, source, target, monitor, observed_phases


def test_legacy_two_phase_journals_every_ownership_action_with_full_batch(tmp_path):
    controller, source, target, monitor, phases = legacy_two_phase_scenario(tmp_path)

    result = controller._execute_d_to_p_two_phase(
        source=source,
        target=target,
        slo_monitor=monitor,
        enter_threshold=0.9,
        exit_threshold=0.9,
        commit_threshold=0.9,
    )

    assert result.success
    assert dict(phases)["/pd_flip/migration/target/commit"] == "target_commit_intent"
    assert dict(phases)["/pd_flip/migration/source/finish"] == "source_finish_intent"
    assert dict(phases)["/pd_flip/migration/target/activate"] == "target_activate_intent"
    record = controller.session_journal.read()
    assert record["batch_rids"] == ["r0", "waiting-r1"]
    assert record["phase"] == "target_active"
    assert record["source_finished"] is True


def test_legacy_two_phase_recovery_journals_abort_with_full_batch(tmp_path):
    controller, source, target, monitor, phases = legacy_two_phase_scenario(
        tmp_path, transfer_result="recovered"
    )

    result = controller._execute_d_to_p_two_phase(
        source=source,
        target=target,
        slo_monitor=monitor,
        enter_threshold=0.9,
        exit_threshold=0.9,
        commit_threshold=0.9,
    )

    assert result.success
    abort_phases = [phase for path, phase in phases if path.endswith("/abort")]
    assert abort_phases == ["abort_intent", "abort_intent"]
    record = controller.session_journal.read()
    assert record["batch_rids"] == ["r0", "waiting-r1"]
    assert record["phase"] == "aborted"
    assert record["source_finished"] is False


def test_legacy_two_phase_partial_abort_requires_operator_recovery(tmp_path):
    controller, source, target, monitor, _ = legacy_two_phase_scenario(
        tmp_path, transfer_result="recovered", fail_source_abort=True
    )

    result = controller._execute_d_to_p_two_phase(
        source=source,
        target=target,
        slo_monitor=monitor,
        enter_threshold=0.9,
        exit_threshold=0.9,
        commit_threshold=0.9,
    )

    assert not result.success
    assert "operator recovery" in result.message
    assert controller.session_journal.read()["phase"] == "abort_incomplete"


def test_progressive_abort_keeps_source_verified_full_batch(tmp_path):
    controller, _ = make_controller(tmp_path, "source_started", "target_prepared")
    source = controller_module.NodeMetrics("source", "http://source", "source")
    target = controller_module.NodeMetrics("target", "http://target", "target")

    def post_worker(records, step, node, path, payload):
        if path == "/pd_flip/migration/source/start":
            return {
                "success": True,
                "manifests": [{"rid": "r0"}, {"rid": "waiting-r1"}],
            }
        return {"success": True}

    controller._post_worker = post_worker
    controller._wait_migration = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("interrupt after source verification")
    )
    controller._abort_two_phase_migration = lambda *args, **kwargs: True

    with pytest.raises(controller_module.ProgressiveAtomicBatchError):
        controller._execute_atomic_batch(
            source,
            target,
            "s",
            ["r0"],
            True,
            next_fsm_phase="role_flip_worker_prefill_intent",
        )

    record = controller.session_journal.read()
    assert record["batch_rids"] == ["r0", "waiting-r1"]
    assert record["phase"] == "aborted"
