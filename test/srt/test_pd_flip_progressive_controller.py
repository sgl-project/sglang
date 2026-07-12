from pathlib import Path
from typing import Optional, get_type_hints

import pytest

from scripts.playground.disaggregation import pd_flip_controller as controller_module


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
