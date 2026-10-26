"""JSON DSPark construction routing; no cost or progress evidence is fabricated."""

import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.manager.env import Envs


def config():
    return {
        "platform": {"accelerator": {"name": "MI355X"}},
        "predictor": {
            "name": "infercast_dspark",
            "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
            "model_revision": "a" * 40,
            "provider_revision": "b" * 40,
            "model_config_root": "model-cache",
            "systems_root": "systems",
            "runtime_profile": {"path": "runtime.json", "sha256": "c" * 64},
            "progress_artifact": {"path": "progress.json", "sha256": "d" * 64},
            "proposal_profile_id": "e" * 64,
        },
    }


@pytest.fixture
def factory(monkeypatch, tmp_path):
    calls = []
    module = ModuleType("sglang_simulator.time_predictor.infercast_dspark")

    def create(model, **kwargs):
        result = SimpleNamespace(model=model, **kwargs)
        calls.append(result)
        return result

    module.InferCastDSparkTimePredictor = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(Envs, "config_path", lambda: str(tmp_path / "sim.json"))
    raw = config()
    monkeypatch.setattr(ConfigManager, "_raw_config", raw)
    return calls, raw, tmp_path


def test_json_factory_uses_config_directory_and_preserves_original(
    factory, monkeypatch
):
    calls, raw, folder = factory
    original = copy.deepcopy(raw)
    # Existing files in another cwd must not change exact asset selection.
    elsewhere = folder / "unrelated"
    elsewhere.mkdir()
    (elsewhere / "runtime.json").write_text("unrelated", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    model, hw, scheduler = object(), object(), object()
    result = ConfigManager.get_inference_time_predictor(model, hw, scheduler)
    assert calls == [result]
    assert result.model is model and result.hw is hw and result.config is scheduler
    assert result.system == "MI355X"
    assert result.systems_root == str((folder / "systems").resolve())
    assert result.model_config_root == str((folder / "model-cache").resolve())
    assert result.runtime_profile == {
        "path": str((folder / "runtime.json").resolve()),
        "sha256": "c" * 64,
    }
    assert result.progress_artifact == {
        "path": str((folder / "progress.json").resolve()),
        "sha256": "d" * 64,
    }
    assert raw == original


@pytest.mark.parametrize("field", ["phase_preparers", "fallback_seconds", "factory"])
def test_json_factory_rejects_opaque_callbacks_and_fake_cost_options(factory, field):
    calls, raw, _ = factory
    raw["predictor"][field] = "unsupported"
    with pytest.raises(ValueError, match="unknown InferCast DSPark"):
        ConfigManager.get_inference_time_predictor(None, None, None)
    assert not calls


def test_malformed_asset_is_passed_to_predictor_validator_not_repaired(factory):
    calls, raw, _ = factory
    raw["predictor"]["runtime_profile"] = {"sha256": "c" * 64}
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.runtime_profile == {"sha256": "c" * 64}
    assert calls == [result]


def test_golden_kind_and_explicit_model_seed_route_without_empirical_asset(factory):
    _, raw, _ = factory
    raw["predictor"].pop("progress_artifact")
    raw["predictor"].update(progress_kind="source_golden351", model_seed=42)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.progress_kind == "source_golden351" and result.model_seed == 42
    assert not hasattr(result, "progress_artifact")


def test_explicit_source_work_option_routes_without_mutating_config(factory):
    _, raw, _ = factory
    raw["predictor"]["native_source_work_policy"] = (
        "native_region_golden351_decode_work_v1"
    )
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert (
        result.native_source_work_policy
        == raw["predictor"]["native_source_work_policy"]
    )
    assert raw == original


def test_native_segment_selector_and_existing_capture_profile_resolve_from_config(
    factory,
):
    _, raw, folder = factory
    raw["predictor"].pop("proposal_profile_id")
    raw["predictor"].update(
        native_segment_profile_id="e" * 64,
        native_capture_profile={"path": "capture.json", "sha256": "f" * 64},
    )
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_segment_profile_id == "e" * 64
    assert result.native_capture_profile == {
        "path": str((folder / "capture.json").resolve()),
        "sha256": "f" * 64,
    }
    assert raw == original


def test_initial_buffer_witness_resolves_from_config_without_mutation(factory):
    _, raw, folder = factory
    raw["predictor"]["native_initial_draft_buffer"] = {
        "path": "warm-before-work-query.json",
        "sha256": "f" * 64,
    }
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_initial_draft_buffer == {
        "path": str((folder / "warm-before-work-query.json").resolve()),
        "sha256": "f" * 64,
    }
    assert raw == original


def test_explicit_hybrid_model_and_output_resolve_from_config(factory):
    _, raw, folder = factory
    raw["predictor"].update(
        cost_mode="HYBRID",
        native_analytic_model={"path": "analytic.json", "sha256": "e" * 64},
        native_cost_evidence_path="run/costs.jsonl",
    )
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.cost_mode == "HYBRID"
    assert result.native_analytic_model == {
        "path": str((folder / "analytic.json").resolve()),
        "sha256": "e" * 64,
    }
    assert result.native_cost_evidence_path == str(
        (folder / "run/costs.jsonl").resolve()
    )
    assert raw == original


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("filename", ["costs.jsonl", "costs", "costs.jsonl.gz"])
def test_joint_ledger_path_uses_explicit_attention_rank(factory, rank, filename):
    _, raw, folder = factory
    raw["predictor"].update(
        native_joint_source_binding={"root": "source"},
        native_cost_evidence_path=filename,
    )
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(
        None, None, None, attention_dp_rank=rank
    )
    path = folder / filename
    assert result.native_cost_evidence_path == str(
        path.with_name(f"{path.stem}.dp{rank}{path.suffix}").resolve()
    )
    assert not hasattr(result, "attention_dp_rank")
    assert raw == original


@pytest.mark.parametrize("rank", [None, False, True, -1, 2, "1", 1.0])
def test_joint_ledger_rejects_unknown_or_invalid_rank_before_constructor(factory, rank):
    calls, raw, _ = factory
    raw["predictor"].update(
        native_joint_source_binding={"root": "source"},
        native_cost_evidence_path="costs.jsonl",
    )
    with pytest.raises(ValueError, match="attention-DP rank 0 or 1"):
        ConfigManager.get_inference_time_predictor(
            None, None, None, attention_dp_rank=rank
        )
    assert not calls


def test_rank_does_not_change_adp1_path_or_ledger_disabled_joint(factory):
    _, raw, folder = factory
    raw["predictor"]["native_cost_evidence_path"] = "costs.jsonl"
    for rank in (None, 0, 1):
        result = ConfigManager.get_inference_time_predictor(
            None, None, None, attention_dp_rank=rank
        )
        assert result.native_cost_evidence_path == str(
            (folder / "costs.jsonl").resolve()
        )
    raw["predictor"]["native_joint_source_binding"] = {"root": "source"}
    raw["predictor"]["native_cost_evidence_path"] = None
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_cost_evidence_path is None
    assert raw == original


@pytest.mark.parametrize("value", ["", " ", "."])
def test_joint_does_not_turn_blank_or_directory_path_into_fresh_file(factory, value):
    _, raw, folder = factory
    raw["predictor"].update(
        native_joint_source_binding={"root": "source"},
        native_cost_evidence_path=value,
    )
    result = ConfigManager.get_inference_time_predictor(
        None, None, None, attention_dp_rank=0
    )
    assert result.native_cost_evidence_path == ConfigManager.resolve_infercast_path(
        value
    )
    assert not list(folder.glob("*.dp0*"))


def test_explicit_native_source_root_resolves_without_rewriting_descriptors(factory):
    _, raw, folder = factory
    binding = {
        "root": "native-source",
        "profile": {"path": "_meta/profile.json", "sha256": "a" * 64},
        "cohort": {"path": "_meta/cohort.json", "sha256": "b" * 64},
        "stack": {"path": "_meta/stack.json", "sha256": "c" * 64},
    }
    raw["predictor"]["native_source_binding"] = binding
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_source_binding == {
        **binding,
        "root": str((folder / "native-source").absolute()),
    }
    assert raw == original


def test_source_root_link_identity_is_left_for_factory_validation(factory, monkeypatch):
    _, raw, folder = factory
    alias, target = folder / "source-alias", folder / "actual-source"
    resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, *a, **kw: target if path == alias else resolve(path, *a, **kw),
    )
    raw["predictor"]["native_source_binding"] = {"root": str(alias)}
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_source_binding["root"] == str(alias.absolute())


@pytest.mark.parametrize("root", ["", " "])
def test_blank_source_root_is_not_repaired_into_config_directory(factory, root):
    _, raw, _ = factory
    raw["predictor"]["native_source_binding"] = {"root": root}
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_source_binding == {"root": root}


@pytest.mark.parametrize("invalid", [123, "", " "])
def test_model_config_root_is_optional_and_malformed_values_are_not_repaired(
    factory, invalid
):
    calls, raw, _ = factory
    raw["predictor"].pop("model_config_root")
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert not hasattr(result, "model_config_root")
    raw["predictor"]["model_config_root"] = invalid
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.model_config_root == invalid
    assert calls[-1] is result


@pytest.mark.parametrize("relative", [True, False])
def test_model_config_root_keeps_requested_link_for_native_binding(
    factory, monkeypatch, relative
):
    _, raw, folder = factory
    alias, target = folder / "cache-alias", folder / "actual-cache"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        return target if path == alias else original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    raw["predictor"]["model_config_root"] = alias.name if relative else str(alias)
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.model_config_root == str(alias.absolute())
    assert result.model_config_root != str(target)
    assert raw == original
