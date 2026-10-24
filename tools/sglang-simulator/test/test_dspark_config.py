"""JSON DSPark construction routing; no cost or progress evidence is fabricated."""

import copy
import sys
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
