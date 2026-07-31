from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.manager.env import Envs


def test_predictor_path_resolves_from_sim_config_ancestor(tmp_path, monkeypatch):
    repro = tmp_path / "repro"
    config_path = repro / "configs" / "model" / "simulator.json"
    predictor_path = repro / "workloads" / "replay.json"
    config_path.parent.mkdir(parents=True)
    predictor_path.parent.mkdir(parents=True)
    config_path.write_text("{}")
    predictor_path.write_text("{}")

    unrelated_cwd = tmp_path / "workspace"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setattr(Envs, "config_path", classmethod(lambda cls: str(config_path)))

    resolved = ConfigManager.resolve_config_relative_path("workloads/replay.json")
    assert resolved == str(predictor_path)


def test_predictor_absolute_path_is_unchanged(tmp_path):
    predictor_path = tmp_path / "replay.json"
    predictor_path.write_text("{}")

    resolved = ConfigManager.resolve_config_relative_path(str(predictor_path))
    assert resolved == str(predictor_path)
