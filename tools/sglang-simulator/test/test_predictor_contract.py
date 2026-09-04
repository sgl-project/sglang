import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang_simulator.compat import apply_simulator_server_args
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.manager.state import StateManager
from sglang_simulator.simulation.sglang.scheduler import (
    build_predictor_batch,
    predict_schedule_batch,
)
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor import PredictorError, ScheduleBatch


def test_optional_predictors_are_not_imported_at_startup():
    src = Path(__file__).resolve().parents[1] / "src"
    script = """
import builtins
import sys

blocked = {"aiconfigurator", "infercast"}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise ImportError(f"blocked optional dependency: {name}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import sglang_simulator.time_predictor
import sglang_simulator.simulation.manager.config
assert not blocked.intersection(sys.modules)
"""
    env = dict(os.environ, PYTHONPATH=str(src))
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )


def test_simulator_disables_unmodeled_speculative_forwards():
    server_args = SimpleNamespace(speculative_algorithm="EAGLE")
    apply_simulator_server_args(server_args)
    assert server_args.speculative_algorithm is None


def test_predictor_paths_resolve_from_config_directory(tmp_path, monkeypatch):
    config = tmp_path / "config" / "simulator.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Envs, "config_path", classmethod(lambda cls: str(config)))
    monkeypatch.chdir(tmp_path)

    assert ConfigManager.resolve_config_relative_path("systems") == str(
        (config.parent / "systems").resolve()
    )


def test_config_manager_passes_infercast_configuration(tmp_path, monkeypatch):
    import sglang_simulator.time_predictor.infercast as infercast_module

    captured = {}

    class Predictor:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    config_path = tmp_path / "simulator.json"
    monkeypatch.setattr(Envs, "config_path", classmethod(lambda cls: str(config_path)))
    monkeypatch.setattr(infercast_module, "InferCastTimePredictor", Predictor)
    monkeypatch.setattr(
        ConfigManager,
        "_raw_config",
        {
            "predictor": {
                "name": "infercast",
                "model_id": "model",
                "systems_root": "systems",
                "attn_kernel_impl": "eager",
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "provider_revision": "a" * 40,
            }
        },
    )
    ConfigManager.get_inference_time_predictor(
        ModelInfo(),
        AcceleratorInfo(
            name="mi350x",
            vendor="AMD",
            hbm_capacity_gb=1,
            hbm_bandwidth_gb=1,
        ),
        SchedulerConfig(),
    )
    assert captured["systems_root"] == str((tmp_path / "systems").resolve())
    assert captured["model_id"] == "model"
    assert captured["provider_revision"] == "a" * 40


def _batch(mode, **fields):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode),
        reqs=[SimpleNamespace(), SimpleNamespace()],
        **fields,
    )


def test_prepared_extend_metadata_is_authoritative():
    batch = _batch(
        "MIXED",
        extend_lens=[1, 8],
        prefix_lens=[100, 0],
    )
    assert build_predictor_batch(batch).request_info() == [[1, 100], [8, 0]]


def test_prepared_decode_metadata_is_authoritative():
    batch = _batch("DECODE", seq_lens_cpu=[100, 201])
    assert build_predictor_batch(batch).request_info() == [[1, 100], [1, 201]]


def test_idle_batch_makes_no_predictor_work():
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(name="IDLE"),
        reqs=[],
    )
    assert build_predictor_batch(batch).is_empty()


def test_unsupported_forward_mode_fails_explicitly():
    with pytest.raises(PredictorError) as exc_info:
        build_predictor_batch(_batch("TARGET_VERIFY"))
    assert exc_info.value.code == "unsupported_forward_mode"


class _InvalidPredictor:
    def predict_infer_time(self, batch):
        return -1


def test_invalid_prediction_does_not_count_or_advance_time():
    StateManager.reset()
    with pytest.raises(PredictorError) as exc_info:
        predict_schedule_batch(_InvalidPredictor(), ScheduleBatch())
    assert exc_info.value.code == "invalid_provider_output"
    assert StateManager.get_iteration() == 0
    assert StateManager.get_global_clock() == 0
