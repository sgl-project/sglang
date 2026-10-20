import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.sglang.prepared_batch import capture_prepared_batch
from sglang_simulator.simulation.sglang.scheduler import _request_prefix_len
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor import PredictorError


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


def test_infercast_systems_resolve_from_config_directory(tmp_path, monkeypatch):
    config = tmp_path / "config" / "simulator.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Envs, "config_path", classmethod(lambda cls: str(config)))
    monkeypatch.chdir(tmp_path)

    assert ConfigManager.resolve_infercast_path("systems") == str(
        (config.parent / "systems").resolve()
    )


def test_existing_predictor_paths_keep_cwd_and_ancestor_resolution(
    tmp_path, monkeypatch
):
    config = tmp_path / "config" / "nested" / "simulator.json"
    config.parent.mkdir(parents=True)
    monkeypatch.setattr(Envs, "config_path", classmethod(lambda cls: str(config)))
    asset = tmp_path / "replay.json"
    asset.write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert ConfigManager.resolve_config_relative_path("replay.json") == str(asset)
    monkeypatch.chdir(config.parent)
    assert ConfigManager.resolve_config_relative_path("replay.json") == str(asset)
    assert ConfigManager.resolve_config_relative_path("missing.json") == "missing.json"


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
            "platform": {"accelerator": {"name": "mi350x"}},
            "predictor": {
                "name": "infercast",
                "model_id": "model",
                "systems_root": "systems",
                "attn_kernel_impl": "eager",
                "decode_attn_kernel_impl": "cuda_graph",
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "model_revision": "b" * 40,
                "provider_revision": "a" * 40,
            },
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
    assert captured["system"] == "mi350x"
    assert captured["provider_revision"] == "a" * 40
    assert captured["model_revision"] == "b" * 40
    assert "contract_version" not in captured
    assert "reduction_policy" not in captured
    assert captured["decode_attn_kernel_impl"] == "cuda_graph"


@pytest.mark.parametrize("field", ["contract_version", "reduction_policy"])
def test_config_manager_rejects_obsolete_infercast_configuration(field, monkeypatch):
    monkeypatch.setattr(
        ConfigManager,
        "_raw_config",
        {
            "platform": {"accelerator": {"name": "mi355x"}},
            "predictor": {"name": "infercast", field: "obsolete"},
        },
    )

    with pytest.raises(ValueError, match="obsolete InferCast predictor"):
        ConfigManager.get_inference_time_predictor(
            ModelInfo(),
            AcceleratorInfo(
                name="mi355x",
                vendor="AMD",
                hbm_capacity_gb=1,
                hbm_bandwidth_gb=1,
            ),
            SchedulerConfig(),
        )


@pytest.mark.parametrize(
    "field", ["prefill_scale_factor", "decode_scale_factor", "typo"]
)
def test_config_manager_rejects_unknown_infercast_configuration(field, monkeypatch):
    monkeypatch.setattr(
        ConfigManager,
        "_raw_config",
        {"predictor": {"name": "infercast", field: 1.0}},
    )
    with pytest.raises(ValueError, match="unknown InferCast predictor"):
        ConfigManager.get_inference_time_predictor(ModelInfo(), None, SchedulerConfig())


def _batch(mode, **fields):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode),
        batch_size=2,
        **fields,
    )


def test_prepared_extend_metadata_is_authoritative():
    batch = _batch(
        "MIXED",
        seq_lens_cpu=[101, 8],
        extend_seq_lens_cpu=[1, 8],
        extend_prefix_lens_cpu=[100, 0],
    )
    assert capture_prepared_batch(batch).predictor_batch().request_info() == [
        [1, 100],
        [8, 0],
    ]


def test_prepared_decode_metadata_is_authoritative():
    batch = _batch("DECODE", seq_lens_cpu=[100, 201])
    assert capture_prepared_batch(batch).predictor_batch().request_info() == [
        [1, 99],
        [1, 200],
    ]


def test_request_prefix_length_supports_current_and_legacy_cores():
    assert _request_prefix_len(SimpleNamespace(prefix_len=64)) == 64
    assert _request_prefix_len(SimpleNamespace(prefix_indices=[0, 1, 2])) == 3


@pytest.mark.parametrize("value", [True, 1.5, SimpleNamespace(item=lambda: 1.5)])
def test_prepared_lengths_are_not_coerced_to_integers(value):
    with pytest.raises(PredictorError) as exc_info:
        capture_prepared_batch(
            _batch(
                "EXTEND",
                seq_lens_cpu=[8, 8],
                extend_seq_lens_cpu=[value, 8],
                extend_prefix_lens_cpu=[0, 0],
            )
        )
    assert exc_info.value.code == "invalid_batch"


@pytest.mark.parametrize("mode", ["IDLE", "TARGET_VERIFY", "DRAFT_EXTEND", None])
def test_unsupported_forward_mode_fails_explicitly(mode):
    with pytest.raises(PredictorError) as exc_info:
        capture_prepared_batch(_batch(mode))
    assert exc_info.value.code == "unsupported_forward_mode"
