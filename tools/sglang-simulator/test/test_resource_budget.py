import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from sglang_simulator.simulation.resource_budget import estimate_infercast_budget
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.data_type import DataType
from sglang_simulator.spec.model import ModelInfo

_REVISION = "a" * 40


def _config(**overrides):
    return {
        "provider": "infercast",
        "provider_revision": _REVISION,
        "model_id": "new-org/new-dense-model",
        "policy": {
            "schema_version": 1,
            "mem_fraction_static": 0.8,
            "runtime_reserve_bytes": 2 * (1 << 30),
        },
        **overrides,
    }


def _target(**overrides):
    return SchedulerConfig(
        **{
            "data_type": DataType.BF16,
            "tp_size": 8,
            "ep_size": 2,
            "pp_size": 1,
            "kv_cache_data_type": DataType.FP8,
            "page_size": 16,
            **overrides,
        }
    )


def _hw(capacity=288):
    return AcceleratorInfo(
        name="target-mi355x",
        vendor="AMD",
        hbm_capacity_gb=capacity,
        hbm_bandwidth_gb=8000,
    )


def _model(tmp_path):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "hidden_size": 16,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return ModelInfo(
        model_path=tmp_path,
        hf_config=config,
        torch_dtype="bfloat16",
        attention_arch="MHA",
    )


@pytest.fixture
def sdk_double(monkeypatch):
    package = ModuleType("infercast")
    package.__path__ = []
    package.__revision__ = _REVISION
    sdk = ModuleType("infercast.sdk")
    sdk.calls = []

    @dataclass
    class Topology:
        tp_size: int = 1
        ep_size: int = 1
        pp_size: int = 1

        def to_dict(self):
            return {
                "schema_version": 1,
                "tp_size": self.tp_size,
                "ep_size": self.ep_size,
                "pp_size": self.pp_size,
            }

    class Policy:
        @classmethod
        def from_dict(cls, payload):
            if (
                set(payload)
                != {"schema_version", "mem_fraction_static", "runtime_reserve_bytes"}
                or type(payload["schema_version"]) is not int
                or payload["schema_version"] != 1
            ):
                raise ValueError("invalid budget policy")
            result = cls()
            result.payload = dict(payload)
            return result

        def to_dict(self):
            return dict(self.payload)

    def build(model_id, **kwargs):
        sdk.calls.append(("build", model_id, kwargs))
        sdk.resources.provenance.config_sha256 = hashlib.sha256(
            kwargs["path"].read_bytes()
        ).hexdigest()
        return sdk.resources

    def estimate(resources, **kwargs):
        sdk.calls.append(("estimate", resources, kwargs))
        return sdk.budget

    sdk.resources = SimpleNamespace(
        digest="b" * 64,
        family="LLAMA",
        tensors=(
            SimpleNamespace(
                role="embedding", storage=SimpleNamespace(data_dtype="bfloat16")
            ),
        ),
        provenance=SimpleNamespace(
            to_dict=lambda: {
                "verified_revision": None,
                "config_sha256": sdk.resources.provenance.config_sha256,
            }
        ),
    )
    sdk.budget = SimpleNamespace(
        max_cache_tokens=1024,
        status="fits_declared_budget",
        ranks=(
            SimpleNamespace(
                to_dict=lambda: {
                    "schema_version": 1,
                    "rank_id": 0,
                    "resident_data_bytes": 1234,
                    "available_cache_bytes": 1000,
                    "max_cache_tokens": 1024,
                    "tensors": [{"name": "large-ledger"}],
                }
            ),
        ),
    )
    sdk.ResourceTopology, sdk.MemoryBudgetPolicy = Topology, Policy
    sdk.build_model_resources, sdk.estimate_memory_budget = build, estimate
    monkeypatch.setitem(sys.modules, "infercast", package)
    monkeypatch.setitem(sys.modules, "infercast.sdk", sdk)
    return sdk


def test_target_topology_local_config_and_explicit_policy(sdk_double, tmp_path):
    model = _model(tmp_path)
    config = _config(model_revision="d" * 40, shared_expert_tp_size=4)
    tokens, metrics = estimate_infercast_budget(model, _hw(), _target(), config)

    assert tokens == 1024
    method, model_id, arguments = sdk_double.calls[0]
    assert method == "build" and model_id == config["model_id"]
    assert arguments["path"] == tmp_path / "config.json"
    assert arguments["topology"].to_dict() == {
        "schema_version": 1,
        "tp_size": 8,
        "ep_size": 2,
        "pp_size": 1,
    }
    assert arguments["kv_cache_dtype"] == "FP8"
    assert arguments["kv_page_size"] == 16
    assert arguments["revision"] == "d" * 40
    assert arguments["shared_expert_tp_size"] == 4
    assert sdk_double.calls[1][2]["device_capacity_bytes"] == 288 * (1 << 30)
    assert sdk_double.calls[1][2]["policy"].to_dict() == config["policy"]
    assert metrics["provider"] == "infercast"
    assert metrics["resource_digest"] == "b" * 64
    assert metrics["max_total_tokens"] == tokens
    assert "tensors" not in metrics["ranks"][0]
    assert metrics["provenance"]["verified_revision"] is None
    json.dumps(metrics, allow_nan=False)


def test_default_page_size_and_revision_are_explicit(sdk_double, tmp_path):
    estimate_infercast_budget(
        _model(tmp_path), _hw(), _target(page_size=None), _config()
    )
    assert sdk_double.calls[0][2]["kv_page_size"] == 1
    assert sdk_double.calls[0][2]["revision"] is None
    assert sdk_double.calls[0][2]["shared_expert_tp_size"] is None


@pytest.mark.parametrize(
    "fields",
    [
        {"extra": 1},
        {"provider": "aiconfigurator"},
        {"provider_revision": "main"},
        {"provider_revision": "A" * 40},
        {"model_id": ""},
        {"model_id": " padded "},
        {"model_revision": "main"},
        {"policy": None},
    ],
)
def test_bad_configuration_fails_before_sdk_queries(sdk_double, tmp_path, fields):
    with pytest.raises(ValueError):
        estimate_infercast_budget(_model(tmp_path), _hw(), _target(), _config(**fields))
    assert sdk_double.calls == []


@pytest.mark.parametrize("actual_revision", [None, "main", "c" * 40])
def test_provider_source_revision_must_match(sdk_double, tmp_path, actual_revision):
    sys.modules["infercast"].__revision__ = actual_revision
    with pytest.raises(ValueError, match="does not match installed"):
        estimate_infercast_budget(_model(tmp_path), _hw(), _target(), _config())
    assert sdk_double.calls == []


@pytest.mark.parametrize("capacity", [None, True, 0, -1, math.nan, math.inf, "288"])
def test_target_memory_is_required_without_host_fallback(
    sdk_double, tmp_path, capacity
):
    with pytest.raises(ValueError, match="hbm_capacity_gb"):
        estimate_infercast_budget(_model(tmp_path), _hw(capacity), _target(), _config())
    assert sdk_double.calls == []


@pytest.mark.parametrize("field", ["dp_size", "cp_size"])
@pytest.mark.parametrize("value", [2, True])
def test_unsupported_target_topologies_fail_closed(sdk_double, tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        estimate_infercast_budget(
            _model(tmp_path), _hw(), _target(**{field: value}), _config()
        )
    assert sdk_double.calls == []


@pytest.mark.parametrize(
    "status,tokens",
    [
        ("over_budget", 0),
        ("fits_declared_budget", 0),
        ("fits_declared_budget", True),
        ("fits_declared_budget", -1),
    ],
)
def test_exhausted_or_invalid_capacity_is_never_published(
    sdk_double, tmp_path, status, tokens
):
    sdk_double.budget.status, sdk_double.budget.max_cache_tokens = status, tokens
    with pytest.raises(ValueError, match="no usable KV capacity"):
        estimate_infercast_budget(_model(tmp_path), _hw(), _target(), _config())


def test_sdk_failure_propagates_without_fallback(sdk_double, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("unsupported resource family")

    sdk_double.build_model_resources = fail
    with pytest.raises(ValueError, match="unsupported resource family"):
        estimate_infercast_budget(_model(tmp_path), _hw(), _target(), _config())
    assert sdk_double.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 32),
        ("num_hidden_layers", 3),
        ("num_attention_heads", 16),
        ("intermediate_size", 64),
        ("quantization_config", {"quant_method": "fp8"}),
    ],
)
def test_effective_model_config_cannot_change_resource_structure(
    sdk_double, tmp_path, field, value
):
    model = _model(tmp_path)
    model.hf_config[field] = value
    with pytest.raises(ValueError, match="geometry differs"):
        estimate_infercast_budget(model, _hw(), _target(), _config())
    assert [call[0] for call in sdk_double.calls] == ["build"]


def test_loaded_geometry_cannot_override_materialized_config(sdk_double, tmp_path):
    model = _model(tmp_path)
    model.hidden_size = 32
    with pytest.raises(ValueError, match="geometry differs"):
        estimate_infercast_budget(model, _hw(), _target(), _config())


def test_effective_dtype_cannot_override_resource_storage(sdk_double, tmp_path):
    model = _model(tmp_path)
    model.torch_dtype = "float32"
    with pytest.raises(ValueError, match="torch_dtype differs"):
        estimate_infercast_budget(model, _hw(), _target(), _config())


def test_target_dtype_cannot_override_resource_storage(sdk_double, tmp_path):
    with pytest.raises(ValueError, match="target data_type differs"):
        estimate_infercast_budget(
            _model(tmp_path), _hw(), _target(data_type=DataType.FP32), _config()
        )


@pytest.mark.parametrize("dtype", [None, DataType.BF16])
def test_target_auto_or_matching_dtype_is_supported(sdk_double, tmp_path, dtype):
    tokens, _ = estimate_infercast_budget(
        _model(tmp_path), _hw(), _target(data_type=dtype), _config()
    )
    assert tokens == 1024


def test_materialized_hf_config_object_is_supported(sdk_double, tmp_path):
    model = _model(tmp_path)
    config = model.hf_config
    model.hf_config = SimpleNamespace(to_dict=lambda: dict(config))
    tokens, _ = estimate_infercast_budget(model, _hw(), _target(), _config())
    assert tokens == 1024


def test_loaded_attention_layout_must_match_resource_family(sdk_double, tmp_path):
    model = _model(tmp_path)
    model.attention_arch = "MLA"
    with pytest.raises(ValueError, match="attention layout differs"):
        estimate_infercast_budget(model, _hw(), _target(), _config())


def test_policy_unknown_fields_are_rejected(sdk_double, tmp_path):
    config = _config()
    config["policy"]["hidden_reserve"] = 1
    with pytest.raises(ValueError, match="invalid budget policy"):
        estimate_infercast_budget(_model(tmp_path), _hw(), _target(), config)
    assert sdk_double.calls == []


def test_module_import_keeps_optional_sdk_and_aic_lazy():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'infercast', 'aiconfigurator', 'torch'}:
        raise AssertionError('unexpected optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import sglang_simulator.simulation.resource_budget
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH=str(source)),
        check=True,
        capture_output=True,
        text=True,
    )


def test_real_sdk_small_dense_budget(tmp_path, monkeypatch):
    pytest.importorskip("infercast.sdk.resources.models")
    import infercast

    # Exercise actual resource contracts without depending on the wheel build used by CI.
    monkeypatch.setattr(infercast, "__revision__", _REVISION)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    target = _target(
        tp_size=2, ep_size=1, pp_size=2, kv_cache_data_type=DataType.BF16, page_size=4
    )
    model = ModelInfo(
        model_path=tmp_path,
        hf_config=config,
        torch_dtype="bfloat16",
        attention_arch="MHA",
    )
    budget_config = _config(
        policy={
            "schema_version": 1,
            "mem_fraction_static": 0.8,
            "runtime_reserve_bytes": 1 << 20,
        }
    )
    count, metrics = estimate_infercast_budget(model, _hw(1), target, budget_config)
    assert count > 0 and count % 4 == 0
    assert len(metrics["ranks"]) == 4
    assert metrics["topology"]["tp_size"] == 2
    assert all("tensors" not in rank for rank in metrics["ranks"])
    assert metrics["provenance"]["verified_revision"] is None
    assert count == min(rank["max_cache_tokens"] for rank in metrics["ranks"])
    model.torch_dtype = "float32"
    with pytest.raises(ValueError, match="torch_dtype differs"):
        estimate_infercast_budget(model, _hw(1), target, budget_config)
