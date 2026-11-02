# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType as Task


class MultiConfig(PipelineConfig):
    supported_task_types = (Task.T2V, Task.I2V)


@pytest.mark.parametrize("prefix", ["", "custom"])
def test_request_task_and_capabilities_are_not_pipeline_overrides(prefix):
    config = MultiConfig(task_type=Task.T2V)
    stem = prefix + "." if prefix else ""
    args = {
        stem + "task_type": "i2v",
        stem + "supported_task_types": (Task.T2I,),
        stem + "dit_precision": "fp32",
    }
    config.update_config_from_dict(args, prefix)
    assert config.task_type == Task.T2V
    assert config.get_supported_task_types() == (Task.T2V, Task.I2V)
    assert config.dit_precision == "fp32"
    assert args[stem + "task_type"] == "i2v"


def test_config_construction_preserves_nested_default_and_request_task(monkeypatch):
    monkeypatch.setattr(
        registry,
        "get_model_info",
        lambda *a, **kw: SimpleNamespace(pipeline_config_cls=MultiConfig),
    )
    kwargs = {
        "model_path": "test-model",
        "task_type": "i2v",
        "pipeline_config": {"task_type": Task.T2V, "supported_task_types": (Task.I2V,)},
    }
    config = PipelineConfig.from_kwargs(kwargs)
    assert config.task_type == Task.T2V
    assert config.get_supported_task_types() == (Task.T2V, Task.I2V)
    assert kwargs["task_type"] == "i2v"


def test_request_cannot_expand_legacy_capabilities():
    config = PipelineConfig(task_type=Task.T2V)
    config.update_config_from_dict({"task_type": "i2v"})
    assert config.get_supported_task_types() == (Task.T2V,)
    with pytest.raises(ValueError, match="Unsupported task_type"):
        config.resolve_task_type("i2v")
