# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.adapter_loader import (
    AdapterLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    ComponentLoader,
    GenericComponentLoader,
    PipelineComponentLoader,
    PlainStateDictComponentLoader,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


class _ProjectionPipeline(ComposedPipelineBase):
    _required_config_modules = ["projection"]
    component_loaders = {"projection": PlainStateDictComponentLoader}

    def initialize_pipeline(self, server_args):
        pass

    def create_pipeline_stages(self, server_args):
        pass


@pytest.fixture
def checkpoint(tmp_path):
    component = tmp_path / "projection"
    component.mkdir()
    config = {
        "_class_name": "PlainLoaderTestProjection",
        "_diffusers_version": "0",
        "_name_or_path": "unused",
        "in_features": 4,
        "out_features": 3,
    }
    (component / "config.json").write_text(json.dumps(config))
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ProjectionPipeline",
                "_diffusers_version": "0",
                "projection": ["diffusers", "PlainLoaderTestProjection"],
                "scheduler": None,
            }
        )
    )
    weights = {"weight": torch.arange(12).reshape(3, 4).float(), "bias": torch.ones(3)}
    save_file(weights, component / "model.safetensors")
    with patch.dict(ModelRegistry.registered_models):
        ModelRegistry.register_model("PlainLoaderTestProjection", nn.Linear)
        yield component, weights


@pytest.fixture
def server_args():
    return SimpleNamespace(
        component_paths={},
        component_weights_paths={},
        component_precisions={},
        component_quantizations={},
        component_direct_gpu_weight_loading=set(),
        model_paths={},
        model_subfolder=None,
        revision=None,
        pipeline_config=SimpleNamespace(
            dit_precision="bf16", native_only_components=[]
        ),
        should_start_component_on_cpu=lambda _name: True,
        should_direct_gpu_weight_load_component=lambda _name: False,
        should_use_fsdp_for_component=lambda _name: False,
        resolve_component_attention_backend=lambda *_names: (None, None),
        requested_component_attention_backend=lambda _name: None,
    )


def _load(component, server_args, **kwargs):
    with patch.object(current_platform, "get_available_gpu_memory", return_value=16.0):
        model, _ = PipelineComponentLoader.load_component(
            "projection",
            str(component),
            "diffusers",
            server_args,
            loader_cls=PlainStateDictComponentLoader,
            **kwargs,
        )
    return model


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("precision", [None, "fp32"])
def test_pipeline_loads_strict_state_dict(checkpoint, server_args, sharded, precision):
    component, weights = checkpoint
    if sharded:
        for key, weight in weights.items():
            save_file({key: weight}, component / f"{key}.safetensors")
        (component / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: f"{key}.safetensors" for key in weights}})
        )
        # The index must win over this unrelated full checkpoint.
        save_file({"wrong": torch.zeros(1)}, component / "model.safetensors")
    if precision is not None:
        server_args.component_precisions["projection"] = precision

    pipeline = object.__new__(_ProjectionPipeline)
    pipeline.model_path = str(component.parent)
    pipeline.server_args = server_args
    pipeline._disagg_role = RoleType.MONOLITHIC
    pipeline.memory_usages = {}
    with patch.object(current_platform, "get_available_gpu_memory", return_value=16.0):
        model = pipeline.load_modules(server_args)["projection"]

    expected_dtype = torch.float32 if precision else torch.bfloat16
    assert not model.training
    assert model.weight.device.type == "cpu"
    assert model.weight.dtype == expected_dtype
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, weights[key].to(expected_dtype))
    assert server_args.model_paths == {"projection": str(component)}


def test_weight_override_and_architecture_fallback(checkpoint, server_args, tmp_path):
    component, weights = checkpoint
    config_path = component / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("_class_name")
    config_path.write_text(json.dumps(config))
    override = tmp_path / "override.safetensors"
    replacement = {key: value + 1 for key, value in weights.items()}
    save_file(replacement, override)
    server_args.component_weights_paths["projection"] = str(override)
    model = _load(
        component, server_args, component_architecture="PlainLoaderTestProjection"
    )
    torch.testing.assert_close(model.weight, replacement["weight"].bfloat16())


@pytest.mark.parametrize(
    "failure", ["missing", "unexpected", "shape", "config", "quantized"]
)
def test_explicit_loader_never_falls_back(checkpoint, server_args, failure):
    component, weights = checkpoint
    if failure == "missing":
        weights.pop("bias")
    elif failure == "unexpected":
        weights["extra"] = torch.zeros(1)
    elif failure == "shape":
        weights["bias"] = torch.zeros(4)
    else:
        config_path = component / "config.json"
        config = json.loads(config_path.read_text())
        if failure == "config":
            config["unsupported_argument"] = True
        else:
            config["quantization_config"] = {"quant_method": "fp8"}
        config_path.write_text(json.dumps(config))
    save_file(weights, component / "model.safetensors")
    with patch.object(ComponentLoader, "load_native") as native:
        with pytest.raises(
            (RuntimeError, TypeError, ComponentCheckpointUnsupportedError)
        ):
            _load(component, server_args)
    native.assert_not_called()


def test_explicit_selection_does_not_change_other_pipelines():
    selected = ComponentLoader.for_component_type(
        "duration_head_2", "ltx2", loader_cls=PlainStateDictComponentLoader
    )
    assert type(selected) is PlainStateDictComponentLoader
    assert selected.component_type == "duration_head_2"
    assert isinstance(
        ComponentLoader.for_component_type("duration_head_2", "ltx2"), AdapterLoader
    )
    assert isinstance(
        ComponentLoader.for_component_type("projection", "diffusers"),
        GenericComponentLoader,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_residency_and_forward(checkpoint, server_args):
    component, weights = checkpoint
    server_args.should_start_component_on_cpu = lambda _name: False
    model = _load(component, server_args)
    assert model.weight.device.type == "cuda"
    inputs = torch.ones(2, 4, device=model.weight.device, dtype=torch.bfloat16)
    expected = nn.functional.linear(
        inputs, weights["weight"].to(inputs), weights["bias"].to(inputs)
    )
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
