# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
)
from sglang.multimodal_gen.runtime.models.dits.llada_image import (
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQModel,
    LLaDAImageTextProjectionModel,
)
from sglang.multimodal_gen.runtime.pipelines.llada_image import LLaDAImagePipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# ServerArgs uses the physical device to resolve residency policies.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_COMPONENTS = {
    "queryformer": (LLaDAImageQueryFormerModel, {"num_queries": 2}),
    "text_projection": (LLaDAImageTextProjectionModel, {"projection_dim": 4}),
    "sigvq": (
        LLaDAImageSigVQModel,
        {
            "image_size": 4,
            "patch_size": 2,
            "codebook_size": 8,
            "codebook_embed_dim": 4,
            "semantic_embed_dim": 4,
        },
    ),
}


@pytest.fixture(params=list(_COMPONENTS))
def checkpoint(request, tmp_path):
    name = request.param
    model_cls, extra_config = _COMPONENTS[name]
    config = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        **extra_config,
    )
    torch.manual_seed(0)
    reference = model_cls(**config)
    weights = {
        key: tensor.contiguous() for key, tensor in reference.state_dict().items()
    }
    component = tmp_path / name
    component.mkdir()
    config.update(_class_name=model_cls.__name__, _diffusers_version="0")
    (component / "config.json").write_text(json.dumps(config))
    save_file(weights, component / "model.safetensors")
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "LLaDAImagePipeline",
                "_diffusers_version": "0",
                name: ["diffusers", model_cls.__name__],
                "scheduler": None,
            }
        )
    )
    return name, component, weights


def _load(name, root, **overrides):
    server_args = ServerArgs(
        model_path=str(root),
        component_residency={name: COMPONENT_OFFLOAD},
        **overrides,
    )
    pipeline = object.__new__(LLaDAImagePipeline)
    pipeline.model_path = str(root)
    pipeline.server_args = server_args
    pipeline._disagg_role = RoleType.MONOLITHIC
    pipeline._required_config_modules = [name]
    pipeline.memory_usages = {}
    return pipeline.load_modules(server_args)[name], server_args


@pytest.mark.parametrize(
    "layout",
    ["single", "sharded", "weight_override", "component_override", "architecture"],
)
def test_auxiliary_checkpoint_loading(checkpoint, tmp_path, layout):
    name, component, weights = checkpoint
    root = component.parent
    overrides = {}
    expected_dtype = torch.bfloat16
    if layout in ("sharded", "weight_override"):
        overrides["component_precisions"] = {name: "fp32"}
        expected_dtype = torch.float32
    if layout == "sharded":
        names = list(weights)
        partitions = (names[::2], names[1::2])
        weight_map = {}
        for index, keys in enumerate(partitions):
            filename = f"shard-{index}.safetensors"
            save_file({key: weights[key] for key in keys}, component / filename)
            weight_map.update({key: filename for key in keys})
        (component / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        save_file({"unrelated": torch.zeros(1)}, component / "model.safetensors")
    elif layout == "weight_override":
        weights = {key: value + 1 for key, value in weights.items()}
        override = tmp_path / "override.safetensors"
        save_file(weights, override)
        overrides["component_weights_paths"] = {name: str(override)}
    elif layout == "component_override":
        override = tmp_path / "alternate-component"
        shutil.copytree(component, override)
        component = override
        weights = {key: value + 1 for key, value in weights.items()}
        save_file(weights, component / "model.safetensors")
        overrides["component_paths"] = {name: str(component)}
    elif layout == "architecture":
        config = json.loads((component / "config.json").read_text())
        config.pop("_class_name")
        (component / "config.json").write_text(json.dumps(config))

    model, server_args = _load(name, root, **overrides)

    assert isinstance(model, _COMPONENTS[name][0])
    assert not model.training
    assert server_args.model_paths[name] == str(component)
    actual = model.state_dict()
    assert actual.keys() == weights.keys()
    for key, value in actual.items():
        assert value.device.type == "cpu"
        assert value.dtype == expected_dtype
        torch.testing.assert_close(
            value, weights[key].to(expected_dtype), rtol=0, atol=0
        )


def test_incomplete_auxiliary_checkpoint_does_not_fallback(checkpoint):
    name, component, weights = checkpoint
    weights.pop(next(iter(weights)))
    save_file(weights, component / "model.safetensors")
    with patch.object(ComponentLoader, "load_native") as native:
        with pytest.raises(ComponentCheckpointUnsupportedError, match="Missing:"):
            _load(name, component.parent)
    native.assert_not_called()
