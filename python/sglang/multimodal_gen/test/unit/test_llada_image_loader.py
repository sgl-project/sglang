# SPDX-License-Identifier: Apache-2.0

import json
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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ServerArgs residency resolution requires CUDA",
)


@pytest.mark.parametrize(
    "name,model_cls,extra",
    [
        ("queryformer", LLaDAImageQueryFormerModel, {"num_queries": 2}),
        ("text_projection", LLaDAImageTextProjectionModel, {"projection_dim": 4}),
        (
            "sigvq",
            LLaDAImageSigVQModel,
            {
                "image_size": 4,
                "patch_size": 2,
                "codebook_size": 8,
                "codebook_embed_dim": 4,
                "semantic_embed_dim": 4,
            },
        ),
    ],
)
@pytest.mark.parametrize("missing_weight", [False, True])
def test_auxiliary_checkpoint_loading(tmp_path, name, model_cls, extra, missing_weight):
    config = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        **extra,
    )
    torch.manual_seed(0)
    weights = {
        key: value.contiguous()
        for key, value in model_cls(**config).state_dict().items()
    }
    component = tmp_path / name
    component.mkdir()
    (component / "config.json").write_text(
        json.dumps(dict(config, _class_name=model_cls.__name__, _diffusers_version="0"))
    )
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
    if missing_weight:
        weights.pop(next(iter(weights)))
    save_file(weights, component / "model.safetensors")
    args = ServerArgs(
        model_path=str(tmp_path), component_residency={name: COMPONENT_OFFLOAD}
    )
    pipeline = object.__new__(LLaDAImagePipeline)
    pipeline.model_path, pipeline.server_args = str(tmp_path), args
    pipeline._disagg_role = RoleType.MONOLITHIC
    pipeline._required_config_modules = [name]
    pipeline.memory_usages = {}
    if missing_weight:
        with (
            patch.object(ComponentLoader, "load_native") as native,
            pytest.raises(ComponentCheckpointUnsupportedError, match="Missing:"),
        ):
            pipeline.load_modules(args)
        native.assert_not_called()
        return
    model = pipeline.load_modules(args)[name]
    assert isinstance(model, model_cls) and not model.training
    assert args.model_paths[name] == str(component)
    assert model.state_dict().keys() == weights.keys()
    for key, value in model.state_dict().items():
        assert value.device.type == "cpu" and value.dtype == torch.bfloat16
        torch.testing.assert_close(value, weights[key].bfloat16(), rtol=0, atol=0)
