# SPDX-License-Identifier: Apache-2.0
"""Frozen native encoder decisions reuse the ordinary materializer."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.component_loaders import text_encoder_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    ResolvedTextEncoderLoad,
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.models.encoders.base import (
    EncoderTensorParallelMixin,
)


class TinyEncoder(EncoderTensorParallelMixin, torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(config.width))

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            with torch.no_grad():
                self.get_parameter(name).copy_(value)
            loaded.add(name)
        return loaded


def recipe(tmp_path):
    return ResolvedTextEncoderLoad(
        model_cls=TinyEncoder,
        config=SimpleNamespace(
            width=3, quant_config=None, arch_config=SimpleNamespace()
        ),
        hf_config={"width": 3},
        model_path=str(tmp_path),
        weights_path=str(tmp_path),
        server_args=SimpleNamespace(num_gpus=1),
        component_name="text_encoder",
        dtype="bf16",
        component_starts_on_cpu=False,
        weight_files=(str(tmp_path / "model.safetensors"),),
    )


def test_encoder_recipe_freeze_detaches_mutable_resolution(tmp_path):
    resolved = recipe(tmp_path)
    frozen = resolved.freeze()
    resolved.config.width = 999
    resolved.hf_config["width"] = 999
    resolved.server_args.num_gpus = 8
    first = frozen.thaw()
    assert first.config.width == first.hf_config["width"] == 3
    assert first.server_args.num_gpus == 1
    first.config.width = 777
    assert frozen.thaw().config.width == 3
    with pytest.raises(AttributeError):
        frozen._payload = b"changed"


def test_encoder_materialization_uses_exact_class_config_and_files(tmp_path):
    resolved = recipe(tmp_path)
    save_file({"weight": torch.tensor([1.0, 2.0, 3.0])}, resolved.weight_files[0])
    # Discovery of this new file would fail the load; the frozen list wins.
    save_file({"unexpected": torch.zeros(2)}, str(tmp_path / "extra.safetensors"))
    frozen = resolved.freeze()
    loader = TextEncoderLoader()
    with (
        patch.object(
            text_encoder_loader,
            "get_local_torch_device",
            return_value=torch.device("cpu"),
        ),
        patch.object(text_encoder_loader, "get_folding_tp_group", return_value=None),
        patch.object(
            text_encoder_loader, "use_tensor_parallel_group", return_value=nullcontext()
        ),
        patch.object(
            text_encoder_loader.ModelRegistry,
            "resolve_model_cls",
            side_effect=AssertionError("rediscovery"),
        ),
        patch.object(
            loader, "_get_all_weights", side_effect=AssertionError("file rediscovery")
        ),
    ):
        model = loader.materialize_prepared(frozen.thaw())
    torch.testing.assert_close(
        model.weight, torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    )
    assert type(model) is TinyEncoder


def test_ordinary_encoder_uses_same_resolver_and_materializer(tmp_path):
    loader = TextEncoderLoader()
    resolved = recipe(tmp_path)
    with (
        patch.object(loader, "prepare_customized", return_value=resolved) as prepare,
        patch.object(
            loader, "materialize_prepared", return_value="model"
        ) as materialize,
    ):
        assert (
            loader.load_customized("path", resolved.server_args, "text_encoder")
            == "model"
        )
    prepare.assert_called_once_with("path", resolved.server_args, "text_encoder", None)
    materialize.assert_called_once_with(resolved)
