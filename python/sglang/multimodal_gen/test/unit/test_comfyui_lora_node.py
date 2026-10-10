# SPDX-License-Identifier: Apache-2.0
"""SGLD LoRA: loader nodes and binding the sampled MODEL's LoRAs to the worker."""

import copy
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock, patch

import pytest
import torch

# Initialize the quantization registry before importing the LoRA layer modules.
import sglang.multimodal_gen.runtime.layers.quantization  # noqa: F401
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline


def _load_nodes():
    """Import the real nodes.py with the ComfyUI modules it needs stubbed."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_full_path = lambda folder, name: f"/{folder}/{name}"
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api.input = comfy_api_input
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }
    with mock.patch.dict(sys.modules, stubs):
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
    return nodes


class _Model:
    """ModelPatcher stand-in: clone() copies patches like SGLDModelPatcher.clone."""

    def __init__(self, executor, patches=None):
        self.model = SimpleNamespace(diffusion_model=executor)
        self.patches = dict(patches or {})
        self.model_options = {}

    def clone(self):
        clone = _Model(self.model.diffusion_model, self.patches)
        clone.model_options = copy.deepcopy(self.model_options)
        return clone


def test_chained_lora_nodes_keep_every_lora() -> None:
    nodes = _load_nodes()
    calls = []
    executor = SimpleNamespace(set_lora=lambda **kwargs: calls.append(kwargs))
    base = _Model(executor)

    loader = nodes.SGLDLoraLoader()
    (first,) = loader.load_lora(base, "style.safetensors", 1.0, nickname="style")
    (second,) = loader.load_lora(first, "detail.safetensors", 0.8, nickname="detail")

    assert calls == []
    desired = second.model_options["sgld_lora_input"]
    assert desired["lora_nickname"] == ["style", "detail"]
    assert desired["strength"] == [1.0, 0.8]
    assert desired["lora_path"] == [
        "/loras/style.safetensors",
        "/loras/detail.safetensors",
    ]
    assert "style" not in base.patches  # the upstream model is not mutated
    assert set(second.patches) == {"style", "detail"}


class _TupleLinear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class _Pipeline(LoRAPipeline):
    def create_pipeline_stages(self, args):
        return None


def _one_layer_pipeline():
    """Identity 2x2 layer; adapter A adds to output 0 and B to output 1."""
    linear = _TupleLinear(2, 2, bias=False)
    linear.weight.data.copy_(torch.eye(2))
    layer = BaseLayerWithLoRA(linear)
    p = object.__new__(_Pipeline)
    p.modules = {"transformer": torch.nn.Module()}
    p.modules["transformer"].add_module("linear", layer)
    p.server_args = SimpleNamespace(
        lora_alpha=None, lora_merge_mode="merge", model_path="/model"
    )
    p.lora_initialized = True
    p.lora_layers = {"linear": layer}
    p.lora_layers_transformer_2, p.lora_layers_critic = {}, {}
    p.cur_adapter_name, p.cur_adapter_path, p.cur_adapter_strength = {}, {}, {}
    p.cur_adapter_config, p.is_lora_merged = {}, {}
    p._temporarily_disable_offload = lambda *a, **k: nullcontext([])
    p.loaded_adapter_paths = {"A": "/A", "B": "/B"}
    p.loaded_adapter_alphas = {"A": None, "B": None}
    p.lora_adapters = {
        name: {
            "linear.lora_A": torch.tensor([[1.0, 1.0]]),
            "linear.lora_B": torch.tensor([[float(i == 0)], [float(i == 1)]]),
        }
        for i, name in enumerate("AB")
    }
    return p, layer


def _executor(generator):
    ex = object.__new__(SGLDiffusionExecutor)
    torch.nn.Module.__init__(ex)
    ex._ensure_runtime, ex._lora_input, ex._run_id = None, None, 0
    ex.generator = generator
    return ex


def _lora_input(state):
    return {
        "lora_nickname": [name for name, _ in state],
        "lora_path": [None] * len(state),
        "strength": [strength for _, strength in state],
        "target": ["transformer"] * len(state),
    }


def _sampler(state):
    options = {"sgld_lora_input": _lora_input(state)} if state else {}
    return SimpleNamespace(model_patcher=SimpleNamespace(model_options=options))


def test_each_sampler_run_merges_exactly_its_models_loras() -> None:
    """The worker is shared, so every run must leave only the LoRAs of the MODEL
    it samples merged: adding, repeating, dropping and re-weighting a LoRA."""
    p, layer = _one_layer_pipeline()
    ex = _executor(p)
    runs = [
        [("A", 0.5)],
        [("A", 0.5)],
        [("A", 0.5), ("B", 0.25)],
        [("B", 0.25)],
        [("A", 1.0)],
        [],
    ]
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        for state in runs:
            out = ex.sampler_sample_wrapper(
                lambda *a, **k: layer(torch.ones(1, 2))[0], _sampler(state)
            )
            expected = torch.ones(1, 2)
            for name, strength in state:
                expected[0, "AB".index(name)] += 2 * strength
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_failed_lora_switch_is_not_recorded_as_active() -> None:
    ex = _executor(Mock())
    ex._lora_input = _lora_input([("A", 0.5)])
    ex.generator.set_lora.side_effect = ValueError("bad adapter")
    with pytest.raises(ValueError, match="bad adapter"):
        ex.sampler_sample_wrapper(lambda *a: None, _sampler([("B", 0.5)]))
    assert ex._lora_input is None
    ex.generator.unmerge_lora_weights.assert_called_once()
