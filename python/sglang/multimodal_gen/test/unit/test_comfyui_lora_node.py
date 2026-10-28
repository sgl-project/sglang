# SPDX-License-Identifier: Apache-2.0
"""SGLD LoRA: loader nodes and binding the sampled MODEL's LoRAs to the worker."""

import copy
import functools
import os
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


@functools.cache
def _load_nodes():
    """Import the real nodes.py with the ComfyUI modules it needs stubbed."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_filename_list = lambda folder: ["a.safetensors"]
    folder_paths.get_full_path = lambda folder, name: f"/{folder}/{name}"
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api.input = comfy_api_input
    comfy = types.ModuleType("comfy")
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.utils = types.ModuleType("comfy.utils")
    stubs = {
        "comfy": comfy,
        "comfy.model_management": comfy.model_management,
        "comfy.utils": comfy.utils,
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
    }
    with mock.patch.dict(sys.modules, stubs):
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
    return nodes


def test_sgld_unet_list_adds_nested_gguf_without_registering_it(monkeypatch) -> None:
    """Adding .gguf to the global registry made ComfyUI's stock loader offer
    files it cannot load."""
    nodes = _load_nodes()
    registry = {"diffusion_models": (["/models"], {".safetensors"})}
    files = ["dit.safetensors", "flux/dit-Q4_K_M.gguf", "notes.txt"]
    for name, value in {
        "folder_names_and_paths": registry,
        "get_filename_list": lambda folder: ["dit.safetensors"],
        "get_folder_paths": lambda folder: registry[folder][0],
        "recursive_search": lambda folder: (files, {}),
        "filter_files_extensions": lambda names, exts: [
            n for n in names if os.path.splitext(n)[1] in exts
        ],
    }.items():
        monkeypatch.setattr(nodes.folder_paths, name, value, raising=False)
    assert nodes._list_unet_names_including_gguf() == [
        "dit.safetensors",
        "flux/dit-Q4_K_M.gguf",
    ]
    assert registry["diffusion_models"][1] == {".safetensors"}


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
    assert loader.INPUT_TYPES()["optional"]["merge_mode"][1]["default"] == "dynamic"
    (first,) = loader.load_lora(base, "style.safetensors", 1.0, nickname="style")
    (second,) = loader.load_lora(first, "detail.safetensors", 0.8, nickname="detail")

    assert calls == []
    desired = second.model_options["sgld_lora_input"]
    assert desired["lora_nickname"] == ["style", "detail"]
    assert desired["strength"] == [1.0, 0.8]
    assert desired["merge_mode"] == "dynamic"
    assert desired["lora_path"] == [
        "/loras/style.safetensors",
        "/loras/detail.safetensors",
    ]
    assert "style" not in base.patches  # the upstream model is not mutated
    assert set(second.patches) == {"style", "detail"}
    with pytest.raises(ValueError, match="same merge_mode"):
        loader.load_lora(first, "detail.safetensors", merge_mode="merge")


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


def _lora_input(state, merge_mode="merge"):
    return {
        "lora_nickname": [name for name, _ in state],
        "lora_path": [None] * len(state),
        "strength": [strength for _, strength in state],
        "target": ["transformer"] * len(state),
        "merge_mode": merge_mode,
    }


def _sampler(state, merge_mode="merge"):
    options = {"sgld_lora_input": _lora_input(state, merge_mode)} if state else {}
    return SimpleNamespace(model_patcher=SimpleNamespace(model_options=options))


def test_each_sampler_run_uses_exactly_its_models_loras() -> None:
    """The worker is shared, so every run must leave only the LoRAs of the MODEL
    it samples active: adding, repeating, dropping and re-weighting a LoRA."""
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
            expected = torch.ones(1, 2)
            for name, strength in state:
                expected[0, "AB".index(name)] += 2 * strength
            # Same adapters, different mode: the worker must rebind each time.
            for mode in ("dynamic", "merge", "dynamic"):
                out = ex.sampler_sample_wrapper(
                    lambda *a: layer(torch.ones(1, 2))[0], _sampler(state, mode)
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                assert layer.merged == (bool(state) and mode == "merge")
                if not layer.merged:
                    torch.testing.assert_close(
                        layer.weight, torch.eye(2), rtol=0, atol=0
                    )


def test_failed_lora_switch_is_not_recorded_as_active() -> None:
    ex = _executor(Mock())
    ex._lora_input = _lora_input([("A", 0.5)])
    ex.generator.set_lora.side_effect = ValueError("bad adapter")
    with pytest.raises(ValueError, match="bad adapter"):
        ex.sampler_sample_wrapper(lambda *a: None, _sampler([("B", 0.5)]))
    assert ex._lora_input is None
    ex.generator.unmerge_lora_weights.assert_called_once()


def test_chained_lora_targets_that_overlap_are_rejected() -> None:
    """The worker keeps one group per overlapping target, so 'all' chained with
    'transformer' would silently drop one LoRA."""
    loader = _load_nodes().SGLDLoraLoader()
    (first,) = loader.load_lora(_Model(None), "a.safetensors", 1.0, "a", "all")
    with pytest.raises(ValueError, match="target 'all'"):
        loader.load_lora(first, "b.safetensors", 1.0, "b", "transformer")


@pytest.mark.parametrize(
    "mode, missing", [("dynamic", "A"), ("merge", "A"), ("cached", "B")]
)
def test_partial_adapter_coverage_replaces_previous_lora(mode, missing):
    p, layer = _one_layer_pipeline()
    p._apply_lora_to_layers(p.lora_layers, ["A"], ["/A"], 0, [1.0], merge_weights=False)
    p.lora_adapters[missing] = {}
    cache = SimpleNamespace(get=lambda *a: None, put=lambda name, weight: weight)
    p._apply_lora_to_layers(
        p.lora_layers,
        ["A", "B"],
        ["/A", "/B"],
        0,
        [0.5, 0.25],
        clear_existing=True,
        merge_weights=mode != "dynamic",
        merge_cache=cache if mode == "cached" else None,
    )
    expected = [[1.0, 1.5]] if missing == "A" else [[2.0, 1.0]]
    torch.testing.assert_close(
        layer(torch.ones(1, 2))[0], torch.tensor(expected), rtol=0, atol=0
    )
    assert len(layer.lora_weights_list) == 1
    assert layer.merged == (mode != "dynamic")


def test_dynamic_reactivation_updates_all_adapter_strengths():
    p, layer = _one_layer_pipeline()
    p.server_args.lora_merge_mode = "dynamic"
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline.dist.get_rank",
        return_value=0,
    ):
        p.set_lora(**_lora_input([("A", 1.0), ("B", 0.25)], "dynamic"))
        p.unmerge_lora_weights()
        p.merge_lora_weights(strength=0.5)
    torch.testing.assert_close(
        layer(torch.ones(1, 2))[0], torch.tensor([[2.0, 2.0]]), rtol=0, atol=0
    )
    assert not layer.merged
