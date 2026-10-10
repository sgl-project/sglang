# SPDX-License-Identifier: Apache-2.0
"""SGLDLoraLoader: chained LoRA nodes must keep every LoRA active."""

import sys
import types
from types import SimpleNamespace
from unittest import mock


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

    def clone(self):
        return _Model(self.model.diffusion_model, self.patches)


def test_chained_lora_nodes_keep_every_lora() -> None:
    nodes = _load_nodes()
    calls = []
    executor = SimpleNamespace(set_lora=lambda **kwargs: calls.append(kwargs))
    base = _Model(executor)

    loader = nodes.SGLDLoraLoader()
    (first,) = loader.load_lora(base, "style.safetensors", 1.0, nickname="style")
    (second,) = loader.load_lora(first, "detail.safetensors", 0.8, nickname="detail")

    # The worker's set_lora replaces the active set, so the last call must
    # carry both LoRAs.
    assert calls[-1]["lora_nickname"] == ["style", "detail"]
    assert calls[-1]["strength"] == [1.0, 0.8]
    assert calls[-1]["lora_path"] == [
        "/loras/style.safetensors",
        "/loras/detail.safetensors",
    ]
    assert "style" not in base.patches  # the upstream model is not mutated
    assert set(second.patches) == {"style", "detail"}


def test_switching_and_dropping_loras_sends_exact_active_set() -> None:
    nodes = _load_nodes()
    calls = []
    executor = SimpleNamespace(set_lora=lambda **kwargs: calls.append(kwargs))
    base = _Model(executor)
    loader = nodes.SGLDLoraLoader()

    def names(call):
        return call["lora_nickname"]

    (a,) = loader.load_lora(base, "a.safetensors", 1.0, nickname="a")
    (ab,) = loader.load_lora(a, "b.safetensors", 0.5, nickname="b")
    (abc,) = loader.load_lora(ab, "c.safetensors", 0.7, nickname="c")
    assert names(calls[-1]) == ["a", "b", "c"]

    # Reducing: a branch that stops after "a" must send only "a".
    (a2,) = loader.load_lora(base, "a.safetensors", 1.0, nickname="a")
    assert names(calls[-1]) == ["a"]

    # Switching: a different chain from the same base drops the earlier ones.
    (xy,) = loader.load_lora(
        loader.load_lora(base, "x.safetensors", 1.0, nickname="x")[0],
        "y.safetensors",
        0.3,
        nickname="y",
    )
    assert names(calls[-1]) == ["x", "y"]
    assert calls[-1]["strength"] == [1.0, 0.3]

    # Going back to an earlier chain restores exactly its set.
    loader.load_lora(ab, "d.safetensors", 1.0, nickname="d")
    assert names(calls[-1]) == ["a", "b", "d"]

    # Re-applying the same nickname replaces its strength, not duplicates it.
    (a3,) = loader.load_lora(abc, "a.safetensors", 0.2, nickname="a")
    assert names(calls[-1]) == ["a", "b", "c"]
    assert calls[-1]["strength"] == [0.2, 0.5, 0.7]

    # No branch leaked into the shared base or its siblings.
    assert base.patches == {}
    assert set(a.patches) == {"a"} and set(ab.patches) == {"a", "b"}
    assert set(a2.patches) == {"a"} and set(xy.patches) == {"x", "y"}
