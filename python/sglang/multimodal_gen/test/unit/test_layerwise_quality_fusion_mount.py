from contextlib import nullcontext

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import FUSED_LINEAR_GELU
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    layerwise_offload as layerwise_offload_mod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadManager,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    toggle_quality_fusions,
)


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _FakeEvent:
    def record(self, _stream) -> None:
        return None

    def synchronize(self) -> None:
        return None


class _FakeDeviceModule:
    Stream = _FakeStream
    Event = _FakeEvent

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def current_stream() -> _FakeStream:
        return _FakeStream()

    @staticmethod
    def stream(_stream):
        return nullcontext()


class _FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(8, 16, dtype=torch.bfloat16)
        FUSED_LINEAR_GELU.mark(self, "proj_in")


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = _FeedForward()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


def _stream(model: _Model) -> LayerwiseOffloadManager:
    return LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=2,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )


def _patch_cpu_device(monkeypatch) -> None:
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")


def _enabled_sites(model: _Model) -> int:
    return sum(bool(block.ff._sgl_fused_gelu_enabled) for block in model.blocks)


def test_fusion_mounts_while_layers_are_streamed_out(monkeypatch):
    _patch_cpu_device(monkeypatch)
    model = _Model()
    manager = _stream(model)
    manager.release_layer(0)
    manager.release_layer(1)
    assert model.blocks[0].ff.proj_in.weight.dim() == 1

    toggle_quality_fusions([model], "lossless")

    assert _enabled_sites(model) == 2
    assert getattr(model, "_sgl_fused_gelu_enabled_last_rejection", None) is None


def test_mounted_site_reads_the_streamed_in_weight(monkeypatch):
    _patch_cpu_device(monkeypatch)
    model = _Model()
    expected = model.blocks[0].ff.proj_in.weight.detach().clone()
    manager = _stream(model)
    manager.release_layer(0)
    toggle_quality_fusions([model], "lossless")

    manager.prefetch_layer(0, non_blocking=False)

    linear = model.blocks[0].ff.proj_in
    assert FUSED_LINEAR_GELU.is_enabled(model.blocks[0].ff)
    assert linear.weight.shape == (16, 8)
    assert torch.equal(linear.weight, expected)


def test_unmanaged_non_2d_weight_is_still_rejected():
    model = _Model()
    model.blocks[0].ff.proj_in.weight.data = torch.empty(1, dtype=torch.bfloat16)

    toggle_quality_fusions([model], "lossless")

    assert _enabled_sites(model) == 0
    assert "non-2D weight" in str(model._sgl_fused_gelu_enabled_last_rejection)
