from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _ModelOptFp8OffloadAdapter,
)
from sglang.multimodal_gen.runtime.managers import (
    layerwise_offload as layerwise_offload_mod,
)
from sglang.multimodal_gen.runtime.managers.layerwise_offload import (
    LayerwiseOffloadManager,
)


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _FakeEvent:
    def record(self, _stream) -> None:
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


class _DummyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.weight = torch.nn.Parameter(base.t())
        self.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float32))


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DummyBlock()])


def test_layerwise_offload_preserves_non_contiguous_stride(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    model = _DummyModel()
    original_weight = model.blocks[0].weight.detach().clone()
    original_stride = model.blocks[0].weight.stride()
    assert not model.blocks[0].weight.is_contiguous()

    manager = LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=1,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    meta = manager._weight_metadata[0]["blocks.0.weight"]
    assert meta["preserve_strides"] is True

    restored_weight = model.blocks[0].weight.data
    assert restored_weight.shape == original_weight.shape
    assert restored_weight.stride() == original_stride
    assert not restored_weight.is_contiguous()
    assert torch.equal(restored_weight, original_weight)

    manager.release_layer(0)
    manager.prefetch_layer(0, non_blocking=False)

    reloaded_weight = model.blocks[0].weight.data
    assert reloaded_weight.stride() == original_stride
    assert not reloaded_weight.is_contiguous()
    assert torch.equal(reloaded_weight, original_weight)


def test_modelopt_fp8_adapter_keeps_layerwise_offload_enabled():
    server_args = SimpleNamespace(
        dit_cpu_offload=True,
        dit_layerwise_offload=True,
    )
    quant_config = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)

    _ModelOptFp8OffloadAdapter._maybe_disable_incompatible_dit_offload_modes(
        server_args=server_args,
        quant_config=quant_config,
    )

    assert server_args.dit_cpu_offload is False
    assert server_args.dit_layerwise_offload is True


def test_layerwise_offload_aligns_contiguous_tensor_offsets(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    class _AlignedDummyBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(9, dtype=torch.float32).reshape(3, 3)
            )
            self.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float32))

    class _AlignedDummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([_AlignedDummyBlock()])

    model = _AlignedDummyModel()
    original_weight = model.blocks[0].weight.detach().clone()
    original_bias = model.blocks[0].bias.detach().clone()

    manager = LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=1,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    weight_meta = manager._weight_metadata[0]["blocks.0.weight"]
    bias_meta = manager._weight_metadata[0]["blocks.0.bias"]
    assert weight_meta["preserve_strides"] is False
    assert bias_meta["preserve_strides"] is False
    assert weight_meta["offset"] == 0
    assert bias_meta["offset"] % 8 == 0

    restored_weight = model.blocks[0].weight.data
    restored_bias = model.blocks[0].bias.data
    assert restored_weight.data_ptr() % 32 == 0
    assert restored_bias.data_ptr() % 32 == 0
    assert torch.equal(restored_weight, original_weight)
    assert torch.equal(restored_bias, original_bias)
