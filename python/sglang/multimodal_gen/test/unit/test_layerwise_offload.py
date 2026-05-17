from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _ModelOptFp8OffloadAdapter,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    component_resident_strategies as component_resident_strategies_mod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    layerwise_offload as layerwise_offload_mod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
    build_component_residency_strategy,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    LayerwiseOffloadStrategy,
    ResidentStrategy,
    VanillaD2HStrategy,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    LayerwiseOffloadManager,
    configure_layerwise_offload_modules,
    is_layerwise_offloaded_module,
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


class _NestedDummyModel(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["encoder.blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _DummyModel()


class _SharedBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "cache", torch.arange(12, dtype=torch.float32).reshape(6, 2)
        )


class _SharedBufferLayer(torch.nn.Module):
    def __init__(self, shared: _SharedBuffer) -> None:
        super().__init__()
        self.shared = shared
        self.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))


class _SharedBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = _SharedBuffer()
        self.blocks = torch.nn.ModuleList(
            [_SharedBufferLayer(shared), _SharedBufferLayer(shared)]
        )


class _NestedEncoderDummyModel(_NestedDummyModel):
    layerwise_offload_default_enabled = False


class _LayerwiseComponent(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["blocks"]

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DummyBlock()])
        self.layerwise_offload_managers = [SimpleNamespace(enabled=enabled)]


def _server_args(**kwargs):
    defaults = dict(
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_offload_prefetch_size=1,
        pin_cpu_memory=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


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


def test_layerwise_offload_keeps_shared_buffers_resident(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    model = _SharedBufferModel()
    original_cache = model.blocks[0].shared.cache.detach().clone()

    manager = LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=2,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    assert not any(
        "cache" in name
        for metadata in manager._weight_metadata.values()
        for name in metadata
    )
    manager.release_layer(0)

    cache = model.blocks[1].shared.cache
    assert torch.equal(cache, original_cache)
    assert torch.equal(cache.index_select(0, torch.tensor([2])), original_cache[2:3])


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


def test_layerwise_capability_selects_layerwise_strategy_for_any_component():
    module = _LayerwiseComponent(enabled=True)

    assert is_layerwise_offloaded_module(module)
    strategy = build_component_residency_strategy(
        "text_encoder", module, _server_args(text_encoder_cpu_offload=True)
    )

    assert isinstance(strategy, LayerwiseOffloadStrategy)


def test_layerwise_configuration_uses_legacy_default_components(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    layerwise_module = _NestedDummyModel()
    modules = {
        "text_encoder": layerwise_module,
        "text_encoder_alias": layerwise_module,
        "scheduler": object(),
    }

    configured = configure_layerwise_offload_modules(modules, _server_args())

    assert configured == ["text_encoder"]
    assert is_layerwise_offloaded_module(layerwise_module)


def test_layerwise_configuration_filters_by_component_name(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    text_encoder = _NestedEncoderDummyModel()
    transformer = _NestedDummyModel()
    vae = _NestedDummyModel()
    modules = {
        "custom_encoder_name": text_encoder,
        "custom_transformer_name": transformer,
        "custom_vae_name": vae,
    }

    configured = configure_layerwise_offload_modules(
        modules, _server_args(), component_names=["custom_encoder_name"]
    )

    assert configured == ["custom_encoder_name"]
    assert is_layerwise_offloaded_module(text_encoder)
    assert not is_layerwise_offloaded_module(transformer)
    assert not is_layerwise_offloaded_module(vae)


def test_layerwise_configuration_default_marker_extends_legacy_defaults(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    text_encoder = _NestedEncoderDummyModel()
    text_encoder_2 = _NestedEncoderDummyModel()
    transformer = _NestedDummyModel()
    vae = _NestedEncoderDummyModel()
    audio_vae = _NestedEncoderDummyModel()
    condition_image_encoder = _NestedEncoderDummyModel()
    modules = {
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "transformer": transformer,
        "vae": vae,
        "audio_vae": audio_vae,
        "condition_image_encoder": condition_image_encoder,
    }

    configured = configure_layerwise_offload_modules(
        modules, _server_args(), component_names=["default", "text_encoder", "vae"]
    )

    assert configured == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
        "audio_vae",
        "condition_image_encoder",
    ]
    assert is_layerwise_offloaded_module(text_encoder)
    assert is_layerwise_offloaded_module(text_encoder_2)
    assert is_layerwise_offloaded_module(transformer)
    assert is_layerwise_offloaded_module(vae)
    assert is_layerwise_offloaded_module(audio_vae)
    assert is_layerwise_offloaded_module(condition_image_encoder)


def test_layerwise_configuration_all_selects_every_capable_component(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    text_encoder = _NestedEncoderDummyModel()
    transformer = _NestedDummyModel()
    modules = {
        "custom_encoder_name": text_encoder,
        "custom_transformer_name": transformer,
        "scheduler": object(),
    }

    configured = configure_layerwise_offload_modules(
        modules, _server_args(), component_names=["all"]
    )

    assert configured == ["custom_encoder_name", "custom_transformer_name"]
    assert is_layerwise_offloaded_module(text_encoder)
    assert is_layerwise_offloaded_module(transformer)


def test_component_cpu_offload_strategy_remains_flag_driven():
    strategy = build_component_residency_strategy(
        "text_encoder", _DummyModel(), _server_args(text_encoder_cpu_offload=True)
    )
    assert isinstance(strategy, VanillaD2HStrategy)

    strategy = build_component_residency_strategy(
        "unknown_component", _DummyModel(), _server_args(text_encoder_cpu_offload=True)
    )
    assert isinstance(strategy, ResidentStrategy)


def test_resident_strategy_prepares_local_device_without_dtype(monkeypatch):
    calls = []

    def fake_module_to_local_device(module, *, dtype=None):
        calls.append((module, dtype))

    monkeypatch.setattr(
        component_resident_strategies_mod,
        "_module_to_local_device",
        fake_module_to_local_device,
    )
    module = _DummyModel()

    ResidentStrategy().prepare_for_use(
        module,
        ComponentUse(stage_name="DenoisingStage", component_name="transformer"),
        SimpleNamespace(),
    )

    assert calls == [(module, None)]


def test_resident_strategy_keeps_fsdp_managed_module_owned_by_fsdp(monkeypatch):
    calls = []

    def fake_module_to_local_device(module, *, dtype=None):
        calls.append((module, dtype))

    monkeypatch.setattr(
        component_resident_strategies_mod,
        "_module_to_local_device",
        fake_module_to_local_device,
    )
    module = type("FSDPDummyModel", (_DummyModel,), {})()

    ResidentStrategy().prepare_for_use(
        module,
        ComponentUse(stage_name="TextEncodingStage", component_name="text_encoder"),
        SimpleNamespace(),
    )

    assert calls == []


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
