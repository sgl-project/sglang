from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _ModelOptFp8OffloadAdapter,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    component_residency_strategies as component_residency_strategies_mod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    layerwise_offload as layerwise_offload_mod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
    build_component_residency_strategy,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
    LayerwiseOffloadStrategy,
    ResidentStrategy,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    LayerwiseOffloadManager,
    compute_streamed_layers,
    configure_layerwise_offload_modules,
    get_layerwise_offload_component_names_for_pipeline,
    is_layerwise_offloaded_module,
    is_resident_layerwise_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    RESIDENCY_POLICY_LEADING,
    RESIDENCY_POLICY_STRIDED,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


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


class _NestedSameNamedBlocksModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_refiner = _DummyModel()
        self.blocks = torch.nn.ModuleList([_DummyBlock()])


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


class _OrderedLinearLayer(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class _ReverseLayerwiseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                _OrderedLinearLayer(2.0),
                _OrderedLinearLayer(3.0),
                _OrderedLinearLayer(5.0),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            x = block(x)
        return x


class _NestedEncoderDummyModel(_NestedDummyModel):
    layerwise_offload_dit_group_enabled = False


class _LayerwiseComponent(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["blocks"]

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DummyBlock()])
        self.layerwise_offload_managers = [SimpleNamespace(enabled=enabled)]


class _TestServerArgs(SimpleNamespace):
    canonical_residency_mode = ServerArgs.canonical_residency_mode
    explicit_residency_mode = ServerArgs.explicit_residency_mode
    _legacy_component_offload_flag = staticmethod(
        ServerArgs._legacy_component_offload_flag
    )
    residency_mode = ServerArgs.residency_mode
    is_arg_explicitly_set = ServerArgs.is_arg_explicitly_set
    is_explicit_layerwise_offload_component = (
        ServerArgs.is_explicit_layerwise_offload_component
    )
    should_cpu_offload_component = ServerArgs.should_cpu_offload_component
    record_component_layerwise_capability = (
        ServerArgs.record_component_layerwise_capability
    )
    _parse_component_value_map = staticmethod(ServerArgs._parse_component_value_map)
    layerwise_tuning_for = ServerArgs.layerwise_tuning_for


def _server_args(**kwargs):
    defaults = dict(
        component_residency=None,
        disagg_role=RoleType.MONOLITHIC,
        _required_resident_components=set(),
        _component_layerwise_capabilities={},
        _explicit_arg_names=set(),
        cpu_offload_components=None,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        layerwise_offload_components=None,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_offload_prefetch_size=1,
        dit_layerwise_resident_layers=0.0,
        dit_layerwise_residency_policy=RESIDENCY_POLICY_LEADING,
        layerwise_prefetch_size={},
        layerwise_resident_layers={},
        layerwise_residency_policy={},
        pin_cpu_memory=False,
        # the pin budget ranks candidates by bytes x steps, and reads the step
        # count off the pipeline's sampling defaults
        pipeline_class_name=None,
    )
    defaults.update(kwargs)
    return _TestServerArgs(**defaults)


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


def test_layerwise_offload_uses_normal_tensors_under_inference_mode(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    model = _DummyModel()
    manager = LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=1,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    with torch.inference_mode():
        manager.release_layer(0)
        manager.prefetch_layer(0, non_blocking=False)

    assert model.blocks[0].weight._version >= 0
    assert model.blocks[0].bias._version >= 0


def test_layerwise_offload_does_not_capture_nested_same_named_layers(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    model = _NestedSameNamedBlocksModel()
    refiner_weight = model.token_refiner.blocks[0].weight.detach().clone()
    manager = LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=1,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    managed_names = {
        name for metadata in manager._weight_metadata.values() for name in metadata
    }
    assert managed_names
    assert all(name.startswith("blocks.") for name in managed_names)
    assert torch.equal(model.token_refiner.blocks[0].weight, refiner_weight)


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


def test_layerwise_offload_loads_current_layer_for_reverse_execution(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    model = _ReverseLayerwiseModel()
    x = torch.ones(1, 2, dtype=torch.float32)
    expected = model(x)

    LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=3,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=1,
    )

    assert torch.equal(model(x), expected)


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


def test_modelopt_fp8_adapter_does_not_change_online_fp8_offload():
    server_args = SimpleNamespace(
        dit_cpu_offload=True,
        dit_layerwise_offload=False,
        quantization="fp8",
    )

    _ModelOptFp8OffloadAdapter._maybe_disable_incompatible_dit_offload_modes(
        server_args=server_args,
        quant_config=Fp8Config(),
    )

    assert server_args.dit_cpu_offload is True


def test_layerwise_capability_selects_layerwise_strategy_for_any_component():
    module = _LayerwiseComponent(enabled=True)

    assert is_layerwise_offloaded_module(module)
    strategy = build_component_residency_strategy(
        "text_encoder", module, _server_args(text_encoder_cpu_offload=True)
    )

    assert isinstance(strategy, LayerwiseOffloadStrategy)


def test_layerwise_pipeline_selection_uses_dit_group(monkeypatch):
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

    selected = get_layerwise_offload_component_names_for_pipeline(modules)
    configured = configure_layerwise_offload_modules(modules, _server_args())

    assert selected == ["text_encoder", "text_encoder_alias"]
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


def test_layerwise_configuration_default_group_selects_non_dit_defaults(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    text_encoder = _NestedEncoderDummyModel()
    text_encoder_2 = _NestedEncoderDummyModel()
    transformer = _NestedDummyModel()
    image_encoder = _NestedEncoderDummyModel()
    vae = _NestedEncoderDummyModel()
    audio_vae = _NestedEncoderDummyModel()
    vocoder = _NestedEncoderDummyModel()
    spatial_upsampler = _NestedEncoderDummyModel()
    condition_image_encoder = _NestedEncoderDummyModel()
    modules = {
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "transformer": transformer,
        "image_encoder": image_encoder,
        "vae": vae,
        "audio_vae": audio_vae,
        "vocoder": vocoder,
        "spatial_upsampler": spatial_upsampler,
        "condition_image_encoder": condition_image_encoder,
    }

    configured = configure_layerwise_offload_modules(
        modules, _server_args(), component_names=["default"]
    )

    assert get_layerwise_offload_component_names_for_pipeline(modules, ["default"]) == [
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "vae",
        "condition_image_encoder",
    ]
    assert configured == [
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "vae",
        "condition_image_encoder",
    ]
    assert is_layerwise_offloaded_module(text_encoder)
    assert is_layerwise_offloaded_module(text_encoder_2)
    assert not is_layerwise_offloaded_module(transformer)
    assert is_layerwise_offloaded_module(image_encoder)
    assert is_layerwise_offloaded_module(vae)
    assert not is_layerwise_offloaded_module(audio_vae)
    assert not is_layerwise_offloaded_module(vocoder)
    assert not is_layerwise_offloaded_module(spatial_upsampler)
    assert is_layerwise_offloaded_module(condition_image_encoder)

    for component_name, module in (
        ("audio_vae", audio_vae),
        ("vocoder", vocoder),
        ("spatial_upsampler", spatial_upsampler),
    ):
        configured = configure_layerwise_offload_modules(
            modules, _server_args(), component_names=[component_name]
        )
        assert configured == [component_name]
        assert is_layerwise_offloaded_module(module)


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


def test_explicit_layerwise_all_rejects_unsupported_modules():
    modules = {
        "text_encoder": _NestedEncoderDummyModel(),
        "unsupported_adapter": torch.nn.Linear(2, 2),
        "scheduler": object(),
    }

    with pytest.raises(ComponentResidencyError, match="unsupported_adapter"):
        configure_layerwise_offload_modules(
            modules, _server_args(), component_names=["all"]
        )


def test_explicit_layerwise_dit_rejects_unsupported_dit():
    modules = {"transformer": torch.nn.Linear(2, 2)}

    with pytest.raises(ComponentResidencyError, match="transformer"):
        configure_layerwise_offload_modules(
            modules, _server_args(), component_names=["dit"]
        )


def test_explicit_layerwise_exact_selector_rejects_non_module(monkeypatch):
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "is_cpu", lambda: False)
    server_args = _server_args(component_residency={"scheduler": LAYERWISE_OFFLOAD})

    with pytest.raises(ComponentResidencyError, match="scheduler"):
        configure_layerwise_offload_modules(
            {"scheduler": object()}, server_args, warn_missing=False
        )


def test_auto_layerwise_skips_unsupported_component(monkeypatch):
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "is_cpu", lambda: False)
    server_args = _server_args(
        layerwise_offload_components=["text_encoder"],
        text_encoder_cpu_offload=True,
    )

    configured = configure_layerwise_offload_modules(
        {"text_encoder": torch.nn.Linear(2, 2)},
        server_args,
        component_names=["text_encoder"],
        warn_missing=False,
    )

    assert configured == []
    assert server_args.residency_mode("text_encoder") == COMPONENT_OFFLOAD


def test_canonical_selector_does_not_make_auto_layerwise_selection_strict(
    monkeypatch,
):
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "is_cpu", lambda: False)
    server_args = _server_args(
        component_residency={"transformer": RESIDENT},
        layerwise_offload_components=["text_encoder"],
        text_encoder_cpu_offload=True,
    )

    configured = configure_layerwise_offload_modules(
        {"text_encoder": torch.nn.Linear(2, 2)},
        server_args,
        warn_missing=True,
    )

    assert configured == []
    assert server_args.residency_mode("text_encoder") == COMPONENT_OFFLOAD


def test_legacy_cpu_offload_flag_selects_component_offload_strategy():
    strategy = build_component_residency_strategy(
        "text_encoder", _DummyModel(), _server_args(text_encoder_cpu_offload=True)
    )
    assert isinstance(strategy, ComponentOffloadStrategy)

    strategy = build_component_residency_strategy(
        "unknown_component", _DummyModel(), _server_args(text_encoder_cpu_offload=True)
    )
    assert isinstance(strategy, ResidentStrategy)


def test_component_residency_strategy_selection_is_direct():
    for mode, strategy_type in (
        (RESIDENT, ResidentStrategy),
        (COMPONENT_OFFLOAD, ComponentOffloadStrategy),
    ):
        strategy = build_component_residency_strategy(
            "text_encoder",
            _DummyModel(),
            _server_args(component_residency={"text_encoder": mode}),
        )
        assert isinstance(strategy, strategy_type)


def test_explicit_layerwise_requires_component_support():
    server_args = _server_args(component_residency={"text_encoder": LAYERWISE_OFFLOAD})

    with pytest.raises(ValueError, match="did not enable layerwise offload"):
        build_component_residency_strategy("text_encoder", _DummyModel(), server_args)


def test_resident_strategy_prepares_local_device_without_dtype(monkeypatch):
    calls = []

    def fake_module_to_local_device(module, *, dtype=None):
        calls.append((module, dtype))

    monkeypatch.setattr(
        component_residency_strategies_mod,
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
        component_residency_strategies_mod,
        "_module_to_local_device",
        fake_module_to_local_device,
    )
    monkeypatch.setattr(
        component_residency_strategies_mod,
        "is_fsdp_managed_module",
        lambda _module: True,
    )
    module = _DummyModel()

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


# ---------------------------------------------------------------------------
# --dit-layerwise-resident-layers: keep N leading layers resident (retained
# across denoise steps), streaming only the tail with the prefetch window.
# ---------------------------------------------------------------------------
class _MultiBlockModel(torch.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DummyBlock() for _ in range(n)])


class _ResidentComponent(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["blocks"]

    def __init__(self, n: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DummyBlock() for _ in range(n)])


class _AuxiliaryResidentComponent(_ResidentComponent):
    layerwise_offload_dit_group_enabled = False


class _MultiGroupComponent(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["small_blocks", "large_blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.small_blocks = torch.nn.ModuleList([_DummyBlock()])
        self.large_blocks = torch.nn.ModuleList(
            [_DummyBlock(), _DummyBlock(), _DummyBlock()]
        )
        self.non_layer = torch.nn.Parameter(torch.ones(2))
        self.to_parameter_shapes = []

    def to(self, *args, **kwargs):
        self.to_parameter_shapes.append(
            {name: tuple(param.shape) for name, param in self.named_parameters()}
        )
        return super().to(*args, **kwargs)


def _patch_fake_device(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")


def _resident_manager(
    model,
    *,
    num_layers,
    prefetch_size=1,
    resident_layers=0,
    residency_policy=RESIDENCY_POLICY_LEADING,
):
    return LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=num_layers,
        enabled=True,
        pin_cpu_memory=False,
        prefetch_size=prefetch_size,
        resident_layers=resident_layers,
        residency_policy=residency_policy,
    )


def _arm_residency(manager):
    """Mimic the first-layer pre-hook: arm the resident set, then pin it."""
    manager._activate_residency()
    manager.prepare_for_next_req(non_blocking=False)


def test_resident_layers_stay_pinned_until_stage_teardown(monkeypatch):
    _patch_fake_device(monkeypatch)
    manager = _resident_manager(
        _MultiBlockModel(4), num_layers=4, prefetch_size=1, resident_layers=2
    )
    # The resident set is armed on the first forward, not at construction.
    assert manager._retained_layers == 0

    _arm_residency(manager)
    assert manager._retained_layers == 2
    assert {0, 1} <= manager._gpu_layers

    # A non-force release keeps the leading resident layers pinned across steps.
    manager.release_layer(0)
    manager.release_layer(1)
    assert {0, 1} <= manager._gpu_layers

    # force=True (teardown) overrides the retention.
    manager.release_layer(0, force=True)
    assert 0 not in manager._gpu_layers
    manager.release_all()  # ends the denoise stage: residents go too
    assert not manager._gpu_layers


def test_resident_layers_off_by_default_streams_everything(monkeypatch):
    _patch_fake_device(monkeypatch)
    manager = _resident_manager(
        _MultiBlockModel(4), num_layers=4, prefetch_size=1, resident_layers=0
    )
    _arm_residency(manager)

    assert manager._retained_layers == 0
    assert manager.holds_residents is False

    manager.prefetch_layer(2, non_blocking=False)
    manager.release_layer(2)  # no residents -> released like plain streaming
    assert 2 not in manager._gpu_layers


def test_prepare_for_next_req_repins_residents(monkeypatch):
    _patch_fake_device(monkeypatch)
    manager = _resident_manager(
        _MultiBlockModel(6), num_layers=6, prefetch_size=1, resident_layers=3
    )
    _arm_residency(manager)
    manager.release_all()
    assert not manager._gpu_layers

    # The next denoise re-pins the resident set (union of prefetch window + residents).
    manager.prepare_for_next_req(non_blocking=False)
    assert {0, 1, 2} <= manager._gpu_layers


def _record_prepare(manager, monkeypatch):
    """Log the order of prefetches and stream waits inside prepare_for_next_req.

    The order is the contract: prepare_for_next_req runs from the layer-0
    pre-hook on every denoise step, so anything issued before the blocking
    wait_stream is something the compute stream will sit and wait for.
    """
    log: list = []
    original = manager.prefetch_layer

    def spy(layer_idx, non_blocking=True):
        log.append(("prefetch", layer_idx, non_blocking))
        return original(layer_idx, non_blocking=non_blocking)

    class _RecordingStream(_FakeStream):
        def wait_stream(self, _stream) -> None:
            log.append(("wait_stream",))

    monkeypatch.setattr(manager, "prefetch_layer", spy)
    monkeypatch.setattr(
        _FakeDeviceModule, "current_stream", staticmethod(_RecordingStream)
    )
    return log


def test_blocking_prepare_waits_for_residents_only(monkeypatch):
    # Regression: the head of the stream used to be issued before the
    # wait_stream. wait_stream drains the whole copy stream, so under `leading`
    # that made every denoise step block on layer `resident_layers` -- a full
    # transfer that is not needed for another `resident_layers` layers, while
    # layer 0 was already pinned. Costs one layer transfer per step on the
    # default path, which is the path nobody passes a flag to get.
    _patch_fake_device(monkeypatch)
    manager = _resident_manager(
        _MultiBlockModel(6), num_layers=6, prefetch_size=1, resident_layers=3
    )
    _arm_residency(manager)
    manager.release_all()

    log = _record_prepare(manager, monkeypatch)
    manager.prepare_for_next_req(non_blocking=False)

    wait = log.index(("wait_stream",))
    before = [entry for entry in log[:wait] if entry[0] == "prefetch"]
    after = [entry for entry in log[wait:] if entry[0] == "prefetch"]
    assert [entry[1] for entry in before] == [0, 1, 2]
    assert all(entry[2] is False for entry in before)
    # Layer 3 is the first streamed layer, and it is issued after the wait and
    # asynchronously, so the pre-hook's own wait_event is what blocks on it.
    assert [entry[1] for entry in after] == [3]
    assert all(entry[2] is True for entry in after)


def test_warmup_prepare_prefetches_the_layer_that_runs_first(monkeypatch):
    # At load time residency is not armed yet, so there is no resident set and
    # the forward will start at layer 0 whatever the policy says. Deriving the
    # head of the stream from the policy here would warm layer
    # `resident_layers` under `leading`, i.e. not the one about to run.
    _patch_fake_device(monkeypatch)
    for policy in (RESIDENCY_POLICY_LEADING, RESIDENCY_POLICY_STRIDED):
        manager = _resident_manager(
            _MultiBlockModel(6),
            num_layers=6,
            prefetch_size=1,
            resident_layers=3,
            residency_policy=policy,
        )
        assert manager._head_of_stream() == [0], policy


def test_configure_resolves_residency_policy(monkeypatch):
    _patch_fake_device(monkeypatch)
    comp = _ResidentComponent(8)
    comp.configure_layerwise_offload(
        _server_args(dit_layerwise_residency_policy=RESIDENCY_POLICY_STRIDED)
    )
    assert comp.layerwise_offload_managers[0].residency_policy == (
        RESIDENCY_POLICY_STRIDED
    )


def test_configure_offloads_all_layer_groups_before_moving_non_layers(monkeypatch):
    _patch_fake_device(monkeypatch)
    model = _MultiGroupComponent()
    initialization_order = []
    initialize_layer_weights = LayerwiseOffloadManager._initialize_layer_weights

    def record_initialization(manager):
        initialization_order.append(manager.layers_attr_str)
        initialize_layer_weights(manager)

    monkeypatch.setattr(
        LayerwiseOffloadManager,
        "_initialize_layer_weights",
        record_initialization,
    )

    model.configure_layerwise_offload(_server_args())

    assert initialization_order == ["large_blocks", "small_blocks"]
    assert [
        manager.layers_attr_str for manager in model.layerwise_offload_managers
    ] == ["small_blocks", "large_blocks"]
    assert len(model.to_parameter_shapes) == 1
    shapes_at_move = model.to_parameter_shapes[0]
    assert shapes_at_move["non_layer"] == (2,)
    for name, shape in shapes_at_move.items():
        if name != "non_layer":
            assert shape == (1,), name


def test_holds_residents_reflects_configuration(monkeypatch):
    _patch_fake_device(monkeypatch)
    resident = _resident_manager(_MultiBlockModel(3), num_layers=3, resident_layers=2)
    streaming = _resident_manager(_MultiBlockModel(3), num_layers=3, resident_layers=0)
    assert resident.holds_residents is True
    assert streaming.holds_residents is False


def test_is_resident_layerwise_module_detector():
    class _Comp(torch.nn.Module, LayerwiseOffloadableModuleMixin):
        pass

    comp = _Comp()
    comp.layerwise_offload_managers = [SimpleNamespace(holds_residents=True)]
    assert is_resident_layerwise_module(comp) is True

    comp.layerwise_offload_managers = [SimpleNamespace(holds_residents=False)]
    assert is_resident_layerwise_module(comp) is False


def test_configure_resolves_resident_layers_absolute(monkeypatch):
    _patch_fake_device(monkeypatch)
    comp = _ResidentComponent(8)
    comp.configure_layerwise_offload(_server_args(dit_layerwise_resident_layers=3))
    assert comp.layerwise_offload_managers[0].resident_layers == 3


def test_configure_resolves_resident_layers_ratio(monkeypatch):
    _patch_fake_device(monkeypatch)
    comp = _ResidentComponent(8)
    comp.configure_layerwise_offload(_server_args(dit_layerwise_resident_layers=0.5))
    # 0.5 * 8 = 4 leading layers resident
    assert comp.layerwise_offload_managers[0].resident_layers == 4


def test_auxiliary_layerwise_components_ignore_dit_tuning(monkeypatch):
    _patch_fake_device(monkeypatch)
    comp = _AuxiliaryResidentComponent(8)
    comp.configure_layerwise_offload(
        _server_args(
            dit_offload_prefetch_size=3,
            dit_layerwise_resident_layers=0.5,
        )
    )

    manager = comp.layerwise_offload_managers[0]
    assert manager.prefetch_size == 1
    assert manager.resident_layers == 0


class _MixinBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(9, dtype=torch.float32).reshape(3, 3)
        )
        self.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float32))


class _MixinModel(torch.nn.Module, LayerwiseOffloadableModuleMixin):
    layer_names = ["blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_MixinBlock() for _ in range(3)])


def _configure_mixin_model(monkeypatch) -> _MixinModel:
    _patch_fake_device(monkeypatch)
    model = _MixinModel()
    model.configure_layerwise_offload(_server_args())
    assert is_layerwise_offloaded_module(model)
    return model


def test_disable_offload_short_circuits_residency_release(monkeypatch):
    """disable_offload() must make later layerwise calls no-ops.

    Regression test: a ComponentResidencyManager strategy built while the
    module was offloaded (the offload_during_compile window) keeps calling
    release_all() on use-site switches. After disable_offload() removed the
    hooks, those releases replaced restored weights with (1,) placeholders
    that nothing ever swapped back in, crashing dual-DiT models (Wan2.2-A14B
    boundary experts, Ideogram-4 paired towers) on the first real request.
    """
    model = _configure_mixin_model(monkeypatch)
    model.disable_offload()

    assert not is_layerwise_offloaded_module(model)
    for name, param in model.named_parameters():
        assert tuple(param.shape) != (1,), name

    # The exact call path the residency strategy takes on use-site switches.
    LayerwiseOffloadStrategy().finish_use(
        model,
        ComponentUse(stage_name="test", component_name="transformer"),
        SimpleNamespace(),
    )
    model.prepare_for_next_req()
    for name, param in model.named_parameters():
        assert tuple(param.shape) != (1,), name


def test_enable_offload_rearms_after_disable(monkeypatch):
    model = _configure_mixin_model(monkeypatch)
    # blocks[2] holds a placeholder right after configure; the real values are
    # what _MixinBlock was constructed with.
    original = torch.arange(9, dtype=torch.float32).reshape(3, 3)

    model.disable_offload()
    assert not is_layerwise_offloaded_module(model)

    model.enable_offload()
    assert is_layerwise_offloaded_module(model)

    manager = model.layerwise_offload_managers[0]
    manager.release_layer(2)
    assert tuple(model.blocks[2].weight.shape) == (1,)
    manager.prefetch_layer(2, non_blocking=False)
    assert torch.equal(model.blocks[2].weight.data, original)


# ---------------------------------------------------------------------------
# --dit-layerwise-residency-policy: which layers stay resident.
#
# `leading` keeps 0..r-1, so every transfer lands in one burst at the tail of
# the step. `strided` spreads them, so the same bytes move at 1/(n/(n-r)) of the
# peak rate and each transfer gets that many layers of compute to hide behind.
# That matters when the model also runs a collective per layer: measured on
# 8 GPUs the ulysses SendRecv total went 2589.7 -> 4016.0 ms once the DiT
# streamed, for identical collective volume.
# ---------------------------------------------------------------------------
def test_leading_policy_keeps_todays_prefix_layout():
    # Regression guard: `leading` is the default, and changing it would silently
    # re-time every existing deployment.
    for num_layers, resident in ((4, 2), (50, 35), (12, 1), (7, 6)):
        assert compute_streamed_layers(
            num_layers=num_layers,
            resident_layers=resident,
            policy=RESIDENCY_POLICY_LEADING,
        ) == tuple(range(resident, num_layers))


def test_strided_policy_layout_is_pinned_for_the_h3_dit():
    # 50 layers with 35 resident is the measured MiniMax-H3 operating point.
    # Pinned exactly so a later "simplification" of the ramp cannot quietly
    # change the schedule this policy exists to produce.
    assert compute_streamed_layers(
        num_layers=50, resident_layers=35, policy=RESIDENCY_POLICY_STRIDED
    ) == (0, 3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43, 47)


def test_both_policies_partition_the_stack():
    for num_layers in range(1, 33):
        for resident in range(0, num_layers + 1):
            for policy in (RESIDENCY_POLICY_LEADING, RESIDENCY_POLICY_STRIDED):
                streamed = compute_streamed_layers(
                    num_layers=num_layers, resident_layers=resident, policy=policy
                )
                # Every layer is either streamed or resident, never both and
                # never neither -- a gap here would strand a layer with no
                # weights at forward time.
                assert len(streamed) == len(set(streamed)) == num_layers - resident
                assert set(streamed) <= set(range(num_layers))


def test_policies_agree_at_the_degenerate_ends():
    for num_layers in (1, 4, 50):
        for resident in (0, num_layers):
            assert compute_streamed_layers(
                num_layers=num_layers,
                resident_layers=resident,
                policy=RESIDENCY_POLICY_LEADING,
            ) == compute_streamed_layers(
                num_layers=num_layers,
                resident_layers=resident,
                policy=RESIDENCY_POLICY_STRIDED,
            )


def test_strided_release_keeps_the_spread_resident_set(monkeypatch):
    _patch_fake_device(monkeypatch)
    # 8 layers, 4 resident -> streams every other layer.
    manager = _resident_manager(
        _MultiBlockModel(8),
        num_layers=8,
        resident_layers=4,
        residency_policy=RESIDENCY_POLICY_STRIDED,
    )
    streamed = set(manager._streamed_order)
    resident = set(range(8)) - streamed
    assert streamed == {0, 2, 4, 6}

    _arm_residency(manager)
    for layer_idx in range(8):
        manager.prefetch_layer(layer_idx, non_blocking=False)
    for layer_idx in range(8):
        manager.release_layer(layer_idx)

    # Non-force release frees exactly the streamed layers, whatever their index.
    assert manager._gpu_layers == resident
    manager.release_all()
    assert not manager._gpu_layers


def test_next_streamed_skips_residents_and_wraps(monkeypatch):
    _patch_fake_device(monkeypatch)
    manager = _resident_manager(
        _MultiBlockModel(8),
        num_layers=8,
        resident_layers=4,
        residency_policy=RESIDENCY_POLICY_STRIDED,
    )
    # Streamed = {0, 2, 4, 6}. From layer 1 the next transfer to issue is 2, not
    # 1+1 handled as "the following index" -- under this policy the immediate
    # successor is usually resident and prefetching it would be a no-op.
    assert manager._next_streamed(after=1, count=1) == [2]
    assert manager._next_streamed(after=2, count=2) == [4, 6]
    # Past the last streamed layer it wraps into the next step.
    assert manager._next_streamed(after=6, count=2) == [0, 2]
    # -1 is the "before the step starts" probe used when priming.
    assert manager._next_streamed(after=-1, count=1) == [0]


class _RunnableBlockModel(torch.nn.Module):
    """Blocks that actually run, so the registered hooks fire for real."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_OrderedLinearLayer(1.0) for _ in range(n)])


def test_strided_forward_leaves_exactly_the_resident_set(monkeypatch):
    _patch_fake_device(monkeypatch)
    model = _RunnableBlockModel(8)
    manager = _resident_manager(
        model,
        num_layers=8,
        resident_layers=4,
        residency_policy=RESIDENCY_POLICY_STRIDED,
    )

    hidden = torch.ones(1, 2)
    for _ in range(2):  # two denoise steps: residents must survive the first
        for layer in model.blocks:
            hidden = layer(hidden)

    # The pre-hook on layer 0 arms residency and primes; the post-hooks release
    # only streamed layers. After two full steps the GPU should hold the
    # resident set plus whatever the prefetch window pulled in ahead.
    resident = set(range(8)) - set(manager._streamed_order)
    assert resident <= manager._gpu_layers
    assert len(manager._gpu_layers) <= len(resident) + manager.prefetch_size


def test_layerwise_tuning_defaults_match_the_group():
    """No per-component entry: the DiT group keeps its knobs, auxiliaries do not."""
    args = _server_args(
        dit_offload_prefetch_size=3,
        dit_layerwise_resident_layers=20,
        dit_layerwise_residency_policy=RESIDENCY_POLICY_STRIDED,
    )
    assert args.layerwise_tuning_for("transformer", dit_group=True) == (
        3.0,
        20.0,
        RESIDENCY_POLICY_STRIDED,
    )
    assert args.layerwise_tuning_for("text_encoder", dit_group=False) == (
        0.0,
        0.0,
        RESIDENCY_POLICY_LEADING,
    )


def test_layerwise_tuning_per_component_entry_wins():
    """An auxiliary component can be tuned; its layers cost the same per pass."""
    args = _server_args(
        dit_offload_prefetch_size=3,
        dit_layerwise_resident_layers=20,
        layerwise_prefetch_size="text_encoder=2",
        layerwise_resident_layers="text_encoder=4",
        layerwise_residency_policy={"text_encoder": RESIDENCY_POLICY_STRIDED},
    )
    assert args.layerwise_tuning_for("text_encoder", dit_group=False) == (
        2.0,
        4.0,
        RESIDENCY_POLICY_STRIDED,
    )
    # an entry for one component leaves every other component alone
    assert args.layerwise_tuning_for("vae", dit_group=False) == (
        0.0,
        0.0,
        RESIDENCY_POLICY_LEADING,
    )
    assert args.layerwise_tuning_for("transformer", dit_group=True)[:2] == (3.0, 20.0)


def test_layerwise_tuning_rejects_unknown_policy():
    args = _server_args(layerwise_residency_policy="vae=sideways")
    with pytest.raises(ValueError, match="unknown residency policy"):
        args.layerwise_tuning_for("vae", dit_group=False)


def test_layerwise_tuning_accepts_json_and_pair_forms():
    pair = _server_args(layerwise_resident_layers="vae=6,text_encoder=2")
    assert pair.layerwise_tuning_for("vae", dit_group=False)[1] == 6.0
    assert pair.layerwise_tuning_for("text_encoder", dit_group=False)[1] == 2.0
    as_json = _server_args(layerwise_resident_layers='{"vae": 6}')
    assert as_json.layerwise_tuning_for("vae", dit_group=False)[1] == 6.0
