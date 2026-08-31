import pathlib
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
    host_memory_budget,
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
        performance_mode="auto",
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


def test_pin_budget_ranks_by_steps_resolved_from_model_index(monkeypatch):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")

    import sglang.multimodal_gen.registry as registry_mod

    class _Sampling:
        num_inference_steps = 50

    monkeypatch.setattr(
        registry_mod,
        "get_model_info",
        lambda *args, **kwargs: SimpleNamespace(sampling_param_cls=_Sampling),
    )

    # Same byte size on purpose: with the steps resolved, the stepped DiT
    # outranks the encoder; with the silent steps=1 fallback the ranking
    # ties and stable sort keeps the encoder first.
    text_encoder = _NestedEncoderDummyModel()
    transformer = _NestedDummyModel()
    modules = {"text_encoder": text_encoder, "transformer": transformer}

    configured = configure_layerwise_offload_modules(
        modules,
        _server_args(model_path="/models/minimax-h3"),
        component_names=["text_encoder", "transformer"],
    )

    assert configured == ["transformer", "text_encoder"], (
        "the pipeline is resolved from model_index when no override is set, "
        "so the stepped DiT must claim the pin budget before the "
        "once-per-request encoder"
    )


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


class _ParkableResidentComponent(_ResidentComponent):
    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.non_layer = torch.nn.Parameter(torch.ones(2))


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


def test_configure_logs_component_start_and_completion(monkeypatch):
    _patch_fake_device(monkeypatch)
    logs = []
    timestamps = iter((10.0, 12.345))
    monkeypatch.setattr(
        layerwise_offload_mod.logger,
        "info",
        lambda message, *args: logs.append(message % args),
    )
    monkeypatch.setattr(
        layerwise_offload_mod,
        "perf_counter",
        lambda: next(timestamps),
    )

    comp = _ResidentComponent(8)
    comp.configure_layerwise_offload(
        _server_args(
            dit_offload_prefetch_size=2,
            dit_layerwise_resident_layers=3,
        ),
        component_name="transformer",
    )

    assert logs[0] == (
        "Configuring layerwise offload for transformer (_ResidentComponent): "
        "blocks (8 layers)"
    )
    assert logs[-1] == (
        "Layerwise offload ready for transformer (_ResidentComponent) in 2.35s: "
        "groups=1, layers=8, prefetch/group=2, resident=3/8, policy=leading"
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


class _FileBackedBlock(torch.nn.Module):
    """A block whose weight is a view into a file, as a loaded checkpoint is."""

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        path.write_bytes(b"\x00" * (64 * 4))
        mapped = torch.from_file(str(path), shared=True, size=64, dtype=torch.float32)
        self.weight = torch.nn.Parameter(mapped.reshape(8, 8), requires_grad=False)


class _TransposedFileBackedBlock(torch.nn.Module):
    """A mapped weight whose layout is not contiguous, as an FP8 weight is."""

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        path.write_bytes(b"\x00" * (64 * 4))
        mapped = torch.from_file(str(path), shared=True, size=64, dtype=torch.float32)
        self.weight = torch.nn.Parameter(mapped.reshape(8, 8).t(), requires_grad=False)


class _FileBackedModel(torch.nn.Module):
    def __init__(
        self,
        path: pathlib.Path,
        num_blocks: int = 1,
        block_cls=_FileBackedBlock,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [block_cls(path.with_name(f"{path.name}.{i}")) for i in range(num_blocks)]
        )


# one _FileBackedBlock weight: 64 float32
_BLOCK_BYTES = 64 * 4


class _MixedBlock(torch.nn.Module):
    """A block that is half checkpoint view, half anonymous memory -- the shape
    of a layer whose qkv was fused at load while the rest stayed mapped."""

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        path.write_bytes(b"\x00" * (64 * 4))
        mapped = torch.from_file(str(path), shared=True, size=64, dtype=torch.float32)
        self.weight = torch.nn.Parameter(mapped.reshape(8, 8), requires_grad=False)
        self.fused = torch.nn.Parameter(
            torch.zeros(8, 8, dtype=torch.float32), requires_grad=False
        )


def test_mapped_layers_ship_through_the_courier(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(tmp_path, monkeypatch, available_gib=0.001)
    model = manager.model
    # the manager has already swapped the parameter for a placeholder; the
    # checkpoint file itself holds zeros, so that is the expected content
    expected = torch.zeros(8, 8)

    manager.prefetch_layer(0, non_blocking=True)
    assert 0 in manager._courier_inflight, "an async prefetch hands the layer over"
    assert (
        0 not in manager._gpu_layers
    ), "the layer is not ready until its tensors are bound on this thread"

    manager.prefetch_layer(0, non_blocking=False)
    assert 0 in manager._gpu_layers and not manager._courier_inflight
    assert torch.equal(
        model.blocks[0].weight.detach().cpu(), expected
    ), "the bytes that went through the courier's slot must be the checkpoint's"


def test_the_courier_kill_switch_forces_the_synchronous_path(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    # The switch is read when the courier would first be built, and
    # initialization itself prefetches -- so set it before the manager exists,
    # the way a deployment sets it before starting the server.
    monkeypatch.setenv("SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER", "1")
    manager = _mapped_manager(tmp_path, monkeypatch, available_gib=0.001)
    manager.release_all()
    manager.prefetch_layer(0, non_blocking=True)
    assert not manager._courier_inflight and manager._mapped_courier is None
    assert (
        0 in manager._gpu_layers
    ), "with the courier disabled the direct synchronous path serves the layer"


def test_release_all_drains_the_courier(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(tmp_path, monkeypatch, available_gib=0.001)
    manager.prefetch_layer(0, non_blocking=True)

    manager.release_all()

    assert not manager._courier_inflight
    assert not manager._gpu_layers
    courier = manager._mapped_courier
    assert courier is not None and not courier._results, (
        "an uncollected layer would keep its device tensors alive inside the "
        "courier for the rest of the process"
    )


def test_a_broken_courier_falls_back_to_the_synchronous_copy(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(tmp_path, monkeypatch, available_gib=0.001)
    model = manager.model
    expected = torch.zeros(8, 8)

    manager.prefetch_layer(0, non_blocking=True)
    # the courier dies with the layer still in flight
    courier = manager._mapped_courier
    courier._broken = True
    with courier._ready:
        courier._results.pop(0, None)
        courier._pending.discard(0)
        courier._ready.notify_all()

    manager.prefetch_layer(0, non_blocking=False)

    assert 0 in manager._gpu_layers
    assert manager._mapped_courier is None, "a failed courier is retired"
    assert torch.equal(model.blocks[0].weight.detach().cpu(), expected)


class _MixedModel(torch.nn.Module):
    def __init__(self, path: pathlib.Path, num_blocks: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [_MixedBlock(path.with_name(f"{path.name}.{i}")) for i in range(num_blocks)]
        )


# one _MixedBlock: 64 float32 mapped + 64 float32 anonymous
_MIXED_BLOCK_BYTES = 2 * 64 * 4


def _mapped_manager(
    tmp_path,
    monkeypatch,
    *,
    available_gib=None,
    available_bytes=None,
    num_blocks=1,
    pin_budget_bytes=None,
    block_cls=_FileBackedBlock,
):
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    if available_bytes is None:
        available_bytes = int(available_gib * 1024**3)
    monkeypatch.setattr(
        host_memory_budget, "host_memory_available_bytes", lambda: available_bytes
    )
    model = _FileBackedModel(
        tmp_path / "weights.bin", num_blocks=num_blocks, block_cls=block_cls
    )
    return LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=num_blocks,
        enabled=True,
        pin_cpu_memory=True,
        pin_budget=(
            host_memory_budget.HostPinBudget(available_bytes=pin_budget_bytes)
            if pin_budget_bytes is not None
            else None
        ),
        prefetch_size=1,
    )


def test_weights_stay_on_the_mapping_when_copies_do_not_fit(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    # the reserve alone exceeds this, so no copy can be afforded
    manager = _mapped_manager(
        tmp_path, monkeypatch, available_gib=0.001, pin_budget_bytes=0
    )
    assert manager._mapped_cpu_weights[0], "expected the weight to stay mapped"
    assert manager._weight_metadata[0]["blocks.0.weight"]["mapped"] is True
    assert not manager._consolidated_cpu_weights.get(0)


def test_weights_are_copied_when_they_fit(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(tmp_path, monkeypatch, available_gib=64)
    assert not manager._mapped_cpu_weights[0], "a copy was affordable"
    assert manager._consolidated_cpu_weights[0]


def test_a_mapped_weight_is_not_written_back(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(
        tmp_path, monkeypatch, available_gib=0.001, pin_budget_bytes=0
    )
    before = manager._mapped_cpu_weights[0]["blocks.0.weight"].clone()
    manager._gpu_layers.add(0)
    manager.sync_layer_to_cpu(0)
    after = manager._mapped_cpu_weights[0]["blocks.0.weight"]
    assert torch.equal(before, after), "writeback must not touch the checkpoint"


def test_the_mapped_store_survives_the_placeholder(tmp_path, monkeypatch):
    """The store must not hold the parameter it is about to have overwritten.

    `_to_local_tensor` returns the parameter itself for anything that is not a
    DTensor, so storing it directly stores the parameter. `Tensor.data = ...`
    then swaps that object's storage in place rather than rebinding a name, so
    the store is left holding the `(1,)` placeholder. Nothing raises: the reload
    does `gpu_tensor.copy_(cpu_tensor)`, and a one-element source broadcasts
    into the full shape, so the layer is silently reconstructed from one value.
    """
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(tmp_path, monkeypatch, available_gib=0.001)
    stored = manager._mapped_cpu_weights[0]
    assert stored, "expected the weight to stay mapped"

    parameters = dict(manager.model.named_parameters())
    for name, tensor in stored.items():
        assert tensor.numel() > 1, (
            f"{name} holds {tensor.numel()} element(s): the store is holding the "
            "placeholder that was assigned to the parameter, not the weight"
        )
        assert tensor is not parameters[name], (
            f"{name} in the store is the parameter object itself, so assigning "
            "to the parameter's .data will overwrite the store"
        )
    assert manager._mapped_bytes == sum(
        t.numel() * t.element_size() for t in stored.values()
    ), "the byte counter and the store must describe the same weights"


def test_only_the_layers_the_budget_covers_are_pinned(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    # The pin budget covers two of the four layers. Available host memory sits
    # above the copy reserve by less than all four layers but more than the
    # two pins, so the rest is demoted to the mapping and the pins stand.
    budget = 2 * 1024**3 + 2 * _BLOCK_BYTES
    manager = _mapped_manager(
        tmp_path,
        monkeypatch,
        available_bytes=4 * 1024**3 + 3 * _BLOCK_BYTES,
        pin_budget_bytes=budget,
        num_blocks=4,
    )
    pinned = {i for i in range(4) if manager._consolidated_cpu_weights.get(i)}
    mapped = {i for i in range(4) if manager._mapped_cpu_weights.get(i)}
    assert pinned == {0, 1}, "the layers the budget covers, taken in index order"
    assert mapped == {2, 3}
    assert not (pinned & mapped), "a layer is in one store or the other"


def test_pins_are_given_back_when_they_do_not_fit_the_host(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    # Four fully mapped layers. The budget covers two pins, but each pin copies
    # its whole layer off the mapping and the host only has room for one plus
    # the reserve: the plan must give the second pin back rather than allocate
    # more than the machine has.
    available = 4 * 1024**3 + int(1.5 * _BLOCK_BYTES)
    manager = _mapped_manager(
        tmp_path,
        monkeypatch,
        available_bytes=available,
        pin_budget_bytes=2 * 1024**3 + 2 * _BLOCK_BYTES,
        num_blocks=4,
    )
    on_mapping = {i for i in range(4) if manager._mapped_cpu_weights.get(i)}
    assert on_mapping == {1, 2, 3}, (
        "two pins add two layers of anonymous memory against room for 1.5, so "
        "the least valuable pin (layer 1) is given back"
    )
    pinned = {i for i in range(4) if manager._consolidated_cpu_weights.get(i)}
    assert pinned == {0}


def test_replacing_an_anonymous_original_is_not_charged_as_new(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    # Half of each layer is a fused anonymous tensor whose store buffer
    # replaces it -- a wash. Only the mapped half is a net addition, so a
    # host with room for the mapped halves alone must still pin every layer.
    block = _MIXED_BLOCK_BYTES
    available = 4 * 1024**3 + int(2.5 * block)
    monkeypatch.setattr(
        layerwise_offload_mod.torch, "get_device_module", lambda: _FakeDeviceModule
    )
    monkeypatch.setattr(layerwise_offload_mod.current_platform, "device_type", "cpu")
    monkeypatch.setattr(
        host_memory_budget, "host_memory_available_bytes", lambda: available
    )
    model = _MixedModel(tmp_path / "weights.bin", num_blocks=4)
    manager = LayerwiseOffloadManager(
        model=model,
        layers_attr_str="blocks",
        num_layers=4,
        enabled=True,
        pin_cpu_memory=True,
        pin_budget=host_memory_budget.HostPinBudget(
            available_bytes=2 * 1024**3 + 3 * block
        ),
        prefetch_size=1,
    )
    pinned = {i for i in range(4) if 0 in manager._consolidated_cpu_weights}
    on_mapping = {i for i in range(4) if manager._mapped_cpu_weights.get(i)}
    assert not on_mapping, (
        "the net addition is three pinned mapped-halves plus one pageable "
        "mapped-half (2 blocks) against room for 2.5 -- charging the replaced "
        "anonymous originals as new would demote, then strip every pin"
    )
    assert all(manager._consolidated_cpu_weights.get(i) for i in range(4))


def test_every_layer_is_pinned_when_the_budget_covers_them(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(
        tmp_path,
        monkeypatch,
        available_gib=64,
        pin_budget_bytes=64 * 1024**3,
        num_blocks=4,
    )
    assert not any(manager._mapped_cpu_weights.get(i) for i in range(4))
    assert all(manager._consolidated_cpu_weights.get(i) for i in range(4))


def test_no_layer_is_pinned_when_the_budget_is_spent(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(
        tmp_path, monkeypatch, available_bytes=0, pin_budget_bytes=0, num_blocks=4
    )
    assert all(manager._mapped_cpu_weights.get(i) for i in range(4))


def test_an_unpinnable_layer_is_still_copied_when_the_copies_fit(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    # no pinned budget at all, but plenty of host memory: a pageable copy is
    # guaranteed resident where a mapping can be dropped and re-read
    manager = _mapped_manager(
        tmp_path,
        monkeypatch,
        available_gib=64,
        pin_budget_bytes=0,
        num_blocks=4,
    )
    assert not any(manager._mapped_cpu_weights.get(i) for i in range(4))
    assert all(manager._consolidated_cpu_weights.get(i) for i in range(4))


def test_mapped_weights_are_visible_to_checksums(tmp_path, monkeypatch):
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(
        tmp_path, monkeypatch, available_gib=0.001, pin_budget_bytes=0
    )
    names = {name for name, _ in manager.iter_cpu_weights()}
    assert "blocks.0.weight" in names


def test_a_non_contiguous_mapped_weight_keeps_its_layout(tmp_path, monkeypatch):
    """Staying mapped costs the layout, so a strided weight must not stay.

    The reload path allocates with `torch.empty(shape)` and copies, which is
    layout-agnostic: values survive, strides do not. ModelOpt FP8 calls its
    transposed layout a correctness requirement, so such a weight has to take
    the strided path even when its storage is a mapping the copies cannot
    afford.
    """
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    manager = _mapped_manager(
        tmp_path,
        monkeypatch,
        available_gib=0.001,
        pin_budget_bytes=0,
        block_cls=_TransposedFileBackedBlock,
    )
    name = "blocks.0.weight"
    assert name not in manager._mapped_cpu_weights[0], (
        "a non-contiguous weight stayed mapped, so its layout is dropped "
        "on reload without any error"
    )
    stored = manager._strided_cpu_weights[0][name]
    assert not stored.is_contiguous()
    assert stored.stride() == (1, 8)
    assert manager._weight_metadata[0][name]["preserve_strides"] is True


def test_refitting_a_mapped_weight_updates_the_store(tmp_path, monkeypatch):
    """A refit must reach a mapped weight without writing to the checkpoint."""
    if not pathlib.Path("/proc/self/maps").exists():
        pytest.skip("needs /proc to tell a mapping from anonymous memory")
    path = tmp_path / "weights.bin.0"
    manager = _mapped_manager(
        tmp_path, monkeypatch, available_gib=0.001, pin_budget_bytes=0
    )
    name = "blocks.0.weight"
    assert manager._weight_metadata[0][name]["mapped"] is True
    on_disk_before = path.read_bytes()

    new_weight = torch.full((8, 8), 3.0)
    updated = manager.update_cpu_weights({name: new_weight})

    assert updated == {name}
    assert torch.equal(manager._mapped_cpu_weights[0][name], new_weight)
    assert path.read_bytes() == on_disk_before, "the checkpoint was written to"


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


def test_non_layer_parking_follows_memory_performance_mode(monkeypatch):
    """The extra transfer per request only pays for itself under memory mode."""
    _patch_fake_device(monkeypatch)
    tight = _ResidentComponent(4)
    tight.configure_layerwise_offload(_server_args(performance_mode="memory"))
    assert tight.park_non_layer_weights_between_uses

    relaxed = _ResidentComponent(4)
    relaxed.configure_layerwise_offload(_server_args(performance_mode="speed"))
    assert not relaxed.park_non_layer_weights_between_uses


def test_parking_leaves_streamed_layer_weights_alone(monkeypatch):
    """Only the parameters no manager streams are moved to the host."""
    comp = _ParkableResidentComponent(4)
    comp.configure_layerwise_offload(_server_args(performance_mode="memory"))
    _headroom(monkeypatch, 0)
    managed = comp._managed_layer_parameter_names()
    assert managed, "the managers should own the block parameters"

    comp.park_non_layer_weights()
    parked = comp._parked_non_layer_weights
    assert not (set(parked) & managed), "a streamed layer weight was parked"
    for name, host_tensor in parked.items():
        assert host_tensor.device.type == "cpu", name

    comp.restore_non_layer_weights()
    restored = dict(comp.named_parameters())
    for name, host_tensor in parked.items():
        assert restored[name].shape == host_tensor.shape


def test_parking_is_a_no_op_outside_memory_mode(monkeypatch):
    comp = _ParkableResidentComponent(4)
    comp.configure_layerwise_offload(_server_args(performance_mode="speed"))
    comp.park_non_layer_weights()
    assert not comp._parked_non_layer_weights


def _headroom(monkeypatch, gib):
    monkeypatch.setattr(
        layerwise_offload_mod.current_platform,
        "get_available_gpu_memory",
        lambda **_: float(gib),
    )
    module = layerwise_offload_mod.torch.get_device_module()
    monkeypatch.setattr(module, "memory_reserved", lambda *_: 0, raising=False)
    monkeypatch.setattr(module, "memory_allocated", lambda *_: 0, raising=False)


def test_parking_is_skipped_when_the_card_has_room(monkeypatch):
    """A component holding a sliver of a large headroom is left alone."""
    comp = _ParkableResidentComponent(4)
    comp.configure_layerwise_offload(_server_args(performance_mode="memory"))
    _headroom(monkeypatch, 400)
    comp.park_non_layer_weights()
    assert not comp._parked_non_layer_weights


def test_parking_happens_when_the_headroom_is_small(monkeypatch):
    comp = _ParkableResidentComponent(4)
    comp.configure_layerwise_offload(_server_args(performance_mode="memory"))
    _headroom(monkeypatch, 0)
    comp.park_non_layer_weights()
    assert comp._parked_non_layer_weights


def test_host_copies_are_given_back_when_room_appears(monkeypatch):
    """Skipping must not leave host memory held for a park that will not happen."""
    comp = _ParkableResidentComponent(4)
    comp.configure_layerwise_offload(_server_args(performance_mode="memory"))
    _headroom(monkeypatch, 0)
    comp.park_non_layer_weights()
    assert comp._parked_non_layer_weights
    comp.restore_non_layer_weights()

    _headroom(monkeypatch, 400)
    comp.park_non_layer_weights()
    assert not comp._parked_non_layer_weights, "host copies should be released"


def test_park_placeholders_are_shared(monkeypatch):
    """One stand-in per (device, dtype), not one allocation per parked weight."""
    comp = _ParkableResidentComponent(4)
    comp.configure_layerwise_offload(_server_args(performance_mode="memory"))
    _headroom(monkeypatch, 0)
    comp.park_non_layer_weights()
    managed = comp._managed_layer_parameter_names()
    stand_ins = {
        id(p) for n, p in comp.named_parameters() if n not in managed and p.numel() == 1
    }
    assert len(stand_ins) <= len(comp._park_placeholders)


def _layer_weight_ok(layer: torch.nn.Module) -> bool:
    return tuple(layer.weight.shape) != (1,)


def test_skip_middle_layers_loads_destination_weights(monkeypatch):
    """Cache-DiT DBCache shape: run Fn, jump to Bn, middle never forwards.

    The destination layer used to see empty(1,) weights when wraparound
    prefetch and the sequential i+1 window desynced. The jump must sync-load
    Bn and leave the skipped gap released.
    """
    _patch_fake_device(monkeypatch)
    model = _RunnableBlockModel(8)
    manager = _resident_manager(model, num_layers=8, prefetch_size=1)

    hidden = torch.ones(1, 2)
    fn_end = 1
    bn_start = 6
    for layer in model.blocks[:fn_end]:
        hidden = layer(hidden)
    for layer in model.blocks[bn_start:]:
        hidden = layer(hidden)

    assert hidden.shape == (1, 2)
    for idx in range(fn_end, bn_start):
        assert idx not in manager._gpu_layers, idx
        assert not _layer_weight_ok(model.blocks[idx]), idx


def test_skip_only_fn_releases_speculative_prefetch_on_next_step(monkeypatch):
    """Fn-only step (Cache-DiT hit with Bn=0) must not leak the i+1 prefetch."""
    _patch_fake_device(monkeypatch)
    model = _RunnableBlockModel(8)
    manager = _resident_manager(model, num_layers=8, prefetch_size=1)

    hidden = torch.ones(1, 2)
    hidden = model.blocks[0](hidden)
    # Layer 0's leading burst prefetches layer 1; that layer never runs.
    assert 1 in manager._gpu_layers

    # Next denoise step's prepare drops leftovers that never posted.
    manager.prepare_for_next_req(non_blocking=False)
    assert 1 not in manager._gpu_layers


def test_last_layer_wraps_to_next_step_head(monkeypatch):
    """A full-stack step may hide the next step's layer 0 behind the last layer."""
    _patch_fake_device(monkeypatch)
    model = _RunnableBlockModel(8)
    manager = _resident_manager(model, num_layers=8, prefetch_size=1)

    hidden = torch.ones(1, 2)
    for layer in model.blocks:
        hidden = layer(hidden)

    assert hidden.shape == (1, 2)
    assert 0 in manager._gpu_layers
    assert 7 not in manager._gpu_layers


def _dbcache_layers(num_layers: int, fn: int, bn: int) -> list[int]:
    """Layers CachedBlocks would call for one DBCache step."""
    fn = min(max(fn, 0), num_layers)
    bn = min(max(bn, 0), num_layers - fn)
    layers = list(range(fn))
    if bn:
        layers.extend(range(num_layers - bn, num_layers))
    return layers


def _run_layer_set(model, layer_indices: list[int]) -> torch.Tensor:
    hidden = torch.ones(1, 2)
    for idx in layer_indices:
        hidden = model.blocks[idx](hidden)
    return hidden


@pytest.mark.parametrize("num_layers", [8, 12])
@pytest.mark.parametrize(
    "fn,bn",
    [
        (1, 0),  # default Cache-DiT hit
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 2),
        (4, 2),
        (3, 5),  # Fn+Bn == 8, no gap when num_layers=8
        (8, 0),  # full stack / miss
    ],
)
@pytest.mark.parametrize("prefetch_size", [1, 2])
@pytest.mark.parametrize(
    "residency_policy",
    [RESIDENCY_POLICY_LEADING, RESIDENCY_POLICY_STRIDED],
)
def test_dbcache_layer_patterns_never_see_empty_weights(
    monkeypatch, num_layers, fn, bn, prefetch_size, residency_policy
):
    """Hit / miss / hit-again under several Cache-DiT Fn/Bn and prefetch windows."""
    if fn + bn > num_layers:
        pytest.skip("Fn+Bn exceeds this stack")
    _patch_fake_device(monkeypatch)
    model = _RunnableBlockModel(num_layers)
    manager = _resident_manager(
        model,
        num_layers=num_layers,
        prefetch_size=prefetch_size,
        residency_policy=residency_policy,
        resident_layers=0,
    )

    hit_layers = _dbcache_layers(num_layers, fn, bn)
    miss_layers = list(range(num_layers))
    gap = [idx for idx in miss_layers if idx not in set(hit_layers)]

    def _assert_gpu_layers_have_real_weights() -> None:
        for idx in range(num_layers):
            on_gpu = idx in manager._gpu_layers
            assert _layer_weight_ok(model.blocks[idx]) is on_gpu, idx

    hidden = _run_layer_set(model, hit_layers)
    assert hidden.shape == (1, 2)
    _assert_gpu_layers_have_real_weights()

    # Speculative Mn prefetch may still sit on GPU until the next prepare.
    manager.prepare_for_next_req(non_blocking=False)
    keep = set(manager._head_of_stream()) | set(manager._retained_set)
    for idx in gap:
        if idx not in keep:
            assert idx not in manager._gpu_layers, idx
            assert not _layer_weight_ok(model.blocks[idx]), idx

    hidden = _run_layer_set(model, miss_layers)
    assert hidden.shape == (1, 2)
    _assert_gpu_layers_have_real_weights()

    manager.prepare_for_next_req(non_blocking=False)
    hidden = _run_layer_set(model, hit_layers)
    assert hidden.shape == (1, 2)
    _assert_gpu_layers_have_real_weights()

    # Two hits in a row (Bn=0 never reaches last layer; still must rematerialize 0).
    manager.prepare_for_next_req(non_blocking=False)
    hidden = _run_layer_set(model, hit_layers)
    assert hidden.shape == (1, 2)
    _assert_gpu_layers_have_real_weights()


@pytest.mark.parametrize(
    "step_kinds",
    [
        # SCM-style: forced compute (full stack) mixed with DBCache hits.
        ("full", "hit10", "hit10", "full", "hit12", "hit10"),
        # TaylorSeer does not change which blocks CachedBlocks calls.
        ("full", "full", "hit20", "hit20", "hit12", "full"),
    ],
)
def test_mixed_scm_and_dbcache_step_schedule(monkeypatch, step_kinds):
    """A request is a sequence of full-stack and skip-compute steps."""
    _patch_fake_device(monkeypatch)
    num_layers = 8
    model = _RunnableBlockModel(num_layers)
    manager = _resident_manager(model, num_layers=num_layers, prefetch_size=2)
    kind_to_layers = {
        "full": list(range(num_layers)),
        "hit10": _dbcache_layers(num_layers, 1, 0),
        "hit12": _dbcache_layers(num_layers, 1, 2),
        "hit20": _dbcache_layers(num_layers, 2, 0),
    }
    for kind in step_kinds:
        hidden = _run_layer_set(model, kind_to_layers[kind])
        assert hidden.shape == (1, 2)
        for idx in range(num_layers):
            on_gpu = idx in manager._gpu_layers
            assert _layer_weight_ok(model.blocks[idx]) is on_gpu, (kind, idx)
        manager.prepare_for_next_req(non_blocking=False)
