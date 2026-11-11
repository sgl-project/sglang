"""CPU contracts for the FlashInfer cuTile NVFP4 adapter."""

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StandardTopK(SimpleNamespace):
    pass


class _StandardTopKWithPacked(SimpleNamespace):
    pass


class _Backend(Enum):
    FLASHINFER_CUTILE = "flashinfer_cutile"


@pytest.fixture
def adapter(monkeypatch):
    """Stub GPU dependency boundaries; execute the real adapter on CPU tensors."""
    names = (
        "sglang.srt.layers.moe.moe_runner.base",
        "sglang.srt.layers.moe.utils",
        "sglang.srt.layers.moe.topk",
        "sglang.srt.layers.moe.token_dispatcher.standard",
        "sglang.srt.runtime_context",
        "flashinfer",
        "flashinfer.fused_moe",
        "flashinfer.fused_moe.core",
    )
    modules = {name: ModuleType(name) for name in names}

    class DispatchCore:
        def __init__(self, config):
            self.config = config

    base = modules["sglang.srt.layers.moe.moe_runner.base"]
    base.DispatchMoeRunnerCore = DispatchCore
    base.MoeQuantInfo = type("MoeQuantInfo", (), {})
    base.MoeRunnerConfig = SimpleNamespace
    utils = modules["sglang.srt.layers.moe.utils"]
    utils.MoeRunnerBackend = _Backend
    utils.get_moe_a2a_backend = lambda: SimpleNamespace(
        is_none=lambda: True, value="none"
    )
    topk = modules["sglang.srt.layers.moe.topk"]
    topk.TopKOutputChecker = SimpleNamespace(
        format_is_standard=lambda value: isinstance(
            value, (_StandardTopK, _StandardTopKWithPacked)
        )
    )
    standard = modules["sglang.srt.layers.moe.token_dispatcher.standard"]
    standard.StandardDispatchOutput = SimpleNamespace
    standard.StandardCombineInput = SimpleNamespace
    context = modules["sglang.srt.runtime_context"]
    context.cutedsl_moe_max_num_tokens = lambda: 512
    context.get_parallel = lambda: SimpleNamespace(moe_ep_size=1)
    context.get_platform = lambda: SimpleNamespace(device_sm=120)
    context.get_server_args = lambda: SimpleNamespace(max_prefill_tokens=512)

    class WeightPack:
        def __init__(self):
            self.views = {}

        def prepare_for(self, name, view):
            self.views[name] = view

    class CuTileConfig:
        @staticmethod
        def prepare_weights(**kwargs):
            return SimpleNamespace(**kwargs)

    provider = modules["flashinfer.fused_moe"]
    provider.CuTileNvfp4Config = CuTileConfig
    provider.MoEWeightPack = WeightPack
    for name in (
        "SwiGLU",
        "MoEActivationPack",
        "MoEConfig",
        "RoutingConfig",
        "QuantConfig",
        "ExpertConfig",
        "BackendOptions",
        "ExecutionConfig",
    ):
        setattr(provider, name, SimpleNamespace)
    provider.QuantFormat = SimpleNamespace(NVFP4="nvfp4")
    modules["flashinfer.fused_moe.core"].RoutingInputMode = SimpleNamespace(
        PackedPrecomputed="packed_precomputed"
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutile.py"
    )
    name = "sglang_cutile_adapter_cpu_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_check_device_support", lambda device: None)
    return module


def _config(**overrides):
    layer = SimpleNamespace(moe_ep_size=1, moe_tp_size=1)
    values = dict(
        num_experts=2,
        num_local_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=64,
        top_k=2,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        num_fused_shared_experts=0,
        apply_router_weight_on_input=False,
        no_combine=False,
        no_combine_keep_router_weight=False,
        silu_mul_keep_fp32=False,
        use_tp_all_gather_activation=False,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        swiglu_limit=None,
        gate_up_interleaved=False,
        routed_scaling_factor=None,
        layer=layer,
        moe_ep_size=1,
        moe_tp_size=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _layer():
    layer = torch.nn.Module()
    values = {
        "w13_weight_packed": torch.cat(
            [torch.full((2, 64, 32), 17), torch.full((2, 64, 32), 34)], dim=1
        ).to(torch.uint8),
        "w13_weight_scale": torch.cat(
            [torch.full((2, 64, 4), 2.0), torch.full((2, 64, 4), 3.0)], dim=1
        ).to(torch.float8_e4m3fn),
        "w13_weight_global_scale": torch.tensor([[2.0, 4.0], [3.0, 6.0]]),
        "w13_input_global_scale": torch.ones(2, 2),
        "w2_weight_packed": torch.full((2, 64, 32), 51, dtype=torch.uint8),
        "w2_weight_scale": torch.full((2, 64, 4), 4.0).to(torch.float8_e4m3fn),
        "w2_weight_global_scale": torch.tensor([5.0, 7.0]),
        "w2_input_global_scale": torch.ones(2),
    }
    for name, value in values.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    layer.num_experts = layer.num_local_experts = 2
    layer.hidden_size = layer.intermediate_size_per_partition = 64
    layer.params_dtype = torch.bfloat16
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_ep_rank = layer.moe_tp_rank = 0
    return layer


def _dispatch(topk_type=_StandardTopK, *, empty=False):
    count = 0 if empty else 2
    return SimpleNamespace(
        hidden_states=torch.ones(count, 64, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=topk_type(
            topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)[:count],
            topk_weights=torch.tensor([[2.0, 3.0], [1.0, 4.0]])[:count],
            router_logits=None,
        ),
    )


def test_weight_preparation_preserves_branch_scale_pairing(adapter):
    layer = _layer()
    original = {
        name: parameter.detach().clone() for name, parameter in layer.named_parameters()
    }
    prepared = adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, _config())
    view = prepared.weight_pack.views["cutile_nvfp4"]

    assert torch.equal(view.w1_fp4[:, :64], original["w13_weight_packed"][:, 64:])
    assert torch.equal(view.w1_fp4[:, 64:], original["w13_weight_packed"][:, :64])
    torch.testing.assert_close(
        view.w1_block_scale[:, :64].float(),
        original["w13_weight_scale"][:, 64:].float(),
    )
    torch.testing.assert_close(
        view.w1_block_scale[:, 64:].float(),
        original["w13_weight_scale"][:, :64].float(),
    )
    torch.testing.assert_close(
        view.w1_global_scale, torch.tensor([[4.0, 2.0], [6.0, 3.0]])
    )
    torch.testing.assert_close(view.w2_global_scale, torch.tensor([5.0, 7.0]))
    assert torch.equal(view.w2_fp4, original["w2_weight_packed"])
    assert view.source_format == "compressed_tensors"
    assert (view.activation.alpha, view.activation.beta, view.activation.limit) == (
        1.702,
        1.0,
        7.0,
    )
    # Source globals are encode scales; the adapter must not invert or mutate them.
    for name, original_value in original.items():
        if getattr(layer, name, None) is not None:
            torch.testing.assert_close(
                getattr(layer, name).float(), original_value.float()
            )


def test_prepared_weights_are_owned_by_each_layer(adapter):
    first, second = _layer(), _layer()
    second.w13_weight_packed.fill_(68)
    one = adapter.prepare_flashinfer_cutile_nvfp4_weights(first, _config())
    two = adapter.prepare_flashinfer_cutile_nvfp4_weights(second, _config())
    assert one.weight_pack is not two.weight_pack
    assert not torch.equal(
        one.weight_pack.views["cutile_nvfp4"].w1_fp4,
        two.weight_pack.views["cutile_nvfp4"].w1_fp4,
    )


@pytest.mark.parametrize("name", ["w13_input_global_scale", "w2_input_global_scale"])
def test_nonunit_activation_global_scale_is_rejected(adapter, name):
    layer = _layer()
    getattr(layer, name).fill_(2.0)
    with pytest.raises(ValueError, match="scale"):
        adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, _config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"params_dtype": torch.float16},
        {"activation": "relu"},
        {"is_gated": False},
        {"no_combine": True},
        {"apply_router_weight_on_input": True},
        {"num_fused_shared_experts": 1},
        {"silu_mul_keep_fp32": True},
        {"num_local_experts": 1},
        {"hidden_size": 65},
        {"intermediate_size_per_partition": 65},
    ],
)
def test_unsupported_runner_config_is_rejected(adapter, overrides):
    with pytest.raises(ValueError):
        adapter.validate_flashinfer_cutile_config(_config(**overrides))


@pytest.mark.parametrize("factor", [None, 2.5])
@pytest.mark.parametrize("topk_type", [_StandardTopK, _StandardTopKWithPacked])
def test_routing_values_and_output_scaling(adapter, monkeypatch, factor, topk_type):
    config = _config(routed_scaling_factor=factor)
    runner = adapter.FlashInferCuTileRunnerCore(config)
    dispatch = _dispatch(topk_type)
    original_weights = dispatch.topk_output.topk_weights.clone()
    original_input = dispatch.hidden_states.clone()

    def execute(activations, weight_pack):
        torch.testing.assert_close(activations.topk_weights, original_weights)
        assert activations.topk_ids.dtype == torch.int32
        totals = (activations.topk_weights * (activations.topk_ids + 1)).sum(
            dim=-1, keepdim=True
        )
        return (activations.hidden_states_q.float() * totals).bfloat16()

    monkeypatch.setattr(runner, "_get_execution", lambda device, num_tokens: execute)
    payload = adapter.FlashInferCuTileMoeQuantInfo(
        weight_pack=object(), device=torch.device("cpu")
    )
    result = runner.run_from_dispatch(dispatch, payload, config).hidden_states
    expected = torch.tensor([8.0, 6.0])[:, None].expand(2, 64)
    if factor is not None:
        expected = expected * factor
    torch.testing.assert_close(result.float(), expected)
    torch.testing.assert_close(dispatch.topk_output.topk_weights, original_weights)
    torch.testing.assert_close(dispatch.hidden_states, original_input)


def test_empty_batch_does_not_create_execution(adapter, monkeypatch):
    config = _config()
    runner = adapter.FlashInferCuTileRunnerCore(config)
    create = Mock(side_effect=AssertionError("empty batch must not create workspace"))
    monkeypatch.setattr(runner, "_get_execution", create)
    dispatch = _dispatch(empty=True)
    payload = adapter.FlashInferCuTileMoeQuantInfo(
        weight_pack=object(), device=torch.device("cpu")
    )
    result = runner.run_from_dispatch(dispatch, payload, config).hidden_states
    assert result.shape == (0, 64)
    assert result.dtype == torch.bfloat16


def test_packed_only_routing_is_rejected(adapter):
    config = _config()
    runner = adapter.FlashInferCuTileRunnerCore(config)
    dispatch = _dispatch()
    dispatch.topk_output = SimpleNamespace(
        packed_topk_ids=torch.zeros(2, 2, dtype=torch.int32)
    )
    payload = adapter.FlashInferCuTileMoeQuantInfo(
        weight_pack=object(), device=torch.device("cpu")
    )
    with pytest.raises((TypeError, ValueError), match="[Ss]tandard|routing|top-?k"):
        runner.run_from_dispatch(dispatch, payload, config)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_weight_global_scale_is_rejected(adapter, value):
    layer = _layer()
    layer.w2_weight_global_scale.fill_(value)
    with pytest.raises(ValueError, match="finite positive"):
        adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, _config())


def test_invalid_global_scale_columns_are_rejected(adapter):
    layer = _layer()
    layer.w13_weight_global_scale = torch.nn.Parameter(
        torch.ones(2, 3), requires_grad=False
    )
    with pytest.raises(ValueError, match="shape|global"):
        adapter.prepare_flashinfer_cutile_nvfp4_weights(layer, _config())


def test_negative_padding_does_not_route_to_an_expert(adapter, monkeypatch):
    config = _config()
    runner = adapter.FlashInferCuTileRunnerCore(config)
    dispatch = _dispatch()
    dispatch.topk_output.topk_ids[0, 1] = -1
    original_ids = dispatch.topk_output.topk_ids.clone()
    original_weights = dispatch.topk_output.topk_weights.clone()

    def execute(activations, weight_pack):
        assert activations.topk_ids[0, 1].item() == 0
        assert activations.topk_weights[0, 1].item() == 0
        totals = (activations.topk_weights * (activations.topk_ids + 1)).sum(
            dim=-1, keepdim=True
        )
        return (activations.hidden_states_q.float() * totals).bfloat16()

    monkeypatch.setattr(runner, "_get_execution", lambda device, num_tokens: execute)
    payload = adapter.FlashInferCuTileMoeQuantInfo(
        weight_pack=object(), device=torch.device("cpu")
    )
    result = runner.run_from_dispatch(dispatch, payload, config).hidden_states
    torch.testing.assert_close(
        result.float(), torch.tensor([2.0, 6.0])[:, None].expand(2, 64)
    )
    assert torch.equal(dispatch.topk_output.topk_ids, original_ids)
    torch.testing.assert_close(dispatch.topk_output.topk_weights, original_weights)


def test_execution_pool_shares_only_matching_stream_and_geometry(adapter, monkeypatch):
    import gc
    import weakref

    stream = [10]
    monkeypatch.setattr(adapter, "_get_max_num_tokens", lambda: 8)
    monkeypatch.setattr(adapter, "_is_capturing", lambda device: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=stream[0]),
    )
    monkeypatch.setattr(
        adapter,
        "_create_moe_layer",
        lambda **kwargs: lambda act, weights: act.hidden_states_q.clone(),
    )
    first = adapter.FlashInferCuTileRunnerCore(_config())
    second = adapter.FlashInferCuTileRunnerCore(_config())
    device = torch.device("cpu")
    shared = first._get_execution(device, 2)
    assert second._get_execution(device, 2) is shared

    stream[0] = 11
    assert first._get_execution(device, 2) is not shared
    stream[0] = 10
    other = adapter.FlashInferCuTileRunnerCore(_config(gemm1_beta=2.0))
    assert other._get_execution(device, 2) is not shared
    with pytest.raises(ValueError, match="capacity"):
        first._get_execution(device, 9)

    reference = weakref.ref(shared)
    del shared, first, second
    gc.collect()
    assert reference() is None


def test_cold_execution_cannot_be_created_during_capture(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_max_num_tokens", lambda: 8)
    monkeypatch.setattr(adapter, "_is_capturing", lambda device: True)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=20)
    )
    create = Mock(side_effect=AssertionError("must not allocate during capture"))
    monkeypatch.setattr(adapter, "_create_moe_layer", create)
    runner = adapter.FlashInferCuTileRunnerCore(_config())
    with pytest.raises(RuntimeError, match="warm"):
        runner._get_execution(torch.device("cpu"), 2)


def test_capture_requires_a_warmed_token_count(adapter, monkeypatch):
    capture = [True]
    monkeypatch.setattr(adapter, "_is_capturing", lambda device: capture[0])
    monkeypatch.setattr(
        adapter,
        "_create_moe_layer",
        lambda **kwargs: lambda act, weights: act.hidden_states_q.clone(),
    )
    execution = adapter._CuTileExecution(
        config=_config(), device=torch.device("cpu"), max_num_tokens=8
    )
    act = SimpleNamespace(hidden_states_q=torch.ones(2, 64))
    with pytest.raises(RuntimeError, match="warm"):
        execution(act, object())
    capture[0] = False
    torch.testing.assert_close(execution(act, object()), act.hidden_states_q)
    capture[0] = True
    torch.testing.assert_close(execution(act, object()), act.hidden_states_q)
    act.hidden_states_q = torch.ones(3, 64)
    with pytest.raises(RuntimeError, match="warm"):
        execution(act, object())


def _load_real_minimax_scheme(monkeypatch, adapter, *, use_cutile):
    """Load the real scheme and parent allocator with only GPU imports stubbed."""
    names = (
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.fused_moe_triton",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.compressed_tensors",
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
        "sglang.srt.layers.quantization.compressed_tensors.schemes",
        "sglang.srt.layers.quantization.fp8",
        "sglang.srt.layers.quantization.utils",
        "sglang.srt.layers.utils.common",
        "sglang.srt.utils",
    )
    modules = {name: ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    moe = modules["sglang.srt.layers.moe"]
    moe.MoeRunner = SimpleNamespace
    moe.MoeRunnerBackend = _Backend
    moe.MoeRunnerConfig = SimpleNamespace
    modules[
        "sglang.srt.layers.moe.fused_moe_triton"
    ].FusedMoeWeightScaleSupported = SimpleNamespace(
        GROUP=SimpleNamespace(value="group"),
        TENSOR=SimpleNamespace(value="tensor"),
    )
    compressed = modules[
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors"
    ]
    compressed.CompressedTensorsFusedMoEMethod = SimpleNamespace
    schemes = modules["sglang.srt.layers.quantization.compressed_tensors.schemes"]
    schemes.CompressedTensorsMoEScheme = type("CompressedTensorsMoEScheme", (), {})
    modules["sglang.srt.layers.quantization.fp8"].Fp8Config = type("Fp8Config", (), {})
    quant_utils = modules["sglang.srt.layers.quantization.utils"]
    for name in (
        "is_layer_skipped",
        "prepare_static_weights_for_trtllm_fp4_moe",
        "reorder_w1w3_to_w3w1",
        "replace_parameter",
        "swizzle_blockscale",
    ):
        setattr(quant_utils, name, lambda *args, **kwargs: None)
    modules["sglang.srt.layers.utils.common"].set_weight_attrs = (
        lambda *args, **kwargs: None
    )
    modules["sglang.srt.utils"].is_hip = lambda: False
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = SimpleNamespace(
        is_flashinfer_cutile=lambda: use_cutile,
        is_flashinfer_trtllm=lambda: False,
        is_flashinfer_trtllm_routed=lambda: False,
    )
    monkeypatch.setattr(
        sys.modules["sglang.srt.layers.moe.utils"],
        "get_moe_runner_backend",
        lambda: backend,
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["sglang.srt.runtime_context"],
        "get_platform",
        lambda: SimpleNamespace(
            is_blackwell=use_cutile, is_hip=False, is_gfx95=False, device_sm=120
        ),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.moe.moe_runner.flashinfer_cutile", adapter
    )
    directory = (
        Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/quantization"
    )

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    parent = load(
        "cutile_test_compressed_parent",
        directory / "compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py",
    )
    schemes.CompressedTensorsW4A4Nvfp4MoE = parent.CompressedTensorsW4A4Nvfp4MoE
    module = load("cutile_test_minimax_scheme", directory / "minimax_m3_nvfp4.py")
    return module.MiniMaxM3Nvfp4MoE()


def _allocate_scheme_layer(monkeypatch, scheme):
    # Give uninitialized float32 storage a deterministic marker. The parent
    # allocator and MiniMax override still execute their real implementations.
    empty = torch.empty

    def marked_empty(*args, **kwargs):
        value = empty(*args, **kwargs)
        if value.dtype == torch.float32:
            value.fill_(13.0)
        return value

    layer = torch.nn.Module()
    with monkeypatch.context() as allocations:
        allocations.setattr(torch, "empty", marked_empty)
        scheme.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
    return layer


def test_dynamic_checkpoint_omitted_input_scales_default_to_one(adapter, monkeypatch):
    scheme = _load_real_minimax_scheme(monkeypatch, adapter, use_cutile=True)
    layer = _allocate_scheme_layer(monkeypatch, scheme)
    torch.testing.assert_close(layer.w13_input_global_scale, torch.ones(2, 2))
    torch.testing.assert_close(layer.w2_input_global_scale, torch.ones(2))
    torch.testing.assert_close(layer.w13_weight_global_scale, torch.full((2, 2), 13.0))
    scheme.moe_runner_config = _config()
    # The actual load-completion path must accept a checkpoint with no input scales.
    scheme.process_weights_after_loading(layer)


@pytest.mark.parametrize("name", ["w13_input_global_scale", "w2_input_global_scale"])
def test_checkpoint_input_scale_override_is_preserved_and_rejected(
    adapter, monkeypatch, name
):
    scheme = _load_real_minimax_scheme(monkeypatch, adapter, use_cutile=True)
    layer = _allocate_scheme_layer(monkeypatch, scheme)
    # Simulate the loader overwriting the newly allocated default with a real value.
    getattr(layer, name).fill_(2.0)
    scheme.moe_runner_config = _config()
    with pytest.raises(ValueError, match="scale"):
        scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(
        getattr(layer, name), torch.full_like(getattr(layer, name), 2.0)
    )


def test_non_cutile_weight_allocation_keeps_existing_behavior(adapter, monkeypatch):
    scheme = _load_real_minimax_scheme(monkeypatch, adapter, use_cutile=False)
    layer = _allocate_scheme_layer(monkeypatch, scheme)
    torch.testing.assert_close(layer.w13_input_global_scale, torch.full((2, 2), 13.0))
    torch.testing.assert_close(layer.w2_input_global_scale, torch.full((2,), 13.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
