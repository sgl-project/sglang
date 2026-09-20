import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_megamoe_module(monkeypatch):
    """Load the adapter with only its small import-time dependencies stubbed."""

    class MoeQuantInfo:
        pass

    class MoeRunnerConfig:
        pass

    def register_fused_func(*_args, **_kwargs):
        return lambda fn: fn

    fake_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.environ": types.ModuleType("sglang.srt.environ"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.moe": types.ModuleType("sglang.srt.layers.moe"),
        "sglang.srt.layers.moe.moe_runner": types.ModuleType(
            "sglang.srt.layers.moe.moe_runner"
        ),
        "sglang.srt.layers.moe.moe_runner.base": types.ModuleType(
            "sglang.srt.layers.moe.moe_runner.base"
        ),
        "sglang.srt.layers.moe.token_dispatcher": types.ModuleType(
            "sglang.srt.layers.moe.token_dispatcher"
        ),
        "sglang.srt.runtime_context": types.ModuleType("sglang.srt.runtime_context"),
        "deep_gemm": types.ModuleType("deep_gemm"),
        "deep_gemm.utils": types.ModuleType("deep_gemm.utils"),
        "deep_gemm.utils.math": types.ModuleType("deep_gemm.utils.math"),
    }
    fake_modules["sglang.srt.environ"].envs = types.SimpleNamespace(
        SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=types.SimpleNamespace(get=lambda: False),
        SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK=types.SimpleNamespace(
            get=lambda: 0
        ),
        SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE=types.SimpleNamespace(
            get=lambda: "bf16"
        ),
        SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=types.SimpleNamespace(
            get=lambda: False
        ),
    )
    runtime_context = fake_modules["sglang.srt.runtime_context"]
    runtime_context.cutedsl_moe_max_num_tokens = lambda: 2048
    base = fake_modules["sglang.srt.layers.moe.moe_runner.base"]
    base.MoeQuantInfo = MoeQuantInfo
    base.MoeRunnerConfig = MoeRunnerConfig
    base.register_fused_func = register_fused_func
    token_dispatcher = fake_modules["sglang.srt.layers.moe.token_dispatcher"]

    class StandardCombineInput:
        def __init__(self, *, hidden_states):
            self.hidden_states = hidden_states

    token_dispatcher.StandardCombineInput = StandardCombineInput
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    _load_autotune_module(monkeypatch)
    module_path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/moe/flashinfer_megamoe.py"
    )
    module_name = "sglang_flashinfer_megamoe_adapter_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_autotune_module(monkeypatch):
    name = "sglang.srt.layers.moe.flashinfer_megamoe_autotune"
    path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/moe/flashinfer_megamoe_autotune.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _profile_fixture(monkeypatch, world_size=1, num_tokens=3):
    from dataclasses import dataclass

    module = _load_autotune_module(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    @dataclass
    class Tensors:
        hidden_states: torch.Tensor
        topk_ids: torch.Tensor
        topk_weights: torch.Tensor
        fc1_alpha: torch.Tensor
        scales: torch.Tensor | None = None

        @property
        def num_tokens(self):
            return self.hidden_states.shape[0]

    @dataclass
    class Config:
        knobs: str | None = None
        kernel_name: str = "test_backend"
        input_norm_const: float = 1.0

    @dataclass
    class Fleet:
        max_tokens_per_rank: int = 128
        num_experts: int = 4
        token_hidden_size: int = 4

    @dataclass
    class Backend:
        megakernel: Config
        transformed_weights: object
        preprocess_weights: bool = False

    tensors = Tensors(
        torch.ones(num_tokens, 4, dtype=torch.bfloat16),
        torch.arange(num_tokens * 2, dtype=torch.int32).reshape(num_tokens, 2) % 4,
        torch.full((num_tokens, 2), 0.5, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
    )
    forward = module.MegaMoeTunedForward(
        types.SimpleNamespace(world_size=world_size, rank=0, process_group=None),
        Fleet(),
        Backend(Config(), object()),
    )
    calls = []

    class Mega:
        def forward(self, inputs, **kwargs):
            calls.append((inputs, kwargs.get("workspace")))
            return inputs.hidden_states

    return module, forward, tensors, Mega(), calls


@pytest.mark.parametrize(
    "decode_tokens,extend_tokens,expected",
    [(0, 0, [1, 2, 3]), (0, 128, [1, 2, 3, 128]), (8, 0, [1, 2, 4, 8])],
)
def test_megamoe_startup_profiles_are_prepared_once(
    monkeypatch, tmp_path, decode_tokens, extend_tokens, expected
):
    module, forward, tensors, mega, calls = _profile_fixture(monkeypatch)
    prepared = []

    def prepare(context, layer, inputs, capacity):
        prepared.append(capacity)
        forward.workspaces[capacity] = object()

    monkeypatch.setattr(forward, "_prepare_profile", prepare)
    with module.megamoe_autotune_context(
        tmp_path / "cache.json", extend_tokens, decode_num_tokens=decode_tokens
    ):
        assert forward(mega, tensors) is tensors.hidden_states
        forward(mega, tensors)
    assert prepared == expected
    selected = min(capacity for capacity in expected if capacity >= 3)
    assert calls[-1][1] is forward.workspaces[selected]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    forward(mega, tensors)
    assert prepared == expected
    assert calls[-1][1] is forward.workspaces[selected]


@pytest.mark.parametrize("local_tokens", [0, 3])
def test_megamoe_profile_selection_uses_all_dp_ranks(monkeypatch, local_tokens):
    module, forward, tensors, mega, calls = _profile_fixture(
        monkeypatch, world_size=4, num_tokens=local_tokens
    )
    dp = types.ModuleType("sglang.srt.layers.dp_attention")
    dp.get_dp_global_num_tokens = lambda: [0, 1, 3, 7]
    monkeypatch.setitem(sys.modules, dp.__name__, dp)
    forward.workspaces = {1: object(), 4: object(), 8: object(), 128: object()}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    forward(mega, tensors)
    assert calls[-1][1] is forward.workspaces[8]


def test_megamoe_without_tuning_preserves_capture_forward(monkeypatch):
    module, forward, tensors, mega, calls = _profile_fixture(monkeypatch)
    forward(mega, tensors)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    forward(mega, tensors)
    assert [workspace for _, workspace in calls] == [None, None]


def test_megamoe_profiles_reuse_winner_across_startup_contexts(monkeypatch, tmp_path):
    import json
    import os
    from dataclasses import replace

    module, first, tensors, _, _ = _profile_fixture(monkeypatch, num_tokens=1)
    payload = {"version": 1, "entries": [{"knobs": {"native_tactic": 7}}]}
    events = []
    staged = []

    class Mega:
        supports_output_view = True

        def __init__(self, **kwargs):
            self.auto = kwargs.get("backend", first.backend).megakernel.knobs == "auto"

        def warmup(self, inputs, workspace=None):
            if self.auto:
                events.append("tune")
                Path(os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"]).write_text(
                    json.dumps(payload)
                )
            else:
                assert workspace is self.workspace
                events.append("warmup")

        def destroy(self):
            events.append("destroy")

        def create_workspace(self, capacity):
            assert capacity == 1
            staged.append(
                json.loads(Path(os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"]).read_text())
            )
            self.workspace = object()
            events.append("create")
            return self.workspace

        def forward(self, inputs, *, workspace, return_workspace_view=False):
            assert workspace is self.workspace
            assert return_workspace_view
            return inputs.hidden_states

    fake_moe = types.ModuleType("flashinfer.moe_ep")
    fake_moe.MoEEpMegaLayer = Mega
    fake_autotuner = types.ModuleType("flashinfer.autotuner")
    fake_autotuner._collect_metadata = lambda: {"runtime": "test"}
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "original-cache.json")

    owners = []
    for phase, scale in (("target", 1.0), ("draft", 2.0)):
        phase_backend = replace(
            first.backend,
            megakernel=replace(first.backend.megakernel, input_norm_const=scale),
        )
        forward = module.MegaMoeTunedForward(
            first.bootstrap, first.fleet_params, phase_backend
        )
        mega = Mega()
        with module.megamoe_autotune_context(
            tmp_path / phase / "cache.json", reuse_cache=False
        ):
            assert forward(mega, tensors) is tensors.hidden_states
        owners.append(mega)
        assert os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"] == "original-cache.json"

    assert events == ["tune", "destroy", "create", "warmup", "create", "warmup"]
    assert staged == [payload, payload]
    assert owners[0].workspace is not owners[1].workspace


@pytest.mark.parametrize("tokens,capacity", [(0, 8), (3, 8), (3, 1)])
def test_megamoe_profile_inputs_retain_runtime_scales(monkeypatch, tokens, capacity):
    module, _, tensors, _, _ = _profile_fixture(monkeypatch, num_tokens=tokens)
    tensors.scales = torch.ones(tokens, 2)
    profile = module._profile_inputs(tensors, capacity, num_experts=4)
    assert profile.hidden_states.shape == (capacity, 4)
    assert profile.topk_ids.shape == profile.topk_weights.shape == (capacity, 2)
    assert profile.hidden_states.dtype == torch.bfloat16
    assert profile.topk_ids.dtype == torch.int32
    assert profile.topk_weights.dtype == torch.float32
    assert profile.fc1_alpha is tensors.fc1_alpha
    assert profile.scales.shape == (capacity, 2)
    assert torch.all(profile.scales == 1)
    assert torch.all((profile.topk_ids >= 0) & (profile.topk_ids < 4))


def test_max_tokens_uses_runtime_context_accessor(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    assert module._resolve_max_tokens_per_rank() == 2048

    runtime_context = sys.modules["sglang.srt.runtime_context"]
    runtime_context.cutedsl_moe_max_num_tokens = lambda: 0
    assert module._resolve_max_tokens_per_rank() == 1024


def test_adapter_keeps_router_ids_int32(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    class FakeMoEEpTensors:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_moe_ep.MoEEpTensors = FakeMoEEpTensors
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.moe_ep = fake_moe_ep
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)

    hidden_states = torch.randn((3, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    topk_weights = torch.randn((3, 2), dtype=torch.float32)
    output = torch.randn_like(hidden_states)

    class Mega:
        def forward(self, tensors):
            self.tensors = tensors
            return output

    mega = Mega()
    dispatch_output = types.SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=types.SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    )
    quant_info = module.FlashInferMegaMoeQuantInfo(mega=mega)
    runner_config = types.SimpleNamespace(routed_scaling_factor=1.0)

    result = module.run_flashinfer_megamoe(
        dispatch_output,
        quant_info,
        runner_config,
    )

    assert mega.tensors.topk_ids.data_ptr() == topk_ids.data_ptr()
    assert mega.tensors.topk_ids.dtype == torch.int32
    assert result.hidden_states is output


@pytest.mark.parametrize("supports_output_view", [False, True])
def test_adapter_requests_workspace_output_view(monkeypatch, supports_output_view):
    module = _load_megamoe_module(monkeypatch)

    class FakeMoEEpTensors:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_moe_ep.MoEEpTensors = FakeMoEEpTensors
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.moe_ep = fake_moe_ep
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)

    hidden_states = torch.randn((2, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    topk_weights = torch.ones((2, 2), dtype=torch.float32)
    output = torch.randn_like(hidden_states)

    class Mega:
        def forward(self, tensors, *, return_workspace_view=False):
            self.tensors = tensors
            self.return_workspace_view = return_workspace_view
            return output

    mega = Mega()
    mega.supports_output_view = supports_output_view
    dispatch_output = types.SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=types.SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    )

    result = module.run_flashinfer_megamoe(
        dispatch_output,
        module.FlashInferMegaMoeQuantInfo(mega=mega),
        types.SimpleNamespace(routed_scaling_factor=1.0),
    )

    assert result.hidden_states is output
    assert mega.tensors.topk_ids.data_ptr() == topk_ids.data_ptr()
    assert mega.tensors.topk_ids.dtype == torch.int32
    assert mega.return_workspace_view is supports_output_view


def test_capture_safe_ue8m0_pack_is_scoped(monkeypatch):
    module = _load_megamoe_module(monkeypatch)

    dgm = sys.modules["deep_gemm.utils.math"]

    def original(value):
        return value

    dgm.pack_ue8m0_to_int = original
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with module._capture_safe_ue8m0_pack():
        assert dgm.pack_ue8m0_to_int is original

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with module._capture_safe_ue8m0_pack():
        assert dgm.pack_ue8m0_to_int is not original
        packed = dgm.pack_ue8m0_to_int(torch.ones(4, dtype=torch.float32))
        assert packed.dtype == torch.int32

    assert dgm.pack_ue8m0_to_int is original


@pytest.mark.parametrize("in_kernel_reduce", [False, True])
def test_w4a16_keeps_weight_scale_storage_without_activation_scales(
    monkeypatch, in_kernel_reduce
):
    module = _load_megamoe_module(monkeypatch)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16, "get", lambda: True
    )
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE,
        "get",
        lambda: in_kernel_reduce,
    )
    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")

    def make_config(
        *, intermediate_size, top_k, gate_up_clamp, enable_in_kernel_fc2_reduce
    ):
        return types.SimpleNamespace(
            intermediate_size=intermediate_size,
            top_k=top_k,
            gate_up_clamp=gate_up_clamp,
            enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
            kernel_name="sm100_bf16_nvfp4_bf16_cutedsl",
            knobs=None,
        )

    class Mega:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def forward(self, tensors):
            self.tensors = tensors
            return tensors.hidden_states

    fake_moe_ep.Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig = make_config
    fake_moe_ep.BootstrapConfig = types.SimpleNamespace
    fake_moe_ep.FleetParams = types.SimpleNamespace
    fake_moe_ep.MegaConfig = types.SimpleNamespace
    fake_moe_ep.MoEEpTensors = types.SimpleNamespace
    fake_moe_ep.MoEEpMegaLayer = Mega
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # No activation-scale fields: W4A16 consumes only weight decode scales.
    layer = types.SimpleNamespace(
        layer_id=0,
        hidden_size=32,
        num_experts=2,
        moe_ep_size=1,
        moe_ep_rank=0,
        intermediate_size_per_partition=64,
        top_k=2,
        moe_runner_config=types.SimpleNamespace(swiglu_limit=None),
        w13_weight=torch.zeros(2, 128, 16, dtype=torch.uint8).transpose(1, 2),
        w2_weight=torch.zeros(2, 32, 32, dtype=torch.uint8).transpose(1, 2),
        w13_weight_scale=torch.zeros(2, 512, dtype=torch.float8_e4m3fn),
        w2_weight_scale=torch.zeros(2, 512, dtype=torch.float8_e4m3fn),
        g1_alphas=torch.tensor([0.25, 0.5], dtype=torch.float32),
        g2_alphas=torch.tensor([0.75, 1.0], dtype=torch.float32),
    )

    mega = module.ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer)
    config = mega.backend.megakernel
    assert config.enable_in_kernel_fc2_reduce is in_kernel_reduce
    fc1, fc2 = mega.backend.transformed_weights
    assert len(fc1) == len(fc2) == 2
    for actual, expected in zip(
        (*fc1, *fc2),
        (
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
        ),
        strict=True,
    ):
        assert actual.data_ptr() == expected.data_ptr()

    # Scales are staged on each forward, so a shared workspace sees this
    # layer's current decode scales after a weight update.
    dispatch_output = types.SimpleNamespace(
        hidden_states=torch.zeros(1, 32, dtype=torch.bfloat16),
        topk_output=types.SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        ),
    )
    quant_info = module.FlashInferMegaMoeQuantInfo(
        mega=mega,
        mega_forward=layer._flashinfer_megamoe_forward,
        fc1_alpha=layer.g1_alphas,
        fc2_alpha=layer.g2_alphas,
    )
    for scale in (2.0, 3.0):
        layer.g1_alphas.fill_(scale)
        layer.g2_alphas.fill_(scale + 1)
        module.run_flashinfer_megamoe(
            dispatch_output,
            quant_info,
            types.SimpleNamespace(routed_scaling_factor=1.0),
        )
        assert mega.tensors.fc1_alpha.data_ptr() == layer.g1_alphas.data_ptr()
        assert mega.tensors.fc2_alpha.data_ptr() == layer.g2_alphas.data_ptr()
        assert mega.tensors.fc1_norm_const is None
        torch.testing.assert_close(mega.tensors.fc1_alpha, torch.full((2,), scale))
        torch.testing.assert_close(mega.tensors.fc2_alpha, torch.full((2,), scale + 1))


@pytest.mark.parametrize("is_gated,activation", [(False, "silu"), (True, "gelu")])
def test_w4a16_rejects_non_swiglu(monkeypatch, is_gated, activation):
    module = _load_megamoe_module(monkeypatch)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16, "get", lambda: True
    )
    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_moe_ep.MoEWeightPack = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)
    layer = types.SimpleNamespace(
        moe_runner_config=types.SimpleNamespace(
            is_gated=is_gated, activation=activation
        )
    )
    with pytest.raises(ValueError, match="only supports SwiGLU"):
        module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)


def test_w4a16_reload_preserves_layer_and_prepared_weight_storage(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16, "get", lambda: True
    )
    # Import the real parameter-binding helper without the full runtime.
    spec = importlib.util.spec_from_file_location(
        "sglang.srt.layers.utils.common",
        Path(module.__file__).parents[1] / "utils/common.py",
    )
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, common)
    spec.loader.exec_module(common)
    hidden, intermediate = 32, 64
    layer = torch.nn.Module()
    layer.hidden_size = hidden
    layer.intermediate_size_per_partition = intermediate
    layer.num_local_experts = layer.num_experts = 2
    layer.moe_ep_size = 1
    layer.quant_config = types.SimpleNamespace(use_per_token_activation=True)
    layer.moe_runner_config = types.SimpleNamespace(
        is_gated=True,
        activation="silu",
        swiglu_limit=None,
        apply_router_weight_on_input=False,
    )
    layer.g1_alphas = layer.g1_alphas_up = torch.ones(2)
    layer.g2_alphas = layer.w13_input_scale_quant = torch.ones(2)
    for prefix, rows, columns in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        for suffix, divisor, dtype in (
            ("weight", 2, torch.uint8),
            ("weight_scale", 16, torch.float8_e4m3fn),
        ):
            layer.register_parameter(
                f"{prefix}_{suffix}",
                torch.nn.Parameter(
                    torch.zeros(2, rows, columns // divisor, dtype=dtype),
                    requires_grad=False,
                ),
            )
    canonical = {name: (p.shape, p.dtype) for name, p in layer.named_parameters()}

    def preprocess(weights, **kwargs):
        result = []
        for weight, scale in (
            (weights.w13, weights.w13_scale),
            (weights.w2, weights.w2_scale),
        ):
            # Model the prepared ABI: transposed FP4 bytes and padded flat scales.
            padded = torch.zeros(
                2, ((scale[0].numel() + 511) // 512) * 512, dtype=scale.dtype
            )
            padded[:, : scale[0].numel()].copy_(scale.flatten(1))
            result.append(
                (
                    weight.clone().view(torch.float4_e2m1fn_x2).transpose(1, 2),
                    padded,
                )
            )
        return tuple(result)

    def make_weight_pack(*, w13, w2, w13_scale, w2_scale):
        return types.SimpleNamespace(
            w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale
        )

    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    fake_moe_ep.MoEWeightPack = make_weight_pack
    fake_moe_ep.preprocess_bf16_nvfp4_cutedsl_mega_weights = preprocess
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)
    module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
    prepared = {
        name: (p.shape, p.stride(), p.dtype, p.data_ptr())
        for name, p in layer.named_parameters()
    }
    mega = types.SimpleNamespace(
        _workspace=object(),
        _transformed_weights={name: p.data for name, p in layer.named_parameters()},
    )
    forward = object()
    layer._flashinfer_megamoe_layer = mega
    layer._flashinfer_megamoe_forward = forward

    def load(param, value):
        for name, p in layer.named_parameters():
            assert (p.shape, p.dtype) == canonical[name]
        param.data.fill_(value)

    loader = module.make_nvfp4_megamoe_weight_loader(layer, load)
    for value in (1, 2):
        for param in layer.parameters():
            loader(param, value)
        module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
        assert layer._flashinfer_megamoe_layer is mega
        assert layer._flashinfer_megamoe_forward is forward
        for name, p in layer.named_parameters():
            assert (p.shape, p.stride(), p.dtype, p.data_ptr()) == prepared[name]
            # Views held by the live backend and captured graphs see new bytes.
            p = mega._transformed_weights[name]
            if p.dtype == torch.float4_e2m1fn_x2:
                assert (p.view(torch.uint8) == value).all()
            else:
                assert (p.float()[:, : canonical[name][0].numel() // 2] == value).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
