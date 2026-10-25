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
    phase_spec = importlib.util.spec_from_file_location(
        "sglang.srt.layers.moe.phase", module_path.with_name("phase.py")
    )
    phase_module = importlib.util.module_from_spec(phase_spec)
    monkeypatch.setitem(sys.modules, phase_spec.name, phase_module)
    phase_spec.loader.exec_module(phase_module)
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
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    tuner = types.SimpleNamespace(
        get_namespaced_records=lambda _: {},
        publish_namespaced_records=lambda *_: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        types.SimpleNamespace(AutoTuner=types.SimpleNamespace(get=lambda: tuner)),
    )

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
        max_tokens_per_rank: int = 4
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
        supports_output_view = False

        def forward(self, inputs, **kwargs):
            calls.append((inputs, kwargs.get("workspace")))
            return inputs.hidden_states

    return module, forward, tensors, Mega(), calls


@pytest.mark.parametrize(
    "decode_tokens,expected", [(3, [1, 2, 3, 128]), (8, [1, 2, 4, 8, 128])]
)
def test_megamoe_startup_profiles_are_prepared_once(
    monkeypatch, decode_tokens, expected
):
    module, forward, tensors, mega, calls = _profile_fixture(monkeypatch, num_tokens=96)
    prepared = []

    def prepare(layer, inputs, capacity):
        prepared.append(capacity)
        forward.workspaces[capacity] = object()

    monkeypatch.setattr(forward, "_prepare_profile", prepare)
    with module.megamoe_autotune_context(decode_tokens, prefill_num_tokens=128):
        assert forward(mega, tensors) is tensors.hidden_states
    # Bounds may exceed the bootstrap fleet. A large initial dummy must not
    # create intermediate prefill profiles.
    assert prepared == expected
    assert calls[-1][1] is forward.workspaces[128]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for count, selected in ((2, 2), (decode_tokens, decode_tokens), (32, 128)):
        inputs = module._profile_inputs(tensors, count, num_experts=4)
        forward(mega, inputs)
        assert prepared == expected
        assert calls[-1][1] is forward.workspaces[selected]
    with pytest.raises(ValueError, match="exceeding its prepared"):
        forward(mega, module._profile_inputs(tensors, 129, num_experts=4))


@pytest.mark.parametrize("local_tokens", [0, 3])
def test_megamoe_profile_selection_uses_all_dp_ranks(monkeypatch, local_tokens):
    module, forward, tensors, mega, calls = _profile_fixture(
        monkeypatch, world_size=4, num_tokens=local_tokens
    )
    dp = types.ModuleType("sglang.srt.layers.dp_attention")
    dp.get_dp_global_num_tokens = lambda: [0, 1, 3, 7]
    monkeypatch.setitem(sys.modules, dp.__name__, dp)
    forward.workspaces = {1: object(), 4: object(), 8: object(), 128: object()}
    forward._profile_capacities = tuple(forward.workspaces)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    forward(mega, tensors)
    assert calls[-1][1] is forward.workspaces[8]


def test_megamoe_without_tuning_preserves_capture_forward(monkeypatch):
    module, forward, tensors, mega, calls = _profile_fixture(monkeypatch)
    forward(mega, tensors)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    forward(mega, tensors)
    assert [workspace for _, workspace in calls] == [None, None]


@pytest.mark.parametrize("new_process", [False, True])
def test_megamoe_profiles_reuse_cached_winner(monkeypatch, new_process):
    import json
    import os
    from dataclasses import replace
    from unittest.mock import Mock

    module, first, tensors, _, _ = _profile_fixture(monkeypatch, num_tokens=1)
    payload = {"version": 1, "entries": [{"knobs": {"native_tactic": 7}}]}
    events = []
    staged = []
    native_paths = []
    require_all_ranks = module.MegaMoeTunedForward._require_all_ranks
    readiness = Mock()

    def require(self, success, message):
        readiness(success)
        require_all_ranks(self, success, message)

    monkeypatch.setattr(module.MegaMoeTunedForward, "_require_all_ranks", require)

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
            path = Path(os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"])
            native_paths.append(path)
            staged.append(json.loads(path.read_text()))
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
    records = {}
    tuner = types.SimpleNamespace(
        get_namespaced_records=Mock(
            side_effect=lambda namespace: records.get(namespace, {}).copy()
        ),
        publish_namespaced_records=Mock(
            side_effect=lambda namespace, values: records.__setitem__(
                namespace, values.copy()
            )
        ),
    )
    fake_autotuner.AutoTuner = types.SimpleNamespace(get=lambda: tuner)
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "original-cache.json")

    owners = []
    for phase in range(2):
        if owners:
            if new_process:
                module._session_winners.clear()  # Only the loaded cache survives.
            else:
                records.clear()  # A new cache context must preserve pooled choices.
        with module.megamoe_autotune_context(
            decode_num_tokens=1,
            prefill_num_tokens=1,
        ):
            for scale in (1.0, 2.0):
                backend = replace(
                    first.backend,
                    megakernel=replace(
                        first.backend.megakernel, input_norm_const=scale
                    ),
                )
                forward = module.MegaMoeTunedForward(
                    first.bootstrap, first.fleet_params, backend
                )
                mega = Mega()
                assert forward(mega, tensors) is tensors.hidden_states
                owners.append(mega)
            assert readiness.call_count == phase + 1
            assert native_paths[-2] == native_paths[-1]
            assert native_paths[-1].is_file()
        assert not native_paths[-1].exists()
        assert os.environ["FLASHINFER_MOE_EP_KNOB_CACHE"] == "original-cache.json"
        assert list(records["sglang_flashinfer_megamoe"].values()) == [payload]
        assert tuner.get_namespaced_records.call_count == phase + 1

    assert tuner.publish_namespaced_records.call_count == (1 if new_process else 2)
    assert events == ["tune", "destroy"] + ["create", "warmup"] * 4
    assert staged == [payload] * 4
    assert len({id(owner.workspace) for owner in owners}) == 4


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
        supports_output_view = False

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
@pytest.mark.parametrize("use_w4a16", [False, True])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_nvfp4_keeps_weight_scale_storage_and_activation_contract(
    monkeypatch, in_kernel_reduce, use_w4a16, activation
):
    module = _load_megamoe_module(monkeypatch)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16, "get", lambda: use_w4a16
    )
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE,
        "get",
        lambda: in_kernel_reduce,
    )
    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")

    def make_config(
        *,
        intermediate_size,
        top_k,
        activation,
        gate_up_clamp,
        enable_in_kernel_fc2_reduce,
        situ_beta=None,
        situ_linear_beta=None,
        apply_topk_in_fc1=False,
        combine_dtype="bf16",
        input_norm_const=None,
    ):
        return types.SimpleNamespace(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation=activation,
            gate_up_clamp=gate_up_clamp,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            apply_topk_in_fc1=apply_topk_in_fc1,
            combine_dtype=combine_dtype,
            input_norm_const=input_norm_const,
            enable_in_kernel_fc2_reduce=enable_in_kernel_fc2_reduce,
            kernel_name=(
                "sm100_bf16_nvfp4_bf16_cutedsl"
                if use_w4a16
                else "sm100_nvfp4_nvfp4_bf16_cutedsl"
            ),
            knobs=None,
        )

    class Mega:
        supports_output_view = False

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def forward(self, tensors, workspace=None):
            self.tensors = tensors
            return tensors.hidden_states

    fake_moe_ep.Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig = make_config
    fake_moe_ep.Nvfp4CutedslMegaMoeConfig = make_config
    fake_moe_ep.BootstrapConfig = types.SimpleNamespace
    fake_moe_ep.FleetParams = types.SimpleNamespace
    fake_moe_ep.MegaConfig = types.SimpleNamespace
    fake_moe_ep.MoEEpTensors = types.SimpleNamespace
    fake_moe_ep.MoEEpMegaLayer = Mega
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    hidden, intermediate = (32, 64) if use_w4a16 else (128, 128)
    # W4A16 consumes only weight decode scales; activation scales exist only
    # in the W4A4 case below.
    layer = types.SimpleNamespace(
        layer_id=0,
        hidden_size=hidden,
        num_experts=2,
        moe_ep_size=1,
        moe_ep_rank=0,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        moe_runner_config=types.SimpleNamespace(
            is_gated=True,
            activation=activation,
            swiglu_limit=8.0,
            gemm1_alpha=4.0 if activation == "situ" else None,
            gemm1_clamp_limit=25.0 if activation == "situ" else None,
        ),
        w13_weight=torch.zeros(
            2, 2 * intermediate, hidden // 2, dtype=torch.uint8
        ).transpose(1, 2),
        w2_weight=torch.zeros(
            2, hidden, intermediate // 2, dtype=torch.uint8
        ).transpose(1, 2),
        w13_weight_scale=torch.zeros(2, 512, dtype=torch.float8_e4m3fn),
        w2_weight_scale=torch.zeros(2, 512, dtype=torch.float8_e4m3fn),
        g1_alphas=torch.tensor([0.25, 0.5], dtype=torch.float32),
        g2_alphas=torch.tensor([0.75, 1.0], dtype=torch.float32),
    )
    if not use_w4a16:
        layer.w13_input_scale_quant = torch.tensor(2.0)
        layer.w2_input_scale_quant = torch.tensor([1.25, 1.5])

    mega = module.ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer)
    config = mega.backend.megakernel
    assert config.enable_in_kernel_fc2_reduce is in_kernel_reduce
    assert config.activation == ("situ" if activation == "situ" else "swiglu")
    assert config.gate_up_clamp == (None if activation == "situ" else 8.0)
    assert config.situ_beta == (4.0 if activation == "situ" else None)
    assert config.situ_linear_beta == (25.0 if activation == "situ" else None)
    assert config.apply_topk_in_fc1 is (not use_w4a16)
    assert config.input_norm_const == (None if use_w4a16 else 2.0)
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
        hidden_states=torch.zeros(1, hidden, dtype=torch.bfloat16),
        topk_output=types.SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        ),
    )
    quant_info = module.FlashInferMegaMoeQuantInfo(
        mega=mega,
        mega_forward=layer._flashinfer_megamoe_nvfp4_states[use_w4a16][1],
        fc1_alpha=layer.g1_alphas,
        fc2_alpha=layer.g2_alphas,
        fc1_norm_const=None if use_w4a16 else layer.w2_input_scale_quant,
    )
    for scale in (2.0, 3.0):
        layer.g1_alphas.fill_(scale)
        layer.g2_alphas.fill_(scale + 1)
        if not use_w4a16:
            layer.w2_input_scale_quant.fill_(scale + 2)
        module.run_flashinfer_megamoe(
            dispatch_output,
            quant_info,
            types.SimpleNamespace(routed_scaling_factor=1.0),
        )
        assert mega.tensors.fc1_alpha.data_ptr() == layer.g1_alphas.data_ptr()
        assert mega.tensors.fc2_alpha.data_ptr() == layer.g2_alphas.data_ptr()
        if use_w4a16:
            assert mega.tensors.fc1_norm_const is None
        else:
            assert mega.tensors.fc1_norm_const is layer.w2_input_scale_quant
            torch.testing.assert_close(
                mega.tensors.fc1_norm_const, torch.full((2,), scale + 2)
            )
        assert mega.tensors.topk_weights is dispatch_output.topk_output.topk_weights
        torch.testing.assert_close(mega.tensors.fc1_alpha, torch.full((2,), scale))
        torch.testing.assert_close(mega.tensors.fc2_alpha, torch.full((2,), scale + 1))


@pytest.mark.parametrize(
    "activation,situ_linear_beta", [("silu", None), ("situ", 25.0), ("situ", None)]
)
@pytest.mark.parametrize("use_w4a16", [False, True])
def test_nvfp4_reload_preserves_layer_and_prepared_weight_storage(
    monkeypatch, activation, situ_linear_beta, use_w4a16
):
    module = _load_megamoe_module(monkeypatch)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16, "get", lambda: use_w4a16
    )
    # Import the real parameter-binding helper without the full runtime.
    spec = importlib.util.spec_from_file_location(
        "sglang.srt.layers.utils.common",
        Path(module.__file__).parents[1] / "utils/common.py",
    )
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, common)
    spec.loader.exec_module(common)
    hidden, intermediate = (32, 64) if use_w4a16 else (128, 128)
    layer = torch.nn.Module()
    layer.hidden_size = hidden
    layer.intermediate_size_per_partition = intermediate
    layer.num_local_experts = layer.num_experts = 2
    layer.moe_ep_size = 1
    layer.quant_config = types.SimpleNamespace(use_per_token_activation=use_w4a16)
    layer.moe_runner_config = types.SimpleNamespace(
        is_gated=True,
        activation=activation,
        swiglu_limit=None,
        gemm1_alpha=4.0 if activation == "situ" else None,
        gemm1_clamp_limit=situ_linear_beta,
        apply_router_weight_on_input=False,
    )
    layer.g1_alphas = layer.g1_alphas_up = torch.ones(2)
    layer.g2_alphas = torch.ones(2)
    layer.w13_input_scale_quant = torch.tensor(1.0)
    layer.w2_input_scale_quant = torch.ones(2)
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

    def preprocess(
        weights,
        *,
        intermediate_size,
        hidden_size,
        gate_up_clamp=None,
        activation_clamp=None,
    ):
        assert (intermediate_size, hidden_size) == (intermediate, hidden)
        result = []
        for weight, scale in (
            (weights.w13, weights.w13_scale),
            (weights.w2, weights.w2_scale),
        ):
            # Both modes expose FP4 weights with their typed view. W4A4
            # exposes prepared FP8 scales as bytes; W4A16 keeps their FP8 view.
            padded = torch.zeros(
                2, ((scale[0].numel() + 511) // 512) * 512, dtype=scale.dtype
            )
            padded[:, : scale[0].numel()].copy_(scale.flatten(1))
            scale_dtype = torch.float8_e4m3fn if use_w4a16 else torch.uint8
            # W4A4 FC2 preprocessing returns a view of the canonical bytes.
            prepared_weight = (
                weight if not use_w4a16 and weight is weights.w2 else weight.clone()
            )
            result.append(
                (
                    prepared_weight.view(torch.float4_e2m1fn_x2).transpose(1, 2),
                    padded.view(scale_dtype),
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
    fake_moe_ep.preprocess_nvfp4_cutedsl_mega_weights = preprocess
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
    layer._flashinfer_megamoe_nvfp4_states[use_w4a16] = (mega, forward)

    def load(param, value):
        for name, p in layer.named_parameters():
            assert (p.shape, p.dtype) == canonical[name]
        param.data.fill_(value)

    loader = module.make_nvfp4_megamoe_weight_loader(layer, load)
    for value in (1, 2):
        for param in layer.parameters():
            loader(param, value)
        module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
        assert layer._flashinfer_megamoe_nvfp4_states[use_w4a16] == (mega, forward)
        for name, p in layer.named_parameters():
            assert (p.shape, p.stride(), p.dtype, p.data_ptr()) == prepared[name]
            # Views held by the live backend and captured graphs see new bytes.
            p = mega._transformed_weights[name]
            if name.endswith("_weight"):
                assert (p.view(torch.uint8) == value).all()
            else:
                scale = p.view(torch.float8_e4m3fn).float()
                assert (scale[:, : canonical[name][0].numel() // 2] == value).all()

    if not use_w4a16:
        # FlashInfer 0.7.1 embeds this scalar in the live kernel configuration;
        # accepting a changed scalar would keep a stale captured quantizer.
        layer.w13_input_scale_quant.fill_(2.0)
        with pytest.raises(ValueError, match="input activation scale"):
            module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
        assert layer._flashinfer_megamoe_nvfp4_states[use_w4a16] == (mega, forward)
        assert layer._flashinfer_megamoe_input_norm_const == 1.0


def test_precapture_warmup_includes_online_methods_and_both_kernels(
    monkeypatch,
):
    module = _load_megamoe_module(monkeypatch)

    class FusedMoE(torch.nn.Module):
        pass

    class NvFp4Method:
        pass

    class OnlineMethod(NvFp4Method):
        pass

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.fused_moe_triton.layer",
        types.SimpleNamespace(FusedMoE=FusedMoE),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.modelopt_quant",
        types.SimpleNamespace(ModelOptNvFp4FusedMoEMethod=NvFp4Method),
    )
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16, "get", lambda: "decode"
    )
    online = FusedMoE()
    online.quant_method = OnlineMethod()
    module._init_flashinfer_megamoe_layer_state(online)
    paired = FusedMoE()
    paired.quant_method = NvFp4Method()
    module._init_flashinfer_megamoe_layer_state(paired)
    paired._flashinfer_megamoe_nvfp4_states[False] = (object(), object())
    ready = FusedMoE()
    ready.quant_method = NvFp4Method()
    module._init_flashinfer_megamoe_layer_state(ready)
    ready._flashinfer_megamoe_nvfp4_states = {
        False: (object(), object()),
        True: (object(), object()),
    }
    calls = []
    monkeypatch.setattr(
        module,
        "ensure_nvfp4_moe_layer_for_flashinfer_megamoe",
        lambda layer, **kwargs: calls.append((layer, kwargs["use_w4a16"])),
    )
    module.warmup_all_flashinfer_megamoe_layers(
        torch.nn.ModuleList([online, paired, ready])
    )
    assert calls == [(online, False), (paired, False)]


@pytest.mark.parametrize("w4a16_phase", ["prefill", "decode"])
@pytest.mark.parametrize("per_token_setting", [False, True, "prefill", "decode"])
def test_phase_specific_nvfp4_shares_weights_and_keeps_reload_addresses(
    monkeypatch, w4a16_phase, per_token_setting
):
    module = _load_megamoe_module(monkeypatch)
    monkeypatch.setattr(
        module.envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16,
        "get",
        lambda: w4a16_phase,
    )
    spec = importlib.util.spec_from_file_location(
        "sglang.srt.layers.utils.common",
        Path(module.__file__).parents[1] / "utils/common.py",
    )
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, common)
    spec.loader.exec_module(common)

    layer = torch.nn.Module()
    layer.hidden_size = layer.intermediate_size_per_partition = 128
    layer.num_experts = layer.num_local_experts = 2
    layer.moe_ep_size = 1
    layer.moe_ep_rank = layer.layer_id = 0
    layer.top_k = 2
    layer.quant_config = types.SimpleNamespace(
        use_per_token_activation=bool(per_token_setting),
        per_token_activation_phase=(
            per_token_setting if isinstance(per_token_setting, str) else None
        ),
    )
    layer.moe_runner_config = types.SimpleNamespace(
        is_gated=True,
        activation="silu",
        swiglu_limit=None,
        apply_router_weight_on_input=False,
    )
    layer.g1_alphas = layer.g1_alphas_up = torch.tensor([0.5, 1.0])
    layer.g2_alphas = torch.tensor([1.5, 2.0])
    layer.w13_input_scale_quant = torch.tensor(2.0)
    layer.w2_input_scale_quant = torch.tensor([4.0, 8.0])
    for name, shape, dtype in (
        ("w13_weight", (2, 256, 64), torch.uint8),
        ("w2_weight", (2, 128, 64), torch.uint8),
        ("w13_weight_scale", (2, 256, 8), torch.float8_e4m3fn),
        ("w2_weight_scale", (2, 128, 8), torch.float8_e4m3fn),
    ):
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False),
        )

    prepared_modes = []

    def preprocess(weights, *, w4a16, **kwargs):
        prepared_modes.append(w4a16)
        return tuple(
            (weight.clone().transpose(1, 2), scale.clone())
            for weight, scale in (
                (weights.w13, weights.w13_scale),
                (weights.w2, weights.w2_scale),
            )
        )

    class Mega:
        supports_output_view = False

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.moe_ep",
        types.SimpleNamespace(
            MoEWeightPack=types.SimpleNamespace,
            BootstrapConfig=types.SimpleNamespace,
            FleetParams=types.SimpleNamespace,
            MegaConfig=types.SimpleNamespace,
            MoEEpMegaLayer=Mega,
            Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig=types.SimpleNamespace,
            Nvfp4CutedslMegaMoeConfig=types.SimpleNamespace,
            preprocess_bf16_nvfp4_cutedsl_mega_weights=lambda weights, **kwargs: (
                preprocess(weights, w4a16=True, **kwargs)
            ),
            preprocess_nvfp4_cutedsl_mega_weights=lambda weights, **kwargs: preprocess(
                weights, w4a16=False, **kwargs
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    if per_token_setting and per_token_setting != w4a16_phase:
        with pytest.raises(ValueError, match="does not support per-token activation"):
            module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
        return

    module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
    assert prepared_modes == [False]
    assert not hasattr(layer, "_flashinfer_megamoe_w4a16")
    assert list(layer.modules()) == [layer]
    torch.testing.assert_close(layer.g1_alphas, torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(layer._w4a16_g1_alphas, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(layer._w4a16_g2_alphas, torch.tensor([6.0, 16.0]))
    weight_names = tuple(layer._flashinfer_megamoe_weight_views)
    assert set(dict(layer.named_parameters())) == set(weight_names) | {
        "_w4a16_g1_alphas",
        "_w4a16_g2_alphas",
    }
    pointers = {name: getattr(layer, name).data_ptr() for name in weight_names}
    alpha_pointers = (
        layer._w4a16_g1_alphas.data_ptr(),
        layer._w4a16_g2_alphas.data_ptr(),
    )

    # Explicit phase selection must initialize both native configurations, so
    # neither phase allocates during capture. The objects share every weight
    # address, and selecting one never replaces layer metadata or the other.
    selected = module.ensure_nvfp4_moe_layer_for_flashinfer_megamoe(
        layer, use_w4a16=True
    )
    states = layer._flashinfer_megamoe_nvfp4_states.copy()
    assert selected is states[True][0]
    assert states[False][0] is not states[True][0]
    for mode in (False, True, False):
        assert (
            module.ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer, use_w4a16=mode)
            is states[mode][0]
        )
        assert layer._flashinfer_megamoe_nvfp4_states == states
        assert layer._flashinfer_megamoe_layer is None
        assert layer._flashinfer_megamoe_forward is None
    for mega, _ in states.values():
        (w13, s13), (w2, s2) = mega.backend.transformed_weights
        for name, value in zip(weight_names, (w13, s13, w2, s2)):
            assert value.data_ptr() == pointers[name]

    loader = module.make_nvfp4_megamoe_weight_loader(
        layer, lambda param: param.data.fill_(2)
    )
    for name in weight_names:
        loader(getattr(layer, name))
    layer.g1_alphas.mul_(2)
    layer.g2_alphas.mul_(2)
    module.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
    assert prepared_modes == [False, False]
    assert layer._flashinfer_megamoe_nvfp4_states == states
    for name in weight_names:
        assert getattr(layer, name).data_ptr() == pointers[name]
        assert (getattr(layer, name).float() == 2).all()
    assert alpha_pointers == (
        layer._w4a16_g1_alphas.data_ptr(),
        layer._w4a16_g2_alphas.data_ptr(),
    )
    torch.testing.assert_close(layer._w4a16_g1_alphas, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(layer._w4a16_g2_alphas, torch.tensor([12.0, 32.0]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
