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
        SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK=types.SimpleNamespace(
            get=lambda: 0
        ),
        SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE=types.SimpleNamespace(
            get=lambda: "bf16"
        ),
        SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=types.SimpleNamespace(
            get=lambda: False
        ),
        SGLANG_FLASHINFER_MEGAMOE_SM120_TRANSPORT_TABLE=types.SimpleNamespace(
            get=lambda: "lsa_where_reachable"
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


def _install_sm120_fakes(monkeypatch, *, dp_counts=None):
    runtime_context = sys.modules["sglang.srt.runtime_context"]
    runtime_context.get_platform = lambda: types.SimpleNamespace(is_sm120=True)
    runtime_context.get_parallel = lambda: types.SimpleNamespace(
        moe_ep_group=types.SimpleNamespace(device_group="ep_group")
    )
    dp_attention = types.ModuleType("sglang.srt.layers.dp_attention")
    dp_attention.get_dp_global_num_tokens = lambda: dp_counts
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.dp_attention", dp_attention)

    def copy_or_rebind_param(module, name, new_value):
        param = getattr(module, name)
        if param.shape == new_value.shape and param.dtype == new_value.dtype:
            param.data.copy_(new_value)
        else:
            param.data = new_value

    layer_utils = types.ModuleType("sglang.srt.layers.utils.common")
    layer_utils.copy_or_rebind_param = copy_or_rebind_param
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.utils.common", layer_utils)


def _install_fake_moe_ep(monkeypatch, **attrs):
    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")
    for name, value in attrs.items():
        setattr(fake_moe_ep, name, value)
    fake_flashinfer = types.ModuleType("flashinfer")
    fake_flashinfer.moe_ep = fake_moe_ep
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)


def test_sm120_forward_stages_the_dp_wide_bucket_with_int64_ids(monkeypatch):
    """Every EP rank must stage the same row bucket, the max over all DP ranks."""
    module = _load_megamoe_module(monkeypatch)
    _install_sm120_fakes(monkeypatch, dp_counts=[3, 0, 7, 5])

    topk_ids = torch.tensor([[0, 5], [-1, -1], [9, 2]], dtype=torch.int32)
    tensors = types.SimpleNamespace(
        hidden_states=torch.zeros((3, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones((3, 2), dtype=torch.float32),
    )
    output_view = torch.empty((3, 4), dtype=torch.bfloat16)

    class Mega:
        def stage_inputs(self, t, *, compile_tokens_per_rank):
            self.staged = (t, compile_tokens_per_rank)

        def compute_staged(self, *, output):
            self.output = output
            return output_view

    mega = Mega()
    result = module._forward_megamoe_bucketed(mega, tensors)

    staged, bucket = mega.staged
    assert bucket == 7
    assert staged.topk_ids.dtype == torch.int64
    assert torch.equal(staged.topk_ids, topk_ids.to(torch.int64))
    assert mega.output is None
    assert result is output_view


def test_sm120_forward_requires_dp_token_counts(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    _install_sm120_fakes(monkeypatch, dp_counts=None)

    with pytest.raises(RuntimeError, match="DP-attention token counts"):
        module._forward_megamoe_bucketed(object(), object())


def test_sm120_fp4_layer_builds_cake_with_one_bucketed_workspace(monkeypatch):
    """SM120 must build the cake config and skip startup tuning, whose
    per-capacity workspaces would each compile a kernel and map NCCL windows."""
    module = _load_megamoe_module(monkeypatch)
    _install_sm120_fakes(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    built = {}

    def CakeConfig(**fields):
        return types.SimpleNamespace(knobs=None, **fields)

    def record(name):
        def build(**kwargs):
            built[name] = kwargs
            return types.SimpleNamespace(**kwargs)

        return build

    _install_fake_moe_ep(
        monkeypatch,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig=CakeConfig,
        BootstrapConfig=record("bootstrap"),
        FleetParams=record("fleet"),
        MegaConfig=record("backend"),
        MoEEpMegaLayer=record("layer"),
    )
    cake_weights = object()
    layer = types.SimpleNamespace(
        _flashinfer_megamoe_layer=None,
        _flashinfer_megamoe_cake_weights=cake_weights,
        layer_id=0,
        moe_ep_size=4,
        moe_ep_rank=1,
        num_experts=288,
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        top_k=6,
        moe_runner_config=types.SimpleNamespace(swiglu_limit=10.0),
    )

    module.ensure_fp4_moe_layer_for_flashinfer_megamoe(layer)

    config = built["backend"]["megakernel"]
    assert config == CakeConfig(
        intermediate_size=2048,
        top_k=6,
        swiglu_limit=10,
        transport_table="lsa_where_reachable",
    )
    assert built["backend"]["preprocess_weights"] is False
    assert built["backend"]["transformed_weights"] is cake_weights
    assert built["fleet"]["num_experts"] == 288
    assert built["bootstrap"]["process_group"] == "ep_group"
    assert layer._flashinfer_megamoe_forward is module._forward_megamoe_bucketed


@pytest.mark.parametrize("limit, expected", [(None, 0), (0.0, 0), (10.0, 10)])
def test_cake_swiglu_limit_is_compiled_in_as_an_integer(monkeypatch, limit, expected):
    module = _load_megamoe_module(monkeypatch)

    assert module._cake_swiglu_limit(limit) == expected


@pytest.mark.parametrize("limit", [7.5, -1.0])
def test_cake_swiglu_limit_rejects_unrepresentable_values(monkeypatch, limit):
    module = _load_megamoe_module(monkeypatch)

    with pytest.raises(ValueError, match="integral swiglu_limit"):
        module._cake_swiglu_limit(limit)


def test_ue8m0_exponent_bytes_are_exact(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    scales = torch.tensor([2.0**-3, 1.0, 2.0**4], dtype=torch.float32)

    assert module._ue8m0_exponent_bytes(scales).tolist() == [124, 127, 131]
    e8m0 = scales.to(torch.float8_e8m0fnu)
    assert module._ue8m0_exponent_bytes(e8m0).data_ptr() == e8m0.data_ptr()
    with pytest.raises(ValueError, match="power-of-two"):
        module._ue8m0_exponent_bytes(torch.tensor([0.75], dtype=torch.float32))


def test_sm120_prepare_keeps_one_resident_copy_of_the_experts(monkeypatch):
    """FC1 must be written back into w13 and both FC operands must alias the
    layer weights; the float32 checkpoint scales must not stay resident."""
    from dataclasses import dataclass

    module = _load_megamoe_module(monkeypatch)
    _install_sm120_fakes(monkeypatch)

    # Stands in for FlashInfer's frozen TransformedCakeMegaWeights dataclass.
    @dataclass(frozen=True)
    class Transformed:
        w1_fp4: torch.Tensor
        w1_sfb: torch.Tensor
        w2_fp4: torch.Tensor
        w2_sfb: torch.Tensor

    seen = {}

    def preprocess(pack, *, hidden_size, intermediate_size):
        seen["pack"] = pack
        return Transformed(
            w1_fp4=pack.w13.flip(1).contiguous(),
            w1_sfb=torch.arange(4, dtype=torch.int32),
            w2_fp4=pack.w2,
            w2_sfb=torch.arange(2, dtype=torch.int32),
        )

    _install_fake_moe_ep(
        monkeypatch,
        MoEWeightPack=lambda **kwargs: types.SimpleNamespace(**kwargs),
        preprocess_sm120_mxfp4_mxfp8_cake_mega_weights=preprocess,
    )
    layer = torch.nn.Module()
    w13 = torch.arange(2 * 4 * 2, dtype=torch.int8).reshape(2, 4, 2)
    expected_fc1 = w13.view(torch.uint8).flip(1).clone()
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(
        torch.ones((2, 2, 1), dtype=torch.int8), requires_grad=False
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.full((2, 4, 1), 2.0**-3), requires_grad=False
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.full((2, 2, 1), 2.0), requires_grad=False
    )
    layer.hidden_size = 64
    layer.intermediate_size_per_partition = 64

    module.prepare_fp4_moe_weights_for_flashinfer_megamoe(layer)

    cake = layer._flashinfer_megamoe_cake_weights
    assert torch.equal(layer.w13_weight.data.view(torch.uint8), expected_fc1)
    assert cake.w1_fp4.data_ptr() == layer.w13_weight.data_ptr()
    assert cake.w2_fp4.data_ptr() == layer.w2_weight.data_ptr()
    assert cake.w1_sfb.data_ptr() == layer.w13_weight_scale_inv.data_ptr()
    assert cake.w2_sfb.data_ptr() == layer.w2_weight_scale_inv.data_ptr()
    assert layer.w13_weight_scale_inv.dtype == torch.int32
    assert seen["pack"].w13_scale.dtype == torch.uint8
    assert seen["pack"].w13_scale.unique().tolist() == [124]
    assert seen["pack"].w2_scale.unique().tolist() == [128]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
