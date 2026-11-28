import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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
        SGLANG_FLASHINFER_MEGAMOE_DECODE_MAX_TOKENS_PER_RANK=types.SimpleNamespace(
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
    import enum
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

    class FinalizeMode(enum.Enum):
        AUTO = "auto"

    @dataclass
    class Config:
        knobs: str | None = None
        kernel_name: str = "test_backend"
        input_norm_const: float = 1.0
        # flashinfer's MegaMoE config carries enum fields such as finalize_mode.
        finalize_mode: FinalizeMode = FinalizeMode.AUTO

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


@pytest.mark.parametrize("keep_persistent_cache", [False, True])
def test_megamoe_profiles_reuse_cached_winner(monkeypatch, keep_persistent_cache):
    from dataclasses import replace
    from unittest.mock import Mock

    module, first, tensors, _, _ = _profile_fixture(monkeypatch, num_tokens=1)
    initial_knobs = {
        "native_tactic": 7,
        "mma_tiler_mnk": (256, 128, 256),
        "cluster_shape_mnk": (2, 1, 1),
        "epi_flag_batch": (2, 4),
    }
    retuned_knobs = {**initial_knobs, "native_tactic": 11}
    tuning_results = []
    namespace = "sglang_flashinfer_megamoe_knobs"
    events = []
    workspace_knobs = []

    class Mega:
        supports_output_view = True

        def __init__(self, **kwargs):
            self.auto = kwargs.get("backend", first.backend).megakernel.knobs == "auto"
            self.tuned = False
            self.destroyed = False

        def warmup(self, inputs, workspace=None):
            if self.auto:
                events.append("tune")
                self.knobs = (retuned_knobs if tuning_results else initial_knobs).copy()
                tuning_results.append(self.knobs)
                self.tuned = True
            else:
                assert workspace is self.workspace
                events.append("warmup")

        def get_tuned_knobs(self):
            assert self.auto and self.tuned and not self.destroyed
            events.append("get")
            return self.knobs.copy()

        def destroy(self):
            self.destroyed = True
            events.append("destroy")

        def create_workspace(self, capacity, *, knobs):
            assert capacity == 1
            workspace_knobs.append(knobs.copy())
            self.workspace = types.SimpleNamespace(knobs=knobs.copy())
            events.append("create")
            return self.workspace

        def forward(self, inputs, *, workspace, return_workspace_view=False):
            assert workspace is self.workspace
            assert return_workspace_view
            return inputs.hidden_states

    fake_moe = types.ModuleType("flashinfer.moe_ep")
    fake_moe.MoEEpMegaLayer = Mega
    fake_autotuner = types.ModuleType("flashinfer.autotuner")
    # The old namespace contains a native cache envelope, not a knob dict.
    # It must not supply workspace settings after the format change.
    records = {
        "sglang_flashinfer_megamoe": {
            f"{first._geometry_key}:1": {
                "version": 2,
                "entries": [{"knobs": {"native_tactic": 99}}],
            }
        }
    }
    tuner = types.SimpleNamespace(
        get_namespaced_records=Mock(
            side_effect=lambda namespace: records.get(namespace, {}).copy()
        ),
        publish_namespaced_records=Mock(
            side_effect=lambda namespace, values: records.__setitem__(
                namespace, json.loads(json.dumps(values))
            )
        ),
    )
    fake_autotuner.AutoTuner = types.SimpleNamespace(get=lambda: tuner)
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_autotuner)

    owners = []
    for phase in range(2):
        if phase and not keep_persistent_cache:
            records.pop(namespace)
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
        expected_knobs = (
            retuned_knobs if phase and not keep_persistent_cache else initial_knobs
        )
        assert list(records[namespace].values()) == [
            json.loads(json.dumps(expected_knobs))
        ]
        assert tuner.get_namespaced_records.call_count == phase + 1
        tuner.get_namespaced_records.assert_called_with(namespace)

    expected_tunes = 1 if keep_persistent_cache else 2
    assert len(tuning_results) == expected_tunes
    assert tuner.publish_namespaced_records.call_count == expected_tunes
    first_context_events = ["tune", "get", "destroy"] + ["create", "warmup"] * 2
    assert events == first_context_events + (
        ["create", "warmup"] * 2 if keep_persistent_cache else first_context_events
    )
    assert workspace_knobs == [initial_knobs] * 2 + [expected_knobs] * 2
    # Retuning after a cache clear cannot change already prepared workspaces.
    assert [owner.workspace.knobs for owner in owners[:2]] == [initial_knobs] * 2
    assert len({id(owner.workspace) for owner in owners}) == 4


@pytest.mark.parametrize("failure", ["warmup", "get_tuned_knobs"])
def test_megamoe_tuning_failure_destroys_temporary_layer(monkeypatch, failure):
    _, forward, tensors, _, _ = _profile_fixture(monkeypatch, num_tokens=1)
    events = []

    class Mega:
        def __init__(self, **kwargs):
            pass

        def warmup(self, inputs):
            events.append("warmup")
            if failure == "warmup":
                raise RuntimeError(failure)

        def get_tuned_knobs(self):
            events.append("get_tuned_knobs")
            raise RuntimeError(failure)

        def destroy(self):
            events.append("destroy")

    fake_moe = types.ModuleType("flashinfer.moe_ep")
    fake_moe.MoEEpMegaLayer = Mega
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe)

    with pytest.raises(RuntimeError, match=failure):
        forward._tune(tensors, capacity=1)

    assert events == (
        ["warmup", "destroy"]
        if failure == "warmup"
        else ["warmup", "get_tuned_knobs", "destroy"]
    )


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

        def forward(self, tensors, *, workspace=None):
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
        def forward(self, tensors, *, workspace=None, return_workspace_view=False):
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
        mega_forward=layer._flashinfer_megamoe_forward,
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
        assert layer._flashinfer_megamoe_layer is mega
        assert layer._flashinfer_megamoe_forward is forward
        assert layer._flashinfer_megamoe_input_norm_const == 1.0


_SM107_WEIGHTS = (
    "flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp4_mxfp4_bf16_cutedsl.weights"
)


def _install_fake_fp4_moe_ep(monkeypatch, calls, *, with_sm107=True):
    """flashinfer.moe_ep with the FP8 x FP4 (DeepGEMM) and, optionally, the SM107
    MXFP4 x MXFP4 MegaMoE pieces the FP4-expert path uses."""
    fake_moe_ep = types.ModuleType("flashinfer.moe_ep")

    def deepgemm_preprocess(weights, *, intermediate_size, hidden_size):
        calls.append(("deepgemm", weights, intermediate_size, hidden_size))
        return (weights.w13, weights.w13_scale), (weights.w2, weights.w2_scale)

    def deepgemm_config(*, intermediate_size, top_k, activation_clamp=None):
        return types.SimpleNamespace(
            kernel_name="sm100_fp8_fp4_bf16_deepgemm",
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
        )

    class Mega:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_moe_ep.MoEWeightPack = lambda **kwargs: types.SimpleNamespace(**kwargs)
    fake_moe_ep.preprocess_mega_weights = deepgemm_preprocess
    fake_moe_ep.DeepGemmMegaMoeConfig = deepgemm_config
    fake_moe_ep.BootstrapConfig = types.SimpleNamespace
    fake_moe_ep.FleetParams = types.SimpleNamespace
    fake_moe_ep.MegaConfig = types.SimpleNamespace
    fake_moe_ep.MoEEpMegaLayer = Mega
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", fake_moe_ep)
    if not with_sm107:
        return

    def sm107_config(*, intermediate_size, top_k, gate_up_clamp=None, knobs=None):
        return types.SimpleNamespace(
            kernel_name="sm107_mxfp4_mxfp4_bf16_cutedsl",
            intermediate_size=intermediate_size,
            top_k=top_k,
            gate_up_clamp=gate_up_clamp,
            knobs=knobs,
        )

    def expert_major(like, value_offset, dtype):
        # Expert-major like FlashInfer's tensors, each expert's slice one dense
        # block with the two inner dims stored transposed.
        num_experts, rows, cols = like.shape
        values = torch.arange(num_experts * rows * cols) + value_offset
        storage = values.remainder(251).to(torch.uint8).view(num_experts, cols, rows)
        return storage.transpose(1, 2).view(dtype)

    def sm107_preprocess(weights, *, intermediate_size, hidden_size):
        calls.append(("sm107", weights, intermediate_size, hidden_size))
        return (
            (
                expert_major(weights.w13, 0, torch.uint8),
                expert_major(weights.w13_sf, 1, torch.float8_e8m0fnu),
            ),
            (
                expert_major(weights.w2, 2, torch.uint8),
                expert_major(weights.w2_sf, 3, torch.float8_e8m0fnu),
            ),
        )

    fake_moe_ep.PrequantizedMoEWeights = lambda w13, w2, w13_sf, w2_sf: (
        types.SimpleNamespace(w13=w13, w2=w2, w13_sf=w13_sf, w2_sf=w2_sf)
    )
    fake_moe_ep.Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig = sm107_config
    parts = _SM107_WEIGHTS.split(".")
    for end in range(3, len(parts)):
        name = ".".join(parts[:end])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    weights_module = types.ModuleType(_SM107_WEIGHTS)
    weights_module.preprocess_mega_weights = sm107_preprocess
    monkeypatch.setitem(sys.modules, _SM107_WEIGHTS, weights_module)


def _fp4_expert_layer(*, w4a4, num_local=2, hidden=64, intermediate=32):
    param = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
    return types.SimpleNamespace(
        layer_id=3,
        hidden_size=hidden,
        num_experts=2 * num_local,
        moe_ep_size=1,
        moe_ep_rank=0,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        _mega_moe_w4a4=w4a4,
        moe_runner_config=types.SimpleNamespace(
            is_gated=True, activation="silu", swiglu_limit=10.0
        ),
        # FP4 weights packed two per byte; FP32 power-of-two scales per 32 values.
        w13_weight=param(
            torch.zeros(num_local, 2 * intermediate, hidden // 2, dtype=torch.uint8)
        ),
        w2_weight=param(
            torch.zeros(num_local, hidden, intermediate // 2, dtype=torch.uint8)
        ),
        w13_weight_scale_inv=param(
            torch.full((num_local, 2 * intermediate, hidden // 32), 0.5)
        ),
        w2_weight_scale_inv=param(
            torch.full((num_local, hidden, intermediate // 32), 4.0)
        ),
    )


def test_mxfp4_w4a4_runs_sm107_megamoe_on_expert_byte_views(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    calls = []
    _install_fake_fp4_moe_ep(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    layer = _fp4_expert_layer(w4a4=True)

    module.prepare_fp4_moe_weights_for_flashinfer_megamoe(layer)

    ((kind, weights, intermediate, hidden),) = calls
    assert (kind, intermediate, hidden) == ("sm107", 32, 64)
    # FP32 scales reach FlashInfer as E8M0 exponents: 0.5 -> 126, 4.0 -> 129.
    assert weights.w13_sf.dtype == weights.w2_sf.dtype == torch.float8_e8m0fnu
    assert bool((weights.w13_sf.view(torch.uint8) == 126).all())
    assert bool((weights.w2_sf.view(torch.uint8) == 129).all())

    # Each routed-expert parameter is an (E, bytes) view of FlashInfer's tensor,
    # so an EPLB copy into one expert's row moves the kernel's weights.
    transformed = layer._flashinfer_megamoe_mxfp4_weights
    (l1_w, l1_sf), (l2_w, l2_sf) = transformed
    for name, tensor in (
        ("w13_weight", l1_w),
        ("w13_weight_scale_inv", l1_sf),
        ("w2_weight", l2_w),
        ("w2_weight_scale_inv", l2_sf),
    ):
        view = getattr(layer, name)
        assert view.dtype == torch.uint8
        assert tuple(view.shape) == (2, tensor[0].numel())
        assert view.is_contiguous()
        for expert in range(2):
            assert view[expert].data_ptr() == tensor[expert].data_ptr()
        view.data[1].fill_(7)
        assert bool((tensor[1].view(torch.uint8) == 7).all())
        assert not bool((tensor[0].view(torch.uint8) == 7).all())

    mega = module.ensure_fp4_moe_layer_for_flashinfer_megamoe(layer)
    config = mega.backend.megakernel
    assert config.kernel_name == "sm107_mxfp4_mxfp4_bf16_cutedsl"
    assert (config.intermediate_size, config.top_k) == (32, 2)
    assert config.gate_up_clamp == 10.0
    # knobs=None: the backend's startup tuning picks them per capacity profile.
    assert config.knobs is None
    assert isinstance(layer._flashinfer_megamoe_forward, module.MegaMoeTunedForward)
    # The kernel gets FlashInfer's own tensors, not the byte views.
    assert mega.backend.transformed_weights is transformed
    assert mega.backend.preprocess_weights is False
    assert mega.fleet_params.max_tokens_per_rank == 2048
    assert module.ensure_fp4_moe_layer_for_flashinfer_megamoe(layer) is mega


def test_fp4_experts_without_w4a4_keep_deepgemm_megamoe(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    calls = []
    _install_fake_fp4_moe_ep(monkeypatch, calls)
    common = types.ModuleType("sglang.srt.layers.utils.common")

    def copy_or_rebind_param(owner, name, value):
        getattr(owner, name).data = value

    common.copy_or_rebind_param = copy_or_rebind_param
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.utils",
        types.ModuleType("sglang.srt.layers.utils"),
    )
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    layer = _fp4_expert_layer(w4a4=False)
    w13 = layer.w13_weight.data

    module.prepare_fp4_moe_weights_for_flashinfer_megamoe(layer)
    mega = module.ensure_fp4_moe_layer_for_flashinfer_megamoe(layer)

    assert [call[0] for call in calls] == ["deepgemm"]
    assert not hasattr(layer, "_flashinfer_megamoe_mxfp4_weights")
    assert mega.backend.megakernel.kernel_name == "sm100_fp8_fp4_bf16_deepgemm"
    assert mega.backend.megakernel.activation_clamp == 10.0
    (fc1_w, _), _ = mega.backend.transformed_weights
    assert fc1_w.data_ptr() == w13.data_ptr()


def test_mxfp4_w4a4_needs_flashinfer_sm107_megamoe(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    _install_fake_fp4_moe_ep(monkeypatch, [], with_sm107=False)
    with pytest.raises(ImportError, match="FlashInfer 0.7.5 or later"):
        module.prepare_fp4_moe_weights_for_flashinfer_megamoe(
            _fp4_expert_layer(w4a4=True)
        )


def test_mxfp4_scales_convert_to_e8m0(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    for dtype in (torch.float32, torch.bfloat16):
        scales = torch.tensor([1.0, 0.5, 4.0, 2.0**-20], dtype=dtype)
        assert module._as_e8m0(scales).view(torch.uint8).tolist() == [
            127,
            126,
            129,
            107,
        ]
    raw = torch.tensor([127, 3], dtype=torch.uint8)
    assert module._as_e8m0(raw).data_ptr() == raw.data_ptr()
    for bad in (torch.tensor([3.0]), torch.tensor([-1.0])):
        with pytest.raises(ValueError, match="powers of two"):
            module._as_e8m0(bad)


def test_expert_byte_views_need_dense_expert_slices(monkeypatch):
    module = _load_megamoe_module(monkeypatch)
    with pytest.raises(ValueError, match="is not dense"):
        module._expert_byte_slabs(
            torch.zeros(2, 4, 8, dtype=torch.uint8)[:, :, :4], 2, "w13_weight"
        )
    with pytest.raises(ValueError, match="unexpected"):
        module._expert_byte_slabs(torch.zeros(2, 4), 2, "w13_weight_scale_inv")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
