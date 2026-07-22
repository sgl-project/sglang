from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer_module
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner import runner as runner_module
from sglang.srt.layers.moe.moe_runner.base import (
    DispatchMoeRunnerCore,
    MoeQuantInfo,
    MoeRunnerConfig,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    RegisteredMoeRunnerBackend,
    register_moe_runner_backend_name,
    resolve_moe_runner_backend,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-c-test-cpu")


class _TestDispatchRunnerCore(DispatchMoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig, backend):
        super().__init__(config)
        self._backend = backend
        self.calls = []

    @property
    def runner_backend(self):
        return self._backend

    def run_from_dispatch(
        self,
        dispatch_output,
        quant_info,
        runner_config,
        hooks=None,
    ):
        self.calls.append((dispatch_output, quant_info, runner_config, hooks))
        return StandardCombineInput(dispatch_output.hidden_states + 1)


@pytest.fixture
def isolated_runner_registries(monkeypatch):
    from sglang.srt.layers.moe import utils as moe_utils

    monkeypatch.setattr(moe_utils, "_REGISTERED_MOE_RUNNER_BACKEND_NAMES", set())
    monkeypatch.setattr(runner_module, "_CUSTOM_RUNNER_CORE_FACTORIES", {})
    monkeypatch.setattr(
        runner_module, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE
    )


def test_registered_runner_core_uses_standard_dispatch(
    isolated_runner_registries,
) -> None:
    backend_name = "test_dispatch_extension"
    runner_module.register_moe_runner_core(
        backend_name,
        lambda config: _TestDispatchRunnerCore(
            config, resolve_moe_runner_backend(backend_name)
        ),
    )

    backend = resolve_moe_runner_backend(backend_name)
    assert isinstance(backend, RegisteredMoeRunnerBackend)
    assert backend.value == backend_name

    runner = runner_module.MoeRunner(backend, MoeRunnerConfig())
    dispatch_output = StandardDispatchOutput(
        hidden_states=torch.zeros(1, 2),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int64),
            router_logits=torch.zeros(1, 1),
        ),
    )
    quant_info = MoeQuantInfo()

    result = runner.run(dispatch_output, quant_info)

    assert torch.equal(result.hidden_states, torch.ones(1, 2))
    assert runner.runner_core.calls == [
        (dispatch_output, quant_info, runner.config, None)
    ]
    with pytest.raises(ValueError, match="already registered"):
        runner_module.register_moe_runner_core(backend_name, _TestDispatchRunnerCore)


def test_runner_core_registration_can_override_builtin_backend(
    isolated_runner_registries,
) -> None:
    backend = MoeRunnerBackend.FLASHINFER_CUTLASS
    runner_module.register_moe_runner_core(
        backend.value,
        lambda config: _TestDispatchRunnerCore(config, backend),
    )

    runner = runner_module.MoeRunner(backend, MoeRunnerConfig())

    assert isinstance(runner.runner_core, _TestDispatchRunnerCore)
    assert runner.runner_core.runner_backend is backend


def test_runner_backend_names_must_be_builtin_or_registered(
    isolated_runner_registries,
) -> None:
    assert resolve_moe_runner_backend("triton") is MoeRunnerBackend.TRITON
    assert MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM.is_flashinfer_trtllm()
    assert MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN.is_marlin()
    with pytest.raises(ValueError, match="neither built in nor registered"):
        resolve_moe_runner_backend("unknown_backend")
    with pytest.raises(ValueError, match="must not be empty"):
        register_moe_runner_backend_name("")
    with pytest.raises(ValueError, match="already built in"):
        register_moe_runner_backend_name("triton")


def test_fused_moe_uses_explicit_quant_method_for_full_lifecycle(monkeypatch) -> None:
    calls = []
    method = UnquantizedFusedMoEMethod()
    monkeypatch.setattr(
        method, "create_weights", lambda **kwargs: calls.append("weights")
    )

    def create_runner(layer, config) -> None:
        calls.append("runner")
        method.runner = SimpleNamespace()

    monkeypatch.setattr(method, "create_moe_runner", create_runner)
    monkeypatch.setattr(
        fused_moe_layer_module,
        "get_parallel",
        lambda: SimpleNamespace(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        ),
    )
    monkeypatch.setattr(
        fused_moe_layer_module,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.AUTO,
    )
    monkeypatch.setattr(
        fused_moe_layer_module,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.NONE,
    )
    monkeypatch.setattr(
        fused_moe_layer_module,
        "get_server_args",
        lambda: SimpleNamespace(ep_join_mode="none", moe_runner_backend="auto"),
    )
    monkeypatch.setattr(
        fused_moe_layer_module,
        "create_kt_config_from_server_args",
        lambda server_args, layer_id: None,
    )
    monkeypatch.setattr(
        fused_moe_layer_module,
        "create_moe_dispatcher",
        lambda config: SimpleNamespace(),
    )
    monkeypatch.setattr(fused_moe_layer_module, "print_info_once", lambda message: None)

    layer = FusedMoE(
        num_experts=2,
        hidden_size=4,
        intermediate_size=8,
        layer_id=0,
        quant_method=method,
    )

    assert layer.quant_method is method
    assert layer.runner is method.runner
    assert calls == ["weights", "runner"]


def test_lora_uses_quant_method_contract_for_registered_backend(
    monkeypatch, isolated_runner_registries
) -> None:
    backend_name = "test_lora_extension"
    register_moe_runner_backend_name(backend_name)
    backend = resolve_moe_runner_backend(backend_name)
    quant_info = object()
    quant_calls = []

    class FakeQuantMethod:
        def get_moe_quant_info(self, layer, runner_backend):
            quant_calls.append((layer, runner_backend))
            return quant_info

    base_layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(base_layer)
    base_layer.quant_method = FakeQuantMethod()
    base_layer.moe_runner_config = MoeRunnerConfig()
    base_layer.dispatcher = object()
    base_layer.num_local_experts = 2
    base_layer.should_fuse_routed_scaling_factor_in_topk = False
    base_layer.moe_tp_size = 1
    base_layer.moe_tp_rank = 0
    base_layer.intermediate_size_per_partition = 8
    base_layer.runner = SimpleNamespace(runner_backend=backend)
    lora_backend = SimpleNamespace(is_moe_lora=False)
    created_runners = []
    monkeypatch.setattr(
        runner_module,
        "MoeRunner",
        lambda selected_backend, config, lora_enabled: created_runners.append(
            (selected_backend, config, lora_enabled)
        )
        or object(),
    )

    wrapper = FusedMoEWithLoRA(base_layer, lora_backend)

    assert wrapper._quant_info is quant_info
    assert quant_calls == [(base_layer, backend)]
    assert created_runners == [(backend, base_layer.moe_runner_config, True)]
