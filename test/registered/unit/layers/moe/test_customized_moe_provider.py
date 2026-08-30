from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe import customized as customized_module
from sglang.srt.layers.moe.customized import (
    CustomizedMoELayer,
    get_customized_moe_provider,
    register_customized_moe_provider,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class _Method(FusedMoEMethodBase, torch.nn.Module):
    loads_expert_weights = False

    def __init__(self, events):
        super().__init__()
        self.events = events

    def create_weights(self, **kwargs):
        self.events.append(("weights", kwargs["num_experts"]))

    def create_moe_runner(self, layer, moe_runner_config):
        self.events.append(("runner", moe_runner_config.layer_id))
        self.runner = SimpleNamespace()

    def apply(self, layer, dispatch_output):
        raise AssertionError("not exercised by this lifecycle test")


class _Dispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        raise AssertionError("not exercised by this lifecycle test")

    def combine(self, combine_input):
        raise AssertionError("not exercised by this lifecycle test")


class _Provider:
    def __init__(self, events):
        self.events = events

    def prepare_layer(self, *, layer, prefix, native_method, runner_config):
        self.events.append(
            ("prepare", prefix, type(native_method), runner_config.layer_id)
        )
        method = _Method(self.events)
        return CustomizedMoELayer(
            method=method,
            dispatcher_factory=lambda config: _Dispatcher(),
        )


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch):
    monkeypatch.setattr(customized_module, "_provider", None)


def test_customized_provider_owns_full_layer_lifecycle() -> None:
    events = []
    register_customized_moe_provider(_Provider(events))

    with (
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.AUTO,
            a2a_backend=MoeA2ABackend.CUSTOMIZED,
        ),
        get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ),
    ):
        layer = FusedMoE(
            num_experts=32,
            hidden_size=8,
            intermediate_size=16,
            layer_id=3,
            prefix="model.layers.3.mlp.experts",
            quant_method=UnquantizedFusedMoEMethod(),
        )

    assert isinstance(layer.quant_method, _Method)
    assert isinstance(layer.dispatcher, _Dispatcher)
    assert layer.quant_method.loads_expert_weights is False
    assert [event[0] for event in events] == ["prepare", "weights", "runner"]


def test_registration_is_single_owner_and_required() -> None:
    with pytest.raises(RuntimeError, match="requires a registered"):
        get_customized_moe_provider()

    provider = _Provider([])
    register_customized_moe_provider(provider)
    assert get_customized_moe_provider() is provider
    with pytest.raises(ValueError, match="already registered"):
        register_customized_moe_provider(_Provider([]))
