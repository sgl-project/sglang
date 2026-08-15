from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.config_backend import MoeLoraConfigBackend
from sglang.srt.lora.moe.execution_plan import ActivationFamily
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _FakeRunner:
    def __init__(self, key: str, provider: str, workspace: object) -> None:
        self.key = key
        self.provider = provider
        self.workspace = workspace
        self.validations = 0
        self.runs = 0

    def validate_factors(self, **_kwargs) -> None:
        self.validations += 1

    def run(self, _dispatch_output, _batch, *, output_dtype=None):
        self.runs += 1
        return self.key, output_dtype


def _batch(**overrides):
    values = {
        "is_shared_outer": False,
        "physical_rank": 64,
        "active_rank": 4,
        "has_active_lora": True,
        "use_cuda_graph": False,
        "is_prefill": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _factors():
    tensor = torch.empty(1)
    return {
        "gate_up_lora_a": tensor,
        "gate_up_lora_b": tensor,
        "down_lora_a": tensor,
        "down_lora_b": tensor,
        "is_shared_outer": False,
    }


def test_binding_constructs_and_validates_every_choice_before_selection(
    monkeypatch,
) -> None:
    created: list[_FakeRunner] = []

    def fake_from_layer(
        _base_layer,
        *,
        provider_name,
        execution_plan,
        launch_config,
        workspace,
    ):
        del execution_plan, launch_config
        runner = _FakeRunner(f"choice-{len(created)}", provider_name, workspace)
        created.append(runner)
        return runner

    monkeypatch.setattr(
        "sglang.srt.lora.moe.config_backend.MoeLoraRunner.from_layer",
        staticmethod(fake_from_layer),
    )
    backend = MoeLoraConfigBackend(
        object(),
        capability=(9, 0),
        hidden_size=2048,
        num_local_experts=256,
        activation=ActivationFamily.SWIGLU,
    )
    backend.bind_factors(**_factors())

    assert backend.is_bound
    assert len(created) == len(backend.choices) > 0
    assert all(runner.validations == 1 for runner in created)
    assert len({id(runner.workspace) for runner in created}) == 1

    count_after_bind = len(created)
    choice = backend.select(_batch(), num_tokens=4)
    backend.select(_batch(is_prefill=True), num_tokens=2048)
    assert len(created) == count_after_bind
    output = backend.run_selected(
        choice,
        SimpleNamespace(hidden_states=torch.empty(4, 8)),
        _batch(),
        output_dtype=torch.bfloat16,
    )
    assert output[1] is torch.bfloat16
    assert len(created) == count_after_bind


def test_eager_and_graph_use_the_physical_kernel_rank(monkeypatch) -> None:
    def fake_from_layer(
        _base_layer,
        *,
        provider_name,
        execution_plan,
        launch_config,
        workspace,
    ):
        del execution_plan, launch_config
        return _FakeRunner("choice", provider_name, workspace)

    monkeypatch.setattr(
        "sglang.srt.lora.moe.config_backend.MoeLoraRunner.from_layer",
        staticmethod(fake_from_layer),
    )
    backend = MoeLoraConfigBackend(
        object(),
        capability=(9, 0),
        activation=ActivationFamily.SWIGLU,
        hidden_size=2048,
        num_local_experts=256,
    )
    backend.bind_factors(**_factors())

    eager = backend.select(_batch(active_rank=4), num_tokens=17)
    graph = backend.select(
        _batch(active_rank=4, physical_rank=64, use_cuda_graph=True),
        num_tokens=17,
    )
    assert "indexed_down_a" in eager.key
    assert "indexed_down_a" in graph.key

    # The rank-sensitive GB300 decode boundary (decode.gab.tiny caps the rank
    # at 16) demonstrates that both paths describe the padded resident GEMM K
    # dimension, not the logical rank: a logical rank of 4 would qualify for
    # the tiny scenario, but the resident physical rank must not.
    gb300 = MoeLoraConfigBackend(
        object(),
        capability=(10, 0),
        activation=ActivationFamily.SWIGLU,
        hidden_size=2048,
        num_local_experts=256,
    )
    gb300.bind_factors(**_factors())
    tiny = gb300.select(
        _batch(active_rank=4, physical_rank=16),
        num_tokens=4,
    )
    assert "decode.gab.tiny" in tiny.key
    eager_decode = gb300.select(
        _batch(active_rank=4, physical_rank=64),
        num_tokens=4,
    )
    graph_decode = gb300.select(
        _batch(active_rank=4, physical_rank=64, use_cuda_graph=True),
        num_tokens=4,
    )
    assert "decode.gab.tiny" not in eager_decode.key
    assert "decode.gab.tiny" not in graph_decode.key
    assert eager_decode.key == graph_decode.key


def test_eager_base_only_uses_preconstructed_runner(
    monkeypatch,
) -> None:
    created = 0

    def fake_from_layer(
        _base_layer,
        *,
        provider_name,
        execution_plan,
        launch_config,
        workspace,
    ):
        nonlocal created
        del execution_plan, launch_config
        created += 1
        return _FakeRunner("choice", provider_name, workspace)

    monkeypatch.setattr(
        "sglang.srt.lora.moe.config_backend.MoeLoraRunner.from_layer",
        staticmethod(fake_from_layer),
    )
    backend = MoeLoraConfigBackend(
        object(),
        capability=(9, 0),
        activation=ActivationFamily.SWIGLU,
        hidden_size=2048,
        num_local_experts=256,
    )
    backend.bind_factors(**_factors())
    created_at_bind = created

    choice = backend.select(_batch(has_active_lora=False), num_tokens=4)
    assert created == created_at_bind
    output = backend.run_selected(
        choice,
        SimpleNamespace(hidden_states=torch.empty(4, 8)),
        _batch(has_active_lora=False),
    )
    assert output[0] == "choice"
    assert created == created_at_bind


def test_failed_initial_binding_is_transactional(monkeypatch) -> None:
    call_count = 0

    def fake_from_layer(
        _base_layer,
        *,
        provider_name,
        execution_plan,
        launch_config,
        workspace,
    ):
        nonlocal call_count
        del execution_plan, launch_config
        call_count += 1
        runner = _FakeRunner(f"choice-{call_count}", provider_name, workspace)
        if call_count == 2:

            def fail_validation(**_kwargs) -> None:
                raise NotImplementedError("unsupported plan")

            runner.validate_factors = fail_validation
        return runner

    monkeypatch.setattr(
        "sglang.srt.lora.moe.config_backend.MoeLoraRunner.from_layer",
        staticmethod(fake_from_layer),
    )
    backend = MoeLoraConfigBackend(
        object(),
        capability=(9, 0),
        activation=ActivationFamily.SWIGLU,
        hidden_size=2048,
        num_local_experts=256,
    )
    with pytest.raises(NotImplementedError, match="unsupported plan"):
        backend.bind_factors(**_factors())
    assert call_count > 1
    assert not backend.is_bound
    assert backend.choices == ()
