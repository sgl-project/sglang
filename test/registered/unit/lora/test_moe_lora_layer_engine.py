"""Bind-once semantics of the per-layer MoE LoRA engine.

The engine resolves plans, tiles, and the runner exactly once, at the first
weight bind — everything selection-shaped happens there, and the forward
path is a phase lookup plus an M-bucket tile pick. These tests fake the
runner and the device capability so the contract is pinned without CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.moe_lora_runner import MoeLoraLayerEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _FakeRunner:
    def __init__(self, provider_names, workspace: object) -> None:
        self.providers = dict.fromkeys(provider_names)
        self.workspace = workspace
        self.prepared: list[str] = []
        self.runs = 0

    def prepare_plan(self, _plan, *, provider_name, is_shared_outer):
        del is_shared_outer
        self.prepared.append(provider_name)

    def run(
        self,
        _dispatch_output,
        _batch,
        *,
        plan,
        launch_config,
        provider_name,
        output_dtype=None,
    ):
        del plan
        self.runs += 1
        return provider_name, launch_config, output_dtype


def _base_layer(*, hidden_size=2048, num_local_experts=256):
    return SimpleNamespace(
        w2_weight=SimpleNamespace(
            device=torch.device("cuda"), shape=(num_local_experts, hidden_size, 768)
        ),
        num_local_experts=num_local_experts,
        moe_runner_config=SimpleNamespace(activation="silu"),
    )


def _engine(monkeypatch, *, capability=(9, 0), created=None, runner_cls=_FakeRunner):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )

    def fake_from_layer(_base_layer, *, provider_names, workspace):
        runner = runner_cls(provider_names, workspace)
        if created is not None:
            created.append(runner)
        return runner

    monkeypatch.setattr(
        "sglang.srt.lora.moe.moe_lora_runner.MoeLoraRunner.from_layer",
        staticmethod(fake_from_layer),
    )
    return MoeLoraLayerEngine(_base_layer(), workspace=object())


def _batch(**overrides):
    values = {"is_prefill": False}
    values.update(overrides)
    return SimpleNamespace(**values)


def _dispatch(num_tokens: int):
    return SimpleNamespace(hidden_states=torch.empty(num_tokens, 8))


def test_binding_resolves_and_prepares_every_phase_once(monkeypatch) -> None:
    created: list[_FakeRunner] = []
    engine = _engine(monkeypatch, created=created)
    assert not engine.is_bound

    engine.ensure_bound(is_shared_outer=False, physical_rank=64)
    assert engine.is_bound
    assert len(created) == 1
    runner = created[0]
    # one plan per phase, each prepared against its provider on ONE runner
    assert runner.prepared == [sel.provider for sel in engine._selected.values()]
    assert set(runner.providers) == {sel.provider for sel in engine._selected.values()}

    # a repeated identical bind is a no-op; a changed constant is an error
    engine.ensure_bound(is_shared_outer=False, physical_rank=64)
    assert len(created) == 1 and len(runner.prepared) == 2
    with pytest.raises(ValueError, match="layout changed"):
        engine.ensure_bound(is_shared_outer=True, physical_rank=64)
    with pytest.raises(ValueError, match="rank changed"):
        engine.ensure_bound(is_shared_outer=False, physical_rank=16)
    with pytest.raises(ValueError, match="positive"):
        engine.ensure_bound(is_shared_outer=False, physical_rank=0)


def test_run_routes_by_phase_and_buckets_by_batch_size(monkeypatch) -> None:
    engine = _engine(monkeypatch, capability=(10, 0))
    engine.ensure_bound(is_shared_outer=False, physical_rank=64)

    decode_provider, tiny_launch, dtype = engine.run(
        _dispatch(4), _batch(), output_dtype=torch.bfloat16
    )
    assert decode_provider == "cutedsl"
    assert dtype == torch.bfloat16
    prefill_provider, _, _ = engine.run(_dispatch(4096), _batch(is_prefill=True))
    assert prefill_provider == "cutedsl_contiguous"

    # the M-bucket pick is per forward: the gb300 decode ladder at rank 64
    # serves different tiles at 4 and 17 tokens
    _, mse_launch, _ = engine.run(_dispatch(16), _batch())
    _, large_launch, _ = engine.run(_dispatch(17), _batch())
    assert tiny_launch.gate_b["BLOCK_SIZE_N"] == 128
    assert mse_launch.gate_b["BLOCK_SIZE_N"] == 512
    assert large_launch.gate_b["BLOCK_SIZE_N"] == 256


def test_run_before_binding_is_an_error(monkeypatch) -> None:
    engine = _engine(monkeypatch)
    with pytest.raises(RuntimeError, match="bound before running"):
        engine.run(_dispatch(4), _batch())


def test_failed_initial_binding_is_transactional(monkeypatch) -> None:
    class _FailingRunner(_FakeRunner):
        def prepare_plan(self, _plan, *, provider_name, is_shared_outer):
            del is_shared_outer
            self.prepared.append(provider_name)
            if len(self.prepared) == 2:
                raise NotImplementedError("unsupported plan")

    created: list[_FakeRunner] = []
    engine = _engine(monkeypatch, created=created, runner_cls=_FailingRunner)
    with pytest.raises(NotImplementedError, match="unsupported plan"):
        engine.ensure_bound(is_shared_outer=False, physical_rank=64)
    assert len(created) == 1
    assert not engine.is_bound
    # the next bind starts clean and may succeed on a fixed runner
    with pytest.raises(NotImplementedError):
        engine.ensure_bound(is_shared_outer=False, physical_rank=64)
    assert len(created) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
