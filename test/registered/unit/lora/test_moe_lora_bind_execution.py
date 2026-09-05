from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.execution_plan import Phase
from sglang.srt.lora.moe.moe_lora_runner import MoeLoraRunner
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class _FakeProvider:
    hidden_size = 2048
    intermediate_size = 768
    num_local_experts = 256
    gate_up_slices = 2
    contract = SimpleNamespace(lora_delta_dtype=torch.bfloat16)


def _base_layer(*, hidden_size=2048, num_local_experts=256):
    return SimpleNamespace(
        w2_weight=SimpleNamespace(
            device=torch.device("cuda"), shape=(num_local_experts, hidden_size, 768)
        ),
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        moe_runner_config=SimpleNamespace(
            activation="silu", top_k=8, routed_scaling_factor=1.0, is_gated=True
        ),
    )


def _from_layer(
    monkeypatch,
    *,
    capability=(9, 0),
    built=None,
    provider_cls=_FakeProvider,
    is_shared_outer=False,
    physical_rank=64,
):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(MoeLoraRunner, "_admit", staticmethod(lambda layer: None))

    def fake_build(_base_layer, *, base_gemm_rows, vendor):
        if built is not None:
            built.append((base_gemm_rows, vendor))
        return provider_cls()

    monkeypatch.setattr(MoeLoraRunner, "_build_provider", staticmethod(fake_build))
    # from_layer reads --moe-lora-base-gemm. The test therefore publishes a
    # context that holds the shipped default. Production code then needs no
    # fallback for a missing server.
    with get_context().override_server_args():
        return MoeLoraRunner.from_layer(
            _base_layer(),
            workspace=object(),
            is_shared_outer=is_shared_outer,
            physical_rank=physical_rank,
        )


def test_from_layer_resolves_plans_and_builds_each_row_order_once(
    monkeypatch,
) -> None:
    built: list[tuple[str, str]] = []
    runner = _from_layer(monkeypatch, built=built)
    row_orders = {sel.base_gemm_rows for sel in runner.plans.values()}
    assert built == [(rows, "cutedsl") for rows, _ in dict.fromkeys(built)]
    assert {rows for rows, _ in built} == row_orders
    assert set(runner.providers) == row_orders
    assert set(runner.tiles) == set(runner.plans)


def test_phase_rows_and_tile_buckets(monkeypatch) -> None:
    runner = _from_layer(monkeypatch, capability=(10, 0))
    assert runner.plans[Phase.DECODE].base_gemm_rows == "expert_major"
    assert runner.plans[Phase.PREFILL].base_gemm_rows == "route_major"
    decode = runner.tiles[Phase.DECODE]
    assert decode.config_for(4).gate_up_b["BLOCK_SIZE_N"] == 128
    assert decode.config_for(16).gate_up_b["BLOCK_SIZE_N"] == 512
    assert decode.config_for(17).gate_up_b["BLOCK_SIZE_N"] == 256


def test_plan_validation_failure_propagates(monkeypatch) -> None:
    class _NonGatedProvider(_FakeProvider):
        gate_up_slices = 1

    with pytest.raises(ValueError, match="gate/up slices"):
        _from_layer(monkeypatch, provider_cls=_NonGatedProvider)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
