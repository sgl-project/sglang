from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.moe.runner import MoeLoraRunner
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _FakeProvider:
    hidden_size = 2048
    intermediate_size = 768
    num_local_experts = 256
    gate_up_slices = 2
    contract = SimpleNamespace(lora_delta_dtype=torch.bfloat16)


def _from_layer(monkeypatch, provider_cls):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(MoeLoraRunner, "_admit", staticmethod(lambda layer: "bf16"))

    def fake_build(_base_layer, *, base_gemm_rows, vendor, family):
        return provider_cls()

    monkeypatch.setattr(MoeLoraRunner, "_build_provider", staticmethod(fake_build))
    # from_layer reads the published MoE runner backend; publish the engine's
    # default name so the builds see its vendor.
    with get_context().override_server_args():
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL):
            return MoeLoraRunner.from_layer(
                SimpleNamespace(
                    w2_weight=SimpleNamespace(
                        device=torch.device("cuda"), shape=(256, 2048, 768)
                    ),
                    num_local_experts=256,
                    hidden_size=2048,
                    moe_runner_config=SimpleNamespace(
                        activation="silu",
                        top_k=8,
                        routed_scaling_factor=1.0,
                        is_gated=True,
                    ),
                ),
                workspace=object(),
                is_shared_outer=False,
                physical_rank=64,
            )


def test_plan_validation_failure_propagates(monkeypatch) -> None:
    class _NonGatedProvider(_FakeProvider):
        gate_up_slices = 1

    with pytest.raises(ValueError, match="gate/up slices"):
        _from_layer(monkeypatch, _NonGatedProvider)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
