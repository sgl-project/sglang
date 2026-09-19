"""CPU checks for PPU Kimi-K3 MoE compatibility paths."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.moe.moe_runner import acext as acext_module


def test_acext_situ_falls_back_before_importing_acext():
    sentinel = object()
    runner_config = SimpleNamespace(activation="situ")
    with (
        patch.object(acext_module, "is_ppu", return_value=True),
        patch.object(acext_module.logger, "info_once", create=True),
        patch.object(
            acext_module,
            "fused_experts_none_to_triton",
            return_value=sentinel,
        ) as fallback,
    ):
        result = acext_module.fused_experts_none_to_acext(
            "dispatch", "quant", runner_config
        )

    assert result is sentinel
    fallback.assert_called_once_with("dispatch", "quant", runner_config)
