"""Tests for scheduler MoE configuration discovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerMoeConfig(CustomTestCase):
    def _run_init(self, *, hf_config, hf_text_config):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = object()
        scheduler.model_config = SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=hf_text_config,
        )

        with (
            patch.object(scheduler_module, "initialize_moe_config") as init_moe,
            patch.object(scheduler_module, "initialize_fp8_gemm_config"),
            patch.object(scheduler_module, "initialize_fp4_gemm_config"),
            patch.object(scheduler_module, "initialize_bf16_gemm_config"),
            patch.object(
                scheduler_module,
                "require_mlp_sync",
                return_value=False,
            ),
        ):
            scheduler.init_moe_gemm_config()

        return scheduler, init_moe

    def test_uses_canonical_text_config_for_llm_config_vlm(self):
        outer_config = SimpleNamespace()
        llm_config = SimpleNamespace(num_experts_per_tok=8)

        scheduler, init_moe = self._run_init(
            hf_config=outer_config,
            hf_text_config=llm_config,
        )

        init_moe.assert_called_once_with(scheduler.server_args)
        self.assertFalse(scheduler.require_mlp_sync)

    def test_canonical_text_config_takes_precedence(self):
        outer_config = SimpleNamespace(text_config=SimpleNamespace())
        llm_config = SimpleNamespace(num_experts_per_tok=8)

        scheduler, init_moe = self._run_init(
            hf_config=outer_config,
            hf_text_config=llm_config,
        )

        init_moe.assert_called_once_with(scheduler.server_args)

    def test_skips_moe_initialization_for_non_moe_text_config(self):
        outer_config = SimpleNamespace()
        text_config = SimpleNamespace()

        _, init_moe = self._run_init(
            hf_config=outer_config,
            hf_text_config=text_config,
        )

        init_moe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
