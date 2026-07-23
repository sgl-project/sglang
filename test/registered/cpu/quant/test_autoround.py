"""
Usage:
SGLANG_USE_CPU_ENGINE=1 python3 -m unittest test_autoround

CPU accuracy test for AutoRound INT4 checkpoints. Covers both AutoRound packing
formats (auto_round:auto_gptq / auto_round:auto_awq) by launching a server and
running an MMLU eval. AutoRound INT4 CPU inference uses the Intel AMX backend,
so the test is skipped on AMD CPUs and other non-AMX CPU hosts.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.utils import cpu_has_amx_support, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cpu_ci(est_time=330, suite="base-a-test-cpu")


class TestAutoRoundCPUConfig(CustomTestCase):
    def test_gptq_defaults_are_explicit(self):
        quant_config = AutoRoundConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
            }
        )

        gptq_kwargs = quant_config.get_gptq_config_kwargs(4, 128)
        self.assertFalse(gptq_kwargs["desc_act"])
        self.assertFalse(gptq_kwargs["lm_head_quantized"])
        self.assertEqual(gptq_kwargs["dynamic"], {})

    def test_gptq_desc_act_is_rejected(self):
        quant_config = AutoRoundConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "desc_act": True,
            }
        )

        with self.assertRaisesRegex(ValueError, "desc_act=False only"):
            quant_config.get_gptq_config_kwargs(4, 128)


@unittest.skipUnless(
    cpu_has_amx_support(),
    "AutoRound INT4 CPU inference requires the Intel AMX CPU backend.",
)
class TestAutoRoundCPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmlu(self):
        device = "cpu"
        for model in DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST:
            with self.subTest(model=model):
                print(f"\n[INFO] Launching server for model: {model}")
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--trust-remote-code", "--quantization", "auto-round"],
                    device=device,
                )

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmlu",
                        num_examples=32,
                        num_threads=32,
                        device=device,
                    )
                    metrics = run_eval(args)
                    self.assertGreaterEqual(metrics["score"], 0.25)
                finally:
                    kill_process_tree(process.pid)
                    print(f"[INFO] Server for {model} stopped.")


if __name__ == "__main__":
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    unittest.main()
