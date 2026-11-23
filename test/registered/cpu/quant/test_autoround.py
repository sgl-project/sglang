"""CPU accuracy test for AutoRound INT4 checkpoints in both packing formats
(auto_round:auto_gptq / auto_round:auto_awq)."""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import cpu_has_amx_support, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cpu_ci(est_time=183, suite="extra-tp-test-cpu-intel")


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

    def test_fused_layer_follows_16bit_shards(self):
        """Without the loader's packed_modules_mapping, the fused in_proj_ba is
        not found in extra_config, falls back to 4-bit and is built as int4,
        so the dense in_proj_a/b checkpoint shards cannot be loaded into it."""
        prefix = "model.language_model.layers.0.linear_attn"
        quant_config = AutoRoundConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "extra_config": {
                    f"{prefix}.in_proj_a": {"bits": 16, "data_type": "fp"},
                    f"{prefix}.in_proj_b": {"bits": 16, "data_type": "fp"},
                },
                "packed_modules_mapping": {"in_proj_ba": ["in_proj_b", "in_proj_a"]},
            }
        )

        layer = LinearBase.__new__(LinearBase)
        quant_method = quant_config.get_quant_method(layer, f"{prefix}.in_proj_ba")

        self.assertIsInstance(quant_method, UnquantizedLinearMethod)


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
                # Auto-sizing can size the KV cache off the full per-NUMA-node
                # memory estimate (100+ GB) for these tiny 0.5B checkpoints;
                # cap it explicitly regardless of SGLANG_CPU_OMP_THREADS_BIND.
                other_args = [
                    "--trust-remote-code",
                    "--quantization",
                    "auto-round",
                    "--dtype",
                    "bfloat16",
                    "--mem-fraction-static",
                    "0.3",
                ]
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=other_args,
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
                    metrics = run_sgl_eval(args)
                    self.assertGreaterEqual(metrics["score"], 0.25)
                finally:
                    kill_process_tree(process.pid)
                    print(f"[INFO] Server for {model} stopped.")


if __name__ == "__main__":
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    unittest.main()
