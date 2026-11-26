"""CPU tests for AutoRound W4/W8 checkpoints and both packing formats."""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.cpu.quantization.gptq_kernels import (
    GPTQIntelAMXLinearKernel,
    _gptq_w8_to_channelwise_w8a8,
)
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
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


@unittest.skipUnless(
    cpu_has_amx_support(),
    "AutoRound W8 CPU inference requires the Intel AMX CPU backend.",
)
class TestAutoRoundCPUW8Kernel(CustomTestCase):
    def test_w8_load_time_conversion_matches_amx_gemm(self):
        torch.manual_seed(0)
        output_channels, input_channels, group_size = 128, 256, 128
        groups = input_channels // group_size
        qvalue = torch.randint(
            0, 256, (input_channels, output_channels), dtype=torch.int32
        )
        scales = torch.rand(groups, output_channels, dtype=torch.float16) * 0.01 + 0.001
        packed = torch.zeros((input_channels // 4, output_channels), dtype=torch.int32)
        qvalue_u8 = qvalue.to(torch.uint8)
        for byte_index in range(4):
            packed |= qvalue_u8[byte_index::4].to(torch.int32) << (8 * byte_index)
        qzeros = torch.full((groups, output_channels // 4), 127, dtype=torch.int32)
        expected_weight, expected_scale = _gptq_w8_to_channelwise_w8a8(
            packed, qzeros, scales, is_symmetric=True
        )

        module = torch.nn.Module()
        module.qweight = torch.nn.Parameter(packed, requires_grad=False)
        module.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        module.scales = torch.nn.Parameter(scales, requires_grad=False)
        module.g_idx = torch.nn.Parameter(
            torch.arange(input_channels, dtype=torch.int32), requires_grad=False
        )
        kernel = GPTQIntelAMXLinearKernel(SimpleNamespace(weight_bits=8, is_sym=True))
        kernel.process_weights_after_loading(module)
        self.assertTrue(module.use_intel_amx_backend)
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            self.assertFalse(hasattr(module, name))

        x = torch.randn(4, input_channels, dtype=torch.bfloat16)
        output = kernel.apply(module, x)

        x_fp32 = x.float()
        x_scale = x_fp32.abs().amax(dim=1, keepdim=True) / 127.0
        x_quant = torch.round(x_fp32 / x_scale).clamp(-127, 127)
        reference = (
            (x_quant @ expected_weight.float().T) * x_scale * expected_scale.flatten()
        )
        relative_error = (output.float() - reference).norm() / reference.norm()
        self.assertLess(relative_error.item(), 0.01)


@unittest.skipUnless(
    cpu_has_amx_support(),
    "AutoRound W4/W8 CPU inference requires the Intel AMX CPU backend.",
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
