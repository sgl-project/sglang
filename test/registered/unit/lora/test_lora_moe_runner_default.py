"""Startup checks for explicit MoE LoRA runners."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.lora_hook import check_lora_moe_runner_args
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")


def _check(
    backend, *, quantization=None, resolved_quantization=None, is_fp4_experts=False
):
    args = ServerArgs(
        model_path="dummy",
        quantization=quantization,
        lora_backend="triton_v2",
        moe_runner_backend=backend,
        enable_lora=True,
    )
    args._model_config = SimpleNamespace(
        quantization=resolved_quantization or quantization,
        is_fp4_experts=is_fp4_experts,
    )
    check_lora_moe_runner_args(args)


class TestLoraMoeExplicitRunner(CustomTestCase):
    def test_explicit_runners_admit_the_served_quantizations(self):
        for kwargs in (
            dict(quantization="fp8"),
            dict(quantization="fp8", is_fp4_experts=True),
            dict(quantization="mxfp8"),
            dict(quantization="w8a8_fp8"),
            dict(quantization="compressed-tensors"),
            # ModelOpt's CLI name resolves to the checkpoint's NVFP4 method.
            dict(quantization="modelopt", resolved_quantization="modelopt_fp4"),
            dict(resolved_quantization="modelopt_mixed"),
        ):
            with self.subTest(**kwargs):
                _check("lora_cutedsl", **kwargs)
        # With no flag, the checkpoint's own method is checked.
        with self.assertRaisesRegex(ValueError, "cannot serve awq MoE weights"):
            _check("lora_cutedsl", resolved_quantization="awq")


if __name__ == "__main__":
    unittest.main()
