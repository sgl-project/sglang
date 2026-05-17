"""
Torch native attention backend test for Intel XPU.

Tests torch native attention backend with MMLU benchmark to verify
attention correctness on XPU.

Based on test/registered/attention/test_torch_native_attention_backend.py
adapted for XPU Stage B (proven passing in HTML report).

Usage:
python3 -m unittest test_torch_native_attention_xpu.TestTorchNativeAttnXPU.test_mmlu
"""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_xpu_ci(est_time=140, suite="stage-b-test-1-gpu-xpu")


class TestTorchNativeAttnXPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.backend = sgl.Runtime(
            model_path=cls.model,
            device="xpu",
            attention_backend="torch_native",
            mem_fraction_static=0.7,
            disable_radix_cache=True,
        )
        sgl.set_default_backend(cls.backend)

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()

    def test_mmlu(self):
        args = type(
            "Args",
            (),
            {
                "base_url": self.backend.url,
                "model": self.model,
                "eval_name": "mmlu",
                "num_examples": 64,
                "num_threads": 32,
            },
        )()

        metrics = run_eval(args)
        assert metrics["accuracy"] >= 0.5


if __name__ == "__main__":
    unittest.main()
