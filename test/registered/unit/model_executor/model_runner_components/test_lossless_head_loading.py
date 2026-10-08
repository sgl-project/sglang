"""Exercise ModelOpt's method selection before the lossless-head load hook."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestLosslessHeadLoading(CustomTestCase):
    def test_modelopt_unquantized_head_prepares_after_loading(self):
        config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "packed_modules_mapping": {},
                "quantized_layers": {
                    "model.layers.0.experts.up_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    }
                },
            }
        )
        head = ParallelLMHead.__new__(ParallelLMHead)
        torch.nn.Module.__init__(head)
        method = config.get_quant_method(head, prefix="lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)
        with (
            patch.dict(os.environ, {"SGLANG_SM120_LOSSLESS_BF16_HEAD": "1"}),
            patch("sglang.srt.layers.quantization.unquant._is_cpu", False),
            patch(
                "sglang.kernels.ops.gemm.sm120_lossless_bf16_head.prepare_lossless_bf16_head"
            ) as prepare,
        ):
            method.process_weights_after_loading(head)
            prepare.assert_called_once_with(head)


if __name__ == "__main__":
    unittest.main()
