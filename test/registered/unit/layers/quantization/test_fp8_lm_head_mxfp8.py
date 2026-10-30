"""An MXFP8 lm_head goes through Fp8LinearMethod.process_weights_after_loading.

ParallelLMHead is a VocabParallelEmbedding, not a LinearBase, so it has no
``deepgemm_mxfp8_input`` attribute. Block-quant weight processing is mocked so
the test runs on CPU; the rest of process_weights_after_loading is real.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.test.test_utils import CustomTestCase


class TestFp8LmHeadMxfp8(CustomTestCase):
    def test_process_weights_after_loading_on_mxfp8_lm_head(self):
        lm_head = ParallelLMHead(256, 128, enable_tp=False)
        method = Fp8LinearMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, use_mxfp8=True)
        )
        self.assertTrue(method.block_quant)
        with patch.object(
            Fp8LinearMethod, "process_weights_after_loading_block_quant"
        ) as block_quant:
            method.process_weights_after_loading(lm_head)
        block_quant.assert_called_once_with(lm_head)
        self.assertFalse(getattr(lm_head, "deepgemm_mxfp8_input", False))


if __name__ == "__main__":
    unittest.main(verbosity=3)
