import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dense_attention import (
    DenseAttentionCase,
    make_dense_cases,
    run_dense_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTorchNativeDenseAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dense_cases("torch_native")
    RUNNER_EAGER_CASES = (
        DenseAttentionCase(
            name="runner_eager_decode_page_boundary",
            backend="torch_native",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_projected_dense_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)

    def test_runner_mode_eager_cases(self):
        for case in self.RUNNER_EAGER_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
