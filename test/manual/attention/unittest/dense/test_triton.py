import sys
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dense_attention import make_dense_cases, run_dense_attention_case


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonDenseAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dense_cases("triton")

    def test_projected_dense_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
