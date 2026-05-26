import sys
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dsa_attention import (
    make_dsa_dense_fallback_cases,
    make_dsa_sparse_cases,
    run_dsa_attention_case,
    run_dsa_sparse_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSAAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dsa_dense_fallback_cases("dsa")
    SPARSE_CASES = make_dsa_sparse_cases("dsa")

    def test_mha_one_shot_dense_fallback_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_attention_case(self, case)

    def test_sparse_topk_cases(self):
        for case in self.SPARSE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_sparse_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
