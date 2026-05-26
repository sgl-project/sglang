import sys
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dual_chunk_attention import (
    make_dual_chunk_cases,
    make_dual_chunk_sparse_cases,
    run_dual_chunk_attention_case,
    run_dual_chunk_sparse_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDualChunkFlashAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dual_chunk_cases("dual_chunk_flash_attn")
    SPARSE_CASES = make_dual_chunk_sparse_cases("dual_chunk_flash_attn")

    def test_projected_dual_chunk_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dual_chunk_attention_case(self, case)

    def test_sparse_dual_chunk_attention_cases(self):
        for case in self.SPARSE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dual_chunk_sparse_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
