import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

from utils import DenseAttentionCase, run_dense_attention_case


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonDenseAttentionBackendCorrectness(CustomTestCase):
    CASES = (
        DenseAttentionCase(
            name="mha_extend_exact_page",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 8),
            extend_lens=(16, 8),
        ),
        DenseAttentionCase(
            name="gqa_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_projected_dense_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
