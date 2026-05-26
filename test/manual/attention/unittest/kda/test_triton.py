import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.kda_attention import (
    KDAAttentionCase,
    run_kda_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonKDABackendCorrectness(CustomTestCase):
    CASES = (
        KDAAttentionCase(
            name="kda_extend_zero_prefix_exact_page",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
        ),
    )

    def test_projected_kda_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_kda_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
