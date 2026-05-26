import sys
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.lightning_attention import (
    make_lightning_cases,
    run_lightning_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonLightningBackendCorrectness(CustomTestCase):
    CASES = make_lightning_cases("triton")

    def test_projected_lightning_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_lightning_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
