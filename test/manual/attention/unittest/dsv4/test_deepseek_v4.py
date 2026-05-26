"""DSV4 attention correctness — SWA-only (compress_ratio=0) slice.

This is the first DSV4 unit test. It is intentionally narrow: one eager EXTEND
case on the SWA path. C4 (4x) and C128 (128x) compressor + indexer paths and
speculative modes are explicit follow-ups.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_FLASH_MLA_AVAILABLE = importlib.util.find_spec("flash_mla") is not None

from common.attention_methods.dsv4_attention import (  # noqa: E402
    make_dsv4_cases,
    run_dsv4_attention_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(not _FLASH_MLA_AVAILABLE, "flash_mla is required for DSV4 SWA")
class TestDSV4AttentionBackendCorrectness(CustomTestCase):
    CASES = make_dsv4_cases("dsv4")

    def test_swa_only_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_attention_case(self, case)


if __name__ == "__main__":
    unittest.main()
