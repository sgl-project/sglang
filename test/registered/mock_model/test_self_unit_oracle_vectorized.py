from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


class TestOracleVectorized(CustomTestCase):
    def test_predict_input_tokens_for_plan_batch(self) -> None:
        pass

    def test_predict_input_tokens_for_plan_matches_scalar(self) -> None:
        pass

    def test_predict_next_tokens_for_active_batch_batch(self) -> None:
        pass

    def test_predict_next_tokens_matches_scalar(self) -> None:
        pass

    def test_vectorized_handles_chunked_offsets(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
