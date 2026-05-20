from __future__ import annotations

import unittest

from utils import mock_model_engine_kwargs

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


class TestE2ESpeculativeEagle(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = sgl.Engine(
            model_path="Qwen/Qwen3-0.6B",
            **mock_model_engine_kwargs(speculative_algorithm="EAGLE"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.shutdown()

    def test_spec_eagle_no_canary_violation(self) -> None:
        self.engine.generate(
            input_ids=_fake_prompt(32),
            sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
