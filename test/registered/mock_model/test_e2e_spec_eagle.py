from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.utils import MOCK_MODEL_PATH, run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=99, stage="extra-a", runner_config="1-gpu-small")


class TestE2ESpeculativeEagle(CustomTestCase):
    def test_spec_eagle_no_canary_violation(self) -> None:
        run_mock_model_bench_serving(
            extra_server_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                MOCK_MODEL_PATH,
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--mem-fraction-static",
                "0.45",
            ],
            input_check_enabled=False,
        )


if __name__ == "__main__":
    unittest.main()
