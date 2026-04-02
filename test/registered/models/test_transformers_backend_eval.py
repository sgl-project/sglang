"""A small end-to-end eval coverage for the transformers modeling backend."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=180, suite="stage-b-test-small-1-gpu")


class TestTransformersBackendEval(DefaultServerBase):
    model = "HuggingFaceTB/SmolLM3-3B"
    gsm8k_num_questions = 30
    gsm8k_accuracy_thres = 0.5
    gsm8k_parallel = 30
    other_args = [
        "--model-impl",
        "transformers",
        "--enable-torch-compile",
        "--torch-compile-max-bs",
        "4",
        "--disable-cuda-graph",
    ]

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.gsm8k_num_questions,
            max_new_tokens=512,
            parallel=self.gsm8k_parallel,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_accuracy_thres)


if __name__ == "__main__":
    unittest.main()
