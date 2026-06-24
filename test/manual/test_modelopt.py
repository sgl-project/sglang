import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST_FP8,
    DEFAULT_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST_FP8_REVISION,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEvalFP8ModelOptQuantAccuracy(CustomTestCase):

    def _run_test(self, model, other_args, expected_score):
        base_url = DEFAULT_URL_FOR_TEST
        other_args = other_args or []

        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
                temperature=0.1,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], expected_score)
        finally:
            kill_process_tree(process.pid)

    @unittest.skipIf(
        torch.version.hip is not None, "modelopt quantization unsupported on ROCm"
    )
    def test_mmlu_offline_only(self):
        """Test with offline quantization only."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST_FP8,
            other_args=[
                "--revision",
                DEFAULT_MODEL_NAME_FOR_MODELOPT_QUANT_ACCURACY_TEST_FP8_REVISION,
            ],
            expected_score=0.64,
        )
