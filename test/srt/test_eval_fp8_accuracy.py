import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_FP8_MODEL_NAME_FOR_ACCURACY_TEST,
    DEFAULT_FP8_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEvalFP8Accuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_FP8_MODEL_NAME_FOR_ACCURACY_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        if is_hip():
            # Another threshold for AMD because fp8 dtype is difference
            self.assertGreaterEqual(metrics["score"], 0.609375)
        else:
            self.assertGreaterEqual(metrics["score"], 0.61)


class TestEvalFP8DynamicQuantAccuracy(CustomTestCase):

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

    def test_mmlu_offline_only(self):
        """Test with offline quantization only."""
        self._run_test(
            model=DEFAULT_FP8_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST,
            other_args=[],
            expected_score=0.64,
        )

    def test_mmlu_offline_and_online_override(self):
        """Test with both offline and online quantization."""
        self._run_test(
            model=DEFAULT_FP8_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST,
            other_args=["--quantization", "w8a8_fp8"],
            # inference will use sgl kernel w/ online quant override
            # we observed that the accuracy is higher then offline only
            expected_score=0.64,
        )

    def test_mmlu_online_only(self):
        """Test with online quantization only."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            # inference will use sgl kernel w/ online quantization only
            # we observed that the accuracy is higher then offline only
            other_args=["--quantization", "w8a8_fp8"],
            expected_score=0.64,
        )

    def test_mmlu_fp16_baseline(self):
        """Test with unquantized fp16 baseline."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            other_args=[],
            expected_score=0.64,
        )


if __name__ == "__main__":
    unittest.main()
