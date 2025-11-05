"""
Usage:
python3 -m unittest test_intel_amx_attention_backend.TestIntelAMXAttnBackend.test_latency_default_model
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    intel_amx_benchmark,
    is_in_ci,
    popen_launch_server,
)


class TestIntelAMXAttnBackend(CustomTestCase):

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--mem-fraction-static", "0.3"],
        min_throughput=10,
    )
    def test_latency_mla_model(self):
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--mem-fraction-static", "0.1"],
        min_throughput=40,
    )
    def test_latency_default_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST

    def test_mmlu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.3",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )
            metrics = run_eval(args)
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


class TestDPAttention(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--attention-backend",
            "intel_amx",
            "--mem-fraction-static",
            "0.1",
            "--disable-overlap-schedule",
            "--tp",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_dp_attention_DP2TP2(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=32,
            parallel=32,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
