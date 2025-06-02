"""
Usage:
python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestQwenPPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node.TestFixedBugs.test_chunked_prefill_with_small_bs
"""

import time
import unittest
from types import SimpleNamespace

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch_server,
)


class TestPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                256,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.74)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)


class TestQwenPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23334"  # different ports to avoid conflicts
        cls.model_name = "Qwen/Qwen3-8B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 1% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        cls.model_name = (
            "Qwen/Qwen3-0.6B"  # qwen3 < 8B all have tie_word_embeddings = True
        )

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.38)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 1% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwenMoePPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23336"  # different ports to avoid conflicts
        cls.model_name = "Qwen/Qwen3-30B-A3B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 1% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestFixedBugs(unittest.TestCase):
    def test_chunked_prefill_with_small_bs(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        server_args = ServerArgs(model_path=model)
        bench_args = OneBatchBenchArgs(
            batch_size=(1,),
            input_len=(1,),
            output_len=(1,),
            base_url=DEFAULT_URL_FOR_TEST,
        )
        other_server_args = [
            "--tp-size",
            2,
            "--pp-size",
            2,
            "--chunked-prefill",
            256,
            "--max-running-requests",
            2,
        ]
        run_bench_one_batch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            server_args,
            bench_args,
            other_server_args,
        )


if __name__ == "__main__":
    unittest.main()
