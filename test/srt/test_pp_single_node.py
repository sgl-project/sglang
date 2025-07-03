"""
Usage:
python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestQwenPPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node.TestFixedBugs.test_chunked_prefill_with_small_bs
"""

import os
import time
import unittest
from types import SimpleNamespace

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.distributed import get_pp_indices
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


# Adapted from https://github.com/vllm-project/vllm/blob/v0.9.0/tests/distributed/test_pipeline_partition.py
class TestCustomLayerPartition(unittest.TestCase):
    def test_custom_layer_partition(self):
        def _verify(partition_str, num_layers, pp_size, goldens):
            bak = os.environ.get("SGLANG_PP_LAYER_PARTITION", None)
            os.environ["SGLANG_PP_LAYER_PARTITION"] = partition_str
            for pp_rank, golden in enumerate(goldens):
                self.assertEqual(get_pp_indices(num_layers, pp_rank, pp_size), golden)
            if bak is not None:
                os.environ["VLLM_PP_LAYER_PARTITION"] = bak

        # Even partition
        _verify("5,5,5,5", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Balanced partition
        _verify("4,6,6,4", 20, 4, [(0, 4), (4, 10), (10, 16), (16, 20)])
        # Put reminder somewhere
        _verify("5,6,5,6", 22, 4, [(0, 5), (5, 11), (11, 16), (16, 22)])
        # Invalid partition strings
        with self.assertRaises(ValueError):
            _verify("5,5,5,5,", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        with self.assertRaises(ValueError):
            _verify("5,5,5,a", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Wrong number of partitions
        with self.assertRaises(ValueError):
            _verify("5,5,5", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Wrong number of layers
        with self.assertRaises(ValueError):
            _verify("5,5,5,5", 21, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])


class TestUnevenAutoPartition(unittest.TestCase):
    def test_uneven_auto_partition(self):
        test_cases = [
            # pp_size 2
            (2, 2, 0, (0, 1)),
            (2, 2, 1, (1, 2)),
            (3, 2, 0, (0, 2)),
            (3, 2, 1, (2, 3)),
            # pp_size 3
            (3, 3, 0, (0, 1)),
            (3, 3, 1, (1, 2)),
            (3, 3, 2, (2, 3)),
            (4, 3, 0, (0, 1)),
            (4, 3, 1, (1, 3)),
            (4, 3, 2, (3, 4)),
            (5, 3, 0, (0, 2)),
            (5, 3, 1, (2, 4)),
            (5, 3, 2, (4, 5)),
        ]
        for num_hidden_layers, pp_size, pp_rank, indices in test_cases:
            self.assertEqual(
                indices, get_pp_indices(num_hidden_layers, pp_rank, pp_size)
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
