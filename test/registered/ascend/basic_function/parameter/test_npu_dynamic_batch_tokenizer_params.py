import logging
import threading
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_WEIGHTS_PATH,
    send_concurrent_requests,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


BASE_OTHER_ARGS = [
    "--chunked-prefill-size",
    "256",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    "0.8",
    "--tp-size",
    "4",
    "--enable-dynamic-batch-tokenizer",
    "--log-level",
    "debug",
]


def launch_server_with_params(
    model_name, base_url, batch_size, batch_timeout, extra_args=None
):
    """set batch_size  batch_timeout"""
    other_args = BASE_OTHER_ARGS.copy()
    other_args.extend(
        [
            "--dynamic-batch-tokenizer-batch-size",
            str(batch_size),
            "--dynamic-batch-tokenizer-batch-timeout",
            str(batch_timeout),
        ]
    )
    if extra_args:
        other_args.extend(extra_args)
    return popen_launch_server(
        model_name,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )


class TestBatchSize64Timeout0p001(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_params(
            QWEN3_32B_WEIGHTS_PATH,
            cls.base_url,
            batch_size=64,
            batch_timeout=0.001,
            extra_args=["--disable-radix-cache"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mixed_text_lengths(self):
        short = ["Hi", "OK", "Yes"]
        medium = [
            ("What is the capital of France?", "Paris"),
            ("Explain what a neural network is.", "neural"),
            ("Describe the water cycle briefly.", "water"),
        ]
        long = [
            (
                "Describe the history of the Roman Empire and its influence on modern culture "
                * 3,
                "Roman",
            ),
            (
                "Describe the role of the United Nations in maintaining international peace and security "
                * 3,
                "United Nations",
            ),
        ]
        all_prompts = short + medium + long
        results, lock = [], threading.Lock()

        def send(item):
            if isinstance(item, tuple):
                prompt, keyword = item
            else:
                prompt, keyword = item, None
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                    timeout=60,
                )
                with lock:
                    results.append((resp.status_code, resp.text, keyword))
            except Exception as e:
                with lock:
                    results.append((-1, str(e), keyword))

        threads = [threading.Thread(target=send, args=(item,)) for item in all_prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for status_code, text, keyword in results:
            self.assertEqual(
                status_code, 200, f"Request failed with status {status_code}"
            )
            if keyword:
                self.assertIn(
                    keyword, text, f"Keyword '{keyword}' not found in: {text[:200]}..."
                )

    def test_streaming_requests(self):
        prompts = [
            "The capital of France is",
            "The largest planet is",
            "The speed of light is",
        ]
        results, lock = [], threading.Lock()

        def send_stream(p):
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": p,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                        "stream": True,
                    },
                    stream=True,
                    timeout=60,
                )
                has_content = any(
                    line and line.startswith(b"data: ") and line[6:] != b"[DONE]"
                    for line in resp.iter_lines()
                )
                with lock:
                    results.append((resp.status_code, has_content))
            except Exception:
                with lock:
                    results.append((-1, False))

        threads = [threading.Thread(target=send_stream, args=(p,)) for p in prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for code, has in results:
            self.assertEqual(code, 200, "Streaming request non-200")
            self.assertTrue(has, "Streaming response no content")

    def test_different_sampling_params(self):
        configs = [
            {"temperature": 0.0, "max_new_tokens": 32},
            {"temperature": 0.7, "max_new_tokens": 32},
            {"temperature": 1.0, "max_new_tokens": 32},
            {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 32},
        ]
        payloads = configs * 5
        results, lock = [], threading.Lock()

        def send(sp):
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={"text": "The capital of France is", "sampling_params": sp},
                    timeout=60,
                )
                with lock:
                    results.append(resp.status_code)
            except Exception:
                with lock:
                    results.append(-1)

        threads = [threading.Thread(target=send, args=(p,)) for p in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        success = sum(1 for s in results if s == 200)
        self.assertEqual(
            success,
            len(payloads),
            f"Sampling params: {success}/{len(payloads)} succeeded",
        )

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            max_new_tokens=512,
        )
        metrics = run_eval(args)
        accuracy = 0.86
        self.assertGreaterEqual(
            metrics["score"],
            accuracy,
            f"GSM8K accuracy {metrics['score']} < {accuracy}",
        )
        logger.info(
            f"GSM8K accuracy with batch_size=64 timeout=0.001: {metrics['score']}"
        )
        logger.info(
            f"GSM8K latency: {metrics['latency']:.2f}s, throughput: {metrics['output_throughput']:.2f} tok/s"
        )


class TestBatchSize1Timeout0p005(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_server_with_params(
            QWEN3_32B_WEIGHTS_PATH,
            cls.base_url,
            batch_size=1,
            batch_timeout=0.005,
            extra_args=["--disable-radix-cache"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_high_concurrency(self):
        results = send_concurrent_requests(
            self.base_url, num_requests=20, num_concurrent=4
        )
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, 20, f"Expected 20 successes, got {success}")
        for r in results:
            self.assertIn("Paris", r["text"])

    def test_disable_radix_cache(self):
        results = send_concurrent_requests(
            self.base_url, num_requests=20, num_concurrent=8
        )
        success = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(success, 20, f"Radix cache disabled: {success}/20 succeeded")
        for r in results:
            self.assertIn("Paris", r["text"])

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            max_new_tokens=512,
        )
        metrics = run_eval(args)
        accuracy = 0.86
        self.assertGreaterEqual(
            metrics["score"],
            accuracy,
            f"GSM8K accuracy {metrics['score']} < {accuracy}",
        )
        logger.info(
            f"GSM8K accuracy with batch_size=1 timeout=0.005: {metrics['score']}"
        )
        logger.info(
            f"GSM8K latency: {metrics['latency']:.2f}s, throughput: {metrics['output_throughput']:.2f} tok/s"
        )


if __name__ == "__main__":
    unittest.main()
