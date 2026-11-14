import json
import signal
import threading
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")

TOKENIZER_WORKER_NUM = 4
NUM_WARMUP_REQUESTS = 32
NUM_STREAMS = 2
MAX_NEW_TOKENS = 8192


class TestMultiTokenizerSigtermDrain(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                str(TOKENIZER_WORKER_NUM),
                "--enable-http-worker-reuseport",
                "--incremental-streaming-output",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _generate(self, max_new_tokens: int, stream: bool) -> requests.Response:
        return requests.post(
            self.base_url + "/generate",
            json={
                "text": "Tell me a very long story.",
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "stream": stream,
            },
            stream=stream,
            timeout=60,
        )

    def _read_stream(self, index: int, chunks_seen: list, results: list) -> None:
        try:
            last_payload = None
            for line in self._generate(MAX_NEW_TOKENS, stream=True).iter_lines():
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:") :].strip()
                if payload == b"[DONE]":
                    results[index] = json.loads(last_payload)
                    return
                last_payload = payload
                chunks_seen[index] += 1
            results[index] = "stream closed before [DONE]"
        except Exception as e:
            results[index] = repr(e)

    def test_inflight_streams_finish_after_sigterm(self):
        """With several tokenizer workers, in-flight streams must run to completion
        after SIGTERM, even while other workers have no requests left to drain."""
        for _ in range(NUM_WARMUP_REQUESTS):
            self.assertEqual(self._generate(1, stream=False).status_code, 200)

        chunks_seen = [0] * NUM_STREAMS
        results = [None] * NUM_STREAMS
        threads = [
            threading.Thread(target=self._read_stream, args=(i, chunks_seen, results))
            for i in range(NUM_STREAMS)
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 60
        while min(chunks_seen) < 10 and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertGreaterEqual(min(chunks_seen), 10)

        self.process.send_signal(signal.SIGTERM)
        for thread in threads:
            thread.join(timeout=300)

        for result in results:
            self.assertIsInstance(result, dict, result)
            self.assertEqual(result["meta_info"]["completion_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(self.process.wait(timeout=120), 0)


if __name__ == "__main__":
    unittest.main()
