import json
import os
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-c", runner_config="4-gpu-h100")


class TestDSparkDPSpecPrefillCoordination(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get("SGLANG_TEST_DSPARK_COORDINATION_URL")
        cls.process = None
        cls.model = os.environ.get("SGLANG_TEST_DSPARK_MODEL", "Qwen/Qwen3-14B")
        if cls.base_url is None:
            cls.base_url = DEFAULT_URL_FOR_TEST
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--speculative-algorithm",
                    "DSPARK",
                    "--speculative-draft-model-path",
                    "deepseek-ai/dspark_qwen3_14b_block7",
                    "--tp-size",
                    "2",
                    "--dp-size",
                    "2",
                    "--enable-dp-attention",
                    "--enable-dp-lm-head",
                    "--attention-backend",
                    "trtllm_mha" if is_sm100_supported() else "fa3",
                    "--speculative-draft-attention-backend",
                    "fa4" if is_sm100_supported() else "fa3",
                    "--page-size",
                    "1",
                    "--mem-fraction-static",
                    "0.75",
                    "--cuda-graph-max-bs-decode",
                    "4",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--enable-mixed-chunk",
                    "--chunked-prefill-size",
                    "16384",
                ],
                env={
                    "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION": "1",
                    "SGLANG_RAGGED_VERIFY_MODE": "static",
                    "SGLANG_SIMULATE_ACC_LEN": "-1",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1",
                },
            )
        response = requests.get(cls.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        cls.incremental_streaming = response.json()["incremental_streaming_output"]
        cls.prompt = "Count from 1 to 1000, separated by commas.\n1, 2, 3,"

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def generate(self, prompt, rank, tokens, started=None):
        payload = {
            "text": prompt,
            "routed_dp_rank": rank,
            "stream": started is not None,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
        }
        with requests.post(
            self.base_url + "/generate",
            json=payload,
            stream=started is not None,
            timeout=300,
        ) as response:
            response.raise_for_status()
            if started is None:
                result = response.json()
            else:
                result = None
                chunks = []
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    result = json.loads(data)
                    chunks.append(result["text"])
                    if result.get("meta_info", {}).get("completion_tokens", 0) >= 64:
                        started.set()
                self.assertIsNotNone(result)
                if self.incremental_streaming:
                    result["text"] = "".join(chunks)
        self.assertEqual(result["meta_info"]["completion_tokens"], tokens)
        self.assertTrue(result["text"])
        return result["text"], time.monotonic()

    def test_mixed_prefill_and_speculative_peer_match_isolated_outputs(self):
        expected = {
            rank: self.generate(self.prompt, rank, 1024)[0] for rank in range(2)
        }
        for prefill_rank in range(2):
            long_prompt = (
                f"Fresh context {uuid.uuid4().hex}:\n"
                + "\n".join(
                    f"Entry {i}: apples are fruit; water is liquid; the sky appears blue."
                    for i in range(1536)
                )
                + "\nEnd of context.\n"
                + self.prompt
            )
            started = [threading.Event(), threading.Event()]
            with self.subTest(prefill_rank=prefill_rank), ThreadPoolExecutor(3) as pool:
                decodes = [
                    pool.submit(self.generate, self.prompt, rank, 1024, started[rank])
                    for rank in range(2)
                ]
                for event in started:
                    self.assertTrue(event.wait(90), "Decode did not start")
                self.assertTrue(all(not f.done() for f in decodes))
                prefill_started = time.monotonic()
                prefill = pool.submit(self.generate, long_prompt, prefill_rank, 32)
                results = [f.result(timeout=300) for f in decodes]
                prefill_output, _ = prefill.result(timeout=300)
                for rank, (output, completed) in enumerate(results):
                    self.assertGreater(completed, prefill_started)
                    self.assertEqual(output, expected[rank])
                self.assertEqual(
                    prefill_output, self.generate(long_prompt, prefill_rank, 32)[0]
                )
                for rank in range(2):
                    self.assertEqual(
                        self.generate(self.prompt, rank, 1024)[0], expected[rank]
                    )


if __name__ == "__main__":
    unittest.main()
