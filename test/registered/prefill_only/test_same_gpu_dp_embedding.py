"""E2E test for same-GPU data parallelism on the embedding path.

Launches one server with two colocated DP workers (--dp-size 2
--gpu-id-step 0) and checks the built-in load balancer serves embedding
requests correctly. Functional only: CI runners have no CUDA MPS daemon, so
workers share the GPU by time-slicing here; the throughput benefits of MPS
are documented in the PR, not asserted in CI.
"""

import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


class TestSameGpuDpEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        # No api_key: multi-tokenizer mode (--tokenizer-worker-num > 1)
        # asserts that API keys are unsupported.
        cls.api_key = "unused"
        other_args = [
            "--is-embedding",
            "--dp-size",
            "2",
            "--gpu-id-step",
            "0",
            "--max-total-tokens",
            "32768",
            "--mem-fraction-static",
            "0.35",
            # The recommended combination: without extra tokenizer workers the
            # shared frontend caps colocated-DP throughput. Also exercises the
            # multi-tokenizer BatchEmbeddingOutput repack fixed in this change.
            "--tokenizer-worker-num",
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

    def test_server_reports_dp_size(self):
        info = requests.get(f"{self.base_url}/server_info").json()
        self.assertEqual(info["dp_size"], 2)
        self.assertEqual(info["gpu_id_step"], 0)

    def test_embeddings_across_workers(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        # More requests than workers so round-robin exercises both.
        texts = [f"same-gpu dp request {i}" for i in range(8)]
        dims = set()
        for text in texts:
            resp = client.embeddings.create(model=self.model, input=text)
            self.assertEqual(len(resp.data), 1)
            dims.add(len(resp.data[0].embedding))
        self.assertEqual(len(dims), 1, "all workers must produce same-dim embeddings")

    def test_batch_embedding(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        resp = client.embeddings.create(
            model=self.model, input=["batched one", "batched two", "batched three"]
        )
        self.assertEqual(len(resp.data), 3)


if __name__ == "__main__":
    unittest.main()
