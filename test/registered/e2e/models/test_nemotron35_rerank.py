import base64
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-large")


class TestNemotron35Rerank(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        response = requests.get(
            "https://docs.nvidia.com/nemo-framework/user-guide/24.07/_images/nemo-llm-mm-stack.png",
            timeout=30,
        )
        response.raise_for_status()
        image = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(response.content).decode()
            },
        }
        text = "NeMo Framework offers comprehensive functionalities for developing both LLMs and Multimodal Models."
        cls.documents = [
            [{"type": "text", "text": text}, image],
            text,
            [image],
            "A biological foundation model designed to analyze and generate DNA, RNA, and protein sequences.",
        ]
        # Same-checkpoint Transformers reference; scores may vary slightly by GPU.
        cls.reference = [-1.801326, -2.107775, -1.818881, -2.956283]
        cls.process = popen_launch_server(
            os.environ.get(
                "SGLANG_TEST_NEMOTRON_RERANK_MODEL",
                "nvidia/Nemotron-3.5-Rerank-8B-BF16",
            ),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--revision",
                "f88f8d9104726747791283ab156d2fcdb29e90a7",
                "--context-length",
                "32768",
                "--max-running-requests",
                "4",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def request(self, documents, **kwargs):
        response = requests.post(
            self.base_url + "/v1/rerank",
            json={
                "query": "What is NVIDIA NeMo Framework?",
                "documents": documents,
                **kwargs,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), len(documents))
        self.assertEqual(
            {item["index"] for item in results}, set(range(len(documents)))
        )
        return results

    def test_reference_scores_ranking_and_multimodal_documents(self):
        results = self.request(self.documents, return_documents=True)
        by_index = {item["index"]: item for item in results}
        self.assertGreater(
            min(by_index[index]["score"] for index in (0, 1, 2)),
            by_index[3]["score"],
        )
        for item in results:
            expected = self.reference[item["index"]]
            self.assertAlmostEqual(
                item["score"], expected, delta=0.05 + 0.02 * abs(expected)
            )
        self.assertIsInstance(by_index[0]["document"], list)
        self.assertEqual(by_index[0]["document"][1]["type"], "image_url")
        flushed = requests.post(self.base_url + "/flush_cache", timeout=120)
        self.assertEqual(flushed.status_code, 200, flushed.text)
        after_flush = self.request(self.documents, return_documents=False)
        for after in after_flush:
            self.assertAlmostEqual(
                by_index[after["index"]]["score"], after["score"], delta=0.02
            )

    def test_distinct_concurrent_requests_match_batched_scores(self):
        batch = {x["index"]: x["score"] for x in self.request(self.documents)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda doc: self.request([doc]), self.documents))
        for index, result in enumerate(results):
            self.assertAlmostEqual(result[0]["score"], batch[index], delta=0.02)


if __name__ == "__main__":
    unittest.main()
