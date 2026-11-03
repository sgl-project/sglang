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
from sglang.test.vlm_utils import IMAGE_MAN_IRONING_URL

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-large")


class TestNemotron35Rerank(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        response = requests.get(IMAGE_MAN_IRONING_URL, timeout=30)
        response.raise_for_status()
        image = {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,"
                + base64.b64encode(response.content).decode()
            },
        }
        cls.query = "What is the man doing?"
        cls.image = image
        text = "A man is ironing clothes on the back of a car."
        cls.documents = [
            [{"type": "text", "text": text}, image],
            text,
            [image],
            "A biological foundation model designed to analyze and generate DNA, RNA, and protein sequences.",
        ]
        # Same-checkpoint Transformers reference; scores may vary slightly by GPU.
        cls.reference = [-1.954972, -2.034311, -2.185751, -3.345560]
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
                "query": self.query,
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

    def test_literal_control_strings_match_reference(self):
        for text, expected in [
            ("see <s>old</s> price", -2.853922),
            ("[IMG] figure", -2.661679),
        ]:
            with self.subTest(text=text):
                response = requests.post(
                    self.base_url + "/v1/rerank",
                    json={
                        "query": "What is shown?",
                        "documents": [[{"type": "text", "text": text}, self.image]],
                    },
                    timeout=120,
                )
                self.assertEqual(response.status_code, 200, response.text)
                results = response.json()
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["index"], 0)
                self.assertAlmostEqual(
                    results[0]["score"], expected, delta=0.05 + 0.02 * abs(expected)
                )

    def test_unsupported_image_inputs_return_http_400(self):
        for query, document, message in [
            ([self.image], "plain text", "Queries must be text"),
            (self.query, [self.image, self.image], "At most one image per document"),
        ]:
            with self.subTest(message=message):
                response = requests.post(
                    self.base_url + "/v1/rerank",
                    json={"query": query, "documents": [document]},
                    timeout=120,
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(message, response.json()["message"])

    def test_distinct_concurrent_requests_match_batched_scores(self):
        batch = {x["index"]: x["score"] for x in self.request(self.documents)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda doc: self.request([doc]), self.documents))
        for index, result in enumerate(results):
            self.assertAlmostEqual(result[0]["score"], batch[index], delta=0.02)


if __name__ == "__main__":
    unittest.main()
