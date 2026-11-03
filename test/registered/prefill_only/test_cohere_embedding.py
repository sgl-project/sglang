"""Real HTTP/model integration for /v2/embed using an existing embedding model.

Run on a supported SGLang inference host. Unlike the CPU route tests, this
launches the full server and compares /v2/embed against /v1/embeddings.
"""

import base64
import json
import unittest

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=220, stage="base-b", runner_config="1-gpu-small")


class TestCohereEmbedding(CustomTestCase):
    model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
    extra_server_args = []

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--is-embedding", "--context-length", "256"]
            + cls.extra_server_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def post(self, path, body):
        response = requests.post(self.base_url + path, json=body, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_batch_and_all_representations_match_existing_endpoint(self):
        texts = ["A red apple on the table.", "A spaceship approaches Mars."]
        actual = self.post(
            "/v2/embed",
            {
                "texts": texts,
                "embedding_types": ["float", "base64", "binary", "ubinary"],
            },
        )
        expected = self.post("/v1/embeddings", {"model": self.model, "input": texts})
        vectors = np.asarray(actual["embeddings"]["float"], dtype=np.float32)
        np.testing.assert_allclose(
            vectors,
            [item["embedding"] for item in expected["data"]],
            atol=1e-5,
            rtol=1e-5,
        )
        for row, encoded in zip(vectors, actual["embeddings"]["base64"]):
            np.testing.assert_array_equal(
                row, np.frombuffer(base64.b64decode(encoded), dtype="<f4")
            )
        bits = np.packbits(vectors >= 0, axis=-1, bitorder="big")
        self.assertEqual(actual["embeddings"]["ubinary"], bits.tolist())
        self.assertEqual(
            actual["embeddings"]["binary"], (bits.astype(np.int16) - 128).tolist()
        )
        self.assertEqual(actual["texts"], texts)
        self.assertEqual(
            actual["meta"]["billed_units"]["input_tokens"],
            expected["usage"]["prompt_tokens"],
        )

    def test_truncation_matches_explicit_token_ids(self):
        text = "The quick brown fox jumps over the lazy dog near the river."
        ids = self.tokenizer.encode(text)
        self.assertGreater(len(ids), 5)
        for policy, selected in (("END", ids[:5]), ("START", ids[-5:])):
            with self.subTest(policy=policy):
                actual = self.post(
                    "/v2/embed", {"texts": [text], "truncate": policy, "max_tokens": 5}
                )
                expected = self.post(
                    "/v1/embeddings", {"model": self.model, "input": selected}
                )
                np.testing.assert_allclose(
                    actual["embeddings"]["float"][0],
                    expected["data"][0]["embedding"],
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertEqual(actual["meta"]["billed_units"]["input_tokens"], 5)
        response = requests.post(
            self.base_url + "/v2/embed",
            json={"texts": [text], "truncate": "NONE", "max_tokens": 5},
            timeout=120,
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["code"], 400)

    def test_request_validation_and_existing_api_errors(self):
        # Exercise errors before the endpoint, shared strict JSON parsing, and
        # app-level CORS middleware on the actual running inference server.
        headers = {"content-type": "application/json", "origin": "https://example.com"}
        for payload in ('{"texts":[]}', "{", '{"texts":["hello"],"unused":NaN}'):
            with self.subTest(payload=payload):
                response = requests.post(
                    self.base_url + "/v2/embed",
                    data=payload,
                    headers=headers,
                    timeout=120,
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["error"]["code"], 400)
                self.assertEqual(response.json()["error"]["type"], "BadRequestError")
                self.assertIn(
                    response.headers["access-control-allow-origin"],
                    ("*", "https://example.com"),
                )
        response = requests.post(self.base_url + "/v1/embeddings", json={}, timeout=120)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["object"], "error")
        self.assertNotIn("error", response.json())

    def test_token_budget_edge_cases(self):
        text = "A small input."
        expected = self.post("/v2/embed", {"texts": [text]})["embeddings"]["float"]
        for policy in ("END", "START"):
            response = self.post(
                "/v2/embed", {"texts": [text], "truncate": policy, "max_tokens": -1}
            )
            np.testing.assert_allclose(
                response["embeddings"]["float"], expected, atol=1e-5, rtol=1e-5
            )
        for policy, budget in (
            ("END", 0),
            ("START", 0),
            ("NONE", 0),
            ("NONE", -1),
            ("END", 100000),
        ):
            response = requests.post(
                self.base_url + "/v2/embed",
                json={"texts": [text], "truncate": policy, "max_tokens": budget},
                timeout=120,
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(response.json()["error"]["code"], 400)
        response = self.post(
            "/v2/embed", {"texts": [text], "truncate": "NONE", "max_tokens": 100000}
        )
        np.testing.assert_allclose(
            response["embeddings"]["float"], expected, atol=1e-5, rtol=1e-5
        )


class TestCohereQwen3Embedding(TestCohereEmbedding):
    model = "Qwen/Qwen3-Embedding-0.6B"
    # As in test_openai_embedding.py, declare the capability explicitly: this
    # SGLang baseline reads these flags from config instead of model cards.
    extra_server_args = [
        "--json-model-override-args",
        json.dumps({"is_matryoshka": True, "matryoshka_dimensions": [128]}),
    ]

    def test_output_dimension_uses_existing_pooler(self):
        actual = self.post(
            "/v2/embed", {"texts": ["A red apple."], "output_dimension": 128}
        )
        expected = self.post(
            "/v1/embeddings",
            {"model": self.model, "input": ["A red apple."], "dimensions": 128},
        )
        vector = actual["embeddings"]["float"][0]
        self.assertEqual(len(vector), 128)
        np.testing.assert_allclose(
            vector, expected["data"][0]["embedding"], atol=1e-5, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
