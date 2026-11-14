"""Real HTTP/model integration for /v2/embed using an existing embedding model.

Run on a supported SGLang inference host. Unlike the CPU route tests, this
launches the full server and compares /v2/embed against /v1/embeddings.
"""

import base64
import copy
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

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")


class TestCohereEmbedding(CustomTestCase):
    model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
    embedding_dimension = 1536
    model_revision = None
    requires_embedding_flag = True
    extra_server_args = []

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, revision=cls.model_revision
        )
        server_args = ["--context-length", "256"]
        if cls.requires_embedding_flag:
            server_args.append("--is-embedding")
        if cls.model_revision is not None:
            server_args.extend(["--revision", cls.model_revision])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args + cls.extra_server_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def post(self, path, body):
        response = requests.post(self.base_url + path, json=body, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIsInstance(payload, dict)
        self.assertNotIn("error", payload)
        self.assertNotEqual(payload.get("object"), "error")
        return payload

    def assert_vectors_close(self, actual, expected, *, batch_size, dimension=None):
        dimension = dimension or self.embedding_dimension
        matrices = []
        for values in (actual, expected):
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), batch_size)
            for row in values:
                self.assertIsInstance(row, list)
                self.assertEqual(len(row), dimension)
                self.assertTrue(all(type(value) in (int, float) for value in row))
            matrix = np.asarray(values, dtype=np.float64)
            self.assertEqual(matrix.shape, (batch_size, dimension))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertTrue((np.linalg.norm(matrix, axis=-1) > 0).all())
            matrices.append(matrix)
        # Check shape and finiteness explicitly: broadcasting or matching NaNs
        # must not let two incorrect endpoint responses pass the parity check.
        np.testing.assert_allclose(
            matrices[0], matrices[1], atol=1e-5, rtol=1e-5, equal_nan=False
        )

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
        self.assert_vectors_close(
            actual["embeddings"]["float"],
            [item["embedding"] for item in expected["data"]],
            batch_size=len(texts),
        )
        vectors = np.asarray(actual["embeddings"]["float"], dtype=np.float32)
        self.assertEqual(len(actual["embeddings"]["base64"]), len(texts))
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
        for policy, side in (("END", "right"), ("START", "left")):
            with self.subTest(policy=policy):
                tokenizer = copy.deepcopy(self.tokenizer)
                tokenizer.truncation_side = side
                selected = tokenizer(text, truncation=True, max_length=5)["input_ids"]
                actual = self.post(
                    "/v2/embed", {"texts": [text], "truncate": policy, "max_tokens": 5}
                )
                expected = self.post(
                    "/v1/embeddings", {"model": self.model, "input": selected}
                )
                self.assert_vectors_close(
                    actual["embeddings"]["float"],
                    [item["embedding"] for item in expected["data"]],
                    batch_size=1,
                )
                self.assertEqual(actual["meta"]["billed_units"]["input_tokens"], 5)
                if policy == "END":
                    default = self.post("/v2/embed", {"texts": [text], "max_tokens": 5})
                    self.assert_vectors_close(
                        default["embeddings"]["float"],
                        [item["embedding"] for item in expected["data"]],
                        batch_size=1,
                    )
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
            self.assert_vectors_close(
                response["embeddings"]["float"], expected, batch_size=1
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
        self.assert_vectors_close(
            response["embeddings"]["float"], expected, batch_size=1
        )


class TestCohereQwen3Embedding(TestCohereEmbedding):
    model = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension = 1024
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
        self.assert_vectors_close(
            actual["embeddings"]["float"],
            [item["embedding"] for item in expected["data"]],
            batch_size=1,
            dimension=128,
        )


class TestCohereNemotron3Embedding(TestCohereEmbedding):
    model = "nvidia/Nemotron-3-Embed-1B-BF16"
    model_revision = "c0c9fea93ea424587517f2c59e20db9f1d6bf615"
    embedding_dimension = 2048
    # The merged adapter must recognize the checkpoint's embedding capability.
    requires_embedding_flag = False

    def test_task_prefixes_match_existing_endpoint(self):
        texts = [
            "How does GPU memory work?",
            "GPU memory stores tensors during inference.",
        ]
        # These are the exact prompt names and prefixes in the pinned
        # config_sentence_transformers.json; the checkpoint has no chat template.
        self.assertFalse(self.tokenizer.chat_template)
        for input_type, prefix in (("query", "query: "), ("document", "passage: ")):
            with self.subTest(input_type=input_type):
                actual = self.post(
                    "/v2/embed", {"texts": texts, "input_type": input_type}
                )
                prefixed = [prefix + text for text in texts]
                expected = self.post(
                    "/v1/embeddings", {"model": self.model, "input": prefixed}
                )
                self.assert_vectors_close(
                    actual["embeddings"]["float"],
                    [item["embedding"] for item in expected["data"]],
                    batch_size=len(texts),
                )
                self.assertEqual(
                    actual["meta"]["billed_units"]["input_tokens"],
                    sum(len(self.tokenizer.encode(text)) for text in prefixed),
                )

    def test_unsupported_task_prefix_is_rejected(self):
        response = requests.post(
            self.base_url + "/v2/embed",
            json={"texts": ["hello"], "input_type": "search_query"},
            timeout=120,
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Unsupported input_type", response.json()["error"]["message"])


if __name__ == "__main__":
    unittest.main()
