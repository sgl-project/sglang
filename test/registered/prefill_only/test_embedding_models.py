# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import base64
import json
import multiprocessing as mp
import random
import unittest
from typing import Optional

import numpy as np
import requests
import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_similarities,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Embedding model tests
register_amd_ci(
    est_time=73,
    suite="base-b-test-1-gpu-small-amd",
    disabled="see https://github.com/sgl-project/sglang/issues/11127",
)
register_cuda_ci(est_time=465, stage="base-b", runner_config="1-gpu-small")

MODEL_TO_CONFIG = {
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": (1, 1e-5),
    "intfloat/e5-mistral-7b-instruct": (1, 1e-5),
    "Qwen/Qwen3-Embedding-8B": (1, 1e-5),
    # Temporarily disable: HF reference path in runners.py runs this Qwen2-VL
    # fine-tune with bidirectional attention (the non-sentence-transformers
    # branch in _get_sentence_transformer_embedding_model does not pass
    # is_causal=True), while SGLang's Qwen2-VL embedding is always causal —
    # producing ~0.30 cosine diffs vs HF on short prompts.
    # See https://github.com/sgl-project/sglang/actions/runs/25224929325/job/73966043206
    # "marco/mcdse-2b-v1": (1, 1e-5),
    # Temporarily disable before this model is fixed
    # "jason9693/Qwen2.5-1.5B-apeach": (1, 1e-5),
}
MODELS = [(key, *MODEL_TO_CONFIG[key]) for key in MODEL_TO_CONFIG]

TORCH_DTYPES = [torch.float16]


class TestEmbeddingModels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        config = AutoConfig.from_pretrained(model_path)
        max_length = getattr(config, "max_position_embeddings", 2048)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        truncated_prompts = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if len(tokens.input_ids[0]) > max_length:
                truncated_text = tokenizer.decode(
                    tokens.input_ids[0][: max_length - 1], skip_special_tokens=True
                )
                truncated_prompts.append(truncated_text)
            else:
                truncated_prompts.append(prompt)
        return truncated_prompts

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        prefill_tolerance,
        matryoshka_dim: Optional[int] = None,
    ) -> None:
        truncated_prompts = self._truncate_prompts(prompts, model_path)

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
            matryoshka_dim=matryoshka_dim,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        attention_backend = "triton" if is_in_amd_ci() else None
        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend=attention_backend,
            json_model_override_args=(
                {"matryoshka_dimensions": [matryoshka_dim]} if matryoshka_dim else None
            ),
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                truncated_prompts, dimensions=matryoshka_dim
            )

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            print("similarity diff", abs(similarity - 1))

            if len(prompts[i]) <= 1000:
                assert torch.all(abs(similarity - 1) < prefill_tolerance), (
                    "embeddings are not all close"
                )

    def test_prefill_logits(self):
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS, model, tp_size, torch_dtype, prefill_tolerance
                )

    def test_matryoshka_embedding(self):
        models_to_test = [
            (
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                *MODEL_TO_CONFIG["Alibaba-NLP/gte-Qwen2-1.5B-instruct"],
            )
        ]

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS,
                    model,
                    tp_size,
                    torch_dtype,
                    prefill_tolerance,
                    matryoshka_dim=128,
                )

    def test_embedding_encoding_format(self):
        model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        tp_size = MODEL_TO_CONFIG[model][0]
        torch_dtype = torch.float16
        prompts = self._truncate_prompts(DEFAULT_PROMPTS, model)

        with SRTRunner(
            model,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as srt_runner:
            default_outputs = srt_runner.forward(prompts)
            tensor_outputs = srt_runner.forward(prompts, encoding_format="tensor")

        for default_emb, tensor_emb in zip(
            default_outputs.embed_logits, tensor_outputs.embed_logits
        ):
            self.assertNotIsInstance(default_emb, torch.Tensor)
            self.assertIsInstance(tensor_emb, torch.Tensor)
            self.assertEqual(tensor_emb.dtype, torch_dtype)
            similarity = get_similarities(default_emb, tensor_emb.float())
            self.assertTrue(torch.all(abs(similarity - 1) < 1e-5))


class _CohereEmbeddingServer(CustomTestCase):
    model = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST
    embedding_dimension = 1536
    model_revision = None
    requires_embedding_flag = True
    extra_server_args = []

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model, revision=cls.model_revision
        )
        # Keep cache and scheduling conditions consistent for strict endpoint parity.
        server_args = [
            "--context-length",
            "256",
            "--disable-radix-cache",
            "--max-running-requests",
            "1",
        ]
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
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

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


class TestCohereEmbedding(_CohereEmbeddingServer):
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


class TestCohereQwen3Embedding(_CohereEmbeddingServer):
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


class TestCohereNemotron3Embedding(_CohereEmbeddingServer):
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


if __name__ == "__main__":
    unittest.main()
