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
"""
Unit tests for LoRA support in embedding models.

Validates that EmbeddingReqInput correctly handles LoRA fields through
normalization, batching, and request splitting.
"""

import unittest

from sglang.srt.managers.io_struct import EmbeddingReqInput, TokenizedEmbeddingReqInput
from sglang.srt.entrypoints.openai.protocol import EmbeddingRequest
from sglang.srt.sampling.sampling_params import SamplingParams


class TestEmbeddingLoraSupport(unittest.TestCase):
    """Test LoRA support in embedding request structures."""

    def test_embedding_req_input_lora_fields_exist(self):
        """Test that EmbeddingReqInput has lora_path and lora_id fields."""
        req = EmbeddingReqInput(text="Hello")
        self.assertIsNone(req.lora_path)
        self.assertIsNone(req.lora_id)

    def test_embedding_req_input_lora_path_normalization_single_to_batch(self):
        """Test that single lora_path is expanded to match batch size."""
        req = EmbeddingReqInput(
            text=["Hello", "World", "Test"],
            lora_path="my-adapter",
        )
        req.normalize_batch_and_arguments()

        self.assertEqual(req.lora_path, ["my-adapter", "my-adapter", "my-adapter"])

    def test_embedding_req_input_lora_path_normalization_list_preserved(self):
        """Test that list of lora_paths is preserved during normalization."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_path=["adapter1", "adapter2"],
        )
        req.normalize_batch_and_arguments()

        self.assertEqual(req.lora_path, ["adapter1", "adapter2"])

    def test_embedding_req_input_lora_path_list_mismatch_raises_error(self):
        """Test that mismatched lora_path list length raises ValueError."""
        req = EmbeddingReqInput(
            text=["Hello", "World", "Test"],
            lora_path=["adapter1"],  # Length 1, but batch size is 3
        )

        with self.assertRaises(ValueError) as ctx:
            req.normalize_batch_and_arguments()

        self.assertIn("lora_path list length", str(ctx.exception))

    def test_embedding_req_input_getitem_extracts_lora_fields(self):
        """Test that __getitem__ correctly extracts LoRA fields."""
        req = EmbeddingReqInput(
            text=["Hello", "World"],
            lora_path=["adapter1", "adapter2"],
            lora_id=["id1", "id2"],
        )
        req.normalize_batch_and_arguments()

        item0 = req[0]
        item1 = req[1]

        self.assertEqual(item0.lora_path, "adapter1")
        self.assertEqual(item0.lora_id, "id1")
        self.assertEqual(item1.lora_path, "adapter2")
        self.assertEqual(item1.lora_id, "id2")

    def test_tokenized_embedding_req_input_has_lora_id(self):
        """Test that TokenizedEmbeddingReqInput has lora_id field."""
        sampling_params = SamplingParams()
        tokenized = TokenizedEmbeddingReqInput(
            input_text="Hello",
            input_ids=[1, 2, 3],
            image_inputs={},
            token_type_ids=[],
            sampling_params=sampling_params,
            lora_id="my-lora-id",
        )

        self.assertEqual(tokenized.lora_id, "my-lora-id")

    def test_embedding_request_protocol_has_lora_path(self):
        """Test that EmbeddingRequest API model has lora_path field."""
        request = EmbeddingRequest(
            input="Hello world",
            model="test-model",
            lora_path="my-adapter",
        )

        self.assertEqual(request.lora_path, "my-adapter")


class TestEmbeddingLoraHFComparison(unittest.TestCase):
    """
    Compare HuggingFace+LoRA vs SGLang+LoRA embedding outputs.

    This test requires:
    - A base embedding model
    - A LoRA adapter for that model
    - GPU access

    Skip if dependencies not available.
    """

    @classmethod
    def setUpClass(cls):
        """Check if required dependencies are available."""
        cls.skip_reason = None

        try:
            import torch
            if not torch.cuda.is_available():
                cls.skip_reason = "CUDA not available"
                return
        except ImportError:
            cls.skip_reason = "torch not installed"
            return

        try:
            from peft import PeftModel
        except ImportError:
            cls.skip_reason = "peft not installed"
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            cls.skip_reason = "sentence-transformers not installed"
            return

    def setUp(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def _get_hf_embedding_with_lora(self, model_path, lora_path, texts, torch_dtype):
        """Get embeddings from HuggingFace model with LoRA adapter."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        from peft import PeftModel

        # Load base model
        base_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).cuda()

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()  # Merge for inference
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Get embeddings
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")

            outputs = model(**inputs)
            # Use last hidden state with mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy().tolist()

    def _get_sglang_embedding_with_lora(self, base_url, model_name, lora_name, texts):
        """Get embeddings from SGLang server with LoRA adapter."""
        import requests

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{base_url}/v1/embeddings",
                json={
                    "model": model_name,
                    "input": text,
                    "lora_path": lora_name,
                }
            )
            response.raise_for_status()
            embeddings.append(response.json()["data"][0]["embedding"])

        return embeddings

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_hf_sglang_lora_embedding_similarity(self):
        """
        Test that HF+LoRA and SGLang+LoRA produce similar embeddings.

        This is a manual test that requires:
        1. Setting environment variables for model/adapter paths
        2. Running an SGLang server with LoRA enabled

        Example:
            export EMBEDDING_MODEL_PATH=/path/to/embedding/model
            export LORA_ADAPTER_PATH=/path/to/lora/adapter
            export SGLANG_BASE_URL=http://localhost:30000
            python -m pytest test/srt/test_embedding_lora_support.py -k test_hf_sglang -v
        """
        import os

        model_path = os.environ.get("EMBEDDING_MODEL_PATH")
        lora_path = os.environ.get("LORA_ADAPTER_PATH")
        base_url = os.environ.get("SGLANG_BASE_URL")

        if not all([model_path, lora_path, base_url]):
            self.skipTest(
                "Set EMBEDDING_MODEL_PATH, LORA_ADAPTER_PATH, and SGLANG_BASE_URL "
                "environment variables to run this test"
            )

        import torch

        test_texts = [
            "Hello world",
            "This is a test sentence for embedding comparison",
        ]

        # Get HF embeddings
        hf_embeddings = self._get_hf_embedding_with_lora(
            model_path, lora_path, test_texts, torch.float16
        )

        # Get SGLang embeddings
        model_name = os.path.basename(model_path)
        lora_name = os.path.basename(lora_path)
        sglang_embeddings = self._get_sglang_embedding_with_lora(
            base_url, model_name, lora_name, test_texts
        )

        # Compare embeddings
        for i, (hf_emb, sgl_emb) in enumerate(zip(hf_embeddings, sglang_embeddings)):
            similarity = self._cosine_similarity(hf_emb, sgl_emb)
            print(f"Text {i}: cosine similarity = {similarity:.6f}")

            # Embeddings should be very similar (cosine > 0.99)
            self.assertGreater(
                similarity, 0.99,
                f"Embedding {i} similarity {similarity:.4f} below threshold"
            )


if __name__ == "__main__":
    unittest.main()
