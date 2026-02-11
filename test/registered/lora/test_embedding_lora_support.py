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

import multiprocessing as mp
import unittest

import numpy as np
import torch

from sglang.srt.entrypoints.openai.protocol import EmbeddingRequest
from sglang.srt.managers.io_struct import EmbeddingReqInput, TokenizedEmbeddingReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

# Test configuration (same model/LoRA as test_lora_hf_sgl_logprob_diff.py)
MODEL_PATH = "meta-llama/Llama-2-7b-hf"
LORA_PATH = "yushengsu/sglang_lora_logprob_diff_without_tuning"
LORA_BACKEND = "triton"
SIMILARITY_THRESHOLD = 0.9999

register_cuda_ci(
    est_time=150,
    suite="nightly-1-gpu",
)


class TestEmbeddingLoraSupport(unittest.TestCase):
    """Test LoRA support in embedding request structures."""

    def test_embedding_lora_fields(self):
        """Test LoRA fields exist and work correctly across all embedding structures."""
        # EmbeddingReqInput: fields exist, normalization expands single to batch, indexing works
        req = EmbeddingReqInput(
            text=["Hello", "World"], lora_path="my-adapter", lora_id=["id1", "id2"]
        )
        self.assertIsNotNone(req.lora_path)
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, ["my-adapter", "my-adapter"])
        self.assertEqual(req[0].lora_path, "my-adapter")
        self.assertEqual(req[1].lora_id, "id2")

        # EmbeddingReqInput: mismatched list length raises error
        req = EmbeddingReqInput(text=["Hello", "World", "Test"], lora_path=["adapter1"])
        with self.assertRaises(ValueError):
            req.normalize_batch_and_arguments()

        # TokenizedEmbeddingReqInput and EmbeddingRequest have lora fields
        tokenized = TokenizedEmbeddingReqInput(
            input_text="Hello",
            input_ids=[1, 2, 3],
            image_inputs={},
            token_type_ids=[],
            sampling_params=SamplingParams(),
            lora_id="my-lora-id",
        )
        self.assertEqual(tokenized.lora_id, "my-lora-id")
        self.assertEqual(
            EmbeddingRequest(
                input="Hello", model="test", lora_path="adapter"
            ).lora_path,
            "adapter",
        )


class TestEmbeddingLoraHFComparison(CustomTestCase):
    """Compare HF+LoRA vs SGLang+LoRA embedding outputs."""

    @classmethod
    def get_hf_embedding_with_lora(cls, model_path, lora_path, texts, torch_dtype):
        """Get embeddings from HuggingFace model with LoRA adapter."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load base model as CausalLM to match adapter's expected structure
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).cuda()

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")

            # Access the inner model (CausalLM wraps the base model)
            outputs = model.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            # Last token pooling with L2 normalization (matching SGLang)
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[
                torch.arange(batch_size, device="cuda"), last_token_indices
            ]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Cleanup
        del model, base_model
        torch.cuda.empty_cache()

        return embeddings.cpu().numpy()

    @classmethod
    def get_sglang_embedding_with_lora(cls, model_path, lora_path, texts, torch_dtype):
        """Get embeddings from SGLang with LoRA adapter."""
        with SRTRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
            lora_paths=[lora_path],
            lora_backend=LORA_BACKEND,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
            trust_remote_code=True,
            mem_fraction_static=0.88,
        ) as runner:
            # Call engine.encode directly with lora_path
            response = runner.engine.encode(prompt=texts, lora_path=lora_path)
            if isinstance(response, list):
                embeddings = [r["embedding"] for r in response]
            else:
                embeddings = [response["embedding"]]

        return np.array(embeddings)

    @staticmethod
    def cosine_similarity(a, b):
        """Compute cosine similarity between vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def test_embedding_lora_hf_sglang_similarity(self):
        """Test that HF+LoRA and SGLang+LoRA produce similar embeddings."""
        test_texts = [
            "Hello world",
            "This is a test sentence for embedding comparison",
        ]

        print(f"\nModel: {MODEL_PATH}")
        print(f"LoRA: {LORA_PATH}")

        # Get SGLang embeddings first (before HF loads model into GPU)
        # This order matches test_lora_hf_sgl_logprob_diff.py and avoids OOM
        print("\nGetting SGLang embeddings...")
        sglang_embeddings = self.get_sglang_embedding_with_lora(
            MODEL_PATH, LORA_PATH, test_texts, torch.float16
        )

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Get HF embeddings
        print("Getting HF embeddings...")
        hf_embeddings = self.get_hf_embedding_with_lora(
            MODEL_PATH, LORA_PATH, test_texts, torch.float16
        )

        # Compare embeddings
        print("\nHF vs SGLang LoRA Embedding Comparison:")
        similarities = []
        for i, (hf_emb, sgl_emb) in enumerate(zip(hf_embeddings, sglang_embeddings)):
            sim = self.cosine_similarity(hf_emb, sgl_emb)
            similarities.append(sim)
            print(f"  Text {i}: cosine similarity = {sim:.6f}")
            self.assertGreater(
                sim,
                SIMILARITY_THRESHOLD,
                f"Text {i} similarity {sim:.6f} below threshold {SIMILARITY_THRESHOLD}",
            )

        avg_similarity = np.mean(similarities)
        print(f"  Average similarity: {avg_similarity:.6f}")
        print(f"  Threshold: {SIMILARITY_THRESHOLD}")

        self.assertGreater(
            avg_similarity,
            SIMILARITY_THRESHOLD,
            f"Average similarity {avg_similarity:.4f} below threshold {SIMILARITY_THRESHOLD}",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main()
