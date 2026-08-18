# Copyright 2023-2026 SGLang Team
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

"""LoRA embedding parity against an independent Hugging Face oracle."""

import multiprocessing as mp
import unittest

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

register_cuda_ci(est_time=150, stage="nightly", runner_config="1-gpu-large")

MODEL_PATH = "meta-llama/Llama-2-7b-hf"
LORA_PATH = "yushengsu/sglang_lora_logprob_diff_without_tuning"
SIMILARITY_THRESHOLD = 0.9999


class TestEmbeddingLoRAParity(CustomTestCase):
    """Guard the end-to-end embedding request and LoRA execution path."""

    @staticmethod
    def _hf_embeddings(texts):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda()
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            hidden_states = model.model(
                **inputs, output_hidden_states=True
            ).hidden_states[-1]
            last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
            embeddings = hidden_states[
                torch.arange(hidden_states.shape[0], device="cuda"),
                last_token_indices,
            ]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        result = embeddings.cpu().numpy()
        del model, base_model
        torch.cuda.empty_cache()
        return result

    @staticmethod
    def _sglang_embeddings(texts):
        def extract_embeddings(response):
            if not isinstance(response, list):
                response = [response]
            return np.asarray([item["embedding"] for item in response])

        with SRTRunner(
            MODEL_PATH,
            torch_dtype=torch.float16,
            model_type="embedding",
            lora_paths=[LORA_PATH],
            lora_backend="triton",
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
            trust_remote_code=True,
            mem_fraction_static=0.88,
        ) as runner:
            base_response = runner.engine.encode(prompt=texts, lora_path=None)
            lora_response = runner.engine.encode(prompt=texts, lora_path=LORA_PATH)

        return extract_embeddings(base_response), extract_embeddings(lora_response)

    def test_hf_sglang_embedding_similarity(self):
        """Dropping LoRA at any embedding handoff must fail external parity."""
        texts = [
            "Hello world",
            "This is a test sentence for embedding comparison",
        ]

        base_embeddings, sglang_embeddings = self._sglang_embeddings(texts)
        self.assertFalse(
            np.allclose(base_embeddings, sglang_embeddings, rtol=1e-4, atol=1e-5),
            "The requested adapter had no observable effect on embeddings",
        )
        torch.cuda.empty_cache()
        hf_embeddings = self._hf_embeddings(texts)

        self.assertEqual(sglang_embeddings.shape, hf_embeddings.shape)
        similarities = np.sum(hf_embeddings * sglang_embeddings, axis=1) / (
            np.linalg.norm(hf_embeddings, axis=1)
            * np.linalg.norm(sglang_embeddings, axis=1)
        )
        np.testing.assert_array_less(
            np.full_like(similarities, SIMILARITY_THRESHOLD), similarities
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main()
