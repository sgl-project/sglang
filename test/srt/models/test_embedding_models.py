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

import multiprocessing as mp
import random
import unittest

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities, is_in_ci

MODELS = [
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1, 1e-5),
    ("intfloat/e5-mistral-7b-instruct", 1, 1e-5),
    ("marco/mcdse-2b-v1", 1, 1e-5),
    ("Qwen/Qwen3-Embedding-8B", 1, 1e-5),
    # Temporarily disable before this model is fixed
    # ("jason9693/Qwen2.5-1.5B-apeach", 1, 1e-5),
]
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
    ) -> None:
        truncated_prompts = self._truncate_prompts(prompts, model_path)

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as srt_runner:
            srt_outputs = srt_runner.forward(truncated_prompts)

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            print("similarity diff", abs(similarity - 1))

            if len(prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"

    def test_prefill_logits(self):
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS, model, tp_size, torch_dtype, prefill_tolerance
                )


if __name__ == "__main__":
    unittest.main()
