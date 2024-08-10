"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest

import torch

from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import get_similarities

MODELS = [("intfloat/e5-mistral-7b-instruct", 1)]
TORCH_DTYPES = [torch.float16]


class TestEmbeddingModels(unittest.TestCase):

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
    ) -> None:
        with HFRunner(
            model_path, torch_dtype=torch_dtype, is_generation_model=False
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation_model=False,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts,
            )

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarities = torch.tensor(get_similarities(hf_logits, srt_logits))

            tolerance = 1e-2
            assert torch.all(
                abs(similarities - 1) < tolerance
            ), f"embeddings not all close"

    def test_prefill_logits(self):
        for model, tp_size in MODELS:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS, model, tp_size, torch_dtype
                )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
