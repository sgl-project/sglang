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

MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", 1),
    ("google/gemma-2-2b", 1),
]
TORCH_DTYPES = [torch.float16]


class TestGenerationModels(unittest.TestCase):

    def assert_close_prefill_logits_and_output_strs(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        max_new_tokens,
    ) -> None:
        with HFRunner(
            model_path, torch_dtype=torch_dtype, is_generation_model=True
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation_model=True,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

        for i in range(len(prompts)):
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])

            tolerance = 3e-2
            assert torch.all(
                abs(hf_logprobs - srt_logprobs) < tolerance
            ), f"prefill logprobs not all close"

        assert hf_outputs.output_strs == srt_outputs.output_strs

    def test_prefill_logits(self):
        for model, tp_size in MODELS:
            for torch_dtype in TORCH_DTYPES:
                max_new_tokens = 8
                self.assert_close_prefill_logits_and_output_strs(
                    DEFAULT_PROMPTS,
                    model,
                    tp_size,
                    torch_dtype,
                    max_new_tokens,
                )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
