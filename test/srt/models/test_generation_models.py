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

import multiprocessing as mp
import unittest

import torch

from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner

MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", 1, 1.1, 3e-2, 4e-2, 1),
    ("google/gemma-2-2b", 1, 3, 3e-2, 5e-2, 1),
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1, None, 6e-2, 4e-2, 1),
]
TORCH_DTYPES = [torch.float16]


def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def calculate_rouge_l(output_strs_list1, output_strs_list2):
    rouge_l_scores = []

    for s1, s2 in zip(output_strs_list1, output_strs_list2):
        lcs_len = lcs(s1, s2)
        precision = lcs_len / len(s1) if len(s1) > 0 else 0
        recall = lcs_len / len(s2) if len(s2) > 0 else 0
        if precision + recall > 0:
            fmeasure = (2 * precision * recall) / (precision + recall)
        else:
            fmeasure = 0.0
        rouge_l_scores.append(fmeasure)

    return rouge_l_scores


class TestGenerationModels(unittest.TestCase):
    def assert_close_prefill_logits_and_output_strs(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        max_new_tokens,
        prefill_tolerance,
        output_tolerance,
        rouge_threshold,
        long_context_tolerance,
    ) -> None:
        if model_path == "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
            prompts = prompts[:-1]

        with HFRunner(
            model_path, torch_dtype=torch_dtype, is_generation=True
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts, max_new_tokens=max_new_tokens)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            is_generation=True,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

        for i in range(len(prompts)):
            # input logprobs comparison
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])
            input_len = hf_logprobs.shape[0]
            print(
                "prefill logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(
                    abs(hf_logprobs - srt_logprobs) < prefill_tolerance
                ), f"prefill logprobs are not all close with model_path={model_path} prompts={prompts} prefill_tolerance={prefill_tolerance}"

            # output logprobs comparison
            hf_logprobs = torch.Tensor(hf_outputs.top_output_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_output_logprobs[i])
            # print(
            #     "output logprobs diff",
            #     [
            #         float(torch.max(abs(hf_logprobs[j] - srt_logprobs[j])))
            #         for j in range(max_new_tokens)
            #     ],
            # )
            print(
                "output logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(
                    abs(hf_logprobs - srt_logprobs) < output_tolerance
                ), f"output logprobs are not all close with model_path={model_path} prompts={prompts}... output_tolerance={output_tolerance}"

        # output strings comparison
        print(f"hf_outputs.output_strs={hf_outputs.output_strs}")
        print(f"srt_outputs.output_strs={srt_outputs.output_strs}")
        rouge_l_scores = calculate_rouge_l(
            hf_outputs.output_strs, srt_outputs.output_strs
        )
        print(f"rouge_l_scores={rouge_l_scores}")
        assert all(
            score >= rouge_threshold for score in rouge_l_scores
        ), f"Not all ROUGE-L scores are greater than rouge_threshold={rouge_threshold}"

    def test_prefill_logits_and_output_strs(self):
        for (
            model,
            tp_size,
            long_context_tolerance,
            prefill_tolerance,
            output_tolerance,
            rouge_threshold,
        ) in MODELS:
            for torch_dtype in TORCH_DTYPES:
                max_new_tokens = 32
                self.assert_close_prefill_logits_and_output_strs(
                    DEFAULT_PROMPTS,
                    model,
                    tp_size,
                    torch_dtype,
                    max_new_tokens,
                    prefill_tolerance=prefill_tolerance,
                    output_tolerance=output_tolerance,
                    rouge_threshold=rouge_threshold,
                    long_context_tolerance=long_context_tolerance,
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main()
