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
import unittest

import torch

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase

MODELS = [
    ("LxzGordon/URM-LLaMa-3.1-8B", 1, 4e-2),
    ("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", 1, 4e-2),
]
TORCH_DTYPES = [torch.float16]

# PROMPT = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
# RESPONSE1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
# RESPONSE2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)
RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]


class TestRewardModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_reward_scores(
        self,
        convs,
        model_path,
        tp_size,
        torch_dtype,
        tolerance,
    ) -> None:
        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="reward",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(convs)

        with SRTRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="reward",
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(convs, tokenize=False)
            srt_outputs = srt_runner.forward(prompts)

        hf_scores = torch.tensor(hf_outputs.scores)
        srt_scores = torch.tensor(srt_outputs.scores)
        print(f"{hf_scores=}")
        print(f"{srt_scores=}")

        assert torch.all(
            abs(hf_scores - srt_scores) < tolerance
        ), "reward scores are not all close"

    def test_reward_scores(self):
        for model, tp_size, tolerance in MODELS:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_reward_scores(
                    CONVS, model, tp_size, torch_dtype, tolerance
                )


if __name__ == "__main__":
    unittest.main()
