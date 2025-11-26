import multiprocessing as mp
import unittest

import torch

from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

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


class TestInternlm2_7bReward(CustomTestCase):
    model_path = (
        "/root/.cache/modelscope/hub/models/Shanghai_AI_Laboratory/internlm2-7b-reward"
    )
    torch_dtype = torch.float16
    tolerance = 4e-2
    tp_size = 1

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_assert_close_reward_scores(self):
        with SRTRunner(
            self.model_path,
            torch_dtype=self.torch_dtype,
            model_type="reward",
            trust_remote_code=True,
            disable_cuda_graph=True,
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(CONVS, tokenize=False)
            srt_outputs = srt_runner.forward(prompts)
        srt_scores = torch.tensor(srt_outputs.scores)
        self.assertIsInstance(srt_scores, torch.Tensor)


class TestQwen2_5Apeach(TestInternlm2_7bReward):
    model_path = "/root/.cache/modelscope/hub/models/Howeee/Qwen2.5-1.5B-apeach"


if __name__ == "__main__":
    unittest.main()
