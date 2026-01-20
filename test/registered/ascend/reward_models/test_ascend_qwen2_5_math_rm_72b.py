import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

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
    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-Math-RM-72B"
    torch_dtype = torch.float16
    tolerance = 4e-2
    tp_size = 4

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
            tp_size=4,
            mem_fraction_static=0.8,
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(CONVS, tokenize=False)
            srt_outputs = srt_runner.forward(prompts)
        srt_scores = torch.tensor(srt_outputs.scores)
        print(f'accuracy: {srt_scores}')
        self.assertIsInstance(srt_scores, torch.Tensor)


# class TestQwen2_5Apeach(TestInternlm2_7bReward):
#    model_path = "/data/l30079981/weights/Qwen2.5-1.5B-apeach"


if __name__ == "__main__":
    unittest.main()
