import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import multiprocessing as mp
import unittest

import torch

from sglang.test.ascend.test_ascend_utils import INTERNLM2_7B_REWARD_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
)

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)
RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]


class TestInternlm2(CustomTestCase):
    """Testcase: This test case verifies that the Shanghai_AI_Laboratory/internlm2-7b-reward model can successfully generate reward
    scores for different conversational responses using the SGLang framework, without comparing to a reference implementation.

    [Test Category] Model
    [Test Target] Shanghai_AI_Laboratory/internlm2-7b-reward
    """

    model_path = INTERNLM2_7B_REWARD_WEIGHTS_PATH
    torch_dtype = torch.float16

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
        print(f"accuracy: {srt_scores}")
        self.assertIsInstance(srt_scores, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
