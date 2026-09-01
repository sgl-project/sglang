"""
XPU reward parity test: compares HF and SRT reward scores on Intel XPU.

Usage:
python3 -m unittest test_xpu_reward.TestXPUReward
"""

import gc
import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

register_xpu_ci(est_time=60, suite="nightly-xpu-1-gpu", nightly=True)

MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
TP_SIZE = 1
TOLERANCE = 1.5e-1
TORCH_DTYPE = torch.bfloat16

PROMPT = (
    "What is the range of the numeric output of a sigmoid node in a neural network?"
)
RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]


class TestXPUReward(CustomTestCase):
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

        gc.collect()
        empty_gpu_cache()

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="reward",
            attention_backend="intel_xpu",
            mem_fraction_static=0.55,
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(
                convs,
                tokenize=False,
                return_dict=False,
            )
            srt_outputs = srt_runner.forward(prompts)

        hf_scores = torch.tensor(hf_outputs.scores)
        srt_scores = torch.tensor(srt_outputs.scores)
        self.assertTrue(
            torch.all(torch.abs(hf_scores - srt_scores) < tolerance),
            "reward scores are not all close",
        )

    def test_reward_scores(self):
        self.assert_close_reward_scores(
            CONVS,
            MODEL_PATH,
            TP_SIZE,
            TORCH_DTYPE,
            TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
