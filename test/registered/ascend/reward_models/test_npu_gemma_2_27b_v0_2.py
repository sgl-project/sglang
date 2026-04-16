import logging
import multiprocessing as mp
import os
import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase

logger = logging.getLogger(__name__)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

MODELS = [
    (
        "/root/.cache/modelscope/hub/models/AI-ModelScope/Skywork-Reward-Gemma-2-27B-v0.2",
        1,
        4e-2,
    ),
]
TORCH_DTYPES = [torch.bfloat16]

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
            mem_fraction_static=0.95,
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(
                convs, tokenize=False, return_dict=False
            )
            srt_outputs = srt_runner.forward(prompts)

        hf_scores = torch.tensor(hf_outputs.scores)
        srt_scores = torch.tensor(srt_outputs.scores)
        logger.info(f"{hf_scores=}")
        logger.info(f"{srt_scores=}")

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
    os.environ["SGLANG_NPU_FORWARD_NATIVE_GELUTANH"] = "1"
    os.environ["SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM"] = "1"
    unittest.main()
