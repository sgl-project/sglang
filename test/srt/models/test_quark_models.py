import unittest
import dataclasses

from sglang.test.test_utils import CustomTestCase
from sglang.test.runners import SRTRunner, DEFAULT_PROMPTS
from typing import List
import os

@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1

ALL_MODELS = [
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4"),
    # Memory access fault with Deepseek-R1 on TP=1 on MI300.
    ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp_size=8),
]


class TestQuarkMXFP4DeepseekV3(CustomTestCase):
    def test_load_and_run(self):
        for model_case in ALL_MODELS:
            prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            max_new_tokens = 20

            with SRTRunner(
                model_case.model_path,
                tp_size=model_case.tp_size,
                model_type="generation",
                torch_dtype="auto"
            ) as srt_runner:
                srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

    def test_load_and_run_mem_emulation(self):
        os.environ["SGLANG_QUARK_EMU_MEM_OPT"] = "1"

        for model_case in ALL_MODELS:
            prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            max_new_tokens = 20

            with SRTRunner(
                model_case.model_path,
                tp_size=model_case.tp_size,
                model_type="generation",
                torch_dtype="auto"
            ) as srt_runner:
                srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)
        
        del os.environ['SGLANG_QUARK_EMU_MEM_OPT']


if __name__ == "__main__":
    unittest.main()
