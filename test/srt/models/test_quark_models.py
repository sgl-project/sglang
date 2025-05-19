import unittest
import dataclasses
from typing import Optional
from types import SimpleNamespace
from sglang.test.test_utils import CustomTestCase
from sglang.test.runners import SRTRunner, DEFAULT_PROMPTS
from sglang.test.few_shot_gsm8k import run_eval
import os

from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.srt.utils import kill_process_tree

@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    context_length: Optional[int] = None
    mem_fraction_static: Optional[float] = None

ALL_MODELS = [
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4"),
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp_size=2),
    # Memory access fault with Deepseek-R1 (3 layers) on TP=1 on MI300.
    ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp_size=8),
    ModelCase("fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp_size=1, mem_fraction_static=0.7, context_length=100000),
    ModelCase("fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp_size=8, mem_fraction_static=0.7, context_length=1000000),
]


class TestQuarkMXFP4Loading(CustomTestCase):
    def test_load_and_run(self):
        for model_case in ALL_MODELS:
            prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            max_new_tokens = 20

            with SRTRunner(
                model_case.model_path,
                tp_size=model_case.tp_size,
                model_type="generation",
                torch_dtype="auto",
                mem_fraction_static=model_case.mem_fraction_static,
                context_length=model_case.context_length
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
                torch_dtype="auto",
                mem_fraction_static=model_case.mem_fraction_static,
                context_length=model_case.context_length
            ) as srt_runner:
                srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)
        
        del os.environ['SGLANG_QUARK_EMU_MEM_OPT']

class TestR1MXFP4Accuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Private model.
        cls.model = "amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant"

        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tp",
            "8",
            "--mem-fraction-static",
            "0.9",
            "--context-length",
            "38768"
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=45 * 60,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.96)


if __name__ == "__main__":
    unittest.main()
