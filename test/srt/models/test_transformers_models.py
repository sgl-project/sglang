import dataclasses
import multiprocessing as mp
import unittest
from types import SimpleNamespace
from typing import List

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.runners import DEFAULT_PROMPTS, SRTRunner, check_close_model_outputs
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


class TestTransformersFallbackEndpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--model-impl", "transformers"],
        )
        cls.mmlu_lower_bound = 0.65
        cls.gsm8k_lower_bound = 0.65

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        from sglang.test.run_eval import run_eval

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], self.mmlu_lower_bound)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        from sglang.test.few_shot_gsm8k import run_eval

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], self.gsm8k_lower_bound)


class TestTransformersFallbackTorchAO(TestTransformersFallbackEndpoint):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--model-impl",
                "transformers",
                "--torchao-config",
                "int4wo-128",
            ],
        )
        cls.mmlu_lower_bound = 0.65
        cls.gsm8k_lower_bound = 0.65


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 5e-2
    rouge_l_tolerance: float = 1
    skip_long_prompt: bool = False
    trust_remote_code: bool = False
    torchao_config: str = None
    torch_dtype: torch.dtype = torch.float16


# Popular models that run on the CI
CI_MODELS = [
    ModelCase(DEFAULT_MODEL_NAME_FOR_TEST),
]

ALL_OTHER_MODELS = [
    ModelCase(DEFAULT_MODEL_NAME_FOR_TEST, tp_size=2),
]


class TestTransformersFallbackEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_logits_and_output_strs(
        self,
        prompts: List[str],
        model_case: ModelCase,
    ) -> None:
        model_path = model_case.model_path
        max_new_tokens = 32
        # force to use transformers impl
        with SRTRunner(
            model_path,
            tp_size=model_case.tp_size,
            torch_dtype=model_case.torch_dtype,
            model_type="generation",
            model_impl="transformers",
            trust_remote_code=model_case.trust_remote_code,
            torchao_config=model_case.torchao_config,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

        with SRTRunner(
            model_path,
            tp_size=model_case.tp_size,
            torch_dtype=model_case.torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
            torchao_config=model_case.torchao_config,
        ) as srt_runner:
            srt_transformers_outputs = srt_runner.forward(
                prompts, max_new_tokens=max_new_tokens
            )

        check_close_model_outputs(
            hf_outputs=srt_transformers_outputs,
            srt_outputs=srt_outputs,
            prefill_tolerance=model_case.prefill_tolerance,
            decode_tolerance=model_case.decode_tolerance,
            rouge_l_tolerance=model_case.rouge_l_tolerance,
            debug_text=f"model_path={model_path} prompts={prompts}",
        )

    def test_ci_models(self):
        for model_case in CI_MODELS:
            # Skip long prompts for models that do not have a long context
            prompts = DEFAULT_PROMPTS
            if model_case.skip_long_prompt:
                prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            # Assert the logits and output strs are close
            self.assert_close_logits_and_output_strs(prompts, model_case)

    def test_others(self):
        if is_in_ci():
            return

        # Skip long prompts for models that do not have a long context
        prompts = DEFAULT_PROMPTS
        for model_case in ALL_OTHER_MODELS:
            if model_case.skip_long_prompt:
                prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

            # Assert the logits and output strs are close
            self.assert_close_logits_and_output_strs(prompts, model_case)


if __name__ == "__main__":
    unittest.main()
