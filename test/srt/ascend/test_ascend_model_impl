import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestModeImpl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = (
            "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-11B-Vision-Instruct"
            if is_npu()
            else DEFAULT_MODEL_NAME_FOR_TEST
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                [
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                    "--model-impl",
                    "transformers",
                    "--trust-remote-code",
                    "--mem-fraction-static",
                    0.9,
                ]
                if is_npu()
                else [
                    "--model-impl",
                    "transformers",
                    "--trust-remote-code",
                ]
            ),
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
        print(f"{metrics=}")
        # self.assertGreaterEqual(metrics["score"], self.mmlu_lower_bound)

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
        # self.assertGreater(metrics["accuracy"], self.gsm8k_lower_bound)


if __name__ == "__main__":
    unittest.main()
