import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestModeImpl(CustomTestCase):
    """Testcase：Verify Llama-3.2-1B-Instruct model set --model-impl = transformers and set --prefill-max-requests = 5,
    the mmlu accuracy greater than 0.65 and the gsm8k accuracy more than 0.65.

    [Test Category] Parameter
    [Test Target] --model-impl, --prefill-max-requests
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
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
                    "--prefill-max-requests",
                    5,
                    "--trust-remote-code",
                    "--mem-fraction-static",
                    0.8,
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
            num_shots=5,
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
        self.assertGreater(metrics["accuracy"], self.gsm8k_lower_bound)


if __name__ == "__main__":
    unittest.main()
