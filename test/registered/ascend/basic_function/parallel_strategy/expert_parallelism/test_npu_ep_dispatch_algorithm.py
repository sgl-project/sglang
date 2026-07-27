import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


class TestEpDispatchAlgorithmDynamic(CustomTestCase):
    """Testcase: Verify set the parameter --ep-dispatch-algorithm，the inference accuracy of the model on the
    GSM8K dataset is no less than 0.90

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm
    """

    ep_dispatch_algorithm = "dynamic"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.8",
                "--tp-size",
                "2",
                "--expert-parallel-size",
                "2",
                "--ep-num-redundant-experts",
                "4",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--ep-dispatch-algorithm",
                cls.ep_dispatch_algorithm,
            ],
            env={
                "HCCL_BUFFSIZE": "1024",
                "TRANSFORMERS_VERBOSITY": "error",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.90)


class TestEpDispatchAlgorithmFake(TestEpDispatchAlgorithmDynamic):
    ep_dispatch_algorithm = "fake"


class TestEpDispatchAlgorithmDynamicMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.8",
                "--tp-size",
                "2",
                "--expert-parallel-size",
                "2",
                "--ep-num-redundant-experts",
                "4",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
            ],
            env={
                "HCCL_BUFFSIZE": "1024",
                "TRANSFORMERS_VERBOSITY": "error",
                "SGLANG_ENABLE_SPEC_V2": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
