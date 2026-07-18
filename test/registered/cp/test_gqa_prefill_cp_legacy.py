import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=260, stage="extra-b", runner_config="4-gpu-h100")

GQA_MODEL_PATH = "Qwen/Qwen3-30B-A3B-FP8"

GSM8K_BASELINE_ACCURACY = 0.93


class TestGQACP2TP2EP2(CustomTestCase):

    kv_size_thres = 50059.4  # auto; update_memory_thresholds.py

    @classmethod
    def setUpClass(cls):
        cls.model = GQA_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--moe-dp-size",
                "2",
                "--ep-size",
                "2",
                "--attn-cp-size",
                "2",
                "--enable-prefill-context-parallel",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "32",
                "--trust-remote-code",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
            env={"SGLANG_ENABLE_CP_V2": "0"},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=16000,
            num_threads=128,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


class TestGQACPTP2CP2EP4(CustomTestCase):

    kv_size_thres = 57399.1  # auto; update_memory_thresholds.py

    @classmethod
    def setUpClass(cls):
        cls.model = GQA_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--moe-dp-size",
                "1",
                "--ep-size",
                "4",
                "--attn-cp-size",
                "2",
                "--enable-prefill-context-parallel",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "32",
                "--trust-remote-code",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
            env={"SGLANG_ENABLE_CP_V2": "0"},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=16000,
            num_threads=128,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
