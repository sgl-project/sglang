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

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")

QWEN3_MOE_MODEL_PATH = "Qwen/Qwen3-30B-A3B-FP8"

GSM8K_BASELINE_ACCURACY = 0.93


class _Qwen3MoePPCompatMixin:
    """Launch a Qwen3 MoE server combining PP with another parallel strategy and
    check GSM8K accuracy. Concrete subclasses set ``parallel_args`` (and
    optionally ``server_env``); this mixin is not a TestCase so it is never
    collected on its own.
    """

    model = QWEN3_MOE_MODEL_PATH
    parallel_args: list = []
    server_env = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.parallel_args,
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "32",
                "--trust-remote-code",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
            ],
            env=cls.server_env,
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
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


class TestQwen3MoePPxCP(_Qwen3MoePPCompatMixin, CustomTestCase):
    """PP x CP: pp_size=2 x attn_cp_size=2 (tp_size=2, moe_dp_size=1)."""

    kv_size_thres = 57358.5  # auto; update_memory_thresholds.py
    parallel_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--ep-size",
        "2",
        "--attn-cp-size",
        "2",
        "--enable-prefill-cp",
        "--cp-strategy",
        "zigzag",
    ]
    server_env = {"SGLANG_ENABLE_CP_V2": "1"}


class TestQwen3MoePPxDP(_Qwen3MoePPCompatMixin, CustomTestCase):
    """PP x DP: pp_size=2 x dp_size=2 attention (tp_size=2)."""

    kv_size_thres = 56912.5  # auto; update_memory_thresholds.py
    parallel_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--dp-size",
        "2",
        "--enable-dp-attention",
    ]


if __name__ == "__main__":
    unittest.main()
