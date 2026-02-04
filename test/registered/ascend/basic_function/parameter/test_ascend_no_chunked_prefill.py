import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_bench_serving,
)
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNoChunkedPrefill(CustomTestCase):
    """Testcase: Verify Llama-3.1-8B-Instruct accuracy ≥ 0.65 and serving normal with chunked prefill disabled.

    [Test Category] Parameter
    [Test Target] --chunked-prefill-size
    """
    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--chunked-prefill-size",
            "-1",
            "--disable-radix-cache",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)
        kill_process_tree(self.process.pid)

    def test_no_chunked_prefill_without_radix_cache(self):
        res = run_bench_serving(
            model=LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache", "--chunked-prefill-size", "-1"],
        )

        assert res["completed"] == 10



if __name__ == "__main__":
    unittest.main()
