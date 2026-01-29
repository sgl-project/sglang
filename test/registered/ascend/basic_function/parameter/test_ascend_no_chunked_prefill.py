import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_npu_ci
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import Llama_3_1_8B_Instruct_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNoChunkedPrefill(CustomTestCase):
    """Testcase: Verify Llama-3.1-8B-Instruct accuracy ≥ 0.65 and and serving normal with chunked prefill disabled.

    [Test Category] Parameter
    [Test Target] --chunked-prefill-size
    """
    def test_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, enable_mixed_chunk=False, chunked_prefill_size=-1
        )

    def test_no_chunked_prefill_without_radix_cache(self):
        res = run_bench_serving(
            model=Llama_3_1_8B_Instruct_WEIGHTS_PATH,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache", "--chunked-prefill-size", "-1"],
        )

        assert res["completed"] == 10



if __name__ == "__main__":
    unittest.main()
