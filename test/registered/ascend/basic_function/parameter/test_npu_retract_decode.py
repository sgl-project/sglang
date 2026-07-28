import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import is_in_ci

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestRetractDecode(CustomTestCase):
    """When retract decode feature is enabled, verify that the MMLU dataset accuracy of the Llama-3.1-8B-Instruct model is greater than 0.65.

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT
    """

    other_args = []

    @classmethod
    def setUpClass(cls):
        # Enable retract decode feature for test
        os.environ["SGLANG_TEST_RETRACT"] = "1"

        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--chunked-prefill-size",
            "128",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ] + cls.other_args
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

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

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)
        time.sleep(1)  # wait for mem check

        assert self.process.poll() is None, "Server crashed during test"


class TestRetractDecodePaged(TestRetractDecode):
    """Retract decode with paged mode (--page-size 128) on NPU.

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT;--page-size
    """

    # NPU page_size is 128; use 128 instead of the CUDA default 16.
    other_args = ["--page-size", "128"]


class TestRetractDecodeChunkCache(TestRetractDecode):
    """Retract decode with chunk cache (--disable-radix-cache) on NPU.

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT;--disable-radix-cache
    """

    other_args = ["--disable-radix-cache"]


class TestRetractDecodeChunkCachePaged(TestRetractDecode):
    """Retract decode with chunk cache + paged mode on NPU.

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT;--disable-radix-cache;--page-size
    """

    other_args = ["--disable-radix-cache", "--page-size", "128"]


@unittest.skipIf(is_in_ci(), "Skipped in CI due to long runtime")
class TestRetractDecodeLongOutput(CustomTestCase):
    """Long-output stress test for retract decode on NPU.

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT;long output
    """

    other_args = []

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_TEST_RETRACT"] = "1"

        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--chunked-prefill-size",
            "128",
            "--page-size",
            "128",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ] + cls.other_args
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_long_output_retract(self):
        data = {
            "input_ids": [[233 + i] * 1234 for i in range(256)],
            "sampling_params": {"max_new_tokens": 90000, "ignore_eos": True},
        }
        res = requests.post(f"{self.base_url}/generate", json=data)
        assert res.status_code == 200, f"Request failed: {res.status_code}"
        assert self.process.poll() is None, "Server crashed during test"


@unittest.skipIf(is_in_ci(), "Skipped in CI due to long runtime")
class TestRetractDecodeLongOutputChunkCache(TestRetractDecodeLongOutput):
    """Long-output stress test for retract decode + chunk cache on NPU.

    [Test Category] Parameter
    [Test Target] SGLANG_TEST_RETRACT;long output;--disable-radix-cache
    """

    other_args = ["--disable-radix-cache"]


if __name__ == "__main__":
    unittest.main()
