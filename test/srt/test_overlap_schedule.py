"""
Usage:
SGLANG_IS_IN_CI=true python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
SGLANG_IS_IN_CI=true python3 test_overlap_schedule.py
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestOverlapSchedule(unittest.TestCase):
    def run_mmlu(self, disable_radix_cache, chunked_prefill_size=32):
        other_args = ["--chunked-prefill-size", str(chunked_prefill_size)]
        if disable_radix_cache:
            other_args += ["--disable-radix-cache"]
        other_args += ["--enable-overlap-schedule"]

        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        try:
            metrics = run_eval(args)
            assert metrics["score"] >= 0.65
        finally:
            kill_child_process(process.pid)

    def test_no_radix_attention_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=True, chunked_prefill_size=32)

    def test_no_radix_attention_no_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=True, chunked_prefill_size=-1)

    def test_radix_attention_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=False, chunked_prefill_size=32)

    def test_radix_attention_no_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=False, chunked_prefill_size=-1)


if __name__ == "__main__":
    unittest.main()
    # @unittest.skip("did not support")
