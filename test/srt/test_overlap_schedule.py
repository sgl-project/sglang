"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import threading
import time
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


def read_output(process, output_lines):
    for line in iter(process.stderr.readline, ""):
        print(line, end="", flush=True)
        output_lines.append(line)


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
            return_stdout_stderr=True,
        )

        output_lines = []
        t = threading.Thread(target=read_output, args=(process, output_lines))
        t.start()

        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=128,
        )

        try:
            metrics = run_eval(args)
            assert metrics["score"] >= 0.65
        finally:
            time.sleep(1)
            kill_child_process(process.pid)

        has_new_server = False
        has_leak = False
        for line in output_lines:
            if "The server is fired" in line:
                has_new_server = True
            if "leak" in line:
                has_leak = True

        assert has_new_server
        assert not has_leak

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
