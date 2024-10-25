"""
python3 -m unittest test_chunked_prefill.TestChunkedPrefill.test_mixed_chunked_prefill_without_radix_cache
"""

import random
import threading
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
    run_bench_serving,
)


def read_output(process, output_lines):
    # Read the outputs to prevent blocking on the buffer
    for line in iter(process.stderr.readline, ""):
        print(line, end="", flush=True)
        output_lines.append(line)


class TestChunkedPrefill(unittest.TestCase):
    def run_mmlu(
        self, disable_radix_cache, enable_mixed_chunk, chunked_prefill_size=32
    ):
        other_args = ["--chunked-prefill-size", str(chunked_prefill_size)]
        if disable_radix_cache:
            other_args += ["--disable-radix-cache"]

        if enable_mixed_chunk:
            other_args += ["--enable-mixed-chunk"]

        model = DEFAULT_MODEL_NAME_FOR_TEST
        port = random.randint(4000, 5000)
        base_url = f"http://127.0.0.1:{port}"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=True,
        )

        output_lines = []
        t = threading.Thread(
            target=read_output, args=(process, output_lines), daemon=True
        )
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
            print(f"{metrics=}")
            assert metrics["score"] >= 0.65
        finally:
            time.sleep(1)
            kill_child_process(process.pid)
            kill_child_process(process.pid)

        has_new_server = False
        has_leak = False
        for line in output_lines:
            if "The server is fired" in line:
                has_new_server = True
            if "leak" in line:
                has_leak = True

        assert has_new_server
        # assert not has_leak

    def test_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=False, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=False, enable_mixed_chunk=True)

    def test_chunked_prefill_without_radix_cache(self):
        self.run_mmlu(disable_radix_cache=True, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill_without_radix_cache(self):
        self.run_mmlu(disable_radix_cache=True, enable_mixed_chunk=True)

    def test_no_chunked_prefill(self):
        self.run_mmlu(
            disable_radix_cache=False, enable_mixed_chunk=False, chunked_prefill_size=-1
        )

    def test_no_chunked_prefill_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache", "--chunked-prefill-size", "-1"],
        )

        assert res["completed"] == 10


if __name__ == "__main__":
    unittest.main()
