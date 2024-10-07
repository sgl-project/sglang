import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    run_bench_serving,
)


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
