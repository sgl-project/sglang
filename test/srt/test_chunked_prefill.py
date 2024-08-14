import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_UNIT_TEST,
    popen_launch_server,
)


class TestChunkedPrefill(unittest.TestCase):
    def run_mmlu(self, disable_radix_cache):
        other_args = ["--chunked-prefill-size", "32"]
        if disable_radix_cache:
            other_args += ["--disable-radix-cache"]

        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_UNIT_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=300,
            other_args=other_args,
        )

        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=32,
            num_threads=32,
        )

        try:
            metrics = run_eval(args)
            assert metrics["score"] >= 0.6
        finally:
            kill_child_process(process.pid)

    def test_chunked_prefill(self):
        self.run_mmlu(disable_radix_cache=False)

    def test_chunked_prefill_without_radix_cache(self):
        self.run_mmlu(disable_radix_cache=True)


if __name__ == "__main__":
    unittest.main()
