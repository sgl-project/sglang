import subprocess
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_latency,
)


class TestTritonAttnBackend(unittest.TestCase):
    def test_latency(self):
        output_throughput = run_bench_latency(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "triton",
                "--enable-torch-compile",
            ],
        )

        if is_in_ci():
            assert output_throughput > 154, f"{output_throughput=}"

    def test_mmlu(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "triton"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            assert metrics["score"] >= 0.65
        finally:
            kill_child_process(process.pid)


if __name__ == "__main__":
    unittest.main()
