import os
import unittest

from sglang.bench_serving import run_benchmark
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)


class TestPrefillDelayerThroughput(CustomTestCase):
    def _run_throughput_test(self, with_prefill_delayer: bool):
        os.environ["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

        model = "Qwen/Qwen3-0.6B"
        base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-dp-attention",
            "--dp",
            "8",
            "--chunked-prefill-size",
            "262144",
            "--mem-fraction-static",
            "0.6",
        ]

        with envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.override(
            with_prefill_delayer
        ), envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.override(
            100
        ):
            process = popen_launch_server(
                model,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

        try:
            args = get_benchmark_args(
                base_url=base_url,
                dataset_name="random",
                num_prompts=500,
                random_input_len=30000,
                random_output_len=256,
                request_rate=32,
                tokenizer=model,
            )
            res = run_benchmark(args)
        finally:
            kill_process_tree(process.pid)

        print(f"=== {with_prefill_delayer=} ===")
        print(f"Input throughput: {res['input_throughput']:.2f} token/s")
        print(f"Output throughput: {res['output_throughput']:.2f} token/s")

    def test_1_dp_attention_throughput_with_prefill_delayer(self):
        self._run_throughput_test(with_prefill_delayer=True)

    def test_2_dp_attention_throughput_without_prefill_delayer(self):
        self._run_throughput_test(with_prefill_delayer=False)


if __name__ == "__main__":
    unittest.main()
