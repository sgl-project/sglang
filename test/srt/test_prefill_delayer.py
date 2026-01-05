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
    def test_1_online_serving_has_prefill_delayer(self):
        self._run_throughput_test_online_serving(prefill_delayer=True)

    def test_2_online_serving_no_prefill_delayer(self):
        self._run_throughput_test_online_serving(prefill_delayer=False)

    def test_3_offline_gen_has_prefill_delayer(self):
        self._run_throughput_test_offline_gen(prefill_delayer=True)

    def test_4_offline_gen_no_prefill_delayer(self):
        self._run_throughput_test_offline_gen(prefill_delayer=False)

    def _run_throughput_test_online_serving(self, prefill_delayer: bool):
        self._run_throughput_test(
            debug_name=f"online_serving ({prefill_delayer=})",
            prefill_delayer=prefill_delayer,
            other_launch_args=[
                "--mem-fraction-static",
                "0.6",
            ],
            other_benchmark_args=dict(
                num_prompts=500,
                # trigger chunked prefill
                random_input_len=30000,
                random_output_len=256,
                request_rate=32,
            ),
        )

    def _run_throughput_test_offline_gen(self, prefill_delayer: bool):
        self._run_throughput_test(
            debug_name=f"offline_gen ({prefill_delayer=})",
            prefill_delayer=prefill_delayer,
            other_benchmark_args=dict(
                num_prompts=800,
                random_input_len=30000,
                random_output_len=500,
            ),
            other_launch_args=[
                "--max-total-tokens",
                "200000",
                "--mem-fraction-static",
                "0.6",
            ],
        )

    def _run_throughput_test(
        self,
        debug_name: str,
        prefill_delayer: bool,
        other_launch_args,
        other_benchmark_args,
    ):
        os.environ["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

        model = "Qwen/Qwen3-0.6B"
        base_url = DEFAULT_URL_FOR_TEST

        with envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.override(
            prefill_delayer
        ), envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.override(100):
            process = popen_launch_server(
                model,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--enable-dp-attention",
                    "--dp",
                    "8",
                    "--chunked-prefill-size",
                    "131072",
                    *other_launch_args,
                ],
            )

        try:
            args = get_benchmark_args(
                base_url=base_url,
                dataset_name="random",
                tokenizer=model,
                **other_benchmark_args,
            )
            res = run_benchmark(args)
        finally:
            kill_process_tree(process.pid)

        print(f"=== {debug_name} ===")
        print(f"Input throughput: {res['input_throughput']:.2f} token/s")
        print(f"Output throughput: {res['output_throughput']:.2f} token/s")


if __name__ == "__main__":
    unittest.main()
