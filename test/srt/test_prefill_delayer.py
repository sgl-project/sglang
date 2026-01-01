import unittest

from sglang.bench_serving import run_benchmark
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


class TestPrefillDelayerThroughput(CustomTestCase):
    def _run_throughput_test(self, with_prefill_delayer: bool):
        model = "Qwen/Qwen3-0.6B"
        base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "2",
            "--enable-dp-attention",
            "--dp",
            "2",
        ]

        with envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.override(with_prefill_delayer):
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
                num_prompts=200,
                random_input_len=1024,
                random_output_len=512,
                request_rate=float("inf"),
            )
            res = run_benchmark(args)
        finally:
            kill_process_tree(process.pid)

        label = "With PrefillDelayer" if with_prefill_delayer else "Without PrefillDelayer (baseline)"
        print(f"=== {label} ===")
        print(f"Output throughput: {res['output_throughput']:.2f} token/s")
        print(f"Input throughput: {res['input_throughput']:.2f} token/s")

        self.assertGreater(res["output_throughput"], 0)

    def test_dp_attention_throughput_with_prefill_delayer(self):
        self._run_throughput_test(with_prefill_delayer=True)

    def test_dp_attention_throughput_without_prefill_delayer(self):
        self._run_throughput_test(with_prefill_delayer=False)


if __name__ == "__main__":
    unittest.main()
