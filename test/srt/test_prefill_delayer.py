import os
import unittest
from types import SimpleNamespace

from sglang.bench_serving import run_benchmark
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)


class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    def test_1_has_prefill_delayer(self):
        self._run(prefill_delayer=True)

    def test_2_no_prefill_delayer(self):
        self._run(prefill_delayer=False)

    def _run(self, prefill_delayer: bool):
        _run_throughput_test(
            debug_name=f"online_serving ({prefill_delayer=})",
            prefill_delayer=prefill_delayer,
            other_launch_args=[
                # Not really needed, only to test support non-FCFS algorithms
                "--schedule-policy",
                "lpm",
            ],
            other_benchmark_args=dict(
                num_prompts=500,
                random_input_len=30000,
                random_output_len=256,
                request_rate=32,
            ),
        )


class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    def test_1_has_prefill_delayer(self):
        self._run(prefill_delayer=True)

    def test_2_no_prefill_delayer(self):
        self._run(prefill_delayer=False)

    def _run(self, prefill_delayer: bool):
        _run_throughput_test(
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
            ],
        )


def _run_throughput_test(
    debug_name: str,
    prefill_delayer: bool,
    other_launch_args,
    other_benchmark_args,
):
    model = "Qwen/Qwen3-0.6B"
    base_url = DEFAULT_URL_FOR_TEST

    process = _launch_server(
        prefill_delayer=prefill_delayer,
        model=model,
        base_url=base_url,
        other_args=other_launch_args,
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


class TestPrefillDelayerAccuracy(CustomTestCase):
    def test_1_mgsm_en_has_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=True)

    def test_2_mgsm_en_no_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=False)

    def _run_accuracy_test(self, prefill_delayer: bool):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = _launch_server(
            prefill_delayer=prefill_delayer,
            model=model,
            base_url=base_url,
            other_args=[
                # Not really needed, only to test support non-FCFS algorithms
                "--schedule-policy",
                "lpm",
                # Use this to ensure prefill delayer will be run
                "--max-total-tokens",
                "4096",
            ],
        )
        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mgsm_en",
                num_examples=None,
                num_threads=1024,
            )
            metrics = run_eval(args)
            print(f"=== mgsm_en ({prefill_delayer=}) ===")
            print(f"{metrics=}")
            self.assertGreater(metrics["score"], 0.87)
        finally:
            kill_process_tree(process.pid)


def _launch_server(*, model, base_url, prefill_delayer: bool, other_args):
    os.environ["SGLANG_PREFILL_DELAYER_DEBUG_LOG"] = "1"

    with envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.override(
        prefill_delayer
    ), envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.override(100):
        return popen_launch_server(
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
                "--mem-fraction-static",
                "0.6",
                *(other_args or []),
            ],
        )


if __name__ == "__main__":
    unittest.main()
