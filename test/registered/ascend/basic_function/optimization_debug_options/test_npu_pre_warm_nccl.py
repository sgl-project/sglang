import unittest

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-2-npu-a3", nightly=True)


class TestPreWarmNccl(CustomTestCase):
    """Testcase: verify --pre-warm-nccl server starts and serves correctly

    [Test Category] Parameter
    [Test Target] --pre-warm-nccl
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
    ]

    def _launch(self, with_warmup):
        args = list(self.base_args)
        if with_warmup:
            args.append("--pre-warm-nccl")
        proc = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )
        return proc

    def _run_bench(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            backend="sglang",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=100,
            random_input_len=3500,
            random_output_len=1500,
            request_rate=float("inf"),
        )
        args.warmup_requests = 0
        res = run_benchmark(args)
        self.assertGreater(res["mean_ttft_ms"], 0, "TTFT must be > 0 ms")
        return res

    def test_pre_warm_nccl(self):
        proc1 = self._launch(with_warmup=True)
        try:
            res_warmup = self._run_bench()
        finally:
            kill_process_tree(proc1.pid)

        proc2 = self._launch(with_warmup=False)
        try:
            res_no_warmup = self._run_bench()
        finally:
            kill_process_tree(proc2.pid)

        p99_w = res_warmup["p99_ttft_ms"]
        p99_nw = res_no_warmup["p99_ttft_ms"]
        print(
            f"\n=== TTFT Comparison: --pre-warm-nccl vs default ===\n"
            f"  P99 TTFT: {p99_w:.1f} ms (warmup) vs {p99_nw:.1f} ms (no-warmup)\n"
        )
        self.assertLessEqual(
            p99_w,
            p99_nw,
            f"Expected --pre-warm-nccl P99 TTFT ({p99_w:.1f} ms) <= "
            f"no-warmup ({p99_nw:.1f} ms). NCCL warmup should prime all-reduce "
            f"communication and reduce first-request tail latency.",
        )


if __name__ == "__main__":
    unittest.main()
