import unittest
from types import SimpleNamespace

from sglang.bench_serving import run_benchmark
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    get_benchmark_args,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=443, suite="stage-c-test-8-gpu-h20")


class TestDisaggregationDPAttention(PDDisaggregationServerBase):
    PREFILL_DP_SIZE = 4
    DECODE_DP_SIZE = 4
    LOAD_BALANCE_METHOD = "auto"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Temporarily disable JIT DeepGEMM
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MLA)

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.PREFILL_DP_SIZE),
            "--dp",
            str(cls.PREFILL_DP_SIZE),
            "--enable-dp-attention",
            "--load-balance-method",
            cls.LOAD_BALANCE_METHOD,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.DECODE_DP_SIZE),
            "--dp",
            str(cls.DECODE_DP_SIZE),
            "--enable-dp-attention",
            "--base-gpu-id",
            str(cls.PREFILL_DP_SIZE),
            "--load-balance-method",
            cls.LOAD_BALANCE_METHOD,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1400,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationDPAttentionRoundRobin(TestDisaggregationDPAttention):
    LOAD_BALANCE_METHOD = "round_robin"
    # TODO: add a balancedness metric

    def test_bench_serving(self):
        args = get_benchmark_args(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=1000,
            random_input_len=4096,
            random_output_len=1024,
            request_rate=float("inf"),
            max_concurrency=256,
        )
        result = run_benchmark(args)

        self.assertLess(result["mean_tpot_ms"], 20)
        self.assertEqual(result["completed"], 1000)


class TestDisaggregationDPAttentionTotalRequests(TestDisaggregationDPAttention):
    LOAD_BALANCE_METHOD = "total_requests"
    test_gsm8k = unittest.skip(
        "Covered by base class; this class targets total_requests path."
    )(TestDisaggregationDPAttention.test_gsm8k)

    def test_bench_serving(self):
        args = get_benchmark_args(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=256,
            random_input_len=2048,
            random_output_len=512,
            request_rate=float("inf"),
            max_concurrency=128,
        )
        result = run_benchmark(args)
        self.assertEqual(result["completed"], 256)


class TestDisaggregationDPAttentionTotalTokens(TestDisaggregationDPAttention):
    LOAD_BALANCE_METHOD = "total_tokens"
    test_gsm8k = unittest.skip(
        "Covered by base class; this class targets total_tokens path."
    )(TestDisaggregationDPAttention.test_gsm8k)

    def test_bench_serving(self):
        args = get_benchmark_args(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=256,
            random_input_len=2048,
            random_output_len=512,
            request_rate=float("inf"),
            max_concurrency=128,
        )
        result = run_benchmark(args)
        self.assertEqual(result["completed"], 256)


@unittest.skip(
    "Skip this test until new testing logic in mini-lb has been updated in docker image."
)
class TestDisaggregationDPAttentionExternalRouting(TestDisaggregationDPAttention):
    """Test external DP rank assignment via mini-lb --test-external-dp-routing.

    NOTE: In PD disaggregation the response comes from the decode server,
    so meta_info["dp_rank"] reflects the decode-side DP rank. Prefill DP
    rank correctness is verified implicitly — if the wrong prefill DP
    worker were used, KV transfer would fail and the request would error.
    The mini-lb internally verifies meta_info["dp_rank"] matches the
    assigned decode dp_rank; a mismatch returns HTTP 500.
    """

    @classmethod
    def launch_lb(cls):
        from sglang.test.test_utils import popen_with_error_check

        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--test-external-dp-routing",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)


if __name__ == "__main__":
    unittest.main()
