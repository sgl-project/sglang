import unittest
from types import SimpleNamespace

from sglang.bench_serving import run_benchmark
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
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

register_cuda_ci(est_time=580, suite="stage-c-test-8-gpu-h20")


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
            num_shots=5,
            data_path=None,
            num_questions=1400,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["accuracy"], 0.60)


class TestDisaggregationDPAttentionRoundRobin(TestDisaggregationDPAttention):
    LOAD_BALANCE_METHOD = "round_robin"
    # TODO: add test for other load balance methods
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


if __name__ == "__main__":
    unittest.main()
