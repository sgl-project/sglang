"""B200 per-commit CI: DeepSeek-V4-Flash PD disaggregation (NIXL + Balanced).

Launches two TP=4/DP=4 servers (prefill + decode) with DeepEP + dp-attention
and NIXL KV transfer. Validates end-to-end accuracy via GSM8K through
the load balancer.

Registry: stage-c-test-dsv4-8-gpu-b200 (per-commit, 8x B200: 4 prefill + 4 decode)
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, suite="stage-c-test-dsv4-8-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}


class TestDSV4FlashPDDisaggB200(PDDisaggregationServerBase):
    """PD disaggregation: TP=4, DP=4, DeepEP + dp-attention + NIXL on 8x B200."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.model = try_cached_model(MODEL)

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=SERVER_LAUNCH_TIMEOUT,
            process=cls.process_prefill,
        )
        cls.wait_server_ready(
            cls.decode_url + "/health",
            timeout=SERVER_LAUNCH_TIMEOUT,
            process=cls.process_decode,
        )
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--base-gpu-id",
            "0",
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--mem-fraction-static",
            "0.7",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "2",
            *cls.transfer_backend,
            *cls.rdma_devices,
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=prefill_args,
            env=_DEEPEP_ENV,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--base-gpu-id",
            "4",
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--mem-fraction-static",
            "0.7",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "2",
            *cls.transfer_backend,
            *cls.rdma_devices,
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=decode_args,
            env=_DEEPEP_ENV,
        )

    def test_gsm8k(self):
        """End-to-end PD-disagg accuracy through the load balancer."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"[{type(self).__name__}] GSM8K {metrics=}")
        self.assertGreater(metrics["score"], 0.93)


if __name__ == "__main__":
    unittest.main()
