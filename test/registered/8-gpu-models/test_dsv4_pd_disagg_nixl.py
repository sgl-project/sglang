"""DSv4 Flash PD-disaggregation test with NIXL transfer backend.

Topology (1 H200 node, 8 GPUs total):
  - Prefill: GPU 0-3, tp=4 — pure TP, **no EP** (no deepep), no DP
    attention. Optimized for throughput on long prompts; each rank
    holds the full MoE weights, no all-to-all dispatch traffic.
    Spec config matches decode (PD ferry currently assumes symmetric
    spec on both sides) so the prefill -> decode metadata buffer
    is sized correctly for the spec module's hidden shape.
  - Decode:  GPU 4-7, tp=4 dp=4 enable-dp-attention + deepep + EAGLE
    MTP — optimized for low-latency decode with spec decoding and
    expert parallelism.
  - Mini load balancer fronting both.

Both sides use DSv4 Flash FP8 weights. Transfer backend is NIXL
(the focus of recent nixl/conn.py forward-delta work; this test is
the e2e check that the generic `send_state` / shared buffer-pool
changes do not break PD).
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_cuda_ci(est_time=1200, suite="stage-c-test-8-gpu-h200")


DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    # Decode side runs MTP with num_draft_tokens=4 → dispatch input scales
    # by ~4x, so default 256 overflows once cuda-graph-max-bs * draft > 256.
    # 1024 covers bs=128 * 4 with headroom (no-op on prefill which has no EP).
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
}

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


class TestDSv4FlashPDDisaggNIXL(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.model = DSV4_FLASH_MODEL_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Prefill: TP=4 (no EP, no DP attention). EAGLE config mirrors decode
        # so the metadata buffer is sized for the spec module's hidden shape;
        # PD ferry currently assumes both sides agree on the spec algorithm.
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--base-gpu-id",
            "0",
            "--tp",
            "4",
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--mem-fraction-static",
            "0.7",
            "--disaggregation-decode-tp",
            "4",
            "--disaggregation-decode-dp",
            "4",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            *cls.transfer_backend,
            *cls.rdma_devices,
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def start_decode(cls):
        # Decode: TP=4 + DP=4 attention + deepep EP + EAGLE MTP.
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
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            *cls.transfer_backend,
            *cls.rdma_devices,
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_ENV,
        )

    def test_gsm8k(self):
        """End-to-end PD-disagg accuracy through the LB."""
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=64,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_gsm8k_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
