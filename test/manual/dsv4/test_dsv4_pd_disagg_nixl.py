"""DSV4 Flash PD-disagg with NIXL backend. Both sides run dp-attention
+ deepep + EAGLE MTP so attn_tp_size and the V4 state pool layout are
fully symmetric: same SWA item_len under matching attn_tp, and same
NSA c4/c128 indexer ring buffer size under matching spec status. nixl
`send_state` is page-by-index and has no V4 TP-slice / spec-asymmetric
path, so any layout mismatch would trip the item_len assert in
`nixl/conn.py`."""

import unittest
from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    # MTP num_draft_tokens=4 scales dispatch by ~4x; 256 overflows at bs=128.
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


class TestDSV4FlashPDDisaggNIXL(PDDisaggregationServerBase):
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
