"""End-to-end PD disaggregation x pipeline parallelism x chunked prefill.

Real prefill server (disaggregation_mode=prefill) paired with a real decode server
(disaggregation_mode=decode) through the PD load balancer -- the actual
cross-instance KV transfer is exercised, not faked. The prefill side runs with
pipeline parallelism (pp_size=2) and a small chunked_prefill_size so every prompt
is split into several partial-prefill chunks, each sent across the PD boundary
while pipelined across PP micro-batches. End-to-end accuracy (gsm8k) must hold,
which only happens if the per-chunk KV that the prefill side sends is reassembled
correctly on the decode side.

Manual test: needs 8 GPUs and an RDMA-capable transfer backend. Outside CI, point
it at a backend/devices via SGLANG_TEST_PD_DISAGG_BACKEND / SGLANG_TEST_PD_DISAGG_DEVICES
(see PDDisaggregationServerBase). Run with:

    python3 -m pytest test/manual/e2e/test_pd_pp_chunked_prefill.py
"""

import time
import unittest
from types import SimpleNamespace

from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

# Small enough that a 5-shot gsm8k prompt (~hundreds of tokens) chunks several
# times, so the per-chunk KV-send path is heavily exercised across PD + PP.
_CHUNKED_PREFILL_SIZE = "256"


class TestPDDisaggPPChunkedPrefillAccuracy(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)

        # Non-blocking start of both servers.
        cls.start_prefill()
        cls.start_decode()

        # Block until both are healthy, then bring up the PD load balancer.
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
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
            # Force the chunked-prefill path: every prompt is split into partial
            # prefill chunks, each sent across the PD boundary and pipelined across
            # the two PP micro-batches.
            "--chunked-prefill-size",
            _CHUNKED_PREFILL_SIZE,
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
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
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
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["score"], 0.24)
        # Let the post-request memory check run.
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
