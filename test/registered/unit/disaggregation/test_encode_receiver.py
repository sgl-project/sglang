"""Unit tests for request construction in the encode-disaggregation path."""

import http.client
import socket
import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.disaggregation.encoder.receiver import (
    EncoderBootstrapServer,
    MMReceiverBase,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_extra_key_and_cache_salt_are_forwarded(self):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        req = MMReceiverBase.create_req(receiver, recv_req)

        self.assertEqual(req.extra_key, "classification")
        self.assertEqual(req.cache_salt, "tenant-a")


class TestEncoderBootstrapLifecycle(CustomTestCase):
    def _server(self, port=0):
        return EncoderBootstrapServer("127.0.0.1", port, health_check_interval=0)

    def test_constructor_rejects_an_occupied_port(self):
        with socket.create_server(("127.0.0.1", 0)) as occupied:
            with self.assertRaisesRegex(OSError, "Could not bind port"):
                self._server(occupied.getsockname()[1])

    def test_publishes_served_port_and_releases_listener(self):
        first = self._server()
        port = first.port
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            self.assertEqual((response.status, response.read()), (200, b"OK"))
        finally:
            connection.close()
        first.close()

        replacement = self._server(port)
        replacement.close()

    def test_close_releases_listener_for_restart(self):
        first = self._server()
        port = first.port
        first.close()

        replacement = self._server(port)
        replacement.close()


if __name__ == "__main__":
    unittest.main()
