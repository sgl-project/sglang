"""Unit tests for the encode-disaggregation receiver."""

import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.encoder import receiver as receiver_module
from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverGrpc,
    _select_mm_processor_prompt,
)
from sglang.srt.multimodal.modality import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSelectMMProcessorPrompt(CustomTestCase):
    def test_tokens_in_prompt_is_list(self):
        """A tokens-in prompt reaches the processor as list[int], not array("q");
        HTTP sends input_text=None and the gRPC servicer sends ""."""
        processor = SimpleNamespace(prefer_tokenized_input=False)
        for input_text in (None, ""):
            with self.subTest(input_text=input_text):
                recv_req = SimpleNamespace(
                    input_text=input_text, input_ids=array("q", [1, 2, 3])
                )

                prompt = _select_mm_processor_prompt(recv_req, processor)

                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt, [1, 2, 3])


class TestGrpcThreadedEncodeDispatch(CustomTestCase):
    def test_threaded_dispatch_sends_grpc_encode(self):
        """The threaded dispatch used by zmq_to_scheduler reaches the gRPC encoder
        instead of reporting a dispatch error for every multimodal request."""
        receiver = MMReceiverGrpc.__new__(MMReceiverGrpc)
        receiver.encode_urls = ["grpc://127.0.0.1:1"]
        receiver.host = "127.0.0.1"
        sent = []
        dispatch_errors = []
        dispatch_ready = threading.Event()
        dispatch_ready.set()

        with patch.object(
            receiver_module,
            "_grpc_encode_request",
            side_effect=lambda target, encode_request: sent.append(
                (target, encode_request)
            ),
        ):
            receiver._run_encode_in_thread(
                req_id="rid-0",
                mm_data=[{"modality": Modality.IMAGE, "url": "http://host/a.png"}],
                endpoint_encode="encode",
                num_items_assigned={Modality.IMAGE: [1]},
                time_stats_json="{}",
                scheduler_dispatch_ready=dispatch_ready,
                on_dispatch_error=dispatch_errors.append,
            )

        self.assertEqual(dispatch_errors, [])
        self.assertEqual(len(sent), 1)
        target, encode_request = sent[0]
        self.assertEqual(target, "127.0.0.1:1")
        self.assertEqual(encode_request["mm_items"], ["http://host/a.png"])
        self.assertEqual(encode_request["req_id"], "rid-0")


if __name__ == "__main__":
    unittest.main()
