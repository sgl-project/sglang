"""Unit tests for the encode-disaggregation receiver."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverHTTP,
    WaitingMMRequestStatus,
    _select_mm_processor_prompt,
)
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


class TestWaitingRequestTimeout(CustomTestCase):
    def test_a_result_reached_in_the_timeout_tick_is_kept(self):
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.wait_timeout = 0.0
        receiver.waiting_by_rid = {}
        receiver.tp_group = SimpleNamespace(cpu_group=None)
        receiver._drain_scheduler_embeddings = Mock()
        receiver.create_req = Mock()
        waiting_req = SimpleNamespace(
            rid="r",
            recv_req=SimpleNamespace(rid="r"),
            start_time=0.0,
            status=WaitingMMRequestStatus.PENDING,
            release_resources=Mock(),
            close_recv_socket=Mock(),
        )

        def recv():
            waiting_req.status = WaitingMMRequestStatus.SUCCESS

        waiting_req._try_recv_mm_data = recv
        receiver.waiting_list = [waiting_req]

        with patch("torch.distributed.all_reduce"):
            new_recv_reqs, abort_reqs = receiver._process_waiting_requests([], None)

        self.assertEqual(new_recv_reqs, [waiting_req.recv_req])
        self.assertEqual(abort_reqs, [])
        waiting_req.release_resources.assert_not_called()


if __name__ == "__main__":
    unittest.main()
