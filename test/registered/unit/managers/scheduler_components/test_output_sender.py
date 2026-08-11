"""Unit tests for srt/managers/scheduler_components/output_sender.py.

No server, no model. Covers the SenderWrapper contracts:
- the None-socket no-op gate (scheduler configs without an output socket),
- the http_worker_ipc back-propagation rule for the multi-HTTP-worker case,
  including the branches that must NOT copy (already-routed outputs, batch
  requests, missing recv context).
"""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SOCK_SEND = "sglang.srt.managers.scheduler_components.output_sender.sock_send"


class TestSenderWrapper(CustomTestCase):
    def test_none_socket_sends_nothing(self):
        """Negative gate: schedulers created without an output socket must
        drop outputs silently instead of crashing in sock_send."""
        sender = SenderWrapper(socket=None)
        with patch(_SOCK_SEND) as mock_send:
            sender.send_output(BaseReq(rid="r1"))
        mock_send.assert_not_called()

    def test_http_worker_ipc_is_back_propagated_from_recv(self):
        """Multi-HTTP-worker contract: an output produced without routing
        info inherits the receiving request's http_worker_ipc so the reply
        reaches the worker that owns the client connection."""
        sender = SenderWrapper(socket=Mock())
        recv = BaseReq(rid="r1", http_worker_ipc="worker-3")
        output = BaseReq(rid="r1")
        with patch(_SOCK_SEND) as mock_send:
            sender.send_output(output, recv_obj=recv)
        self.assertEqual(output.http_worker_ipc, "worker-3")
        mock_send.assert_called_once_with(sender.socket, output)

    def test_existing_routing_is_not_overwritten(self):
        """Negative branch: an output that already carries routing info must
        keep it — overwriting would misdeliver the reply."""
        sender = SenderWrapper(socket=Mock())
        recv = BaseReq(rid="r1", http_worker_ipc="worker-3")
        output = BaseReq(rid="r1", http_worker_ipc="worker-7")
        with patch(_SOCK_SEND):
            sender.send_output(output, recv_obj=recv)
        self.assertEqual(output.http_worker_ipc, "worker-7")

    def test_batch_recv_does_not_propagate(self):
        """Negative branch: the back-propagation rule is defined only for
        single-request recv payloads; a batch recv must leave the output
        untouched (per-request routing lives inside the batch entries)."""
        sender = SenderWrapper(socket=Mock())
        output = BaseReq(rid="r1")
        with patch(_SOCK_SEND):
            sender.send_output(output, recv_obj=BaseBatchReq())
        self.assertIsNone(output.http_worker_ipc)

    def test_missing_recv_context_still_sends(self):
        """Completeness: outputs with no recv context (scheduler-initiated
        messages) are sent as-is."""
        sender = SenderWrapper(socket=Mock())
        output = BaseReq(rid="r1")
        with patch(_SOCK_SEND) as mock_send:
            sender.send_output(output)
        self.assertIsNone(output.http_worker_ipc)
        mock_send.assert_called_once_with(sender.socket, output)


if __name__ == "__main__":
    unittest.main()
