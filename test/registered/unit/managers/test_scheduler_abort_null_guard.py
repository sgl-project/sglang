"""Unit tests for the kv_receiver null-guard used by Scheduler.handle_abort_req.

DecodeTransferQueue clears ``decode_req.kv_receiver`` after a successful
KV transfer commit or a metadata-corruption abort. Under an ``abort_all``
wave, ``handle_abort_req`` may iterate the prealloc / transfer queues
and observe an already-nulled receiver; calling ``.abort()`` on ``None``
would raise ``AttributeError`` and kill the scheduler thread. PR #29834
added the same guard for the sibling call site in
``disaggregation/decode.py``; the two DECODE call sites in
``handle_abort_req`` are now guarded by the same shape.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler import _abort_decode_kv_receiver
from sglang.test.test_utils import CustomTestCase


class TestAbortDecodeKVReceiver(CustomTestCase):
    def test_none_receiver_is_noop(self):
        """kv_receiver set to None => helper must not raise."""
        decode_req = SimpleNamespace(kv_receiver=None)
        _abort_decode_kv_receiver(decode_req)

    def test_live_receiver_is_aborted_once(self):
        """kv_receiver present => .abort() called exactly once."""
        receiver = MagicMock()
        decode_req = SimpleNamespace(kv_receiver=receiver)
        _abort_decode_kv_receiver(decode_req)
        receiver.abort.assert_called_once_with()

    def test_receiver_flipped_to_none_between_reads(self):
        """Simulate the exact race: attribute exists but resolves to None."""

        class RacyReq:
            @property
            def kv_receiver(self):
                return None

        _abort_decode_kv_receiver(RacyReq())


if __name__ == "__main__":
    unittest.main()
