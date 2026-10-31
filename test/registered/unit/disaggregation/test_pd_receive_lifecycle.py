"""PD queues set up and judge network transfers alike on every storage.

Both PD queues install a network-transfer waiter on the pool's storage at
construction, static pools included; and the decode queue's judgement of a
failed receive, which the waiter MIN-reduces across ranks every round, stays
done once its release timeout has passed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import NetworkTransferWaiter
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage, kv_storage_of
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStaticStorageNetworkWaiter(CustomTestCase):
    def test_static_storage_takes_both_queues_waiters(self):
        """BUG REGRESSION. The prefill and decode queues register a network
        waiter on `kv_storage_of(...)` at construction; for a static pool that
        is a `StaticKVStorage`, which had no `set_network_waiter`, so every PD
        server without unified memory failed to start."""
        storage = kv_storage_of(SimpleNamespace(), SimpleNamespace(storage=None))
        self.assertIsInstance(storage, StaticKVStorage)
        # The prefill queue's waiter (attn TP and CP groups) and the decode
        # queue's (its gloo group): a static pool never moves a row, so it
        # never calls either.
        storage.set_network_waiter(NetworkTransferWaiter(lambda peer: True, (None,)))
        storage.set_network_waiter(NetworkTransferWaiter(lambda peer: True, ()))


class _FailedReceiver:
    """A receive that failed once ``failed`` is set, with the prefill's drain
    acknowledgements never arriving."""

    def __init__(self):
        self.failed = False
        self.require_staging = False
        self.abort_notified = True
        self.bootstrap_infos = [object()]
        self.kv_mgr = SimpleNamespace(is_abort_release_safe=lambda room, acks: False)

    def poll(self):
        return KVPoll.Failed if self.failed else KVPoll.Transferring


class TestReceiveTimeoutStaysDone(CustomTestCase):
    @staticmethod
    def _rank():
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_staging = False
        queue.enable_deferred_kv_release = True
        queue.enable_host_receive = False
        queue.deferred_kv_release_timeout = 10.0
        queue._receive_failed_at = {}
        receiver = _FailedReceiver()
        decode_req = DecodeRequest(
            req=FakeReq(rid="r", bootstrap_room=7), kv_receiver=receiver
        )
        return queue, receiver, decode_req

    def test_staggered_failures_agree_once_timed_out(self):
        """BUG REGRESSION. A receive that timed out answered done once, then
        restarted its timer. Ranks that saw the failure at different times took
        turns answering done, so the waiter's MIN never agreed and the move
        waited forever."""
        ranks = [self._rank(), self._rank()]
        now = [0.0]
        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic",
            side_effect=lambda: now[0],
        ):
            agreed_at = None
            for step in range(80):
                now[0] = step * 0.5
                ranks[0][1].failed = True
                # The second rank sees the failure one second later.
                ranks[1][1].failed = now[0] >= 1.0
                flags = [queue._receive_done(req) for queue, _, req in ranks]
                if all(flags) and agreed_at is None:
                    agreed_at = now[0]
                if agreed_at is not None:
                    # Done stays done until the release.
                    self.assertEqual(flags, [True, True], now[0])
        self.assertEqual(agreed_at, 11.0)


if __name__ == "__main__":
    unittest.main()
