"""PD queues set up and judge network transfers alike on every storage.

Both PD queues install a network-transfer waiter on the pool's storage at
construction, static pools included; and the decode queue's judgement of a
failed receive, which the waiter MIN-reduces across ranks every round, stays
done once its release timeout has passed. Unified memory refuses staging
buffers at startup.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import NetworkTransferWaiter
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
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
        self.kv_mgr = SimpleNamespace(
            enable_deferred_decode_kv_release=True,
            is_abort_release_safe=lambda room, acks: False,
        )

    def poll(self):
        return KVPoll.Failed if self.failed else KVPoll.Transferring

    def ensure_abort_notified(self, *, force_arm=False):
        self.abort_notified = True


class TestReceiveTimeoutStaysDone(CustomTestCase):
    @staticmethod
    def _rank():
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_staging = False
        queue.enable_deferred_kv_release = True
        queue.enable_host_receive = False
        queue.deferred_kv_release_timeout = 10.0
        queue._deferred_releases = IndexedReqList(
            ListKind.PD_RELEASING, req_of=lambda x: x[0].req
        )
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


class TestUnifiedMemoryRefusesStagingBuffers(CustomTestCase):
    """BUG REGRESSION. With staging buffers on, the decode side read transfer
    metadata back without blocking and recorded the receive as an active
    transfer before its destination was published, so a move that waited on
    it could never finish. Unified memory refuses the staging buffers."""

    def _check(self, mode):
        import msgspec

        from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="dummy")
        for name, value in {
            "enable_unified_memory": True,
            "disaggregation_mode": mode,
            "disaggregation_transfer_backend": "mooncake",
            "pp_size": 1,
            "enable_hisparse": False,
            "disaggregation_decode_enable_offload_kvcache": False,
            "disaggregation_decode_retraction_backup": None,
            "speculative_algorithm": None,
            "speculative_eagle_topk": None,
            "enable_hierarchical_cache": False,
            "enable_lmcache": False,
            "dcp_size": 1,
            "cuda_graph_config": None,
        }.items():
            msgspec.structs.force_setattr(sa, name, value)
        handle_unified_memory_pool(sa)

    def test_staging_buffers_are_refused(self):
        from sglang.srt.environ import envs

        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                with envs.SGLANG_DISAGG_STAGING_BUFFER.override(True):
                    with self.assertRaises(AssertionError) as ctx:
                        self._check(mode)
                self.assertIn("SGLANG_DISAGG_STAGING_BUFFER", str(ctx.exception))
                self.assertIn("heterogeneous-TP staging", str(ctx.exception))
                self._check(mode)


if __name__ == "__main__":
    unittest.main()
