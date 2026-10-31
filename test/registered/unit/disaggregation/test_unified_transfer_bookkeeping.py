"""Bookkeeping of unified-memory PD transfers: every request in flight has a
completion flag, however many there are; a request's transfers end at its row's
release point even when a session keeps the row; and a cleared room leaves
nothing behind in the status path."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import record_network_transfer
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.kv.alloc.signals import CompletionFlagPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Manager(CommonKVManager):
    """The status path of a connection, without its sockets."""

    def __init__(self):
        self.request_status = {}
        self.deferred_bootstrap = None
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}


class TestTransferBookkeeping(CustomTestCase):
    def test_flags_outgrow_one_array(self):
        """REGRESSION: one array of completion flags asserted once every slot
        was in use, one flag per request in flight; the pool grows instead."""
        pool = CompletionFlagPool(slots_per_array=2, device_mapped=False)
        flags = [pool.acquire() for _ in range(5)]
        self.assertEqual(len({(id(f._flags), f._slot) for f in flags}), 5)
        pool.release(flags[0])
        self.assertTrue(flags[0].query(), "a released flag is published")
        again = pool.acquire()
        self.assertFalse(again.query(), "a reused slot starts unpublished")

    def test_a_cleared_room_forgets_its_conclusion(self):
        """REGRESSION: a room cleared before its transfer concluded kept its
        flag in the status path's table forever."""
        manager = _Manager()
        manager.update_status(5, KVPoll.Bootstrapping)
        flag = SimpleNamespace(publish=mock.Mock())
        manager.publish_on_conclusion(5, flag, failed_delay=None)
        self.assertIn(5, manager._conclusion_flags)
        manager.drop_conclusion(5)
        self.assertNotIn(5, manager._conclusion_flags)
        manager.update_status(5, KVPoll.Success)
        flag.publish.assert_not_called()

    def test_a_row_a_session_keeps_ends_its_transfers(self):
        """REGRESSION: a streaming session that kept a finished request's row
        returned before the row release that ends the request's transfers,
        leaking the flag and keeping the rows ordered behind it."""
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="tri",
                full_layer_ids=(0,),
                swa_layer_ids=(1,),
                state_layer_ids=(2,),
                lazy=True,
                device="cpu",
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        storage = allocator.storage
        owner = allocator.full_attn_allocator
        ids = owner.alloc(4)
        page = int(owner.ids.virtual_to_physical[ids[0] // owner.page_size])
        req = SimpleNamespace(
            rid="r0",
            bootstrap_room=1,
            kv=SimpleNamespace(holds_kv=True, is_kv_released=False),
        )
        record_network_transfer(
            allocator,
            req,
            kv_manager=_Manager(),
            writes=True,
            page_indices=np.array([page], dtype=np.int32),
            page_size=allocator.page_size,
        )
        flag, release = storage._network_transfers[req.rid]

        def claim(r):
            r.kv.holds_kv = False
            return True

        tree_cache = SimpleNamespace(
            claim_kv_row=claim,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=allocator,
        )
        release_kv_cache(req, tree_cache, checkpoint=True)
        self.assertTrue(release.released)
        self.assertTrue(flag.query())
        self.assertFalse(storage._network_transfers)


if __name__ == "__main__":
    unittest.main()
