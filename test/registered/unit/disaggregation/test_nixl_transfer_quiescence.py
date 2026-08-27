"""CPU regressions for NIXL transfer quiescence accounting."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlTransferGroupErrOwnership(unittest.TestCase):
    room = 73

    def test_failed_room_skip_preserves_unresolved_group_and_does_not_ack(self):
        mgr = object.__new__(NixlKVManager)
        mgr.request_status = {self.room: KVPoll.Failed}
        mgr.enable_deferred_decode_kv_release = True
        mgr._staging_outstanding = defaultdict(int, {self.room: 1})
        mgr._deferred_ack_targets = {}
        mgr._sent_acks = []
        mgr._send_abort_ack = lambda ip, port, room: mgr._sent_acks.append(
            (ip, port, room)
        )
        mgr.register_deferred_ack_target(self.room, "10.0.0.8", 6000)
        mgr.maybe_send_extra = MagicMock()
        mgr.send_aux = MagicMock()
        chunk = TransferKVChunk(
            room=self.room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=[[1]],
        )
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

        self.assertEqual(mgr._staging_outstanding[self.room], 1)
        self.assertEqual(mgr._sent_acks, [])
        self.assertIn(self.room, mgr._deferred_ack_targets)
        mgr.maybe_send_extra.assert_not_called()
        mgr.send_aux.assert_not_called()


if __name__ == "__main__":
    unittest.main()
