"""Unit tests for KV-transfer replication accounting in PD disaggregation.

In Prefill-CP + Decode-TP (e.g. prefill CP8, decode TP8), by default only prefill
CP rank 0 transfers, replicating its KV to every decode TP rank. get_transfer_metric()
must report bytes put on the wire (logical KV size x fan-out). The fan-out equals
required_dst_info_num -- a topological invariant -- so it is resolved once and cached
on the shared CommonKVManager.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVTransferMetric
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

KV_ITEM_LENS_SUM = 100
STATE_ITEM_LENS_SUM = 7


def _room(fan_out):
    """A room of `fan_out` destinations, each reporting required_dst_info_num.

    All infos in a room report the same required_dst_info_num (the registration
    barrier waits until that many arrive), so the resolver reads it off any one.
    """
    return {
        f"sess{i}": SimpleNamespace(required_dst_info_num=fan_out)
        for i in range(fan_out)
    }


def _make_kv_mgr(is_mla_backend):
    """CommonKVManager bypassing __init__, wiring only the fields the path reads."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.is_mla_backend = is_mla_backend
    mgr.kv_item_lens_sum = KV_ITEM_LENS_SUM
    mgr.state_item_lens_sum = STATE_ITEM_LENS_SUM
    mgr._kv_replica_factor = None if is_mla_backend else 1
    return mgr


class _ConcreteKVSender(CommonKVSender):
    """CommonKVSender is abstract: `poll` and `failure_exception` are backend
    duties, so it cannot be instantiated (not even via __new__, which enforces
    ABC completeness). This stub supplies them so the shared metric path can be
    exercised without pulling in a transfer backend.
    """

    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _make_sender(kv_mgr):
    """CommonKVSender bypassing __init__, wiring only the fields the path reads."""
    sender = _ConcreteKVSender.__new__(_ConcreteKVSender)
    sender._transfer_metric = KVTransferMetric()
    sender._transfer_num_kv_indices = 0
    sender._transfer_num_state_indices = 0
    sender.kv_mgr = kv_mgr
    return sender


class TestKVTransferReplicaMetric(CustomTestCase):
    def test_mla_scales_kv_and_state_bytes_by_fan_out(self):
        # CP rank 0 replicates to 4 decode TP ranks; both kv and state scale.
        mgr = _make_kv_mgr(is_mla_backend=True)
        sender = _make_sender(mgr)

        mgr.resolve_kv_replica_factor(_room(4))
        self.assertEqual(mgr._kv_replica_factor, 4)

        sender._record_transfer_indices(
            np.arange(8, dtype=np.int32), [np.arange(5, dtype=np.int32)]
        )
        expected = (8 * KV_ITEM_LENS_SUM + 5 * STATE_ITEM_LENS_SUM) * 4
        self.assertEqual(sender.get_transfer_metric().transfer_total_bytes, expected)

    def test_mooncake_skips_kv_bytes_on_non_elected_replica_ranks(self):
        room = 17
        peer = SimpleNamespace(
            dst_attn_tp_size=1,
            dst_tp_rank=0,
            requires_dcp_relayout=False,
            staging_base_ptr=0,
            staging_total_size=0,
            staging=None,
        )
        req = SimpleNamespace(room=room, mooncake_session_id="decode:0", is_dummy=False)

        elected_ranks = []
        for rank in range(8):
            mgr = MooncakeKVManager.__new__(MooncakeKVManager)
            mgr.is_mla_backend = False
            mgr.is_hybrid_mla_backend = False
            mgr.enable_staging = False
            mgr.kv_buffer_tensors = None
            mgr.attn_tp_size = 8
            mgr.kv_item_lens_sum = KV_ITEM_LENS_SUM
            mgr.state_item_lens_sum = STATE_ITEM_LENS_SUM
            mgr._kv_replica_factor = 1
            mgr.kv_args = SimpleNamespace(
                engine_rank=rank,
                total_kv_head_num=1,
                kv_head_num=1,
                state_types=[],
            )
            mgr.transfer_infos = {room: {"decode:0": req}}
            mgr.decode_kv_args_table = {"decode:0": peer}

            sender = MooncakeKVSender.__new__(MooncakeKVSender)
            sender.kv_mgr = mgr
            sender.bootstrap_room = room
            sender._transfer_metric = KVTransferMetric()
            sender._transfer_num_kv_indices = 0
            sender._transfer_num_state_indices = 0

            sender._record_transfer_indices(np.arange(6, dtype=np.int32), None)
            total_bytes = sender.get_transfer_metric().transfer_total_bytes
            if mgr.sends_main_kv_for_room(room):
                elected_ranks.append(rank)
                self.assertEqual(total_bytes, 6 * KV_ITEM_LENS_SUM)
            else:
                self.assertIsNone(total_bytes)

                # Extra state still sent by non-elected ranks remains accounted.
                with get_context().override_server_args(enable_kv_cache_sharding=False):
                    sender._record_transfer_indices(
                        np.arange(4, dtype=np.int32), [np.arange(3, dtype=np.int32)]
                    )
                self.assertEqual(
                    sender.get_transfer_metric().transfer_total_bytes,
                    3 * STATE_ITEM_LENS_SUM,
                )

        self.assertEqual(len(elected_ranks), 1)


if __name__ == "__main__":
    unittest.main()
