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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

KV_ITEM_LENS_SUM = 100
# One state component, per-layer item lengths summing to 7.
STATE_ITEM_LENS_PER_COMPONENT = [7]


def _room(fan_out):
    """A room of `fan_out` destinations, each reporting required_dst_info_num.

    All infos in a room report the same required_dst_info_num (the registration
    barrier waits until that many arrive), so the resolver reads it off any one.
    """
    return {
        f"sess{i}": SimpleNamespace(required_dst_info_num=fan_out)
        for i in range(fan_out)
    }


def _make_kv_mgr(is_mla_backend, state_item_lens_per_component=None):
    """CommonKVManager bypassing __init__, wiring only the fields the path reads."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.is_mla_backend = is_mla_backend
    mgr.kv_item_lens_sum = KV_ITEM_LENS_SUM
    mgr.state_item_lens_per_component = (
        STATE_ITEM_LENS_PER_COMPONENT
        if state_item_lens_per_component is None
        else state_item_lens_per_component
    )
    mgr._kv_replica_factor = None if is_mla_backend else 1
    return mgr


def _make_sender(kv_mgr):
    """CommonKVSender bypassing __init__, wiring only the fields the path reads."""
    sender = CommonKVSender.__new__(CommonKVSender)
    sender._transfer_metric = KVTransferMetric()
    sender._transfer_num_kv_indices = 0
    sender._transfer_state_bytes = 0
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
        expected = (8 * KV_ITEM_LENS_SUM + 5 * STATE_ITEM_LENS_PER_COMPONENT[0]) * 4
        self.assertEqual(sender.get_transfer_metric().transfer_total_bytes, expected)

    def test_non_mla_factor_is_one_regardless_of_destinations(self):
        # Non-MLA head slices sum to one logical copy: factor stays 1.
        mgr = _make_kv_mgr(is_mla_backend=False)
        sender = _make_sender(mgr)

        mgr.resolve_kv_replica_factor(_room(8))
        self.assertEqual(mgr._kv_replica_factor, 1)

        sender._record_transfer_indices(np.arange(6, dtype=np.int32), None)
        self.assertEqual(
            sender.get_transfer_metric().transfer_total_bytes, 6 * KV_ITEM_LENS_SUM
        )

    def test_each_state_component_is_charged_its_own_item_length(self):
        """State components ship independent index lists of independent sizes.

        The metric used to flatten both sides -- one summed index count times one
        summed item length -- which is a cross product: every index of every
        component was charged the item length of every OTHER component too. It
        only agreed with reality when a single component was in play, so the
        error grew with the number of components and was invisible to a
        single-component test.

        DeepSeek-V4 ships SWA_RING and C128_STATE together, with a small
        sliding-window index list against a much longer compressed-state one.
        """
        # 2 components: 3 indices at 10 bytes/index, 1 index at 1000 bytes/index.
        mgr = _make_kv_mgr(
            is_mla_backend=False, state_item_lens_per_component=[10, 1000]
        )
        sender = _make_sender(mgr)

        sender._record_transfer_indices(
            np.arange(2, dtype=np.int32),
            [np.arange(3, dtype=np.int32), np.arange(1, dtype=np.int32)],
        )

        expected = 2 * KV_ITEM_LENS_SUM + (3 * 10 + 1 * 1000)
        self.assertEqual(sender.get_transfer_metric().transfer_total_bytes, expected)
        # The flattened form would have charged (3 + 1) * (10 + 1000) = 4040
        # instead of 1030 -- roughly 4x, and independent of which component the
        # indices actually belonged to.
        self.assertNotEqual(
            sender.get_transfer_metric().transfer_total_bytes,
            2 * KV_ITEM_LENS_SUM + (3 + 1) * (10 + 1000),
        )

    def test_a_component_with_no_indices_costs_nothing(self):
        """A `None` slot means that component shipped nothing this transfer.

        Charging it anything at all reintroduces the cross product in miniature:
        under the flattened form an absent component still inflated the per-index
        price paid by its neighbours.
        """
        mgr = _make_kv_mgr(
            is_mla_backend=False, state_item_lens_per_component=[10, 1000]
        )
        sender = _make_sender(mgr)

        sender._record_transfer_indices(
            np.empty(0, dtype=np.int32), [np.arange(4, dtype=np.int32), None]
        )
        self.assertEqual(sender.get_transfer_metric().transfer_total_bytes, 4 * 10)

    def test_unresolved_factor_does_not_crash_metric(self):
        # An empty room or a missing required_dst_info_num leaves the factor
        # unresolved. get_transfer_metric() must still compute -- it must not
        # multiply bytes by None.
        for room in ({}, {"sess0": SimpleNamespace(required_dst_info_num=None)}):
            mgr = _make_kv_mgr(is_mla_backend=True)
            sender = _make_sender(mgr)

            mgr.resolve_kv_replica_factor(room)
            self.assertIsNone(mgr._kv_replica_factor)

            sender._record_transfer_indices(np.arange(4, dtype=np.int32), None)
            self.assertIsInstance(
                sender.get_transfer_metric().transfer_total_bytes, int
            )


if __name__ == "__main__":
    unittest.main()
