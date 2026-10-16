"""The PD owners around each PP ring slot: a rank past the first agrees with
the ids the previous rank sent, the owner applies the agreed ids, and a chunk
sent before its commit waits for its forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.pd_pp_consensus import (
    _DisaggDecodeRing,
    _DisaggPrefillRing,
)
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PP_MODULE = "sglang.srt.disaggregation.pd_pp_consensus"


class _Channel:
    """A middle PP rank: it receives what the previous rank sent, in order."""

    is_first_rank = False
    is_last_rank = False

    def __init__(self, *received):
        self.received = list(received)
        self.sent = []

    def recv_prev(self):
        return self.received.pop(0)

    def send_next(self, data):
        self.sent.append(data)
        return []

    def commit(self, work):
        pass


def _decode_ring(owner, channel):
    return _DisaggDecodeRing(owner, channel, 2, schedule=Mock(), lora_ids_in_use=set)


class TestDisaggDecodeRing(CustomTestCase):
    def test_take_ids_agrees_with_the_previous_rank(self):
        owner = decode_owner_shell(
            retracted_rids=lambda mb_id: ["a", "b"],
            poll_prealloc_rids=lambda: (["x", "y"], ["z"]),
            aborted_prealloc_rids=lambda: {"y"},
            poll_transfer_rids=lambda: ["t1", "t2"],
        )
        channel = _Channel(["b", "c"], (["x", "y", "w"], ["v"]), ["t2", "t3"])
        ring = _decode_ring(owner, channel)

        ring.take_ids(0)

        self.assertEqual(ring.retract_rids, ["b"])
        good, bad = ring.prealloc_rids
        # Good ids need both ranks; failed or aborted ids need one.
        self.assertEqual(good, ["x"])
        self.assertEqual(set(bad), {"v", "y", "z"})
        self.assertEqual(ring.transferred_rids, ["t2"])
        self.assertEqual(ring.rmbs[0], ["b"])
        self.assertIsNone(ring.rmbs[1])

    def test_take_ids_drains_decode_hicache_events_first(self):
        # As the flat loop's intake: finished load-backs unpin their nodes
        # before the slot polls the requests waiting for room.
        calls = []
        owner = decode_owner_shell(
            enable_decode_hicache=True,
            tree_cache=SimpleNamespace(
                check_hicache_events=lambda: calls.append("hicache")
            ),
            retracted_rids=lambda mb_id: calls.append("retract") or [],
            poll_prealloc_rids=lambda: ([], []),
            aborted_prealloc_rids=set,
            poll_transfer_rids=lambda: [],
        )
        channel = _Channel()
        channel.is_first_rank = True
        ring = _decode_ring(owner, channel)

        ring.take_ids(0)
        ring.take_ids(1)

        self.assertEqual(calls, ["hicache", "retract"] * 2)

    @patch(f"{PP_MODULE}.get_disagg")
    def test_exchange_ids_applies_the_consensus_of_the_next_slot(self, get_disagg):
        get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        owner = decode_owner_shell(
            apply_retract_consensus=Mock(return_value=["r"]),
            apply_prealloc_consensus=Mock(return_value=[["p"], []]),
            apply_transfer_consensus=Mock(return_value=["t"]),
        )
        channel = _Channel(["r", "q"], [["p"], []], ["t"])
        ring = _decode_ring(owner, channel)
        ring.rmbs[1] = ring.pmbs[1] = ring.tmbs[1] = []
        # The previous consensus passes on to the next rank.
        ring.consensus_retract_rids = ["old"]

        next_slot = ScheduleSlot(index=1)
        ring.exchange_ids(next_slot, next_first_rank_mb_id=0)

        owner.apply_retract_consensus.assert_called_once_with(["r", "q"], next_slot)
        owner.apply_prealloc_consensus.assert_called_once_with(
            [["p"], []], next_slot, lora_ids_in_use=set
        )
        owner.apply_transfer_consensus.assert_called_once_with(["t"], next_slot)
        self.assertEqual(channel.sent, [["old"]])
        self.assertEqual(ring.consensus_retract_rids, ["r"])
        self.assertEqual(ring.consensus_prealloc_rids, [["p"], []])
        self.assertEqual(ring.release_rids, ["t"])

    @patch(f"{PP_MODULE}.get_disagg")
    def test_exchange_ids_skips_a_slot_that_put_nothing(self, get_disagg):
        get_disagg.return_value = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False
        )
        owner = decode_owner_shell(apply_transfer_consensus=Mock())
        channel = _Channel()
        ring = _decode_ring(owner, channel)

        ring.exchange_ids(ScheduleSlot(index=1), next_first_rank_mb_id=0)

        owner.apply_transfer_consensus.assert_not_called()
        self.assertIsNone(ring.release_rids)


class TestDisaggPrefillRing(CustomTestCase):
    def test_release_waits_for_the_slot_consensus(self):
        owner = prefill_owner_shell(poll_transfer_rids=lambda: ["a", "b"])
        intake, release = Mock(), Mock()
        channel = _Channel(["b", "c"], ["b"])
        ring = _DisaggPrefillRing(
            owner, channel, 2, intake=intake, schedule=Mock(), release=release
        )

        ring.take_ids(0)
        intake.assert_called_once_with()
        self.assertEqual(ring.transferred_rids, ["b"])

        # Slot 1 put nothing to the consensus: nothing is received or released.
        ring.exchange_ids(ScheduleSlot(index=1), next_first_rank_mb_id=0)
        ring.after_commit(next_mb_id=1)
        release.assert_not_called()

        ring.exchange_ids(ScheduleSlot(index=0), next_first_rank_mb_id=1)
        ring.after_commit(next_mb_id=0)
        release.assert_called_once_with(["b"])

        ring.send_ids()
        self.assertEqual(channel.sent, [["b"]])


class TestPPChunkSend(CustomTestCase):
    """A PP ring slot sends the chunk the previous slot launched before that
    chunk's commit; the transfer waits for the forward that writes it."""

    def _send_chunk(self, pp_size):
        enter_scope(self, published_topology(pp_size=pp_size))
        sends = []
        forward_done = object()
        sender = SimpleNamespace(_early_send_wait_event=None)
        owner = prefill_owner_shell(
            enable_overlap=False,
            forward_stream=SimpleNamespace(record_event=lambda: forward_done),
            checkpoint=Mock(),
            check_bootstrap=lambda req: True,
            send_kv_chunk=lambda req: sends.append(sender._early_send_wait_event),
        )
        req = SimpleNamespace(rid="chunked", extend_end=8, prefix_len=4)
        owner.kv_senders[req.rid] = sender

        chunked_req, _ = owner.process_prefill_chunk(req, [], [])

        self.assertIs(chunked_req, req)
        return sends, forward_done

    def test_a_pp_chunk_send_waits_for_its_forward(self):
        sends, forward_done = self._send_chunk(pp_size=2)
        self.assertEqual(sends, [forward_done])

    def test_a_flat_chunk_send_follows_its_commit(self):
        # Without PP the step committed before this schedule: nothing to wait on.
        sends, _ = self._send_chunk(pp_size=1)
        self.assertEqual(sends, [None])


if __name__ == "__main__":
    unittest.main()
