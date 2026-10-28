"""The PD owners around each PP ring slot: a rank past the first agrees with
the ids the previous rank sent, the owner applies the agreed ids, and a chunk
sent before its commit waits for its forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.pd_pp_consensus import (
    _DisaggDecodeRing,
    _DisaggPrefillRing,
)
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PP_MODULE = "sglang.srt.disaggregation.pd_pp_consensus"
DECODE_MODULE = "sglang.srt.disaggregation.decode"


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
    return _DisaggDecodeRing(owner, channel, 2, lora_ids_in_use=set)


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


class _PPHiCache:
    """One PP rank's decode HiCache, reduced to its load-back acks. As in
    UnifiedRadixCache.check_hicache_events, PP0 counts its finished loads at
    the head of its queue and sends the count down the ring, and every rank
    reaps that many acks from its own queue."""

    def __init__(self, pp_rank, sent_counts, dma_time):
        self.pp_rank = pp_rank
        self.sent_counts = sent_counts
        # How long a load takes on this rank, and the rank's clock.
        self.dma_time = dma_time
        self.now = 0.0
        # Per load, when it finishes; the loads not yet reaped.
        self.done_at = []
        self.ack_load_queue = []
        # Read only when a queued load-back starts no new copy.
        self.cache_controller = SimpleNamespace(
            layer_done_counter=SimpleNamespace(producer_index=-1)
        )

    def check_hicache_events(self):
        if self.pp_rank == 0:
            count = 0
            for load in self.ack_load_queue:
                if self.done_at[load] > self.now:
                    break
                count += 1
            self.sent_counts.append(count)
        else:
            count = self.sent_counts.pop(0)
        for _ in range(count):
            # Reaping waits for the load to finish.
            load = self.ack_load_queue.pop(0)
            self.done_at[load] = min(self.done_at[load], self.now)

    def has_free_load_back_slot(self):
        return len(self.ack_load_queue) < 2

    def ready_to_load_host_cache(self):
        self.done_at.append(self.now + self.dma_time)
        self.ack_load_queue.append(len(self.done_at) - 1)
        return len(self.done_at) - 1

    def is_load_back_event_done(self, consumer_index):
        return self.now >= self.done_at[consumer_index]


class _LandedReceiver:
    """A KV receiver whose transfer from the prefill node has landed."""

    def poll(self):
        return KVPoll.Success


def _decode_rank(pp_rank, sent_counts, dma_time, rids):
    tree_cache = _PPHiCache(pp_rank, sent_counts, dma_time)
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = [
        DecodeRequest(
            req=SimpleNamespace(rid=rid, finished_reason=None),
            kv_receiver=_LandedReceiver(),
            metadata_buffer_index=i,
            prefix_match=SimpleNamespace(needs_local_restore=True),
        )
        for i, rid in enumerate(rids)
    ]
    queue.__dict__.update(
        tree_cache=tree_cache,
        enable_decode_hicache=True,
        enable_staging=False,
        enable_host_receive=False,
        enable_hisparse=False,
        gloo_group=None,
        metadata_buffers=SimpleNamespace(clear_slot=lambda idx: None),
        req_to_metadata_buffer_idx_allocator=Mock(),
        _deferred_releases=[],
        _commit_transfer_to_req=Mock(),
    )

    def queue_load_back(decode_req):
        decode_req.hicache_restore_lock = object()
        return True

    queue._try_hicache_queue_load_back = queue_load_back
    return decode_owner_shell(
        enable_decode_hicache=True,
        enable_hisparse=False,
        tree_cache=tree_cache,
        transfer_queue=queue,
        core_lists=SimpleNamespace(decoding=[]),
        attn_cp_cpu_group=None,
        attn_tp_cpu_group=None,
    )


def _local_polls(pollers, *groups, **kwargs):
    return [poller.poll() for poller in pollers]


@patch(f"{DECODE_MODULE}.poll_and_all_reduce", _local_polls)
@patch(f"{DECODE_MODULE}.poll_and_all_reduce_attn_cp_tp_group", _local_polls)
class TestDecodeHiCacheAcrossPPRanks(CustomTestCase):
    """Under PP every rank restores a transferred request's prefix from its own
    host copy. The ranks must queue the same load-backs at the same ring point,
    or the load acks PP0 counts are popped on a rank that has not queued them,
    and must release a request in the same slot, or the microbatch differs."""

    def _run_ring(self, pp0_dma_time, pp1_dma_time):
        sent_counts = []
        pp0 = _decode_rank(0, sent_counts, pp0_dma_time, rids=["a", "b"])
        pp1 = _decode_rank(1, sent_counts, pp1_dma_time, rids=["a", "b"])
        ranks = (pp0, pp1)
        slot = ScheduleSlot(index=0)
        released = []
        for step in range(1, 7):
            # take_ids: every rank polls; then the slot's schedule drains
            # every rank's HiCache events.
            for rank in ranks:
                rank.tree_cache.now = step
            consensus = sorted(
                set(pp0.poll_transfer_rids()) & set(pp1.poll_transfer_rids())
            )
            for rank in ranks:
                rank.check_hicache_events()
            # exchange_ids, half a slot later: PP0 applies the consensus, and
            # the next rank applies the ids PP0 released.
            for rank in ranks:
                rank.tree_cache.now = step + 0.5
            released_pp0 = pp0.apply_transfer_consensus(consensus, slot)
            released_pp1 = pp1.apply_transfer_consensus(released_pp0, slot)
            self.assertEqual(released_pp1, released_pp0)
            released += released_pp0
        self.assertEqual(sorted(released), ["a", "b"])

    def test_a_late_rank_reaps_only_loads_it_queued(self):
        # PP0's load lands before the next slot counts it.
        self._run_ring(pp0_dma_time=0.25, pp1_dma_time=0.25)

    def test_the_ranks_release_a_restored_request_in_one_slot(self):
        # PP0's load lands between the next slot's count and its release;
        # PP1's lands later.
        self._run_ring(pp0_dma_time=1, pp1_dma_time=2)


def _ready_req(rid, *, backup=None, copied=None):
    """A ready request; `copied` is whether its restore copy has finished on
    this rank, None when it restores nothing."""
    event = None if copied is None else SimpleNamespace(query=lambda: copied)
    return SimpleNamespace(
        rid=rid,
        last_node=object(),
        kv=SimpleNamespace(
            kv_committed_len=None, retraction_backup=backup, restore_done_event=event
        ),
        init_next_round_input=Mock(),
        time_stats=MagicMock(),
    )


@patch(f"{DECODE_MODULE}.start_restore_kv_cache")
@patch(f"{DECODE_MODULE}.finish_restore_kv_cache")
class TestHostRestoreAdmission(CustomTestCase):
    """KV received into host memory is copied into its row before admission
    without blocking the scheduler. The ranks sharing a request must admit it
    in the same step, so it waits until every rank's copy has finished; PP
    stages cannot agree, so they admit it and wait for the copy on the host."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_enable_radix_cache=True
        )
        override.install()
        self.addCleanup(override.restore)

    def _admit(self, ready, *, pp_size=1, reduce=None):
        enter_scope(self, published_topology(pp_size=pp_size))
        owner = decode_owner_shell(
            enable_host_receive=True,
            restore_on_forward_stream=False,
            enable_priority_scheduling=False,
            req_to_token_pool=SimpleNamespace(size=8),
            max_running_requests=8,
            core_lists=SimpleNamespace(decoding=[]),
            tree_cache=Mock(),
            attn_cp_cpu_group=None,
            ready=list(ready),
        )
        with patch(
            f"{DECODE_MODULE}.all_reduce_flags",
            reduce or (lambda flags, group, op: flags),
        ):
            admitted = owner.pop_reqs_to_admit(ScheduleSlot())
        return owner, admitted

    def test_a_copy_unfinished_on_a_peer_rank_keeps_the_request_ready(
        self, finish, start
    ):
        restored = _ready_req("restored", copied=True)
        plain = _ready_req("plain")
        owner, admitted = self._admit(
            [restored, plain], reduce=lambda flags, group, op: [0] * len(flags)
        )
        self.assertEqual(admitted, [plain])
        self.assertEqual(owner.ready, [restored])

        owner, admitted = self._admit(owner.ready)
        self.assertEqual(admitted, [restored])
        finish.assert_called_with(restored, owner.tree_cache)

    def test_pp_admits_a_pending_copy_and_waits_on_the_host(self, finish, start):
        restored = _ready_req("restored", copied=False)
        owner, admitted = self._admit([restored], pp_size=2)
        self.assertEqual(admitted, [restored])
        finish.assert_called_once_with(restored, owner.tree_cache)

    def test_a_received_request_starts_its_copy_before_admission(self, finish, start):
        def start_copy(req, tree_cache):
            req.kv.retraction_backup = None
            req.kv.restore_done_event = SimpleNamespace(query=lambda: False)

        start.side_effect = start_copy
        received = _ready_req("received", backup=object())
        owner, admitted = self._admit([received])
        start.assert_called_once_with(received, owner.tree_cache)
        self.assertEqual(admitted, [])
        self.assertEqual(owner.ready, [received])
        finish.assert_not_called()


class TestDisaggPrefillRing(CustomTestCase):
    def test_release_waits_for_the_slot_consensus(self):
        owner = prefill_owner_shell(poll_transfer_rids=lambda: ["a", "b"])
        intake, release = Mock(), Mock()
        channel = _Channel(["b", "c"], ["b"])
        ring = _DisaggPrefillRing(owner, channel, 2, intake=intake, release=release)

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
