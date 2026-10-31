"""Bookkeeping of unified-memory PD transfers: each send or receive is its own
transfer; a page move that must wait for one waits on the host, releasing the
GIL, until every rank's engine is done with that one transfer, so a chunked
request never waits for its own later chunks; a request's transfers end at its
row's release point even when a session keeps the row; and a cleared room
leaves nothing behind in the status path."""

import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.utils import (
    NetworkTransferWaiter,
    record_network_transfer,
)
from sglang.srt.mem_cache.common import release_kv_cache
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


class TestSendCompletion(CustomTestCase):
    """What a page move waits for on a prefill node: one send's rows, until the
    rank's transfer worker is done with that send."""

    def _queued_send(self, manager, room):
        sender = SimpleNamespace(kv_mgr=manager, bootstrap_room=room)
        peer = prefill._send_peer(sender)
        manager.note_send_enqueued(room)
        return peer

    def test_a_send_ends_when_the_worker_finishes_it_not_the_room(self):
        """A chunked request's rows a move needs are read by its earlier
        sends only; waiting for the whole room would wait for chunks the
        scheduler has yet to run. Each send ends when the worker is done with
        it, before the room concludes."""
        manager = _Manager()
        manager.update_status(7, KVPoll.Bootstrapping)
        manager.update_status(7, KVPoll.Transferring)
        first = self._queued_send(manager, 7)
        second = self._queued_send(manager, 7)
        self.assertFalse(prefill._send_done(first))
        manager.note_send_finished(7)
        self.assertTrue(prefill._send_done(first))
        self.assertFalse(prefill._send_done(second))
        self.assertEqual(manager.request_status[7], KVPoll.Transferring)

    def test_a_send_never_queued_or_of_a_failed_room_waits_for_nothing(self):
        manager = _Manager()
        manager.update_status(8, KVPoll.Bootstrapping)
        skipped = prefill._send_peer(SimpleNamespace(kv_mgr=manager, bootstrap_room=8))
        self.assertTrue(prefill._send_done(skipped), "no worker reads its rows")
        queued = self._queued_send(manager, 8)
        self.assertFalse(prefill._send_done(queued))
        manager.update_status(8, KVPoll.Failed)
        self.assertTrue(prefill._send_done(queued))
        self.assertTrue(prefill._send_done(None))

    def test_a_new_room_lifecycle_forgets_the_last_ones_sends(self):
        manager = _Manager()
        manager.update_status(9, KVPoll.Bootstrapping)
        manager.note_send_enqueued(9)
        manager.note_send_finished(9)
        manager.request_status.pop(9)
        manager.update_status(9, KVPoll.Bootstrapping)
        self.assertEqual(manager.send_counts(9), (0, 0))


class TestNetworkTransferWaiter(CustomTestCase):
    def test_the_waiter_releases_the_gil_while_the_engine_reports(self):
        """REGRESSION: a move waited on the GPU for a flag a Python thread
        published; the scheduler then synchronized that stream holding the
        GIL, and the engine's thread could never publish. The waiter waits on
        the host and sleeps between rounds, so a thread that must take the GIL
        to report finishes."""
        done = threading.Event()
        reported = []

        def engine():
            time.sleep(0.02)
            reported.append(True)  # needs the GIL
            done.set()

        waiter = NetworkTransferWaiter(lambda peer: done.is_set(), (None,))
        thread = threading.Thread(target=engine)
        thread.start()
        waiter([SimpleNamespace(peer="send")])
        thread.join()
        self.assertEqual(reported, [True])

    def test_the_waiter_returns_once_every_transfer_is_done(self):
        rounds = defaultdict(int)

        def done(peer):
            rounds[peer] += 1
            return rounds[peer] >= (3 if peer == "slow" else 1)

        NetworkTransferWaiter(done, ())(
            [SimpleNamespace(peer="fast"), SimpleNamespace(peer="slow")]
        )
        self.assertEqual(rounds["slow"], 3)


def _agreeing_rank(rank, path, out):
    """One rank of a two-rank group: rank 1's engine finishes later. Each rank
    records how many of its own rounds said done before the waiter returned."""
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2
    )
    try:
        start = time.monotonic()
        delay = 0.0 if rank == 0 else 0.2
        rounds = []

        def done(peer):
            rounds.append(time.monotonic() - start >= delay)
            return rounds[-1]

        NetworkTransferWaiter(done, (dist.group.WORLD,))([SimpleNamespace(peer=0)])
        out.put((rank, len(rounds), sum(rounds), time.monotonic() - start))
    finally:
        dist.destroy_process_group()


class TestWaiterAgreesAcrossRanks(CustomTestCase):
    def test_every_rank_returns_in_the_same_round(self):
        """A rank whose engine is done keeps waiting until every rank's is:
        the ranks keep one layout, so they must leave a wait together."""
        import multiprocessing as mp
        import os
        import tempfile

        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store")
            procs = [
                ctx.Process(target=_agreeing_rank, args=(rank, path, out))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            results = sorted(out.get(timeout=60) for _ in procs)
            for proc in procs:
                proc.join(timeout=60)
        (_, rounds0, done0, _), (_, rounds1, done1, _) = results
        self.assertEqual(rounds0, rounds1, "the ranks left after different rounds")
        self.assertEqual(done0, rounds0)
        self.assertGreater(rounds0, 1, "rank 0 left before rank 1 was done")
        self.assertEqual(done1, 1, "rank 1 left on its first done round")


class TestReceiveCompletion(CustomTestCase):
    """What a page move waits for on a decode node: a receive, until it
    succeeded, or failed and drained (main's deferred release point)."""

    def _queue(self, *, deferred=True, timeout=30.0):
        return SimpleNamespace(
            enable_staging=False,
            staging_handler=None,
            enable_deferred_kv_release=deferred,
            deferred_kv_release_timeout=timeout,
            _receive_failed_at={},
            _is_host_destination=lambda decode_req: False,
            _required_abort_acks=lambda decode_req: 1,
        )

    def _decode_req(self, poll, *, drained=False):
        receiver = SimpleNamespace(
            require_staging=False,
            poll=lambda: poll,
            kv_mgr=SimpleNamespace(is_abort_release_safe=lambda room, acks: drained),
        )
        return SimpleNamespace(
            kv_receiver=receiver, req=SimpleNamespace(rid="r", bootstrap_room=3)
        )

    def _done(self, queue, decode_req):
        return DecodeTransferQueue._receive_done(queue, decode_req)

    def test_a_receive_ends_at_success_or_a_drained_failure(self):
        queue = self._queue()
        self.assertTrue(self._done(queue, self._decode_req(KVPoll.Success)))
        self.assertFalse(self._done(queue, self._decode_req(KVPoll.Transferring)))
        self.assertFalse(self._done(queue, self._decode_req(KVPoll.Failed)))
        self.assertTrue(
            self._done(queue, self._decode_req(KVPoll.Failed, drained=True))
        )
        self.assertTrue(self._done(queue, SimpleNamespace(kv_receiver=None)))

    def test_an_undrained_failure_ends_at_the_release_timeout(self):
        queue = self._queue(timeout=0.0)
        self.assertTrue(self._done(queue, self._decode_req(KVPoll.Failed)))
        self.assertTrue(
            self._done(self._queue(deferred=False), self._decode_req(KVPoll.Failed))
        )


class TestTransferBookkeeping(CustomTestCase):
    def test_a_row_a_session_keeps_ends_its_transfers(self):
        """REGRESSION: a streaming session that kept a finished request's row
        returned before the row release that ends the request's transfers,
        keeping the rows ordered behind them forever."""
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
        for _ in range(2):
            record_network_transfer(
                allocator,
                req,
                peer=None,
                writes=True,
                page_indices=np.array([page], dtype=np.int32),
                page_size=allocator.page_size,
            )
        transfers = list(storage._network_transfers[req.rid])
        self.assertEqual(len(transfers), 2, "one transfer per receive")

        def claim(r):
            r.kv.holds_kv = False
            return True

        tree_cache = SimpleNamespace(
            claim_kv_row=claim,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=allocator,
        )
        release_kv_cache(req, tree_cache, checkpoint=True)
        self.assertTrue(all(t.released for t in transfers))
        self.assertFalse(storage._network_transfers)

    def test_the_ledger_takes_device_rows_the_send_copies(self):
        """REGRESSION: under host-tensor indices the prefill handed the
        page-move ledger the pinned host copy of a window's rows, which fills
        only when the schedule stream reaches the copy; the ledger read it at
        once. The send takes the host copy, the ledger the device ids."""
        rows = torch.tensor([4, 5, 6])
        payloads = [prefill._DeviceRows(rows), [np.array(1)], None]
        sent = [prefill._rows_on_host(p) for p in payloads]
        recorded = [prefill._rows_on_device(p) for p in payloads]
        self.assertIsInstance(sent[0], np.ndarray)
        self.assertEqual(sent[0].tolist(), [4, 5, 6])
        self.assertIs(recorded[0], rows)
        self.assertEqual(sent[1:], recorded[1:])


if __name__ == "__main__":
    unittest.main()
