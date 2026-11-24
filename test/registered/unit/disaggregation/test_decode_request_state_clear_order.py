"""A PD decode row's DeepSeek-V4 request-scoped state (C128 ring, pending pairs)
is reset before the prefill peer writes the transferred part of it. Two other
writers can land after a reset that is not ordered:

- under overlap, the row's previous owner can still have one redundant forward
  in flight, and it writes the same row;
- the peer's RDMA follows no stream, so once the destination is sent it can
  write the row while the reset is still queued on the device.

Each case delays the device with a sleep kernel so the unordered interleaving
is the one that runs, then checks the row's bytes: the reset's values, with the
peer's write intact and nothing of the old owner left."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.mem_cache import deepseek_v4_memory_pool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

ROW = 3
HEAD_DIM = 128
# ~0.1 s at GPU clocks: long past any host-side launch latency.
SLEEP_CYCLES = 200_000_000
STALE, PEER = 7.0, 5.0
# The part of the row the peer's transfer writes.
PEER_SLICE = slice(HEAD_DIM, HEAD_DIM + 16)


def _pool():
    """A pool whose request-scoped state is one online C128 row per request."""
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    kv_score = torch.full((8, 3 * HEAD_DIM), STALE, device="cuda")
    pool.compress_state_pools = [
        SimpleNamespace(
            request_scoped=True,
            ratio=128,
            kv_score_buffer=SimpleNamespace(kv_score=kv_score),
        )
    ]
    return pool, kv_score


def _queue(pool, *, enable_overlap, defer_publish, forward_stream):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        enable_overlap=enable_overlap,
        forward_stream=forward_stream,
        device_module=torch.cuda,
    )
    queue._defer_publish = defer_publish
    queue.token_to_kv_pool = pool
    queue.transfer_queue = SimpleNamespace(enable_staging=False)
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(alloc=lambda: 0)
    queue._record_receive = mock.Mock()
    queue._num_published_destinations = 0
    return queue


def _entry(on_send=None):
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=ROW), time_stats=mock.Mock())
    receiver = mock.Mock()
    if on_send is not None:
        receiver.send_metadata.side_effect = lambda *args, **kwargs: on_send()
    return DecodeRequest(req=req, kv_receiver=receiver)


def _peer_writes(kv_score):
    """The peer's transfer: ordered after nothing on this side's streams."""
    with torch.cuda.stream(torch.cuda.Stream()):
        kv_score[ROW, PEER_SLICE].fill_(PEER)


def _publish(queue, pool, entry, kv_indices, *, slow_reset=False):
    """What `_publish_destination` does once the payloads are built."""

    def reset_state():
        if slow_reset:
            # The reset is still running on the device when the host moves on.
            torch.cuda._sleep(SLEEP_CYCLES)
        pool.clear_request_scoped_state(ROW)

    queue._send_kv_metadata(
        entry, kv_indices, 1, state_indices=[[0]], reset_state=reset_state
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
@mock.patch.object(deepseek_v4_memory_pool, "ONLINE_C128", True)
class TestRequestStateClearOrder(CustomTestCase):
    def assert_reset_then_peer(self, kv_score):
        torch.cuda.synchronize()
        expected = torch.zeros(3 * HEAD_DIM)
        expected[:HEAD_DIM] = float("-inf")
        expected[PEER_SLICE] = PEER
        self.assertTrue(torch.equal(kv_score[ROW].cpu(), expected))

    def test_a_held_send_lands_after_the_old_owners_forward_and_the_reset(self):
        """Overlap without a deferred publish: the previous owner's redundant
        forward still writes the row when its destination is published."""
        pool, kv_score = _pool()
        forward, schedule = torch.cuda.Stream(), torch.cuda.Stream()
        queue = _queue(
            pool, enable_overlap=True, defer_publish=False, forward_stream=forward
        )
        with torch.cuda.stream(schedule):
            # A first allocation on a stream can reach cudaMalloc, which waits
            # for the whole device; the scheduler's stream has its blocks cached.
            kv_indices = torch.arange(4, device="cuda")
            torch.empty(1 << 20, device="cuda")
        torch.cuda.synchronize()
        with torch.cuda.stream(forward):
            torch.cuda._sleep(SLEEP_CYCLES)
            kv_score[ROW].fill_(STALE)
        entry = _entry()

        # Unlike the legacy default stream, the scheduler's own stream does not
        # wait for the forward stream implicitly.
        with torch.cuda.stream(schedule):
            _publish(queue, pool, entry, kv_indices)

        # The held send goes out once its event is done (`publish_pending_metadata`).
        entry.pending_metadata[0].synchronize()
        entry.kv_receiver.send_metadata.assert_not_called()
        _peer_writes(kv_score)
        self.assert_reset_then_peer(kv_score)

    def test_an_unheld_send_goes_out_after_the_reset_ran(self):
        """No overlap, and a deferred publish: the destination is sent at once,
        while the reset still runs on the device. Reading back device page
        indices after the reset waits for it; empty indices (a HiSparse request
        of 1-3 tokens) and host indices read nothing back, so the send must
        wait."""
        indices = {
            "empty device indices": lambda: torch.empty(
                0, dtype=torch.int64, device="cuda"
            ),
            "device indices": lambda: torch.arange(4, device="cuda"),
            "host indices": lambda: torch.arange(4),
        }
        for enable_overlap, defer_publish in ((False, False), (True, True)):
            for kind, make_indices in indices.items():
                with self.subTest(
                    overlap=enable_overlap, deferred=defer_publish, indices=kind
                ):
                    pool, kv_score = _pool()
                    queue = _queue(
                        pool,
                        enable_overlap=enable_overlap,
                        defer_publish=defer_publish,
                        forward_stream=torch.cuda.Stream(),
                    )
                    entry = _entry(on_send=lambda: _peer_writes(kv_score))
                    publishing = torch.cuda.Stream()
                    with torch.cuda.stream(publishing):
                        kv_indices = make_indices()
                        torch.empty(1 << 20, device="cuda")
                    torch.cuda.synchronize()
                    with torch.cuda.stream(publishing):
                        _publish(queue, pool, entry, kv_indices, slow_reset=True)

                    self.assertIsNone(entry.pending_metadata)
                    entry.kv_receiver.send_metadata.assert_called_once()
                    self.assert_reset_then_peer(kv_score)


if __name__ == "__main__":
    unittest.main()
