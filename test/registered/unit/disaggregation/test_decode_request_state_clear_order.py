"""A PD decode row's DeepSeek-V4 request-state reset lands after the row's
previous owner and before the peer's transfer. Sleep kernels force the
unordered interleaving, then the row's bytes are checked."""

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
# ~0.1 s: far longer than any host-side launch.
SLEEP_CYCLES = 200_000_000
STALE, PEER = 7.0, 5.0
PEER_SLICE = slice(HEAD_DIM, HEAD_DIM + 16)  # what the peer writes


def _pool():
    """One online C128 row per request."""
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
    """The peer's RDMA: ordered after nothing on this side."""
    with torch.cuda.stream(torch.cuda.Stream()):
        kv_score[ROW, PEER_SLICE].fill_(PEER)


def _publish(queue, pool, entry, kv_indices, *, slow_reset=False):
    """Send as `_publish_destination` does; ``slow_reset`` keeps the reset
    running on the device after the host moves on."""

    def reset_state():
        if slow_reset:
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
        """Overlap: the previous owner's forward still writes the row."""
        pool, kv_score = _pool()
        forward, schedule = torch.cuda.Stream(), torch.cuda.Stream()
        queue = _queue(
            pool, enable_overlap=True, defer_publish=False, forward_stream=forward
        )
        with torch.cuda.stream(schedule):
            # Warm the allocator: a first cudaMalloc waits for the whole device.
            kv_indices = torch.arange(4, device="cuda")
            torch.empty(1 << 20, device="cuda")
        torch.cuda.synchronize()
        with torch.cuda.stream(forward):
            torch.cuda._sleep(SLEEP_CYCLES)
            kv_score[ROW].fill_(STALE)
        entry = _entry()

        # Unlike the default stream, it does not wait for other streams.
        with torch.cuda.stream(schedule):
            _publish(queue, pool, entry, kv_indices)

        # The held send goes out once its event is done.
        entry.pending_metadata[0].synchronize()
        entry.kv_receiver.send_metadata.assert_not_called()
        _peer_writes(kv_score)
        self.assert_reset_then_peer(kv_score)

    def test_an_unheld_send_goes_out_after_the_reset_ran(self):
        """No overlap or a deferred publish: the send goes out at once, while
        the reset still runs. A device readback waits for it; empty indices (a
        short HiSparse request) and host indices must wait explicitly."""
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
