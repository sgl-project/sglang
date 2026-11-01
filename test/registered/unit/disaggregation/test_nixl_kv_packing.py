import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.nixl.packed import (
    PackedKVSender,
    packed_page_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _gather(src_pools, pages, page_size, src_token_bytes, dst_token_bytes, head_off):
    """_fused_gather_to_staging_kernel on numpy: per tensor, the head slice of each token of `pages`."""
    out = []
    for pool in src_pools:
        tokens = (np.asarray(pages)[:, None] * page_size + np.arange(page_size)).ravel()
        rows = pool.reshape(-1, src_token_bytes)[tokens]
        out.append(rows[:, head_off : head_off + dst_token_bytes].ravel())
    return np.concatenate(out)


def _slot_addrs(runs, indices):
    """Addresses NIXL resolves for `indices` of a dlist prepped from Nx5 (addr, len, dev, stride, count) runs."""
    addrs = np.concatenate(
        [addr + np.arange(count) * stride for addr, _, _, stride, count in runs]
    )
    return addrs[np.asarray(indices)]


class TestPackedPageIndices(CustomTestCase):
    def test_indices_write_each_decode_rank_its_head_slice(self):
        """Prefill TP2 -> decode TP4: each decode rank receives its half of the prefill heads."""
        num_tensors, num_pages, page_size = 3, 8, 4
        src_heads, head_bytes = 4, 8
        src_token_bytes = src_heads * head_bytes
        rng = np.random.default_rng(0)
        src_pools = [
            rng.integers(0, 256, num_pages * page_size * src_token_bytes, dtype=np.uint8)
            for _ in range(num_tensors)
        ]
        src_pages = np.array([5, 1, 6], dtype=np.int64)
        dst_pages = np.array([0, 7, 3], dtype=np.int64)
        for head_start in (0, 2):
            heads = 2
            dst_token_bytes = heads * head_bytes
            page_bytes = page_size * dst_token_bytes
            staging = _gather(
                src_pools,
                src_pages,
                page_size,
                src_token_bytes,
                dst_token_bytes,
                head_start * head_bytes,
            )
            # The buffer's prepped dlist: page-sized slots from address 0. Decode pools live back
            # to back in one fake address space, one strided run per tensor.
            pool_bytes = num_pages * page_bytes
            src_runs = [(0, page_bytes, 0, page_bytes, len(staging) // page_bytes)]
            dst_runs = [
                (t * pool_bytes, page_bytes, 1, page_bytes, num_pages)
                for t in range(num_tensors)
            ]
            src_idx, dst_idx = packed_page_indices(
                num_tensors=num_tensors, dst_pages=dst_pages, dst_num_slots=num_pages
            )
            self.assertEqual(len(src_idx), num_tensors * len(dst_pages))
            dst = np.zeros(num_tensors * pool_bytes, dtype=np.uint8)
            for s_addr, d_addr in zip(
                _slot_addrs(src_runs, src_idx), _slot_addrs(dst_runs, dst_idx)
            ):
                dst[d_addr : d_addr + page_bytes] = staging[s_addr : s_addr + page_bytes]

            for t in range(num_tensors):
                got = dst[t * pool_bytes : (t + 1) * pool_bytes].reshape(
                    num_pages, page_size, heads, head_bytes
                )
                ref = src_pools[t].reshape(num_pages, page_size, src_heads, head_bytes)
                for sp, dp in zip(src_pages, dst_pages):
                    np.testing.assert_array_equal(
                        got[dp], ref[sp, :, head_start : head_start + heads]
                    )
                untouched = sorted(set(range(num_pages)) - set(dst_pages.tolist()))
                self.assertFalse(got[untouched].any())


def _packer(buffer_bytes, transfer_results, failed_handles=()):
    """A PackedKVSender with a fake agent and no GPU: the gather is a mock."""
    packer = object.__new__(PackedKVSender)
    packer.gpu_id = 0
    packer.inflight = None
    packer.buffer = SimpleNamespace(get_size=lambda: buffer_bytes, get_ptr=lambda: 0x1000)
    packer._gather = Mock()
    packer._src_handles, packer._dst_handles = {}, {}
    packer.prep_dlist = Mock(side_effect=lambda peer, runs, mem: f"dlist-{peer or 'local'}")
    posted = []
    results = iter(transfer_results)

    def make_prepped_xfer(op, src_handle, src_idx, dst_handle, dst_idx):
        handle = f"h{len(posted)}"
        posted.append((handle, len(src_idx)))
        return handle

    packer.agent = SimpleNamespace(
        make_prepped_xfer=make_prepped_xfer,
        transfer=lambda handle: next(results),
    )
    packer.await_handles = Mock(
        side_effect=lambda hs, failure_seen: (True, hs[0] in failed_handles)
    )
    return packer, posted


def _send(packer, *, num_pages, handles=None):
    pages = np.arange(num_pages, dtype=np.int64)
    packer.send(
        peer_name="decode-0",
        src_ptrs=[0, 1],
        dst_ptrs=[10, 11],
        src_pages=pages,
        dst_pages=pages,
        src_page_bytes=256,
        dst_page_bytes=128,
        page_size=4,
        head_offset_bytes=0,
        dst_gpu_id=1,
        dst_num_slots=64,
        handles=[] if handles is None else handles,
    )


class TestPackedKVSender(CustomTestCase):
    def test_batches_by_buffer_and_posts_every_batch_for_the_caller_to_settle(self):
        # 2 tensors x 128-byte pages: a 1 KB buffer holds 4 pages per batch.
        packer, posted = _packer(1024, ["DONE"] * 3)
        handles = []
        _send(packer, num_pages=10, handles=handles)
        self.assertEqual([n for _, n in posted], [8, 8, 4])
        self.assertEqual(handles, ["h0", "h1", "h2"])
        self.assertEqual(packer._gather.call_count, 3)
        self.assertIsNone(packer.inflight)

    def test_dlists_are_prepped_once_per_buffer_and_peer(self):
        """Transfers are built by index from cached dlists, not from raw descriptors per batch."""
        packer, _ = _packer(1024, ["DONE"] * 6)
        _send(packer, num_pages=10)
        _send(packer, num_pages=10)
        self.assertEqual(packer.prep_dlist.call_count, 2)

    def test_a_failed_post_does_not_fail_the_next_room(self):
        packer, posted = _packer(1024, ["ERR", "DONE"], failed_handles={"h0"})
        with self.assertRaises(RuntimeError):
            _send(packer, num_pages=2)
        # The next room on this worker reaches a healthy peer and must succeed.
        _send(packer, num_pages=2)
        self.assertEqual(len(posted), 2)
        self.assertIsNone(packer.inflight)

    def test_a_failed_batch_fails_its_own_room(self):
        packer, _ = _packer(1024, ["DONE", "DONE"], failed_handles={"h0"})
        with self.assertRaises(RuntimeError):
            _send(packer, num_pages=6)

    def test_an_unsettled_leftover_blocks_reuse_of_the_buffer(self):
        packer, _ = _packer(1024, ["DONE"])
        packer.inflight = "stale"
        packer.await_handles = Mock(return_value=(False, False))
        with self.assertRaises(RuntimeError):
            _send(packer, num_pages=2)
        self.assertEqual(packer.inflight, "stale")
        packer._gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
