"""CPU byte-exact tests of page-sharded target/draft DSA PD transfer.

Exercise the production owner filter and Mooncake and NIXL descriptor builders with a
memory-copy transport. No GPU, RDMA engine or model weights are required.
"""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.nixl.buffers import BufferKeys
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mooncake_utils import attach_mooncake_transport, engine_writes
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOM = 7
_INDEX_PAGE_BYTES = 64 * (128 + 4)


def _fragmented_indexer_pages():
    logical_pages = np.array(
        [15, 3, 24, 9, 18, 7, 28, 12, 22, 1, 8, 30, 4],
        dtype=np.int32,
    )
    dst_pages = np.array(
        [29, 5, 16, 8, 1, 27, 3, 20, 7, 15, 10, 31, 0],
        dtype=np.int32,
    )
    rng = np.random.default_rng(92)
    canonical = rng.integers(0, 256, (4, 32, _INDEX_PAGE_BYTES), dtype=np.uint8)
    return logical_pages, dst_pages, canonical


def _manager(rank, size, sources, destinations):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.attn_cp_rank, mgr.attn_cp_size = rank, size
    mgr.kv_shard_rank, mgr.kv_shard_size = rank, size
    mgr.attn_tp_size = mgr.pp_size = 1
    mgr.is_mla_backend = True
    mgr.is_hybrid_mla_backend = False
    mgr.enable_custom_mem_pool = False
    mgr.max_transfer_batch_indices = 0
    mgr.state_strides_validated = set()
    mgr.request_status = {_ROOM: KVPoll.Transferring}
    mgr.kv_args = SimpleNamespace(
        state_types=[StateType.DSA],
        state_data_ptrs=[[x.ctypes.data for x in sources]],
        state_item_lens=[[_INDEX_PAGE_BYTES if len(x) else 0 for x in sources]],
        state_dim_per_tensor=[[]],
        state_layer_ids=[[]],
        prefill_start_layer=0,
    )
    mgr.blocks = []

    def copy_blocks(session, blocks):
        mgr.blocks.extend(blocks)
        for src, dst, length in blocks:
            ctypes.memmove(dst, src, length)
        return 0

    attach_mooncake_transport(
        mgr,
        engine_writes(copy_blocks),
        state=mgr.kv_args.state_data_ptrs,
        peer_state=[[x.ctypes.data for x in destinations]],
        sessions=("cpu",),
    )
    peer = SimpleNamespace(
        dst_state_data_ptrs=[[x.ctypes.data for x in destinations]],
        dst_state_item_lens=[
            [_INDEX_PAGE_BYTES if len(x) else 0 for x in destinations]
        ],
        dst_state_dim_per_tensor=[[]],
        dst_state_layer_ids=[[]],
        dst_attn_tp_size=1,
        requires_dcp_relayout=False,
    )
    return mgr, peer


def _send_state(mgr, peer, src_pages, dst_pages):
    with patch(
        "sglang.srt.disaggregation.mooncake.conn.get_memory",
        return_value=SimpleNamespace(enable_unified_memory=False),
    ):
        return mgr.maybe_send_extra(
            SimpleNamespace(
                room=_ROOM,
                mooncake_session_id="cpu",
                dst_state_indices=[dst_pages],
            ),
            [src_pages],
            executor=None,
            target_rank_registration_info=peer,
        )


class _FragmentedPagesTestMixin:
    def test_fragmented_pages_preserve_every_target_and_draft_byte(self):
        # Three target layers and one draft layer, including packed keys and scales.
        logical, dst_pages, canonical = _fragmented_indexer_pages()
        for size in (4, 8):  # At CP size 8, rank 5 owns no requested pages.
            dst = np.full_like(canonical, 0xA5)
            expected = dst.copy()
            for rank in range(size):
                with self.subTest(cp_size=size, rank=rank):
                    src = np.ascontiguousarray(canonical[:, rank::size])
                    written = self._send_pages(rank, size, src, dst, logical, dst_pages)
                    owned = logical % size == rank
                    expected[:, dst_pages[owned]] = canonical[:, logical[owned]]
                    np.testing.assert_array_equal(dst, expected)
                    self.assertEqual(
                        written, owned.sum() * len(src) * _INDEX_PAGE_BYTES
                    )


class TestDSAShardedPDTransfer(_FragmentedPagesTestMixin, CustomTestCase):
    def setUp(self):
        super().setUp()
        override = patch(
            "sglang.srt.disaggregation.mooncake.conn.get_parallel",
            return_value=SimpleNamespace(enable_kv_cache_sharding=True),
        )
        override.start()
        self.addCleanup(override.stop)

    def _send_pages(self, rank, size, src, dst, logical, dst_pages):
        mgr, peer = _manager(rank, size, src, dst)
        with patch(
            "sglang.srt.disaggregation.common.conn.get_parallel",
            return_value=SimpleNamespace(
                enable_dsa_cache_layer_split=False,
                enable_cp_tp_group_sharing=False,
            ),
        ):
            self.assertEqual(
                mgr._get_dsa_cache_transfer_skip_flags(peer), (False, False)
            )
        self.assertEqual(_send_state(mgr, peer, logical, dst_pages), 0)
        return sum(length for _, _, length in mgr.blocks)

    def test_mismatch_is_rejected_before_any_write(self):
        src = [np.zeros((8, _INDEX_PAGE_BYTES), dtype=np.uint8)]
        dst = [np.full((32, _INDEX_PAGE_BYTES), 0xA5, dtype=np.uint8)]
        for src_pages, dst_pages in (([4, 8], [0]), ([4], [0, 1]), ([], [0])):
            with self.subTest(src=src_pages, dst=dst_pages):
                mgr, peer = _manager(0, 4, src, dst)
                with self.assertRaisesRegex(RuntimeError, "state page count mismatch"):
                    _send_state(mgr, peer, src_pages, dst_pages)
                self.assertEqual(mgr.blocks, [])
                self.assertTrue(np.all(dst[0] == 0xA5))

    def test_disabled_sharding_keeps_all_source_pages(self):
        src = [np.full((8, _INDEX_PAGE_BYTES), 0x19, dtype=np.uint8)]
        dst = [np.full_like(src[0], 0xA5)]
        mgr, peer = _manager(0, 4, src, dst)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.get_parallel",
            return_value=SimpleNamespace(enable_kv_cache_sharding=False),
        ):
            self.assertEqual(_send_state(mgr, peer, [1, 3], [4, 6]), 0)
        np.testing.assert_array_equal(dst[0][[4, 6]], src[0][[1, 3]])

    def test_chunked_kv_delta_does_not_filter_away_full_prefix_indexer_state(self):
        # Decode cached two prefix pages. Latent transfer contains only the
        # suffix; DSA state still addresses the full sequence. Rank 3 owns one
        # prefix indexer page but no suffix KV page. Its empty final KV chunk
        # must retain that state, with independently filtered state metrics.
        src = [np.full((8, _INDEX_PAGE_BYTES), 0x19, dtype=np.uint8)]
        dst = [np.full((32, _INDEX_PAGE_BYTES), 0xA5, dtype=np.uint8)]
        mgr, peer = _manager(3, 4, src, dst)
        mgr.add_transfer_request = Mock()
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = _ROOM
        sender.aux_index = 0
        sender.curr_idx = 0
        sender.num_kv_indices = 3
        sender.trace_ctx = Mock()
        sender.conclude_state = None
        sender._transfer_start_time = None
        sender._early_send_wait_event = None
        sender._transfer_metric = KVTransferMetric()
        sender._transfer_num_kv_indices = 0
        sender._transfer_num_state_indices = 0
        full_pages = np.array([3, 6, 4, 9, 12], dtype=np.int32)
        with patch(
            "sglang.srt.disaggregation.common.conn.get_parallel",
            return_value=SimpleNamespace(
                enable_dsa_cache_layer_split=False, enable_cp_tp_group_sharing=False
            ),
        ):
            sender.send(full_pages[2:4])
            sender.send(full_pages[4:], [full_pages])
        calls = mgr.add_transfer_request.call_args_list
        # The prerequisite skips empty non-final chunks but keeps the final
        # notification and the independently owner-filtered indexer state.
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].args[3])
        self.assertEqual(len(calls[0].args[1]), 0)
        self.assertEqual(sender._transfer_num_kv_indices, 0)
        self.assertEqual(sender._transfer_num_state_indices, 1)
        np.testing.assert_array_equal(calls[0].kwargs["state_indices"][0], full_pages)
        self.assertEqual(_send_state(mgr, peer, full_pages, [7, 2, 4, 8, 11]), 0)
        expected = np.full_like(dst[0], 0xA5)
        expected[7] = src[0][0]  # logical page 3 on rank 3 is local page 0
        np.testing.assert_array_equal(dst[0], expected)


class TestDSANIXLShardedPDTransfer(_FragmentedPagesTestMixin, CustomTestCase):
    def _send_pages(self, rank, size, src, dst, logical, dst_pages):
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.kv_shard_rank, mgr.kv_shard_size = rank, size
        mgr.is_mla_backend = True
        mgr._state_layout_rejections = {}
        mgr._prepared_state_plans = {}
        mgr.kv_args = SimpleNamespace(
            gpu_id=0,
            kv_data_ptrs=[],
            state_types=[StateType.DSA],
            state_layer_ids=[[]],
            state_data_ptrs=[[x.ctypes.data for x in src]],
            state_item_lens=[[_INDEX_PAGE_BYTES] * len(src)],
        )
        mgr._local_keys = BufferKeys.for_buffers([], [], [list(range(len(src)))])
        peer_keys = BufferKeys.for_buffers([], [], [list(range(len(dst)))])
        mgr._peers = {"decode": SimpleNamespace(keys=peer_keys)}
        src_by_key = dict(zip(mgr._local_keys.state[0], src))
        dst_by_key = dict(zip(peer_keys.state[0], dst))
        byte_count = 0

        # Apply exactly the byte ranges submitted to the transport.
        def transfer_copies(peer, copies):
            nonlocal byte_count
            self.assertEqual(peer, "decode")
            for copy in copies:
                s_buf = src_by_key[copy.src].reshape(-1)
                d_buf = dst_by_key[copy.dst].reshape(-1)
                for s, d, n in copy.ranges.tolist():
                    self.assertTrue(0 <= s <= s_buf.size - n)
                    self.assertTrue(0 <= d <= d_buf.size - n)
                    d_buf[d : d + n] = s_buf[s : s + n]
                    byte_count += n
            return [SimpleNamespace(xfer=object(), result=None, error=None)]

        mgr._transfer_copies = transfer_copies
        handles = list(
            mgr.maybe_send_extra(
                "decode",
                [logical],
                [[x.ctypes.data for x in dst]],
                [dst_pages],
                decode_tp_size=1,
                dst_state_item_lens=[[_INDEX_PAGE_BYTES] * len(dst)],
            )
        )
        self.assertEqual(len(handles), int(np.any(logical % size == rank)))
        return byte_count


if __name__ == "__main__":
    unittest.main()
