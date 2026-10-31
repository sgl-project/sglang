"""CPU byte-exact tests of page-sharded target/draft DSA PD transfer.

Exercise the production owner filter and Mooncake descriptor builder with a
memory-copy transport. No GPU, RDMA engine or model weights are required.
"""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOM = 7
_INDEX_PAGE_BYTES = 64 * (128 + 4)


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

    mgr._transfer_data = copy_blocks
    peer = SimpleNamespace(
        dst_state_data_ptrs=[[x.ctypes.data for x in destinations]],
        dst_state_item_lens=[
            [_INDEX_PAGE_BYTES if len(x) else 0 for x in destinations]
        ],
        dst_state_dim_per_tensor=[[]],
        dst_state_layer_ids=[[]],
        dst_attn_tp_size=1,
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


class TestDSAShardedPDTransfer(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = patch(
            "sglang.srt.disaggregation.mooncake.conn.get_parallel",
            return_value=SimpleNamespace(enable_kv_cache_sharding=True),
        )
        override.start()
        self.addCleanup(override.stop)

    def test_fragmented_pages_preserve_every_target_and_draft_byte(self):
        # A full indexer page includes its packed K region AND scale region.
        # Random bytes encode layer, logical page and byte offset independently.
        # The fourth layer represents draft, preventing target/draft aliasing
        # from passing as an apparently correct copy.
        for size in (4, 8):
            with self.subTest(cp_size=size):
                logical_pages = np.array(
                    [15, 3, 24, 9, 18, 7, 28, 12, 22, 1, 8, 30, 4],
                    dtype=np.int32,
                )
                dst_pages = np.array(
                    [29, 5, 16, 8, 1, 27, 3, 20, 7, 15, 10, 31, 0],
                    dtype=np.int32,
                )
                rng = np.random.default_rng(92)
                canonical = rng.integers(
                    0, 256, (4, 32, _INDEX_PAGE_BYTES), dtype=np.uint8
                )
                destinations = [np.full_like(x, 0xA5) for x in canonical]
                expected = [x.copy() for x in destinations]
                for layer in range(4):
                    expected[layer][dst_pages] = canonical[layer][logical_pages]
                byte_count = 0
                for rank in range(size):
                    sources = [np.ascontiguousarray(x[rank::size]) for x in canonical]
                    mgr, peer = _manager(rank, size, sources, destinations)
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
                    self.assertEqual(
                        _send_state(mgr, peer, logical_pages, dst_pages), 0
                    )
                    byte_count += sum(length for _, _, length in mgr.blocks)
                for actual, oracle in zip(destinations, expected):
                    np.testing.assert_array_equal(actual, oracle)
                self.assertEqual(byte_count, len(logical_pages) * 4 * _INDEX_PAGE_BYTES)

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


if __name__ == "__main__":
    unittest.main()
