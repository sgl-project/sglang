import unittest
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.mooncake.buffers import BufferKeys
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMooncakePageDcp(unittest.TestCase):
    def _make_peers(self):
        manager = object.__new__(MooncakeKVManager)
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.pp_size = 2
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        # This PP stage owns only layers 11 and 12. Latent KV and RoPE
        # are local, while index-K and the separately registered draft use
        # allocator-global page slots on decode.
        manager.kv_args = SimpleNamespace(
            page_size=4,
            num_draft_entries=2,
            kv_data_ptrs=[10_000 * (i + 1) for i in range(8)],
            kv_layer_ids=[],
            kv_item_lens=[32, 32, 8, 8, 12, 12, 32, 8],
            dcp_page_global_slots=[False] * 4 + [True] * 4,
            dcp_page_layer_ids=[11, 12] * 3 + [78, 78],
        )
        manager._init_dcp_pack_buffers_once = Mock()
        manager._transfer_copies = Mock(return_value=0)
        manager._local_keys = BufferKeys.for_buffers(
            manager.kv_args.kv_data_ptrs, [], []
        )

        peer = KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=12345,
            mooncake_session_id="decode",
            dst_kv_ptrs=[100_000 * (i + 1) for i in range(14)],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            dst_tp_rank=2,
            dst_attn_tp_size=4,
            dst_kv_item_len=32,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_layer_ids=[],
            dst_state_layer_ids=[],
            dst_dcp_size=4,
            dst_dcp_rank=2,
            dst_kv_item_lens=[32] * 4 + [8] * 4 + [48] * 4 + [128, 32],
            num_draft_entries=2,
            dst_dcp_page_size=4,
            dst_dcp_page_global_slots=[False] * 8 + [True] * 6,
            dst_dcp_page_layer_ids=[9, 10, 11, 12] * 3 + [78, 78],
        )
        manager._peers = {peer.mooncake_session_id: self._make_peer_buffers(peer)}
        return manager, peer

    @staticmethod
    def _make_peer_buffers(peer):
        return SimpleNamespace(
            keys=BufferKeys.for_buffers(
                peer.dst_kv_ptrs, peer.dst_aux_ptrs, peer.dst_state_data_ptrs
            )
        )

    def test_pairs_mixed_width_pipeline_entries_and_sends_whole_pages(self):
        manager, peer = self._make_peers()
        manager._prepare_peer_dcp_layout(peer)

        expected_entries = [
            (0, 2, 32, False),
            (1, 3, 32, False),
            (2, 6, 8, False),
            (3, 7, 8, False),
            (4, 10, 12, True),
            (5, 11, 12, True),
            (6, 12, 32, True),
            (7, 13, 8, True),
        ]
        self.assertTrue(peer.requires_dcp_relayout)
        self.assertEqual(peer.dcp_page_transfer_entries, expected_entries)
        manager._init_dcp_pack_buffers_once.assert_not_called()
        self.assertEqual(
            manager._send_page_interleaved_dcp(
                peer.mooncake_session_id,
                np.array([5, 6, 7, 8, 9], dtype=np.int32),
                np.array([10, 12], dtype=np.int32),
                entries=peer.dcp_page_transfer_entries,
                dst_dcp_size=peer.dst_dcp_size,
                dst_dcp_rank=peer.dst_dcp_rank,
                src_page_offset=1,
                decode_prefix_len=16,
                num_kv_tokens=18,
            ),
            0,
        )
        manager._transfer_copies.assert_called_once()
        session, copies = manager._transfer_copies.call_args.args
        self.assertEqual(session, peer.mooncake_session_id)
        self.assertEqual(len(copies), len(expected_entries))
        for copy, (src, dst, page_bytes, uses_global_slots) in zip(
            copies, expected_entries
        ):
            self.assertEqual(copy.src, manager._local_keys.kv[src])
            self.assertEqual(copy.dst, manager._peers[session].keys.kv[dst])
            # Local rank 2 owns only source page 6. Global buffers also
            # receive the other pages, respecting the destination gap.
            pages = [(5, 41, 3), (8, 48, 2)] if uses_global_slots else [(6, 10, 1)]
            self.assertEqual(copy.ranges.dtype, np.uint64)
            np.testing.assert_array_equal(copy.ranges, np.array(pages) * page_bytes)

    def test_rejects_invalid_destination_stride_before_transfer(self):
        for entry in (6, 10, 12):
            with self.subTest(entry=entry):
                manager, peer = self._make_peers()
                peer.dst_kv_item_lens[entry] += manager.kv_args.page_size

                with self.assertRaises((RuntimeError, ValueError)):
                    manager._prepare_peer_dcp_layout(peer)

                self.assertIsNone(peer.dcp_page_transfer_entries)
                manager._init_dcp_pack_buffers_once.assert_not_called()
                manager._transfer_copies.assert_not_called()

    @staticmethod
    def _make_decode(peer):
        decode = object.__new__(MooncakeKVManager)
        decode.get_session_id = Mock(return_value=peer.mooncake_session_id)
        decode.local_ip = peer.endpoint
        decode.rank_port = peer.dst_port
        decode.attn_tp_size = peer.dst_attn_tp_size
        decode.dcp_size = peer.dst_dcp_size
        decode.dcp_rank = peer.dst_dcp_rank
        decode.enable_staging = False
        decode.exchange_aux_item_lens = False
        decode.kv_args = SimpleNamespace(
            engine_rank=peer.dst_tp_rank,
            kv_data_ptrs=peer.dst_kv_ptrs,
            kv_item_lens=peer.dst_kv_item_lens,
            kv_layer_ids=peer.dst_kv_layer_ids,
            aux_data_ptrs=[],
            state_data_ptrs=[],
            state_item_lens=[],
            state_layer_ids=[],
            draft_rows_dcp_collapsed=False,
            num_draft_entries=peer.num_draft_entries,
            page_size=peer.dst_dcp_page_size,
            dcp_page_global_slots=peer.dst_dcp_page_global_slots,
            dcp_page_layer_ids=peer.dst_dcp_page_layer_ids,
        )
        return decode

    def test_page_layout_survives_receiver_registration_roundtrip(self):
        manager, peer = self._make_peers()
        decode = self._make_decode(peer)
        frames = decode.build_receiver_registration_message()
        received = KVArgsRegisterInfo.from_zmq(frames)
        self.assertEqual(received, peer)
        manager._prepare_peer_dcp_layout(received)
        self.assertIsNotNone(received.dcp_page_transfer_entries)

        legacy_frames = frames[:23]
        legacy = KVArgsRegisterInfo.from_zmq(legacy_frames)
        self.assertEqual(legacy.dst_dcp_page_size, 0)
        self.assertEqual(legacy.dst_dcp_page_global_slots, [])
        self.assertEqual(legacy.dst_dcp_page_layer_ids, [])
        self.assertEqual(legacy.dst_kv_ptrs, peer.dst_kv_ptrs)
        self.assertEqual(legacy.dst_kv_item_lens, peer.dst_kv_item_lens)
        self.assertEqual(legacy.num_draft_entries, 2)
        with self.assertRaisesRegex(RuntimeError, "both peers"):
            manager._prepare_peer_dcp_layout(legacy)

        decode.kv_args.dcp_page_global_slots = None
        self.assertEqual(decode.build_receiver_registration_message(), legacy_frames)

        # Without page-layout metadata, a token-interleaved MLA peer still
        # uses the scalar width handshake and starts the packing path.
        manager.kv_args.dcp_page_global_slots = None
        manager.kv_args.kv_item_lens = [32] * 7 + [8]
        manager._prepare_peer_dcp_layout(legacy)
        self.assertIsNone(legacy.dcp_page_transfer_entries)
        self.assertEqual(legacy.dcp_token_item_lens, [8] * 7 + [2])
        manager._init_dcp_pack_buffers_once.assert_called_once_with(
            4, include_draft=True
        )

    def test_bad_registration_does_not_stop_the_bootstrap_thread(self):
        manager, peer = self._make_peers()
        decode = self._make_decode(peer)
        good_message = decode.build_receiver_registration_message()
        decode.get_session_id.return_value = "bad-decode"
        decode.kv_args.kv_item_lens[6] += manager.kv_args.page_size
        messages = iter([decode.build_receiver_registration_message(), good_message])

        def recv():
            message = next(messages, None)
            if message is None:
                manager._stopped = True
            return message

        manager.server_socket = None
        manager._make_worker_recv = Mock(return_value=recv)
        manager._handle_abort_notification = Mock(return_value=False)
        manager._stopped = False
        manager._worker_threads = []
        manager.session_lock = Lock()
        manager.state_layout_rejections = {}
        manager.state_strides_validated = set()
        manager.decode_kv_args_table = {}
        manager._peers.clear()
        manager._import_peer = Mock(
            side_effect=lambda _, peer: self._make_peer_buffers(peer)
        )
        manager._kv_layout_mismatch = Mock(return_value=None)
        manager._validate_peer_state_layout = Mock(return_value=(True, None))
        manager.check_peer_aux_buffer_count = Mock()
        manager._unblacklist_session = Mock()

        with self.assertLogs("sglang.srt.disaggregation.mooncake.conn", level="ERROR"):
            manager.start_prefill_thread()
            (bootstrap_thread,) = manager._worker_threads
            bootstrap_thread.join(timeout=2)

        self.assertFalse(bootstrap_thread.is_alive())
        self.assertIn("stride differs", manager.state_layout_rejections["bad-decode"])
        self.assertNotIn("decode", manager.state_layout_rejections)
        self.assertIsNone(
            manager.decode_kv_args_table["bad-decode"].dcp_page_transfer_entries
        )
        self.assertIsNotNone(
            manager.decode_kv_args_table["decode"].dcp_page_transfer_entries
        )


if __name__ == "__main__":
    unittest.main()
