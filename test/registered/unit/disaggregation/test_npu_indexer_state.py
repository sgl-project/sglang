"""NPU indexer state keeps all token pages beside token-sharded attention KV."""

import unittest
from copy import copy
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.hardware_backend.npu.disaggregation import NPUMooncakeKVManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mooncake_utils import attach_mooncake_transport, engine_writes
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNpuIndependentIndexerState(CustomTestCase):
    def test_npu_managers_register_and_reuse_pack_buffers(self):
        for manager_class in (NPUMooncakeKVManager, AscendKVManager):
            with self.subTest(manager=manager_class.__name__):
                mgr = object.__new__(manager_class)
                mgr.kv_args = SimpleNamespace(
                    page_size=128,
                    gpu_id=0,
                    kv_item_lens=[131072, 16384, 131072],
                    num_draft_entries=1,
                )
                mgr._dcp_pack_buffers = None
                mgr.transfer_queues = [None, None]
                mgr._register_staging_memory = Mock()

                def buffer(size, gpu_id):
                    result = SimpleNamespace(get_size=lambda: size)
                    result.get_ptr = lambda: id(result)
                    return result

                with (
                    get_context().override_server_args(chunked_prefill_size=250),
                    patch(
                        "sglang.srt.hardware_backend.npu.dcp_pack.NPUDCPPackBuffer",
                        side_effect=buffer,
                    ) as allocate,
                ):
                    mgr._init_dcp_pack_buffers_once(2, include_draft=True)
                    buffers = mgr._dcp_pack_buffers
                    mgr._init_dcp_pack_buffers_once(2, include_draft=True)
                self.assertEqual(allocate.call_count, 2)
                self.assertEqual([b.get_size() for b in buffers], [256 * 1152] * 2)
                self.assertEqual(
                    [call.args for call in mgr._register_staging_memory.call_args_list],
                    [(b.get_ptr(), b.get_size()) for b in buffers],
                )
                self.assertIs(mgr._dcp_pack_buffers, buffers)
                self.assertEqual(mgr._dcp_pack_max_tokens, 256)

    def setUp(self):
        super().setUp()
        for name, flags in (
            ("get_parallel", dict(enable_kv_cache_sharding=False)),
            ("get_memory", dict(enable_unified_memory=False)),
            ("get_spec", dict(speculative_algorithm=None)),
        ):
            context = patch(
                f"sglang.srt.disaggregation.mooncake.conn.{name}",
                return_value=SimpleNamespace(**flags),
            )
            context.start()
            self.addCleanup(context.stop)

    def _manager(self, manager_class, src_ids, dst_ids, widths, rank=0):
        mgr = object.__new__(manager_class)
        mgr.kv_args = SimpleNamespace(
            state_types=[StateType.DSA],
            state_data_ptrs=[[1_000_000 * (i + 1) for i in range(len(src_ids))]],
            state_item_lens=[widths],
            state_layer_ids=[src_ids],
            state_window_tails=[],
            state_dim_per_tensor=[],
        )
        mgr.state_strides_validated = {"decode"}
        mgr.enable_custom_mem_pool = False
        mgr.max_transfer_batch_indices = 0
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = True
        mgr.pp_size = 2
        mgr.attn_tp_size = 1
        mgr.request_status = {17: KVPoll.Transferring}
        writes = Mock(return_value=0)
        info = SimpleNamespace(
            requires_dcp_relayout=True,
            dst_state_data_ptrs=[
                [
                    100_000_000 * (i + 1) + rank * 1_000_000_000
                    for i in range(len(dst_ids))
                ]
            ],
            dst_state_item_lens=[[32768] * 4 + [512] * 4],
            dst_state_layer_ids=[dst_ids],
            dst_state_dim_per_tensor=[],
            dst_attn_tp_size=1,
        )
        attach_mooncake_transport(
            mgr,
            engine_writes(writes),
            state=mgr.kv_args.state_data_ptrs,
            peer_state=info.dst_state_data_ptrs,
            sessions=("decode",),
        )
        return mgr, info, writes

    def test_every_decode_rank_receives_all_indexer_pages_and_scales(self):
        # 529 valid tokens in five physical pages, with a 17-token tail.
        source_pages = np.array([2, 3, 6, 7, 8], dtype=np.int32)
        destination_pages = np.array([5, 9, 10, 14, 17], dtype=np.int32)
        for manager_class in (NPUMooncakeKVManager, AscendKVManager):
            for rank in (0, 1):
                with self.subTest(manager=manager_class.__name__, rank=rank):
                    mgr, info, writes = self._manager(
                        manager_class,
                        src_ids=[1, 3, 1, 3],
                        dst_ids=[0, 1, 2, 3] * 2,
                        widths=[32768, 32768, 512, 512],
                        rank=rank,
                    )
                    req = SimpleNamespace(
                        room=17,
                        mooncake_session_id="decode",
                        dst_state_indices=[destination_pages],
                    )
                    self.assertEqual(
                        mgr.maybe_send_extra(req, [source_pages], None, info), 0
                    )
                    self.assertTrue(info.requires_dcp_relayout)
                    writes.assert_called_once()
                    blocks = writes.call_args.args[1]
                    expected = []
                    for i, j in enumerate((1, 3, 5, 7)):
                        width = mgr.kv_args.state_item_lens[0][i]
                        src = mgr.kv_args.state_data_ptrs[0][i]
                        dst = info.dst_state_data_ptrs[0][j]
                        expected += [
                            (src + int(s) * width, dst + int(d) * width, width)
                            for s, d in zip(source_pages, destination_pages)
                        ]
                    # Neither side's consecutive pages are consecutive on both
                    # peers, so every transfer block is exactly one page.
                    self.assertCountEqual(blocks, expected)
                    self.assertIsNone(mgr.dsa_index_k_dcp_entries(info))

    def test_gpu_only_skips_state_send_for_indexer_relayout(self):
        for relayout in (False, True):
            with self.subTest(relayout=relayout):
                width = 32768
                mgr, info, writes = self._manager(
                    MooncakeKVManager, src_ids=[0], dst_ids=[0], widths=[width]
                )
                info.requires_dcp_relayout = relayout
                info.dst_state_item_lens = [[width]]
                req = SimpleNamespace(
                    room=17,
                    mooncake_session_id="decode",
                    dst_state_indices=[np.array([5], dtype=np.int32)],
                )
                src_pages = [np.array([2], dtype=np.int32)]
                self.assertEqual(mgr.maybe_send_extra(req, src_pages, None, info), 0)
                if relayout:
                    writes.assert_not_called()
                else:
                    writes.assert_called_once()
                    src = mgr.kv_args.state_data_ptrs[0][0]
                    dst = info.dst_state_data_ptrs[0][0]
                    self.assertEqual(
                        writes.call_args.args[1],
                        [(src + 2 * width, dst + 5 * width, width)],
                    )

    def _register(self, mgr, info):
        mgr._stopped = False
        mgr._worker_threads = []
        mgr.server_socket = None

        def recv():
            mgr._stopped = True
            return [b"None", b"", b"", b"decode"]

        with (
            patch.object(mgr, "_make_worker_recv", return_value=recv),
            patch(
                "sglang.srt.disaggregation.mooncake.conn.KVArgsRegisterInfo.from_zmq",
                return_value=info,
            ),
            patch("sglang.srt.disaggregation.mooncake.conn.threading.Thread") as thread,
        ):
            mgr.start_prefill_thread()
            thread.call_args.kwargs["target"]()

    def test_registration_aligns_real_target_and_draft_widths(self):
        for manager_class in (
            MooncakeKVManager,
            NPUMooncakeKVManager,
            AscendKVManager,
        ):
            with self.subTest(manager=manager_class.__name__):
                mgr, _, _ = self._manager(manager_class, [], [], [])
                mgr.kv_args = SimpleNamespace(
                    kv_item_lens=[32, 16, 32, 16],
                    kv_data_ptrs=[100, 200, 300, 400],
                    kv_layer_ids=[1, 1, 7, 7],
                    page_size=4,
                    num_draft_entries=2,
                    state_types=[],
                    aux_data_ptrs=[],
                )
                mgr.dcp_size = 1
                mgr.dcp_rank = 0
                mgr.kv_shard_state_peer_only = False
                mgr.state_tp_size = mgr.attn_tp_size
                mgr.session_lock = Lock()
                mgr.failed_sessions = set()
                mgr.session_failures = {}
                mgr.failed_session_probes = {}
                mgr.state_layout_rejections = {}
                mgr.state_strides_validated = set()
                mgr.decode_kv_args_table = {}
                mgr._init_dcp_pack_buffers_once = Mock()
                info = SimpleNamespace(
                    mooncake_session_id="decode",
                    dst_draft_rows_dcp_collapsed=False,
                    dst_dcp_size=2,
                    dst_dcp_rank=0,
                    dst_kv_layer_ids=[0, 1, 2, 3] * 2 + [7, 7],
                    dst_kv_ptrs=list(range(1000, 1010)),
                    dst_kv_item_len=32,
                    dst_kv_item_lens=[32] * 4 + [16] * 4 + [64, 32],
                    dst_aux_ptrs=[],
                    dst_state_data_ptrs=[],
                    staging_base_ptr=0,
                )
                self._register(mgr, info)
                self.assertTrue(info.requires_dcp_relayout)
                self.assertEqual(info.dcp_token_item_lens, [8, 4, 8, 4])
                self.assertIs(mgr.decode_kv_args_table["decode"], info)
                for dst_entry, src_entry in ((5, 1), (8, 2)):
                    with self.subTest(dst_entry=dst_entry):
                        bad_info = copy(info)
                        bad_info.dst_kv_item_lens = info.dst_kv_item_lens.copy()
                        bad_info.dst_kv_item_lens[dst_entry] = 32
                        with self.assertRaisesRegex(RuntimeError, f"entry {src_entry}"):
                            self._register(mgr, bad_info)
                        self.assertIs(mgr.decode_kv_args_table["decode"], info)
                mgr._init_dcp_pack_buffers_once.assert_called_once_with(
                    2, include_draft=True
                )

    def test_gpu_registration_preserves_positional_and_collapsed_draft_layouts(self):
        mgr = object.__new__(MooncakeKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_item_lens=[32, 32],
            kv_layer_ids=[],
            num_draft_entries=1,
            page_size=4,
            prefill_start_layer=1,
        )
        mgr.dcp_size, mgr.dcp_rank, mgr.pp_size = 1, 0, 2
        mgr.is_mla_backend = True
        mgr._init_dcp_pack_buffers_once = Mock()
        mgr._publish_peer_registration = Mock(return_value=None)
        for widths, collapsed in (
            ([32, 32, 64], False),
            ([32, 32, 32], True),
            ([], False),
        ):
            with self.subTest(widths=widths, collapsed=collapsed):
                info = SimpleNamespace(
                    dst_dcp_size=2,
                    dst_dcp_rank=0,
                    dst_kv_item_len=32,
                    dst_kv_item_lens=widths,
                    dst_kv_layer_ids=[],
                    dst_draft_rows_dcp_collapsed=collapsed,
                )
                self._register(mgr, info)
                self.assertEqual(info.dcp_token_item_lens, [8, 8])
                self.assertIs(mgr._publish_peer_registration.call_args.args[1], info)


if __name__ == "__main__":
    unittest.main()
