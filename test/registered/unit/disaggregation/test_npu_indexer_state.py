"""NPU indexer state keeps all token pages beside token-sharded attention KV."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import KVClassType, TransferBackend, get_kv_class
from sglang.srt.hardware_backend.npu.disaggregation import NPUMooncakeKVManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNpuIndependentIndexerState(CustomTestCase):
    def test_npu_managers_register_and_reuse_pack_buffers(self):
        for manager_class in (NPUMooncakeKVManager, AscendKVManager):
            with self.subTest(manager=manager_class.__name__):
                mgr = object.__new__(manager_class)
                mgr.kv_args = SimpleNamespace(
                    npu_kv_layout="ND", page_size=128, kv_item_lens=[131072, 16384]
                )
                mgr._dcp_pack_buffers = None
                mgr.transfer_queues = [None, None]
                mgr._register_staging_memory = Mock()
                buffers = [object(), object()]
                with (
                    get_context().override_server_args(chunked_prefill_size=250),
                    patch(
                        "sglang.srt.hardware_backend.npu.dcp_pack.init_npu_dcp_pack_buffers",
                        return_value=buffers,
                    ) as allocate,
                ):
                    mgr._init_dcp_pack_buffers_once(2)
                    mgr._init_dcp_pack_buffers_once(2)
                allocate.assert_called_once_with(
                    mgr._register_staging_memory,
                    mgr.kv_args,
                    2,
                    2,
                    256,
                    include_draft=False,
                )
                self.assertIs(mgr._dcp_pack_buffers, buffers)
                self.assertEqual(mgr._dcp_pack_max_tokens, 256)

    def setUp(self):
        super().setUp()
        for name, flags in (
            ("get_parallel", dict(enable_kv_cache_sharding=False)),
            ("get_memory", dict(enable_unified_memory=False)),
        ):
            context = patch(
                f"sglang.srt.disaggregation.mooncake.conn.{name}",
                return_value=SimpleNamespace(**flags),
            )
            context.start()
            self.addCleanup(context.stop)

    def _manager(self, manager_class, src_ids, dst_ids, widths):
        mgr = object.__new__(manager_class)
        mgr.kv_args = SimpleNamespace(
            npu_kv_layout="ND",
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
        mgr._transfer_data = Mock(return_value=0)
        info = SimpleNamespace(
            requires_dcp_relayout=True,
            dst_state_data_ptrs=[[100_000_000 * (i + 1) for i in range(len(dst_ids))]],
            dst_state_item_lens=[[32768] * 4 + [512] * 4],
            dst_state_layer_ids=[dst_ids],
            dst_state_dim_per_tensor=[],
            dst_attn_tp_size=1,
        )
        return mgr, info

    def test_every_decode_rank_receives_all_indexer_pages_and_scales(self):
        # 529 valid tokens in five physical pages, with a 17-token tail.
        source_pages = np.array([2, 3, 6, 7, 8], dtype=np.int32)
        destination_pages = np.array([5, 9, 10, 14, 17], dtype=np.int32)
        for manager_class in (NPUMooncakeKVManager, AscendKVManager):
            for rank in (0, 1):
                with self.subTest(manager=manager_class.__name__, rank=rank):
                    mgr, info = self._manager(
                        manager_class,
                        src_ids=[1, 3, 1, 3],
                        dst_ids=[0, 1, 2, 3] * 2,
                        widths=[32768, 32768, 512, 512],
                    )
                    info.dst_state_data_ptrs[0] = [
                        ptr + rank * 1_000_000_000
                        for ptr in info.dst_state_data_ptrs[0]
                    ]
                    req = SimpleNamespace(
                        room=17,
                        mooncake_session_id="decode",
                        dst_state_indices=[destination_pages],
                    )
                    self.assertEqual(
                        mgr.maybe_send_extra(req, [source_pages], None, info), 0
                    )
                    self.assertTrue(info.requires_dcp_relayout)
                    blocks = mgr._transfer_data.call_args.args[1]
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

    def test_intermediate_chunk_without_state_does_not_resend_indexer(self):
        mgr, info = self._manager(NPUMooncakeKVManager, [0], [0], [32768])
        req = SimpleNamespace(room=17, dst_state_indices=[], mooncake_session_id="d")
        self.assertEqual(mgr.maybe_send_extra(req, [], None, info), 0)
        mgr._transfer_data.assert_not_called()

    def test_same_dcp_layout_uses_only_the_common_state_sender(self):
        mgr, info = self._manager(NPUMooncakeKVManager, [0], [0], [32768])
        info.requires_dcp_relayout = False
        req = SimpleNamespace(room=17, dst_state_indices=[[4]])
        with patch.object(
            MooncakeKVManager, "maybe_send_extra", return_value=0
        ) as send:
            self.assertEqual(mgr.maybe_send_extra(req, [[2]], None, info), 0)
        send.assert_called_once_with(req, [[2]], None, info)
        mgr._transfer_data.assert_not_called()

    def test_heterogeneous_widths_are_checked_against_the_actual_peer(self):
        mgr, _ = self._manager(NPUMooncakeKVManager, [], [], [])
        mgr.kv_args = SimpleNamespace(
            npu_kv_layout="ND",
            kv_item_lens=[32, 16, 32, 16],
            kv_data_ptrs=[100, 200, 300, 400],
            kv_layer_ids=[1, 1, 7, 7],
            page_size=4,
            num_draft_entries=2,
        )
        self.assertEqual(
            mgr.prepare_dcp_token_item_lens([32, 32, None, None], 2), [8, 4, 8, 4]
        )
        info = SimpleNamespace(
            requires_dcp_relayout=True,
            dst_draft_rows_dcp_collapsed=False,
            dst_dcp_size=2,
            dst_kv_layer_ids=[0, 1, 2, 3] * 2 + [7, 7],
            dst_kv_ptrs=list(range(10)),
            dst_kv_item_lens=[32] * 4 + [16] * 4 + [64, 32],
        )
        self.assertIsNone(mgr._kv_layout_mismatch(info))
        info.dst_kv_item_lens[5] = 32
        self.assertIn("item length mismatch", mgr._kv_layout_mismatch(info))

    def test_relayout_rejects_nz_or_collapsed_draft(self):
        mgr, _ = self._manager(NPUMooncakeKVManager, [], [], [])
        for layout, collapsed in (("NZ", False), ("ND", True)):
            with self.subTest(layout=layout, collapsed=collapsed):
                mgr.kv_args.npu_kv_layout = layout
                with self.assertRaisesRegex(ValueError, "ND KV and replicated draft"):
                    mgr.prepare_dcp_token_item_lens([], 2, collapsed)

    def test_factory_selects_npu_adapter_only_on_npu(self):
        with patch("sglang.srt.disaggregation.utils.is_npu", return_value=True):
            self.assertIs(
                get_kv_class(TransferBackend.MOONCAKE, KVClassType.MANAGER),
                NPUMooncakeKVManager,
            )
        with patch("sglang.srt.disaggregation.utils.is_npu", return_value=False):
            self.assertIs(
                get_kv_class(TransferBackend.MOONCAKE, KVClassType.MANAGER),
                MooncakeKVManager,
            )


if __name__ == "__main__":
    unittest.main()
