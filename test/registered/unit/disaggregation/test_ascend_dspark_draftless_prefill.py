import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.ascend.conn import AscendKVManager, AscendStateType
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import state_wire
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="base-a-test-npu")


class TestDraftlessPrefill(unittest.TestCase):
    def manager(self):
        manager = AscendKVManager.__new__(AscendKVManager)
        manager._dspark_cold_start = True
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            state_types=[
                StateType.SWA,
                StateType.DSV4_REQUEST_STATE,
                AscendStateType.DSV4_C128,
                AscendStateType.DSV4_C4_STATE,
                StateType.SWA,
            ],
            num_draft_entries=0,
        )
        return manager

    def test_only_missing_draft_is_allowed(self):
        manager = self.manager()
        wire = state_wire(manager.kv_args)
        info = SimpleNamespace(pp_size=1, spec_kv_layout=None, state_wire=wire[:-1])
        manager._check_prefill_state_entries(info)
        self.assertTrue(manager._draftless_dspark_prefill(info))
        for bad_wire in [wire[:-2], wire[:1] + wire[2:], wire + [["swa", None]]]:
            with self.subTest(wire=bad_wire):
                info.state_wire = bad_wire
                with self.assertRaises(RuntimeError):
                    manager._check_prefill_state_entries(info)
        info.state_wire = wire[:-1]
        manager._dspark_cold_start = False
        with self.assertRaises(RuntimeError):
            manager._check_prefill_state_entries(info)

    def test_reused_slot_resets_for_each_peer(self):
        manager = self.manager()
        wire = state_wire(manager.kv_args)
        info = SimpleNamespace(pp_size=1, spec_kv_layout=None, state_wire=wire[:-1])
        manager.prefill_info_table = {"peer": info}
        pool = SimpleNamespace(_npu_dspark_context_start=torch.tensor([37, 41, 43]))
        manager.initialize_dspark_context("peer", pool, 1, 129)
        self.assertEqual(pool._npu_dspark_context_start.tolist(), [37, 129, 43])
        info.state_wire = wire
        manager.initialize_dspark_context("peer", pool, 1, 500)
        self.assertEqual(pool._npu_dspark_context_start.tolist(), [37, 0, 43])
        info.state_wire = wire[:-1]
        manager.initialize_dspark_context("peer", pool, 1, 256)
        self.assertEqual(pool._npu_dspark_context_start.tolist(), [37, 256, 43])

    def test_attention_never_reads_unwritten_history(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._dsv4_sliding_window_size = 128
        backend.req_to_token = torch.arange(1024).view(2, 512)
        backend.token_to_kv_pool = SimpleNamespace(
            full_to_swa_index_mapping=torch.arange(1024, dtype=torch.int32) + 3000,
            _npu_dspark_context_start=torch.tensor([0, 129]),
        )
        for committed in [0, 1, 7, 127, 128, 129]:
            prefix = 129 + committed
            block = torch.arange(512 + prefix, 512 + prefix + 6)
            indices = backend._build_dspark_sparse_indices(
                seq_lens=torch.tensor([prefix], dtype=torch.int32),
                req_pool_indices=torch.tensor([1], dtype=torch.int32),
                out_cache_loc=block,
                block_size=6,
            )
            history = torch.arange(512 + max(129, prefix - 128), 512 + prefix)
            expected = torch.cat([history, block]).to(torch.int32) + 3000
            for row in indices:
                self.assertTrue(torch.equal(row[row >= 0], expected))

    def test_startup_warmup_has_no_prefill_registration(self):
        manager = self.manager()
        manager.prefill_info_table = {}
        pool = SimpleNamespace(_npu_dspark_context_start=torch.tensor([0, 47]))
        manager.initialize_dspark_context(
            "2.2.2.2:8998", pool, 1, 6, fake_transfer=True
        )
        self.assertEqual(pool._npu_dspark_context_start.tolist(), [0, 6])
        manager._dspark_cold_start = False
        manager.initialize_dspark_context("unregistered", None, 0, 6)

    def test_boundary_exists_before_graph_capture_and_keeps_its_address(self):
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend._is_dspark_draft_worker = True
        backend.req_to_token = torch.zeros((9, 128), dtype=torch.int64)
        backend.token_to_kv_pool = SimpleNamespace(num_req_slots=9, swa_valid_from=None)
        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35",
            return_value=True,
        ):
            backend._init_dspark_context_start()
            self.assertFalse(
                hasattr(backend.token_to_kv_pool, "_npu_dspark_context_start")
            )
        with patch(
            "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35",
            return_value=False,
        ):
            backend._init_dspark_context_start()
            boundary = backend.token_to_kv_pool._npu_dspark_context_start
            self.assertEqual(boundary.tolist(), [0] * 9)
            boundary[1] = 129
            backend._init_dspark_context_start()
            self.assertIs(backend.token_to_kv_pool._npu_dspark_context_start, boundary)
            self.assertEqual(boundary[1], 129)
        self.assertIsNone(backend.token_to_kv_pool.swa_valid_from)


if __name__ == "__main__":
    unittest.main()
