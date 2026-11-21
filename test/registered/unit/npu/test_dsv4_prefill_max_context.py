"""Bounded NPU prefill graphs consume live-width metadata at replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="stage-a-unit-test-npu")


class TestDsv4PrefillMaxContext(unittest.TestCase):
    @staticmethod
    def _backend():
        backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend.device = torch.device("cpu")
        backend.page_size = 128
        backend.max_context_len = 8192
        backend.req_to_token = (torch.arange(8192) + 128).repeat(2, 1)
        backend.req_to_token_pool = SimpleNamespace(
            size=2,
            req_to_token=backend.req_to_token,
            c128_page_size=16,
            req_to_c128_sidecar=torch.tensor([[0, 0, 0, 0], [7, 12, 17, 22]]),
        )
        backend.full_to_swa_index_mapping = torch.arange(8320) + 1280
        backend.token_to_kv_pool = SimpleNamespace()
        backend.is_hybrid_swa = True
        backend.use_sliding_window_kv_pool = False
        backend.use_mla = False
        backend.use_quant_flash_attn = False
        backend.use_flash_mla = False
        backend.use_paged_mtp = False
        backend.dense_mla_dcp_target = False
        backend.dsa_dcp_target = False
        backend._dsv4_has_c4 = backend._dsv4_has_c128 = True
        backend._dsv4_unique_compress_ratios = [4, 128]
        backend._dsv4_compress_ratios = [4, 128]
        backend._is_dspark_algorithm = False
        backend._dsv4_state_pools_by_ratio = {
            ratio: SimpleNamespace(dummy_state_loc=-1) for ratio in (4, 128)
        }
        return backend

    @staticmethod
    def _batch(context_len, query_len=128, max_context=None):
        prefix = context_len - query_len
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            max_seq_len_override=max_context,
            batch_size=1,
            input_ids=torch.zeros(query_len, dtype=torch.int64),
            positions=torch.arange(prefix, context_len),
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([context_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([context_len], dtype=torch.int32),
            extend_seq_lens=torch.tensor([query_len], dtype=torch.int32),
            extend_seq_lens_cpu=[query_len],
            extend_prefix_lens=torch.tensor([prefix], dtype=torch.int32),
            extend_prefix_lens_cpu=[prefix],
            spec_info=None,
            out_cache_loc=torch.arange(query_len),
            out_cache_loc_dsv4=None,
            can_run_tbo=False,
            global_num_token_non_padded_cpu=query_len,
        )

    def test_capture_and_replay_keep_capacity_with_live_width_metadata(self):
        for bound in (512, None):
            with self.subTest(max_context=bound):
                backend = self._backend()
                batches = []
                live_metadata = []

                def initialize(batch):
                    batches.append(batch)
                    AscendAttnBackend.init_forward_metadata(backend, batch)
                    backend._build_npu_compress_metadata(batch)
                    fm = backend.forward_metadata
                    live_metadata.append(fm)
                    cu = torch.tensor([0, batch.input_ids.numel()]).int()
                    fm.actual_seq_lengths_q_pa = cu
                    fm.actual_seq_lengths_q = cu[1:]
                    fm.actual_seq_lengths_kv = batch.seq_lens.int()
                    fm.seqused = batch.extend_seq_lens.int()
                    fm.start_pos = batch.extend_prefix_lens.int()
                    fm.swa_page_table = fm.block_tables_swa
                    fm.kernel_metadata = {"li_quant_metadata": torch.zeros(16).int()}
                    for ratio in (4, 128):
                        capacity = min(cu[-1].item(), cu[-1].item() // ratio + 1)
                        setattr(
                            fm,
                            f"positions_cmp_padding_c{ratio}",
                            torch.zeros(capacity).long(),
                        )

                def state_table(**kwargs):
                    history = 8 if kwargs["ratio"] == 4 else 128
                    return torch.full(
                        (1, history + kwargs["max_input_capacity"]), -1
                    ).int()

                with (
                    get_parallel().override(dcp_enabled=False),
                    patch.object(
                        backend, "init_forward_metadata", side_effect=initialize
                    ),
                    patch.object(backend, "_build_npu_compress_metadata_prefill"),
                    patch.object(
                        backend,
                        "_build_dsv4_prefill_graph_state_table",
                        side_effect=state_table,
                    ),
                ):
                    capture_batch = self._batch(128, max_context=bound)
                    captured = (
                        backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                            capture_batch
                        )
                    )
                    state = captured.dsv4_prefill_graph_state
                    tables = [
                        state.swa_page_table,
                        state.c4_page_table,
                        state.c128_page_table,
                    ]
                    pointers = [table.data_ptr() for table in tables]
                    context_capacity = (
                        backend.max_context_len if bound is None else bound
                    )
                    self.assertEqual(
                        [table.shape[1] for table in tables],
                        [
                            context_capacity // 128,
                            context_capacity // 128,
                            max(1, context_capacity // (128 * 16)),
                        ],
                    )
                    # The temporary full/SWA metadata retains its live width.
                    self.assertEqual(captured.block_tables.shape, (1, 1))
                    self.assertEqual(captured.block_tables_swa.shape, (1, 1))

                    for context_len, query_len in ((384, 65), (128, 128), (512, 128)):
                        live = self._batch(context_len, query_len)
                        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                            captured,
                            live,
                            static_forward_batch=SimpleNamespace(
                                max_seq_len_override=bound
                            ),
                        )
                        raw = live_metadata[-1]
                        width = context_len // 128
                        self.assertIs(batches[-1], live)
                        self.assertIsNone(live.max_seq_len_override)
                        self.assertEqual(raw.swa_page_table.shape, (1, width))
                        self.assertEqual(raw.c4_page_table.shape, (1, width))
                        self.assertIs(backend.forward_metadata, captured)
                        self.assertEqual(
                            captured.actual_seq_lengths_kv.tolist(), [context_len]
                        )
                        self.assertEqual(captured.seqused.tolist(), [query_len])
                        self.assertEqual(
                            captured.start_pos.tolist(), [context_len - query_len]
                        )
                        self.assertEqual(
                            [table.data_ptr() for table in tables], pointers
                        )
                        for dst, src in (
                            (state.swa_page_table, raw.swa_page_table),
                            (state.c4_page_table, raw.c4_page_table),
                            (state.c128_page_table, raw.c128_page_table),
                        ):
                            torch.testing.assert_close(dst[:, : src.shape[1]], src)
                            self.assertTrue((dst[:, src.shape[1] :] == 0).all().item())


if __name__ == "__main__":
    unittest.main()
