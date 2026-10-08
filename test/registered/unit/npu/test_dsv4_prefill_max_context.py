import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="stage-a-unit-test-npu")


class TestDsv4PrefillMaxContext(unittest.TestCase):
    def setUp(self):
        # Exercise real metadata builders with CPU tensors; no model or kernel
        # launch is needed to check table extents and live sequence lengths.
        self.backend = DeepseekV4AscendAttnBackend.__new__(
            DeepseekV4AscendAttnBackend
        )
        backend = self.backend
        backend.device = torch.device("cpu")
        backend.page_size = 128
        backend.max_context_len = 8192
        backend.req_to_token = (torch.arange(8192) + 128).repeat(2, 1)
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=backend.req_to_token,
            c128_page_size=16,
            req_to_c128_sidecar=torch.tensor([[0, 0, 0, 0], [7, 12, 17, 22]]),
        )
        backend.full_to_swa_index_mapping = torch.arange(8320) + 1280
        backend.token_to_kv_pool = SimpleNamespace()
        backend.is_hybrid_swa = True
        backend.use_sliding_window_kv_pool = False
        backend.use_mla = False
        backend._dsv4_unique_compress_ratios = [4, 128]
        backend._dsv4_compress_ratios = [4, 128]

    def batch(self, seq_len, max_context=None, mode=ForwardMode.EXTEND):
        return SimpleNamespace(
            forward_mode=mode,
            max_seq_len_override=max_context,
            batch_size=1,
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            extend_seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            extend_seq_lens_cpu=[seq_len],
            extend_prefix_lens_cpu=[0],
            spec_info=None,
            out_cache_loc=torch.arange(seq_len),
            out_cache_loc_dsv4=None,
        )

    def init_metadata(self, batch):
        AscendAttnBackend.init_forward_metadata(self.backend, batch)
        with (
            patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.is_npu_arch35",
                return_value=False,
            ),
            patch.object(self.backend, "_build_npu_compress_metadata_prefill"),
        ):
            self.backend._build_npu_compress_metadata(batch)
        return self.backend.forward_metadata

    def test_fixed_tables_keep_live_lengths(self):
        for seq_len in (64, 128, 300, 1024):
            with self.subTest(seq_len=seq_len):
                metadata = self.init_metadata(self.batch(seq_len, 1024))
                self.assertEqual(metadata.block_tables.tolist(), [list(range(1, 9))])
                self.assertEqual(
                    metadata.block_tables_swa.tolist(), [list(range(11, 19))]
                )
                self.assertEqual(metadata.c4_page_table.shape, (1, 8))
                self.assertEqual(metadata.c128_page_table.tolist(), [[7]])
                self.assertEqual(metadata.seq_lens.tolist(), [seq_len])
                self.assertEqual(metadata.seq_lens_cpu_int.tolist(), [seq_len])
                self.assertEqual(metadata.extend_seq_lens.tolist(), [seq_len])

    def test_c128_extent_uses_compressed_page_size(self):
        metadata = self.init_metadata(self.batch(128, 4096))
        self.assertEqual(metadata.block_tables.shape, (1, 32))
        self.assertEqual(metadata.c4_page_table.shape, (1, 32))
        self.assertEqual(metadata.c128_page_table.tolist(), [[7, 12]])

    def test_unconfigured_prefill_keeps_live_table_extent(self):
        metadata = self.init_metadata(self.batch(256))
        self.assertEqual(metadata.block_tables.shape, (1, 2))
        self.assertEqual(metadata.c4_page_table.shape, (1, 2))

    def test_decode_keeps_live_table_extent(self):
        batch = self.batch(256, 1024, ForwardMode.DECODE)
        AscendAttnBackend.init_forward_metadata(self.backend, batch)
        self.assertEqual(self.backend.forward_metadata.block_tables.shape, (1, 2))

    def test_backend_without_capability_keeps_live_table_extent(self):
        self.backend.supports_prefill_cuda_graph_max_context_size = False
        metadata = self.init_metadata(self.batch(256, 1024))
        self.assertEqual(metadata.block_tables.shape, (1, 2))
        self.assertEqual(metadata.c4_page_table.shape, (1, 2))

    def test_rejects_context_larger_than_fixed_capacity(self):
        batch = self.batch(513, 512)
        batch.extend_seq_lens_cpu = [64]
        batch.extend_prefix_lens_cpu = [449]
        with self.assertRaisesRegex(ValueError, "smaller than the live context"):
            self.init_metadata(batch)

    def test_rejects_invalid_fixed_capacity(self):
        for max_context in (0, -128, 8320):
            with self.subTest(max_context=max_context):
                with self.assertRaisesRegex(ValueError, "must fit the model"):
                    self.init_metadata(self.batch(128, max_context))


if __name__ == "__main__":
    unittest.main()
