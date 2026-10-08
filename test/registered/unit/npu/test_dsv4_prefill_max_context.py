import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="stage-a-unit-test-npu")


class TestDsv4PrefillMaxContext(unittest.TestCase):
    def setUp(self):
        # Exercise real metadata builders with CPU tensors; no model or kernel
        # launch is needed to check table extents and live sequence lengths.
        self.backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
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
        lengths = [seq_len] if isinstance(seq_len, int) else list(seq_len)
        num_tokens = sum(lengths)
        return SimpleNamespace(
            forward_mode=mode,
            max_seq_len_override=max_context,
            batch_size=len(lengths),
            input_ids=torch.zeros(num_tokens, dtype=torch.int64),
            req_pool_indices=(
                torch.tensor([1]) if len(lengths) == 1 else torch.arange(len(lengths))
            ),
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
            extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=[0] * len(lengths),
            spec_info=None,
            out_cache_loc=torch.arange(num_tokens),
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

    def test_fixed_tables_keep_multiple_request_lengths(self):
        batch = self.batch([300, 212], 384)
        metadata = self.init_metadata(batch)
        self.assertEqual(metadata.block_tables.shape, (2, 3))
        self.assertEqual(metadata.block_tables_swa.shape, (2, 3))
        self.assertEqual(metadata.c4_page_table.shape, (2, 3))
        self.assertEqual(metadata.c128_page_table.shape, (2, 1))
        self.assertEqual(metadata.seq_lens.tolist(), [300, 212])
        self.assertEqual(metadata.extend_seq_lens.tolist(), [300, 212])

    def test_capture_scratch_respects_request_compressor_boundaries(self):
        for lengths, context, c4_count, c128_count in (
            ([384, 128], 384, 128, 4),
            ([300, 212], 300, 128, 3),
            ([127, 1], 127, 31, 0),
        ):
            with self.subTest(lengths=lengths):
                batch = self.batch(lengths, context)
                self.backend.prepare_prefill_graph_capture_batch(batch)
                bundle = batch.out_cache_loc_dsv4
                self.assertIs(bundle.out_full_loc, batch.out_cache_loc)
                self.assertEqual(bundle.out_swa_loc.numel(), sum(lengths))
                self.assertEqual(bundle.out_c4_loc.numel(), c4_count)
                self.assertEqual(bundle.out_c128_loc.numel(), c128_count)
                self.assertEqual(torch.count_nonzero(bundle.out_swa_loc).item(), 0)
                self.assertEqual(torch.count_nonzero(bundle.out_c4_loc).item(), 0)
                self.assertEqual(torch.count_nonzero(bundle.out_c128_loc).item(), 0)

    def test_replay_metadata_keeps_graph_limit_off_serving_batch(self):
        runner = SimpleNamespace(
            _is_full_backend=False,
            use_captured_attn_metadata=False,
            model_runner=SimpleNamespace(attn_backend=self.backend),
        )
        for lengths in ([128], [300, 212], [64, 64], [256]):
            with self.subTest(lengths=lengths):
                batch = self.batch(lengths)
                static_batch = SimpleNamespace(max_seq_len_override=384)
                with (
                    patch.object(
                        self.backend,
                        "init_forward_metadata",
                        side_effect=self.init_metadata,
                    ) as init_metadata,
                    patch.object(
                        self.backend, "prepare_prefill_shared_read_snapshot"
                    ) as snapshot,
                ):
                    PrefillCudaGraphRunner._prepare_forward_metadata_for_replay(
                        runner, batch, static_batch, SimpleNamespace(size=512)
                    )
                metadata_batch = init_metadata.call_args.args[0]
                self.assertIsNot(metadata_batch, batch)
                self.assertIs(metadata_batch.seq_lens, batch.seq_lens)
                self.assertEqual(metadata_batch.max_seq_len_override, 384)
                self.assertIsNone(batch.max_seq_len_override)
                snapshot.assert_called_once_with(metadata_batch, num_qo_tokens=512)
                metadata = self.backend.forward_metadata
                self.assertEqual(metadata.block_tables.shape, (len(lengths), 3))
                self.assertEqual(metadata.c4_page_table.shape, (len(lengths), 3))
                self.assertEqual(metadata.seq_lens.tolist(), lengths)

    def test_unconfigured_replay_uses_serving_batch(self):
        batch = self.batch(256)
        runner = SimpleNamespace(
            _is_full_backend=False,
            use_captured_attn_metadata=False,
            model_runner=SimpleNamespace(attn_backend=self.backend),
        )
        with patch.object(
            self.backend, "init_forward_metadata", side_effect=self.init_metadata
        ) as init_metadata:
            PrefillCudaGraphRunner._prepare_forward_metadata_for_replay(
                runner,
                batch,
                SimpleNamespace(max_seq_len_override=None),
                SimpleNamespace(size=256),
            )
        init_metadata.assert_called_once_with(batch)
        self.assertEqual(self.backend.forward_metadata.block_tables.shape, (1, 2))

    def test_context_limit_is_per_request_including_prefix(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.max_context_size = 384
        runner.max_num_tokens = 512
        runner.capture_num_tokens = [128, 256, 384, 512]
        for lengths, prefix_lengths, num_tokens, expected in (
            ([384], [0], 384, True),
            ([385], [0], 385, False),
            ([300, 212], [0, 0], 512, True),
            ([385, 127], [0, 0], 512, False),
            ([449], [385], 64, False),
        ):
            with self.subTest(lengths=lengths, prefix_lengths=prefix_lengths):
                batch = self.batch(lengths)
                batch.extend_prefix_lens_cpu = prefix_lengths
                self.assertEqual(
                    runner.can_replay_locally(
                        batch_size=len(lengths),
                        num_tokens=num_tokens,
                        input_embeds=None,
                        replace_embeds=None,
                        prefix_lens=prefix_lengths,
                        is_target_verify=False,
                        capture_hidden_mode=CaptureHiddenMode.NULL,
                        return_logprob=False,
                        batch_max_context_len=runner._batch_max_context_len(batch),
                    ),
                    expected,
                )

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
