import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="stage-a-unit-test-npu")


class TestDsv4PrefillMaxContext(unittest.TestCase):
    def test_capture_and_replay_keep_bounded_storage_and_live_lengths(self):
        backend = self.backend
        backend._dsv4_has_c4 = backend._dsv4_has_c128 = True
        backend._dsv4_state_pools_by_ratio = {
            ratio: SimpleNamespace(dummy_state_loc=-1) for ratio in (4, 128)
        }

        def build_metadata(batch):
            fm = self.init_metadata(batch)
            cu = torch.cat(
                [torch.zeros(1, dtype=torch.int32), batch.extend_seq_lens.cumsum(0)]
            ).int()
            fm.actual_seq_lengths_q_pa = cu
            fm.actual_seq_lengths_q = cu[1:]
            fm.actual_seq_lengths_kv = batch.seq_lens.int()
            fm.seqused = batch.extend_seq_lens.int()
            fm.start_pos = batch.extend_prefix_lens.int()
            fm.swa_page_table = fm.block_tables_swa
            fm.kernel_metadata = {"li_quant_metadata": torch.zeros(16).int()}
            for ratio in (4, 128):
                capacity = min(
                    batch.input_ids.numel(), batch.input_ids.numel() // ratio + 1
                )
                setattr(
                    fm, f"positions_cmp_padding_c{ratio}", torch.zeros(capacity).long()
                )

        def state_table(**kwargs):
            history = 8 if kwargs["ratio"] == 4 else 128
            return torch.full((1, history + kwargs["max_input_capacity"]), -1).int()

        capture_batch = self.batch(128, 512)
        capture_batch.positions = torch.arange(128)
        capture_batch.extend_prefix_lens = torch.zeros(1).int()
        capture_batch.can_run_tbo = False
        with (
            patch.object(backend, "init_forward_metadata", side_effect=build_metadata),
            patch.object(
                backend,
                "_build_dsv4_prefill_graph_state_table",
                side_effect=state_table,
            ),
        ):
            captured = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                capture_batch
            )
            state = captured.dsv4_prefill_graph_state
            tensors = [state.swa_page_table, state.c4_page_table, state.c128_page_table]
            pointers = [tensor.data_ptr() for tensor in tensors]
            self.assertEqual(captured.block_tables.shape, (1, 4))
            self.assertEqual(captured.block_tables_swa.shape, (1, 4))
            self.assertEqual(state.swa_page_table.shape, (1, 4))
            self.assertEqual(state.c4_page_table.shape, (1, 4))
            self.assertEqual(state.c128_page_table.shape, (1, 1))

            live = self.batch(384)
            live.input_ids = torch.zeros(65).long()
            live.extend_seq_lens = torch.tensor([65]).int()
            live.extend_prefix_lens = torch.tensor([319]).int()
            live.extend_prefix_lens_cpu = [319]
            live.forward_mode = ForwardMode.EXTEND
            live.can_run_tbo = False
            live.global_num_token_non_padded_cpu = 65
            backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                captured,
                live,
                static_forward_batch=SimpleNamespace(max_seq_len_override=512),
            )

        self.assertIs(backend.forward_metadata, captured)
        self.assertIsNone(live.max_seq_len_override)
        self.assertEqual(captured.actual_seq_lengths_kv.tolist(), [384])
        self.assertEqual(captured.seqused.tolist(), [65])
        self.assertEqual(captured.start_pos.tolist(), [319])
        self.assertEqual([tensor.data_ptr() for tensor in tensors], pointers)

    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(get_parallel().override(dcp_enabled=False))
        # Exercise real metadata builders with CPU tensors; no model or kernel
        # launch is needed to check table extents and live sequence lengths.
        self.backend = DeepseekV4AscendAttnBackend.__new__(DeepseekV4AscendAttnBackend)
        backend = self.backend
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
        backend._dsv4_unique_compress_ratios = [4, 128]
        backend._dsv4_compress_ratios = [4, 128]
        backend._is_dspark_algorithm = False

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

    def test_context_limit_is_per_request_including_prefix(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._captured_attn_metadata_max_bs = 1
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.max_context_size = 384
        runner.max_seq_len = None
        runner.max_num_tokens = 512
        runner.capture_num_tokens = [128, 256, 384, 512]
        for lengths, prefix_lengths, num_tokens, expected in (
            ([384], [0], 384, True),
            ([384], [128], 256, True),
            ([385], [0], 385, False),
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
