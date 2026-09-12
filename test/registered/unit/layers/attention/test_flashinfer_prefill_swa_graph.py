"""SWA graph write-buffer lifetime, without model weights or GPU execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferPrefillSWAGraph(CustomTestCase):
    def make_backend(self, swa=True):
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.use_sliding_window_kv_pool = swa
        backend.workspace_buffer = torch.empty(1, dtype=torch.uint8)
        backend.num_wrappers = 1
        backend.max_context_len = 16
        backend.prefill_backend = "fa2"
        backend.kv_last_page_len = torch.ones(2, dtype=torch.int32)
        backend.indices_updater_prefill = SimpleNamespace(
            num_qo_heads=4, num_kv_heads=1, head_dim=8, update=Mock()
        )
        backend.kv_index_translator = SimpleNamespace(
            sliding_window_write_loc_for=lambda loc: loc + 100
        )
        with (
            patch.object(backend, "_full_cg_prefill_workspace_bytes", return_value=64),
            patch(
                "sglang.srt.layers.attention.flashinfer_backend.BatchPrefillWithPagedKVCacheWrapper",
                create=True,
            ),
        ):
            backend.full_cg_prefill_wrappers = backend._create_full_cg_prefill_wrappers(
                2, 8
            )
        return backend

    def make_batch(self, mode=ForwardMode.EXTEND):
        return SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.arange(2),
            seq_lens=torch.tensor([4, 4]),
            seq_lens_cpu=torch.tensor([4, 4]),
            seq_lens_sum=8,
            encoder_lens=None,
            forward_mode=mode,
            spec_info=None,
            positions=torch.arange(8),
            extend_prefix_lens=torch.zeros(2, dtype=torch.int32),
            out_cache_loc=torch.arange(8),
        )

    def test_prefill_capture_before_decode_and_replay_keep_stable_storage(self):
        backend = self.make_backend()
        batch = self.make_batch()
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        captured = backend.forward_metadata.swa_out_cache_loc
        address = captured.data_ptr()
        self.assertEqual(captured.shape, (8,))
        self.assertFalse(torch.any(captured))

        # Decode capture happens later and must not replace prefill's write target.
        backend.cuda_graph_swa_out_cache_loc = torch.full((2,), 99, dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            swa_out_cache_loc=backend.cuda_graph_swa_out_cache_loc
        )
        for locations in (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([7, 8, 9])):
            batch.out_cache_loc = locations
            backend.init_forward_metadata_out_graph(batch)
            self.assertEqual(captured.data_ptr(), address)
            self.assertTrue(torch.equal(captured[: len(locations)], locations + 100))
            self.assertFalse(torch.any(captured[len(locations) :]))
            self.assertTrue(torch.all(backend.cuda_graph_swa_out_cache_loc == 99))

        batch.out_cache_loc = torch.arange(9)
        with self.assertRaisesRegex(AssertionError, "used 9 > capacity 8"):
            backend.init_forward_metadata_out_graph(batch)

    def test_decode_and_speculative_replay_keep_existing_buffer(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.DLLM_EXTEND,
        ):
            with self.subTest(mode=mode):
                backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
                backend.use_sliding_window_kv_pool = True
                backend.cuda_graph_swa_out_cache_loc = torch.full((4,), 99)
                backend.kv_index_translator = SimpleNamespace(
                    sliding_window_write_loc_for=lambda loc: loc + 100
                )
                backend.indices_updater_decode = SimpleNamespace(update=Mock())
                backend.indices_updater_prefill = SimpleNamespace(update=Mock())
                backend.decode_cuda_graph_metadata = {2: [Mock()]}
                backend.prefill_cuda_graph_metadata = {2: [Mock()]}
                backend.draft_extend_cuda_graph_metadata = {2: [Mock()]}
                backend.disable_cuda_graph_kv_split = False
                backend.dllm_config = SimpleNamespace(block_size=2)
                backend.use_paged = True
                batch = self.make_batch(mode)
                batch.out_cache_loc = torch.tensor([7, 8])
                backend.init_forward_metadata_out_graph(batch)
                self.assertTrue(
                    torch.equal(
                        backend.cuda_graph_swa_out_cache_loc,
                        torch.tensor([107, 108, 0, 0]),
                    )
                )

    def test_full_attention_does_not_allocate_swa_storage(self):
        self.assertIsNone(
            self.make_backend(swa=False).full_cg_prefill_swa_out_cache_loc
        )


if __name__ == "__main__":
    unittest.main()
