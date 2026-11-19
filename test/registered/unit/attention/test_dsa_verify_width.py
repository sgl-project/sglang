"""DSA metadata must describe the live DFlash verification query rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def expand_query_lengths(extend_lens, cache_lens, total, max_extend):
    lengths = torch.cat(
        [
            torch.arange(int(cache - extend) + 1, int(cache) + 1, dtype=torch.int32)
            for extend, cache in zip(extend_lens, cache_lens, strict=True)
        ]
    )
    assert len(lengths) == total
    return lengths


class TestDSAVerifyWidth(CustomTestCase):
    def backend(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.speculative_num_draft_tokens = 12
        backend.req_to_token = torch.arange(256).view(2, 128)
        backend.req_to_token_pool = SimpleNamespace(req_to_token=backend.req_to_token)
        backend.runner_ids = None
        backend._kv_shard_pool = None
        backend.set_dsa_prefill_impl = lambda batch: None
        backend.dsa_decode_impl = "triton"
        backend.use_mha = False
        backend.get_topk_transform_method = lambda mode: None
        backend.dsa_index_kpool = 1
        backend.dsa_index_topk = 64
        backend.physical_page_size = 64
        backend.dsa_shards_index_k = False
        backend._gather_token_table = lambda table, indices, width: table[
            indices, :width
        ]
        backend.get_device_int32_arange = lambda count: torch.arange(
            count, dtype=torch.int32
        )
        backend._transform_table_1_to_real = lambda table: table
        backend._build_topk_v2_plan = lambda lengths: None
        backend._init_kpool_metadata = lambda metadata, batch, **kwargs: metadata
        return backend

    def batch(self, width):
        lengths = torch.tensor([17, 33], dtype=torch.int32)
        return SimpleNamespace(
            batch_size=2,
            seq_lens=lengths,
            seq_lens_cpu=lengths,
            seq_lens_sum=50,
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=DFlashVerifyInput(
                draft_token=torch.empty((2 * max(width, 0),), dtype=torch.int64),
                positions=torch.empty((2 * max(width, 0),), dtype=torch.int64),
                draft_token_num=width,
            ),
        )

    def test_metadata_tracks_b12_b1_b12_without_mutating_configuration(self):
        backend = self.backend()
        for width in (12, 1, 12):
            with (
                self.subTest(width=width),
                patch.object(dsa_backend, "is_cuda", return_value=False),
                patch.object(
                    dsa_backend, "seqlens_expand_triton", expand_query_lengths
                ),
                patch.object(
                    dsa_backend, "pad_dsa_cache_seqlens", lambda batch, lens: lens
                ),
            ):
                batch = self.batch(width)
                backend.init_forward_metadata(batch)
            metadata = backend.forward_metadata
            torch.testing.assert_close(
                metadata.cache_seqlens_int32, batch.seq_lens + width
            )
            self.assertEqual(batch.extend_seq_lens_cpu, [width, width])
            self.assertEqual(metadata.page_table_1.shape[0], 2 * width)
            self.assertEqual(metadata.cu_seqlens_q.tolist(), list(range(2 * width + 1)))
            expected = torch.cat(
                [torch.arange(18, 18 + width), torch.arange(34, 34 + width)]
            ).int()
            torch.testing.assert_close(metadata.dsa_seqlens_expanded, expected)
            self.assertEqual(backend.speculative_num_draft_tokens, 12)

    def test_blackwell_schedule_uses_the_live_query_width(self):
        backend = self.backend()
        for width in (12, 1, 2, 12):
            batch = self.batch(width)
            cache = batch.seq_lens + width
            expanded = expand_query_lengths(
                torch.tensor([width, width]), cache, 2 * width, width
            )
            with patch.object(
                dsa_backend, "get_platform", return_value=SimpleNamespace(is_sm100=True)
            ):
                schedule = backend._build_paged_mqa_schedule_2d_ctx_lens(
                    batch.forward_mode, cache, expanded, 2, draft_token_num=width
                )
            self.assertEqual(tuple(schedule.shape), (2, width))
            self.assertEqual(backend.speculative_num_draft_tokens, 12)

    def test_other_speculative_inputs_keep_the_configured_width(self):
        backend = self.backend()
        batch = self.batch(1)
        batch.spec_info = SimpleNamespace(draft_token_num=1)
        self.assertEqual(backend._runtime_verify_width(batch), 12)

    def test_nonpositive_dflash_width_is_rejected(self):
        backend = self.backend()
        for width in (0, -1):
            with self.subTest(width=width), self.assertRaises(ValueError):
                backend.init_forward_metadata(self.batch(width))


if __name__ == "__main__":
    unittest.main()
