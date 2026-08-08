import unittest

import numpy as np
import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    make_local_attention_virtual_batches,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class TestLocalAttentionMetadata(unittest.TestCase):
    def test_default_preserves_short_prompt_chunk_normalization(self):
        query_start_loc = np.array([0, 1000], dtype=np.int32)
        seq_lens = np.array([1000], dtype=np.int32)
        block_table = torch.arange(128, dtype=torch.int32).reshape(1, 128)

        (
            query_lens,
            query_start_loc_local,
            key_lens,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            attn_chunk_size=8192,
            query_start_loc_np=query_start_loc,
            seq_lens_np=seq_lens,
            block_table=block_table,
            page_size=16,
        )

        np.testing.assert_array_equal(query_lens, np.array([992, 8]))
        np.testing.assert_array_equal(
            query_start_loc_local, np.array([0, 992, 1000], dtype=np.int32)
        )
        np.testing.assert_array_equal(key_lens, np.array([992, 8], dtype=np.int32))
        self.assertEqual(block_table_local.shape, (2, 62))

    def test_short_prompt_does_not_change_attention_chunk_boundary(self):
        query_start_loc = np.array([0, 1000], dtype=np.int32)
        seq_lens = np.array([1000], dtype=np.int32)
        block_table = torch.arange(128, dtype=torch.int32).reshape(1, 128)

        (
            query_lens,
            query_start_loc_local,
            key_lens,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            attn_chunk_size=8192,
            query_start_loc_np=query_start_loc,
            seq_lens_np=seq_lens,
            block_table=block_table,
            page_size=16,
            preserve_attn_chunk_size=True,
        )

        np.testing.assert_array_equal(query_lens, np.array([1000]))
        np.testing.assert_array_equal(
            query_start_loc_local, np.array([0, 1000], dtype=np.int32)
        )
        np.testing.assert_array_equal(key_lens, np.array([1000], dtype=np.int32))
        self.assertEqual(block_table_local.shape, (1, 512))
        torch.testing.assert_close(
            block_table_local[0, :128], torch.arange(128, dtype=torch.int32)
        )

    def test_eager_metadata_converts_token_locations_to_page_indices(self):
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.has_local_attention = True
        backend.use_sliding_window_kv_pool = False
        backend.attention_chunk_size = 8192
        backend.page_size = 16

        metadata = FlashAttentionMetadata()
        metadata.cu_seqlens_q = torch.tensor([0, 1000], dtype=torch.int32)
        metadata.cache_seqlens_int32 = torch.tensor([1000], dtype=torch.int32)

        page_ids = torch.arange(64, dtype=torch.int32) * 17 + 3
        metadata.page_table = (
            page_ids[:, None] * backend.page_size
            + torch.arange(backend.page_size, dtype=torch.int32)
        ).reshape(1, -1)

        backend._maybe_init_local_attn_metadata(
            forwardbatch=None,
            metadata=metadata,
            device=torch.device("cpu"),
        )

        local_metadata = metadata.local_attn_metadata
        self.assertEqual(local_metadata.local_block_table.shape, (1, 512))
        torch.testing.assert_close(
            local_metadata.local_block_table[0, :63], page_ids[:63]
        )


if __name__ == "__main__":
    unittest.main()
