import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4TokenToKVPool,
)


class TestDeepSeekV4IndexerGatherAPI(unittest.TestCase):
    def test_indexer_pool_forwards_batched_gather_arguments(self):
        pool = object.__new__(DeepSeekV4IndexerPool)
        buffer = MagicMock(name="indexer_buffer")
        pool.index_k_with_scale_buffer = [buffer]
        seq_lens = torch.tensor([65, 127], dtype=torch.int32)
        page_indices = torch.tensor([[3, 1], [2, 2]], dtype=torch.int32)
        expected = (MagicMock(name="k"), MagicMock(name="scale"))

        with patch.object(
            index_buf_accessor.GetKAndS, "execute", return_value=expected
        ) as execute:
            actual = pool.get_index_k_scale_buffer(
                layer_id=0,
                seq_len_tensor=seq_lens,
                page_indices=page_indices,
                seq_len_sum=192,
                max_seq_len=127,
            )

        self.assertIs(actual, expected)
        execute.assert_called_once_with(
            pool,
            buffer,
            page_indices=page_indices,
            seq_len_tensor=seq_lens,
            seq_len_sum=192,
            max_seq_len=127,
        )

    def test_token_pool_maps_model_layer_to_c4_layer(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.wait_layer_transfer = MagicMock()
        pool.layer_mapping = {17: (4, 3, MagicMock())}
        pool.c4_indexer_kv_pool = MagicMock()
        seq_lens = torch.tensor([65, 127], dtype=torch.int32)
        page_indices = torch.tensor([[3, 1], [2, 2]], dtype=torch.int32)
        expected = (MagicMock(name="k"), MagicMock(name="scale"))
        pool.c4_indexer_kv_pool.get_index_k_scale_buffer.return_value = expected

        actual = pool.get_index_k_scale_buffer(
            layer_id=17,
            seq_len_tensor=seq_lens,
            page_indices=page_indices,
            seq_len_sum=192,
            max_seq_len=127,
        )

        self.assertIs(actual, expected)
        pool.wait_layer_transfer.assert_called_once_with(17)
        pool.c4_indexer_kv_pool.get_index_k_scale_buffer.assert_called_once_with(
            3,
            seq_lens,
            page_indices,
            192,
            127,
        )


if __name__ == "__main__":
    unittest.main()
