# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.srt.mem_cache.sparsity.algorithms.knorm_algorithm import KnormPageAlgorithm
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class MockConfig:
    def __init__(self, compression_ratio=0.2, page_size=64):
        self.compression_ratio = compression_ratio
        self.page_size = page_size


class MockTokenToKVPool:
    def __init__(self, num_tokens=1024, num_layers=2, num_heads=8, head_dim=64):
        self._k_buffer = {
            i: torch.randn(
                num_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda"
            )
            for i in range(num_layers)
        }

    def get_key_buffer(self, layer_id):
        return self._k_buffer[layer_id]


class MockReqToTokenPool:
    def __init__(self, max_reqs=32, max_tokens=2048, num_physical_tokens=1024):
        self.req_to_token = (
            torch.arange(max_reqs * max_tokens, device="cuda").reshape(
                max_reqs, max_tokens
            )
            % num_physical_tokens
        )


class MockStates:
    def __init__(self, max_reqs=32):
        self.prompt_lens = torch.zeros(max_reqs, dtype=torch.int64, device="cuda")
        self.repr_constructed = torch.zeros(max_reqs, dtype=torch.bool, device="cuda")
        self.last_constructed_page = torch.zeros(
            max_reqs, dtype=torch.int64, device="cuda"
        )


class MockForwardBatch:
    def __init__(self, mode=ForwardMode.EXTEND):
        self.forward_mode = mode


class MockAttnMetadata:
    def __init__(self, cache_seqlens):
        self.cache_seqlens_int32 = cache_seqlens


class TestKnormPageAlgorithm(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.config = MockConfig(compression_ratio=0.2, page_size=64)
        self.algorithm = KnormPageAlgorithm(self.config, self.device)
        self.token_to_kv_pool = MockTokenToKVPool(num_tokens=1024, num_layers=2)
        self.req_to_token_pool = MockReqToTokenPool(
            max_reqs=8, max_tokens=512, num_physical_tokens=1024
        )
        self.states = MockStates(max_reqs=8)

    def test_initialize_representation_pool(self):
        self.algorithm.initialize_representation_pool(
            start_layer=0,
            end_layer=2,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            states=self.states,
        )

        self.assertEqual(len(self.algorithm.page_scores), 2)
        self.assertIsNotNone(self.algorithm.req_to_token_pool)
        self.assertIsNotNone(self.algorithm.states)

    def test_construct_representations(self):
        self.algorithm.initialize_representation_pool(
            0, 2, self.token_to_kv_pool, self.req_to_token_pool, self.states
        )

        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([128, 192], dtype=torch.int64, device="cuda")
        k_buffer = self.token_to_kv_pool.get_key_buffer(0)
        forward_batch = MockForwardBatch(mode=ForwardMode.EXTEND)

        self.states.prompt_lens[req_pool_indices] = seq_lens

        self.algorithm.construct_representations(
            layer_id=1,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            k_buffer=k_buffer,
            forward_batch=forward_batch,
        )

        self.assertTrue(self.states.repr_constructed[0])
        self.assertTrue(self.states.repr_constructed[1])
        # last_constructed_page stores page count, not token position
        # 128 / 64 = 2 pages, 192 / 64 = 3 pages
        self.assertEqual(self.states.last_constructed_page[0].item(), 2)
        self.assertEqual(self.states.last_constructed_page[1].item(), 3)

    def test_update_representations(self):
        self.algorithm.initialize_representation_pool(
            0, 2, self.token_to_kv_pool, self.req_to_token_pool, self.states
        )

        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([192, 256], dtype=torch.int64, device="cuda")
        k_buffer = self.token_to_kv_pool.get_key_buffer(0)
        forward_batch = MockForwardBatch(mode=ForwardMode.DECODE)

        self.states.repr_constructed[req_pool_indices] = True
        # Start from page 2 (was 128 tokens)
        self.states.last_constructed_page[req_pool_indices] = torch.tensor(
            [2, 2], dtype=torch.int64, device="cuda"
        )

        self.algorithm.update_representations(
            layer_id=1,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            k_buffer=k_buffer,
            forward_batch=forward_batch,
        )

        # 192 / 64 = 3 pages, 256 / 64 = 4 pages
        self.assertEqual(self.states.last_constructed_page[0].item(), 3)
        self.assertEqual(self.states.last_constructed_page[1].item(), 4)

    def test_retrieve_topk(self):
        self.algorithm.initialize_representation_pool(
            0, 2, self.token_to_kv_pool, self.req_to_token_pool, self.states
        )

        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        queries = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
        cache_seqlens = torch.tensor([512, 640], dtype=torch.int32, device="cuda")
        sparse_mask = torch.ones(2, dtype=torch.bool, device="cuda")
        attn_metadata = MockAttnMetadata(cache_seqlens=cache_seqlens)

        self.algorithm.page_scores[0] = torch.randn(
            16, 1, dtype=torch.float32, device="cuda"
        )

        out_indices, out_lengths = self.algorithm.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            attn_metadata=attn_metadata,
        )

        self.assertEqual(out_indices.shape[0], 2)
        self.assertEqual(out_lengths.shape[0], 2)
        self.assertTrue((out_lengths > 0).all())


if __name__ == "__main__":
    unittest.main()
