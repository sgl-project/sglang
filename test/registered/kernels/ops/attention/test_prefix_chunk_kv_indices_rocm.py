"""Packed chunked-prefix layout on ROCm: chunk count and the KV indices the GPU kernel gathers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="base-b-test-1-gpu-small-amd-mi35x")

_EXTEND_LEN = 64
_MAX_BS = 4
_MAX_SEQ_LEN = 128


class _Batch(ForwardBatch):
    def __init__(self, *, capacity, **kwargs):
        super().__init__(**kwargs)
        self._capacity = capacity

    def get_max_chunk_capacity(self):
        return self._capacity


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "Requires ROCm")
class TestPackedPrefixChunkKvIndices(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.device = "cuda"
        self.req_to_token = (
            torch.arange(_MAX_BS * _MAX_SEQ_LEN, device=self.device)
            .reshape(_MAX_BS, _MAX_SEQ_LEN)
            .to(torch.int32)
        )
        pool = MLATokenToKVPool(
            size=_MAX_BS * _MAX_SEQ_LEN,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )
        backend = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
            token_to_kv_pool=pool,
            runner_ids=RunnerIds(
                req_to_token=self.req_to_token,
                token_to_kv_pool_allocator=None,
                token_to_kv_pool=pool,
                page_size=1,
                device=self.device,
            ),
        )
        previous = set_forward_context(ForwardContext(attn_backend=backend))
        self.addCleanup(set_forward_context, previous)

    def _prepare(self, *, prefix_lens, capacity):
        bs = len(prefix_lens)
        seq_lens_cpu = [_EXTEND_LEN + prefix for prefix in prefix_lens]
        batch = _Batch(
            capacity=capacity,
            batch_size=bs,
            input_ids=torch.zeros((bs, _EXTEND_LEN), dtype=torch.int64, device="cuda"),
            out_cache_loc=torch.arange(bs * _EXTEND_LEN, device="cuda"),
            seq_lens_sum=sum(seq_lens_cpu),
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(bs, device="cuda"),
            seq_lens=torch.tensor(seq_lens_cpu, device="cuda"),
            seq_lens_cpu=seq_lens_cpu,
            extend_prefix_lens=torch.tensor(prefix_lens, device="cuda"),
            extend_prefix_lens_cpu=prefix_lens,
        )
        batch.prepare_chunked_prefix_cache_info(self.device)
        return batch

    def test_packed_chunks_gather_the_cached_tokens_of_each_request(self):
        for prefix_lens, capacity, num_chunks in (
            ([16, 32, 48, 64], 64, 3),
            ([1, 64], 31, 3),
        ):
            with self.subTest(prefix_lens=prefix_lens, capacity=capacity):
                batch = self._prepare(prefix_lens=prefix_lens, capacity=capacity)
                self.assertEqual(batch.num_prefix_chunks, num_chunks)
                self.assertEqual(batch.prefix_chunk_len, capacity)
                for chunk in range(batch.num_prefix_chunks):
                    expected = [
                        self.req_to_token[
                            req,
                            batch.prefix_chunk_starts[chunk, req].item() : (
                                batch.prefix_chunk_starts[chunk, req].item()
                                + batch.prefix_chunk_seq_lens[chunk, req].item()
                            ),
                        ]
                        for req in range(len(prefix_lens))
                    ]
                    self.assertTrue(
                        torch.equal(
                            batch.prefix_chunk_kv_indices[chunk], torch.cat(expected)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
