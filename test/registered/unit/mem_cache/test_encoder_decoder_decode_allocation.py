"""Encoder-decoder decode must append to, and release, the full cache row."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocation import alloc_for_decode_default
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestEncoderDecoderDecodeAllocation(CustomTestCase):
    def test_append_cross_pages_and_recycle_without_aliasing(self):
        """Decode previously reused encoder slots and returned duplicate pages."""
        for allocator_type in (
            PagedTokenToKVPoolAllocator,
            NPUPagedTokenToKVPoolAllocator,
        ):
            for encoder_lengths in ([1500, 128], [0, 0]):
                with self.subTest(
                    allocator=allocator_type.__name__, encoder_lengths=encoder_lengths
                ):
                    self.check_decode_and_release(allocator_type, encoder_lengths)

    def check_decode_and_release(self, allocator_type, encoder_lengths, device="cpu"):
        page_size = 128
        allocator = allocator_type(
            size=64 * page_size,
            page_size=page_size,
            dtype=torch.float16,
            device=device,
            kvcache=None,
            need_sort=False,
        )
        allocator.debug_mode = True
        table = ReqToTokenPool(2, 2048, device, enable_memory_saver=False)
        request_rows = torch.tensor([1, 2], dtype=torch.int64)
        decoder_lengths = torch.tensor([4, 127], dtype=torch.int64)
        encoder_tensor = torch.tensor(encoder_lengths, dtype=torch.int64)
        requests = [
            SimpleNamespace(kv=ReqKvInfo(req_pool_idx=int(row))) for row in request_rows
        ]
        for index, request in enumerate(requests):
            length = encoder_lengths[index] + int(decoder_lengths[index])
            allocated = allocator.alloc(
                (length + page_size - 1) // page_size * page_size
            )
            table.write(
                (request.kv.req_pool_idx, slice(0, length)), allocated[:length].int()
            )
            request.kv.set_row_len(length)

        tree_cache = SimpleNamespace(
            page_size=page_size,
            req_to_token_pool=table,
            token_to_kv_pool_allocator=allocator,
            supports_prefix_sharing=lambda: False,
        )
        batch = SimpleNamespace(
            model_config=SimpleNamespace(is_encoder_decoder=any(encoder_lengths)),
            seq_lens=decoder_lengths.to(device),
            seq_lens_cpu=decoder_lengths.clone(),
            encoder_lens=encoder_tensor.to(device),
            encoder_lens_cpu=list(encoder_lengths),
            req_pool_indices=request_rows.to(device),
            req_pool_indices_cpu=request_rows.clone(),
            req_to_token_pool=table,
            tree_cache=tree_cache,
            reqs=requests,
            out_cache_loc_dsv4=None,
            maybe_evict_swa=lambda: None,
        )
        for step in range(140):
            prior_lengths = batch.seq_lens.clone()
            prior_slots = torch.cat(
                [
                    table.req_to_token[
                        request.kv.req_pool_idx, : request.kv.kv_allocated_len
                    ]
                    for request in batch.reqs
                ]
            )
            new_slots = alloc_for_decode_default(batch, token_per_req=1)
            self.assertFalse(torch.isin(new_slots, prior_slots).any())
            self.assertEqual(len(torch.unique(new_slots)), len(batch.reqs))
            torch.testing.assert_close(batch.seq_lens, prior_lengths)
            torch.testing.assert_close(batch.seq_lens_cpu, prior_lengths.cpu())
            for index, request in enumerate(batch.reqs):
                self.assertEqual(
                    int(
                        table.req_to_token[
                            request.kv.req_pool_idx, request.kv.kv_allocated_len - 1
                        ]
                    ),
                    int(new_slots[index]),
                )
            batch.seq_lens += 1
            batch.seq_lens_cpu += 1
            if step == 20:
                order = torch.tensor([1, 0])
                batch.seq_lens = batch.seq_lens[order]
                batch.seq_lens_cpu = batch.seq_lens_cpu[order]
                batch.encoder_lens = batch.encoder_lens[order]
                batch.encoder_lens_cpu.reverse()
                batch.req_pool_indices = batch.req_pool_indices[order]
                batch.req_pool_indices_cpu = batch.req_pool_indices_cpu[order]
                batch.reqs.reverse()

        for request in batch.reqs:
            row = table.req_to_token[
                request.kv.req_pool_idx, : request.kv.kv_allocated_len
            ]
            allocator.free_segment(row, start_pos=0)
        self.assertEqual(allocator.available_size(), 64 * page_size)
        all_slots = allocator.alloc(64 * page_size)
        self.assertEqual(len(torch.unique(all_slots)), 64 * page_size)


if __name__ == "__main__":
    unittest.main()
