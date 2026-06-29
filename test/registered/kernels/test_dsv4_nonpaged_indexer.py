import unittest
from types import SimpleNamespace

import torch

from sglang.jit_kernel.dsv4 import topk_transform_512
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    C4IndexerBackendMixin,
)
from sglang.srt.layers.attention.dsv4.metadata import NonPagedIndexerPlan
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

PAGE_SIZE = 64
HEAD_DIM = 128


def _random_fp8(shape):
    value = torch.rand(shape, device="cuda").add_(0.125).to(torch.bfloat16)
    scale = value.abs().float().amax(dim=-1, keepdim=True) / 448.0
    return (value.float() / scale).to(FP8_DTYPE), scale


class TestDeepSeekV4NonPagedIndexer(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nonpaged_matches_paged(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("requires Hopper or newer")
        import deep_gemm  # Import/ABI failure must fail the registered H100 job.

        torch.manual_seed(7)
        query_rows, key_rows, num_heads, num_pages = 256, 1021, 64, 18
        max_seqlen_k = 1024
        page_order = torch.tensor(
            [17, 2, 15, 1, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8],
            dtype=torch.int32,
            device="cuda",
        )
        page_table = page_order.unsqueeze(0).repeat(query_rows, 1)

        k_fp8, k_scale = _random_fp8((num_pages, PAGE_SIZE, 1, HEAD_DIM))
        packed = torch.cat(
            (
                k_fp8.reshape(num_pages, -1).view(torch.uint8),
                k_scale.reshape(num_pages, -1).view(torch.uint8),
            ),
            dim=1,
        )
        paged_cache = packed.view(num_pages, PAGE_SIZE, 1, HEAD_DIM + 4)

        q_fp8, q_scale = _random_fp8((query_rows, 1, num_heads, HEAD_DIM))
        weights = torch.rand(query_rows, num_heads, device="cuda").add_(0.125)
        weights *= q_scale[:, 0, :, 0]
        ke = torch.arange(
            key_rows - query_rows + 1,
            key_rows + 1,
            dtype=torch.int32,
            device="cuda",
        )
        context_lens = ke.unsqueeze(1)
        paged_logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            paged_cache,
            weights,
            context_lens,
            page_table,
            deep_gemm.get_paged_mqa_logits_metadata(
                context_lens, PAGE_SIZE, deep_gemm.get_num_sms()
            ),
            max_seqlen_k,
            clean_logits=False,
        )

        pool = object.__new__(DeepSeekV4IndexerPool)
        pool.index_k_with_scale_buffer = [packed]
        pool.page_size = PAGE_SIZE
        pool.index_head_dim = HEAD_DIM
        pool.quant_block_size = HEAD_DIM
        pool.device = torch.device("cuda")
        plan = NonPagedIndexerPlan(
            page_table=page_table[:1],
            gather_seq_lens=torch.tensor([key_rows], dtype=torch.int32, device="cuda"),
            ks=torch.zeros_like(ke),
            ke=ke,
            seq_len_sum=key_rows,
            max_seq_len=key_rows,
            max_seqlen_k=max_seqlen_k,
            query_rows=query_rows,
        )
        nonpaged_logits = C4IndexerBackendMixin._forward_nonpaged_indexer(
            q_indexer=q_fp8[:, 0],
            weights=weights,
            c4_indexer=SimpleNamespace(layer_id=0),
            token_to_kv_pool=pool,
            plan=plan,
        )

        valid = torch.arange(max_seqlen_k, device="cuda")[None, :] < ke[:, None]
        torch.testing.assert_close(
            nonpaged_logits[valid], paged_logits[valid], rtol=0, atol=0
        )

        def physical_topk(logits):
            physical = torch.empty((query_rows, 512), dtype=torch.int32, device="cuda")
            raw = torch.empty_like(physical)
            topk_transform_512(logits, ke, page_table, physical, PAGE_SIZE, raw)
            return physical.sort(dim=1).values

        torch.testing.assert_close(
            physical_topk(nonpaged_logits),
            physical_topk(paged_logits),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
