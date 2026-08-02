import math
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from sglang.srt.mem_cache.sparsity.backend.visibility_adaptor import (
    FlashAttentionVisibilityAdaptor,
    HBMResidentPlacement,
)
from sglang.srt.mem_cache.sparsity.contracts import Granularity, SelectionResult
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and _is_fa3_supported(), "FA3 requires a supported GPU"
)
class TestKVSparsityFA3Visibility(unittest.TestCase):
    def test_noncontiguous_sparse_page_table_matches_reference_attention(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        query = torch.randn((1, 1, 64), device=device, dtype=dtype)
        key_cache = torch.randn((8, 1, 1, 64), device=device, dtype=dtype)
        value_cache = torch.randn((8, 1, 1, 64), device=device, dtype=dtype)
        req_to_token = torch.zeros((2, 8), device=device, dtype=torch.int32)
        req_to_token[1] = torch.arange(8, device=device, dtype=torch.int32)
        metadata = SimpleNamespace(
            page_table=req_to_token[1:2].clone(),
            cache_seqlens_int32=torch.tensor([8], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        dense_page_table = metadata.page_table.clone()
        adaptor = FlashAttentionVisibilityAdaptor(
            HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
        )
        result = SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=torch.tensor(
                [[0, 1, 6, 7]], device=device, dtype=torch.int32
            ),
            valid_lengths=torch.tensor([4], device=device, dtype=torch.int32),
            visible_kv_lens=torch.tensor([4], device=device, dtype=torch.int32),
            sparse_mask=torch.tensor([True], device=device),
        )

        adaptor.capture_dense_metadata(metadata)
        adaptor.apply(
            result,
            metadata,
            SimpleNamespace(req_pool_indices=torch.tensor([1], device=device)),
        )
        output = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=1,
            causal=False,
            softmax_scale=1 / math.sqrt(64),
            ver=3,
        )

        selected_pages = metadata.page_table[0, :4].long()
        selected_keys = key_cache[selected_pages, 0, 0].float()
        selected_values = value_cache[selected_pages, 0, 0].float()
        weights = torch.softmax(
            query[0, 0].float() @ selected_keys.T / math.sqrt(64), dim=-1
        )
        reference = weights @ selected_values

        torch.testing.assert_close(
            output[0, 0].float(), reference, atol=0.02, rtol=0.02
        )
        adaptor.restore_dense_metadata(metadata)
        torch.testing.assert_close(metadata.page_table, dense_page_table)


if __name__ == "__main__":
    unittest.main()
