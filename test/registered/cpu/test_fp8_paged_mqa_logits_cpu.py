import unittest
from typing import Any

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from utils import precision

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

BLOCK_SIZE = 64
HEAD_DIM = 128
HEAD_DIM_WITH_SCALE_BYTES = 132

FP8_DTYPE = torch.float8_e4m3fn


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    for i in range(batch_size):
        q = q_fp8[i, 0]
        q = q.to(torch.float32)
        q_scale = weight[i]
        seq_len = int(seq_lens[i].item())
        assert seq_len <= max_seq_len
        num_pages = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_pages * block_size
        pages = page_table[i, :num_pages]
        kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
        kvcache = kvcache_fp8[pages]
        SCALE_OFFSET = block_size * head_dim
        kvcache_value = kvcache[..., :SCALE_OFFSET].view(dtype=FP8_DTYPE)
        kvcache_scale = kvcache[..., SCALE_OFFSET:].view(dtype=torch.float32)
        kvcache_value = kvcache_value.to(torch.float32)
        kvcache_scale = kvcache_scale.contiguous()
        kvcache_value = kvcache_value.view(padded_seq_len, head_dim)
        kvcache_scale = kvcache_scale.view(padded_seq_len)
        score = F.linear(kvcache_value, q)
        score = F.relu(score)
        score *= q_scale[None, :]
        score = score.sum(dim=1)
        score *= kvcache_scale
        logits[i, :seq_len] = score[:seq_len]

    return logits


class TestFp8PagedMqaLogitsCPU(CustomTestCase):
    def _make_inputs(
        self,
        *,
        batch_size: int = 3,
        num_heads: int = 4,
        max_seq_len: int = 192,
        num_blocks: int = 8,
        index_dtype: torch.dtype = torch.int32,
        weight_dtype: torch.dtype = torch.float32,
        q_dtype: torch.dtype = torch.bfloat16,
    ):
        torch.manual_seed(2)

        q = (torch.randn(batch_size, 1, num_heads, HEAD_DIM) * 0.25).to(q_dtype)
        q_fp8 = q.to(torch.float8_e4m3fn).contiguous()

        k = (torch.randn(num_blocks, BLOCK_SIZE, HEAD_DIM) * 0.25).to(q_dtype)
        k_fp8 = k.to(torch.float8_e4m3fn).contiguous()
        k_bytes = k_fp8.view(num_blocks, BLOCK_SIZE * HEAD_DIM).view(dtype=torch.uint8)

        scales = torch.rand(num_blocks, BLOCK_SIZE, dtype=torch.float32) * 0.5 + 0.75
        scale_bytes = (
            scales.contiguous().view(num_blocks, BLOCK_SIZE).view(dtype=torch.uint8)
        )

        kvcache = torch.cat([k_bytes, scale_bytes], dim=1).contiguous()
        kvcache = kvcache.view(num_blocks, BLOCK_SIZE, 1, HEAD_DIM_WITH_SCALE_BYTES)

        weight = (
            torch.randn(batch_size, num_heads, dtype=torch.float32)
            .to(weight_dtype)
            .contiguous()
        )
        seq_lens = torch.tensor([0, 65, max_seq_len - 1], dtype=index_dtype)

        pages_per_batch = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        page_table = torch.empty(batch_size, pages_per_batch, dtype=index_dtype)
        page_table[0] = torch.tensor([0, 1, 2], dtype=index_dtype)
        page_table[1] = torch.tensor([3, 4, 5], dtype=index_dtype)
        page_table[2] = torch.tensor([2, 6, 7], dtype=index_dtype)

        return q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len

    def _assert_matches_reference(
        self,
        index_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        q_dtype: torch.dtype = torch.bfloat16,
    ):
        q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len = self._make_inputs(
            index_dtype=index_dtype,
            weight_dtype=weight_dtype,
            q_dtype=q_dtype,
        )

        actual = torch.ops.sgl_kernel.fp8_paged_mqa_logits_cpu(
            q_fp8,
            kvcache,
            weight,
            seq_lens,
            page_table,
            max_seq_len,
            False,
        )
        expected = fp8_paged_mqa_logits_torch(
            q_fp8,
            kvcache,
            weight,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            False,
        )

        self.assertEqual(actual.shape, (seq_lens.numel(), max_seq_len))
        self.assertEqual(actual.dtype, torch.float32)
        atol = rtol = precision[q_dtype]
        for batch_idx, seq_len in enumerate(seq_lens.tolist()):
            if seq_len == 0:
                continue
            torch.testing.assert_close(
                actual[batch_idx, :seq_len],
                expected[batch_idx, :seq_len],
                atol=atol,
                rtol=rtol,
            )

    def test_matches_torch_reference(self):
        self._assert_matches_reference(
            index_dtype=torch.int32,
            weight_dtype=torch.float32,
            q_dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    unittest.main()
