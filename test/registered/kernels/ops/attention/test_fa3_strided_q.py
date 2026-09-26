"""FA3 reads a strided (fused-QKV) q without a copy.

``forward_extend`` reshapes ``qkv[:, :q_size]`` with ``reshape`` instead of
``contiguous().view``. FA3 takes ``q_row_stride`` / ``q_head_stride`` as kernel
arguments, so it must read only the q columns of the fused buffer, bit for bit
like the copied version.
"""

import unittest

import torch

from sglang.kernels.ops.attention.flash_attention import (
    flash_attn_with_kvcache,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# FA3 (ver=3) needs Hopper or newer.
_no_fa3 = not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9

HEAD_DIM = 128


def _paged_kv(total_tokens, num_kv_heads, seqlens, device, dtype):
    """A page_size=1 KV cache plus the index metadata for ``seqlens``."""
    k_cache, v_cache = (
        torch.randn(total_tokens, 1, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
        for _ in range(2)
    )
    max_seqlen = max(seqlens)
    page_table = torch.zeros(len(seqlens), max_seqlen, device=device, dtype=torch.int32)
    start = 0
    for i, seqlen in enumerate(seqlens):
        page_table[i, :seqlen] = torch.arange(
            start, start + seqlen, device=device, dtype=torch.int32
        )
        start += seqlen
    cache_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(len(seqlens) + 1, device=device, dtype=torch.int32)
    cu_seqlens_q[1:] = cache_seqlens.cumsum(0)
    return k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen


class TestFa3StridedQ(CustomTestCase):
    @unittest.skipIf(_no_fa3, "FA3 requires compute capability sm90 (Hopper) or newer")
    def test_fused_qkv_slice_matches_copy_bitwise(self):
        """A strided q slice gives bit-identical output to the copied q.

        The k/v columns of the fused buffer are poisoned with NaN: if FA3
        ignored ``q_row_stride`` and read outside the q columns, the output
        would be NaN rather than merely different.
        """
        torch.manual_seed(0)
        device, dtype = "cuda", torch.bfloat16

        for num_q_heads, num_kv_heads, seqlens in (
            (16, 8, [17, 33]),  # GQA, ragged
            (14, 2, [1, 512]),  # Qwen3-style GQA, wide length spread
            (8, 8, [64]),  # MHA, single sequence
        ):
            with self.subTest(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads):
                total_tokens = sum(seqlens)
                q_size = num_q_heads * HEAD_DIM
                kv_size = num_kv_heads * HEAD_DIM

                # Fused QKV projection output; q is the leading column slice.
                qkv = torch.randn(
                    total_tokens, q_size + 2 * kv_size, device=device, dtype=dtype
                )
                qkv[:, q_size:] = float("nan")
                q = qkv[:, :q_size]

                q_strided = q.reshape(-1, num_q_heads, HEAD_DIM)
                # The whole point of the patch: the reshape copies nothing.
                self.assertEqual(q_strided.data_ptr(), q.data_ptr())
                self.assertFalse(q_strided.is_contiguous())
                self.assertEqual(
                    q_strided.stride(), (q_size + 2 * kv_size, HEAD_DIM, 1)
                )

                (
                    k_cache,
                    v_cache,
                    page_table,
                    cache_seqlens,
                    cu_seqlens_q,
                    max_seqlen,
                ) = _paged_kv(total_tokens, num_kv_heads, seqlens, device, dtype)

                def run(q_heads):
                    return flash_attn_with_kvcache(
                        q=q_heads,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen,
                        softmax_scale=HEAD_DIM**-0.5,
                        causal=True,
                        ver=3,
                    )

                out_strided = run(q_strided)
                out_copy = run(q.contiguous().view(-1, num_q_heads, HEAD_DIM))

                self.assertFalse(torch.isnan(out_strided).any())
                self.assertTrue(torch.equal(out_strided, out_copy))

    def test_reshape_matches_contiguous_view_for_3d_q(self):
        """``reshape`` returns what ``contiguous().view`` returned for a 3-D q.

        The call site is 2-D today, but the reshape has to keep reflowing the
        leading dims the way the copy did rather than splitting the last one,
        and it has to keep accepting every layout the copy accepted.
        """
        num_heads, head_dim = 4, 8
        for q in (
            torch.randn(6, 2, 32),  # contiguous
            torch.randn(6, 2, 64)[..., :32],  # strided rows
            torch.randn(6, 2, 64)[..., ::2],  # strided rows and last dim
        ):
            with self.subTest(stride=q.stride()):
                got = q.reshape(-1, num_heads, head_dim)
                self.assertEqual(got.shape, (12, num_heads, head_dim))
                # Mergeable leading dims: still a view, still zero copy.
                self.assertEqual(got.data_ptr(), q.data_ptr())
                self.assertTrue(
                    torch.equal(got, q.contiguous().view(-1, num_heads, head_dim))
                )

        # Leading dims that cannot be merged are the one layout ``view`` would
        # reject. ``reshape`` copies instead, exactly as the old expression did.
        q = torch.randn(6, 4, 32)[:, :2]
        with self.assertRaises(RuntimeError):
            q.view(-1, num_heads, head_dim)
        got = q.reshape(-1, num_heads, head_dim)
        self.assertNotEqual(got.data_ptr(), q.data_ptr())
        self.assertTrue(torch.equal(got, q.contiguous().view(-1, num_heads, head_dim)))


if __name__ == "__main__":
    unittest.main()
