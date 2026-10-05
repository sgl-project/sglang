import math
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        from aiter.ops.triton.attention.unified_attention import unified_attention
        from aiter.ops.triton.utils.types import e4m3_dtype

        from sglang.kernels.ops.attention.unified_attention_3d_mtp import (
            unified_attention_3d_mtp_func,
        )
    except Exception:
        _RUNNABLE = False


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 with aiter")
class TestUnifiedAttention3dMtp(CustomTestCase):
    def test_matches_aiter(self):
        # TP2 shard of Qwen3.5-397B-A17B: 32 q / 2 kv heads split over two ranks.
        self._check_matches_aiter(num_query_heads=16, num_kv_heads=1)

    def test_matches_aiter_multi_kv_head(self):
        # TP1, the same model unsharded: still 16:1, but two kv heads.
        self._check_matches_aiter(num_query_heads=32, num_kv_heads=2)

    def _check_matches_aiter(self, num_query_heads: int, num_kv_heads: int):
        torch.manual_seed(0)
        device = "cuda"
        query_lens = [4, 2]
        kv_lens_list = [1024, 769]
        head_size = 256
        block_size = 16
        max_kv_len = max(kv_lens_list)
        max_blocks_per_seq = math.ceil(max_kv_len / block_size)
        num_blocks = len(query_lens) * max_blocks_per_seq

        query = torch.randn(
            sum(query_lens),
            num_query_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        ).to(e4m3_dtype)
        value = torch.randn_like(key, dtype=torch.bfloat16).to(e4m3_dtype)
        cu_seqlens_q = torch.tensor(
            [0, query_lens[0], sum(query_lens)],
            device=device,
            dtype=torch.int32,
        )
        seqused_k = torch.tensor(kv_lens_list, device=device, dtype=torch.int32)
        block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
            len(query_lens), max_blocks_per_seq
        )
        k_descale = torch.ones(1, device=device, dtype=torch.float32)
        v_descale = torch.ones(1, device=device, dtype=torch.float32)
        expected = torch.empty_like(query)
        actual = torch.empty_like(query)

        unified_attention(
            q=query,
            k=key,
            v=value,
            out=expected,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max(query_lens),
            seqused_k=seqused_k,
            max_seqlen_k=max_kv_len,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        unified_attention_3d_mtp_func(
            q=query,
            k=key,
            v=value,
            out=actual,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max_kv_len,
            softmax_scale=head_size**-0.5,
            block_table=block_table,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
