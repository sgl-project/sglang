import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from utils import parametrize, precision

from sglang.test.test_utils import CustomTestCase

flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func


torch.manual_seed(1234)


def flash_attn_varlen_ref(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    is_causal,
    enable_gqa,
):
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch = len(cu_k) - 1

    # [T, H, D] -> [1, H, T, D]
    q, k, v = [x.unsqueeze(0).transpose(1, 2) for x in [q, k, v]]

    B, H, T, D = q.shape
    out = torch.empty(B, H, T, v.size(-1), dtype=q.dtype)
    for b in range(batch):
        start_q, end_q = cu_q[b], cu_q[b + 1]
        start_k, end_k = cu_k[b], cu_k[b + 1]

        out[:, :, start_q:end_q, :] = F.scaled_dot_product_attention(
            q[:, :, start_q:end_q, :],
            k[:, :, start_k:end_k, :],
            v[:, :, start_k:end_k, :],
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    # [1, H, T, D] -> [T, H, D]
    return out.transpose(1, 2).squeeze(0)


class TestFlashAttn(CustomTestCase):

    @parametrize(
        batch=[4],
        max_seqlen_q=[35, 96],
        max_seqlen_k=[35, 96],
        num_heads=[16],
        num_heads_kv=[16, 2],
        head_dim=[32, 48],  # test when D is not 32x
        head_dim_v=[32],
        is_causal=[True, False],
    )
    def test_flash_attn_varlen(
        self,
        batch,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        dtype = torch.bfloat16

        # random seqlens for k and kv
        seqlens_q = torch.randint(1, max_seqlen_q, (batch,), dtype=torch.int32)
        seqlens_k = torch.randint(1, max_seqlen_k, (batch,), dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_k = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

        sum_seqlen_q = seqlens_q.sum().item()
        sum_seqlen_k = seqlens_k.sum().item()
        q = torch.randn(sum_seqlen_q, num_heads, head_dim).to(dtype)
        k = torch.randn(sum_seqlen_k, num_heads_kv, head_dim).to(dtype)
        v = torch.randn(sum_seqlen_k, num_heads_kv, head_dim_v).to(dtype)

        out_ref = flash_attn_varlen_ref(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            is_causal=is_causal,
            enable_gqa=num_heads != num_heads_kv,
        )

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlens_q.max().item(),
            seqlens_k.max().item(),
            is_causal,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
