import unittest

import torch
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


def fix_query_key_value_ordering_reshape_cat(
    mixed_qkvz, mixed_ba, num_k_heads, num_v_heads, attn_tp_size, head_k_dim, head_v_dim
):
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        num_k_heads // attn_tp_size,
        (
            head_k_dim
            + head_k_dim
            + (head_v_dim + head_v_dim) * num_v_heads // num_k_heads
        ),
    )
    new_tensor_shape_ba = mixed_ba.size()[:-1] + (
        num_k_heads // attn_tp_size,
        2 * num_v_heads // num_k_heads,
    )

    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        (num_v_heads // num_k_heads * head_v_dim),
        (num_v_heads // num_k_heads * head_v_dim),
    ]
    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads,
    ]
    # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
    # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
    (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
    (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

    # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
    value = value.reshape(value.size(0), -1, head_v_dim)
    z = z.reshape(z.size(0), -1, head_v_dim)
    b = b.reshape(b.size(0), num_v_heads // attn_tp_size)
    a = a.reshape(a.size(0), num_v_heads // attn_tp_size)
    query, key, value = map(lambda x: x.reshape(x.shape[0], -1), (query, key, value))
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    return mixed_qkv, z, b, a


class TestQwen3(CustomTestCase):
    def test_fused_qkvzba_split_reshape_cat(self):
        mixed_qkvz = torch.rand(1024, 12288, dtype=torch.bfloat16)
        mixed_ba = torch.rand(1024, 64, dtype=torch.bfloat16)
        head_k_dim = 128
        head_v_dim = 128
        num_v_heads = 32
        num_k_heads = 16
        attn_tp_size = 1
        mixed_qkv_ref, z_ref, b_ref, a_ref = fix_query_key_value_ordering_reshape_cat(
            mixed_qkvz,
            mixed_ba,
            num_k_heads,
            num_v_heads,
            attn_tp_size,
            head_k_dim,
            head_v_dim,
        )
        num_heads_qk = num_k_heads // attn_tp_size
        num_heads_v = num_v_heads // attn_tp_size
        mixed_qkv, z, b, a = torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_cpu(
            mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_k_dim, head_v_dim
        )
        atol = rtol = precision[mixed_qkv.dtype]
        torch.testing.assert_close(mixed_qkv, mixed_qkv_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(b, b_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(a, a_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
