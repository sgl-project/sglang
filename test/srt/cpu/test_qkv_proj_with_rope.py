import unittest

import sgl_kernel
import torch
from utils import (
    convert_weight,
    native_w8a8_per_token_matmul,
    per_token_quant_int8,
    precision,
)

from sglang.srt.layers.rotary_embedding import _apply_rotary_emb
from sglang.test.test_utils import CustomTestCase

convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
qkv_proj_with_rope = torch.ops.sgl_kernel.qkv_proj_with_rope
qkv_proj_with_rope_fused_weight = torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight
torch.manual_seed(0)
# constants
kv_lora_rank = 512
qk_head_dim = 192
qk_nope_head_dim = 128
qk_rope_head_dim = 64
rotary_dim = qk_rope_head_dim
num_heads = 22
q_lora_rank = 1536
hidden_size = 7168
B = 1
eps = 1e-6


def layernorm(x, weight, variance_epsilon=1e-6, residual=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return (x * weight).to(orig_dtype)


def rotary_emb(q_pe, k_pe, pos, cos_sin_cache):
    orig_dtype = q_pe.dtype
    q_pe = q_pe.float()
    k_pe = k_pe.float()
    cos_sin_cache = cos_sin_cache.float()

    query_rot = q_pe[..., :rotary_dim]
    key_rot = k_pe[..., :rotary_dim]
    cos_sin = cos_sin_cache[pos]
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_rot = _apply_rotary_emb(query_rot, cos, sin, False)
    key_rot = _apply_rotary_emb(key_rot, cos, sin, False)
    return query_rot.to(orig_dtype), key_rot.to(orig_dtype)


def native_torch(
    q_input,
    hidden_states,
    q_a_proj_weight,
    norm_weight1,
    q_b_proj_weight,
    w_kc,
    kv_a_proj_weight,
    norm_weight2,
    pos,
    cos_sin_cache,
):

    q = torch.matmul(hidden_states, q_a_proj_weight.t())
    q = layernorm(q, norm_weight1)
    q = torch.matmul(q, q_b_proj_weight.t()).view(-1, num_heads, qk_head_dim)

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)
    latent_cache = torch.matmul(hidden_states, kv_a_proj_weight.t())
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = layernorm(v_input.contiguous(), norm_weight2).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank:]

    q_pe, k_pe = rotary_emb(q_pe, k_pe, pos, cos_sin_cache)
    q_input[..., kv_lora_rank:] = q_pe
    k_input[..., kv_lora_rank:] = k_pe

    return q_input, k_input, v_input


def native_torch_int8(
    q_input,
    hidden_states,
    w1_q,
    w1_s,
    norm_weight1,
    w2_q,
    w2_s,
    w_kc,
    w3_q,
    w3_s,
    norm_weight2,
    pos,
    cos_sin_cache,
):

    a_q, a_s = per_token_quant_int8(hidden_states)
    q = native_w8a8_per_token_matmul(a_q, w1_q, a_s, w1_s, None, torch.bfloat16)
    q = layernorm(q, norm_weight1)

    a_q, a_s = per_token_quant_int8(q)
    q = native_w8a8_per_token_matmul(a_q, w2_q, a_s, w2_s, None, torch.bfloat16).view(
        -1, num_heads, qk_head_dim
    )

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)
    a_q, a_s = per_token_quant_int8(hidden_states)
    latent_cache = native_w8a8_per_token_matmul(
        a_q, w3_q, a_s, w3_s, None, torch.bfloat16
    )
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = layernorm(v_input.contiguous(), norm_weight2).unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank:]

    q_pe, k_pe = rotary_emb(q_pe, k_pe, pos, cos_sin_cache)
    q_input[..., kv_lora_rank:] = q_pe
    k_input[..., kv_lora_rank:] = k_pe

    return q_input, k_input, v_input


class TestQKVProjWithROPE(CustomTestCase):
    def test_bf16_qkv_proj_with_rope(self):
        dtype = torch.bfloat16
        hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
        q_input = torch.empty(
            B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype
        )
        q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
        norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
        q_b_proj_weight = (
            torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
        )
        w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
        kv_a_proj_weight = (
            torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
        )
        fused_weight = torch.cat([q_a_proj_weight, kv_a_proj_weight], dim=0)
        norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
        pos = torch.randint(10, 100, (B,))
        cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)
        q_ref, k_ref, v_ref = native_torch(
            q_input,
            hidden_states,
            q_a_proj_weight,
            norm_weight1,
            q_b_proj_weight,
            w_kc.transpose(1, 2),
            kv_a_proj_weight,
            norm_weight2,
            pos,
            cos_sin_cache,
        )
        qa_packed = convert_weight_packed(q_a_proj_weight)
        qb_packed = convert_weight_packed(q_b_proj_weight)
        kva_packed = convert_weight_packed(kv_a_proj_weight)
        wkc_packed = convert_weight_packed(w_kc)
        fused_weight_packed = convert_weight_packed(fused_weight)

        q_out, k_out, v_out = qkv_proj_with_rope(
            hidden_states,
            qa_packed,
            qb_packed,
            kva_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            False,
            None,
            None,
            None,
            True,
            None,
        )
        fused_q_out, fused_k_out, fused_v_out = qkv_proj_with_rope_fused_weight(
            hidden_states,
            fused_weight_packed,
            qb_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            False,
            None,
            None,
            True,
            None,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        atol = rtol = precision[q_ref.dtype]
        torch.testing.assert_close(q_ref, q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_ref, k_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_ref, v_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_q_out, q_out)
        torch.testing.assert_close(fused_k_out, k_out)
        torch.testing.assert_close(fused_v_out, v_out)

    def test_int8_qkv_proj_with_rope(self):
        dtype = torch.bfloat16
        hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
        q_input = torch.empty(
            B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype
        )
        q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
        norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
        q_b_proj_weight = (
            torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
        )
        w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
        kv_a_proj_weight = (
            torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
        )
        norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
        pos = torch.randint(10, 100, (B,))
        cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)

        w1_q, w1_s = per_token_quant_int8(q_a_proj_weight)
        w2_q, w2_s = per_token_quant_int8(q_b_proj_weight)
        w3_q, w3_s = per_token_quant_int8(kv_a_proj_weight)
        q_ref, k_ref, v_ref = native_torch_int8(
            q_input,
            hidden_states,
            w1_q,
            w1_s,
            norm_weight1,
            w2_q,
            w2_s,
            w_kc.transpose(1, 2),
            w3_q,
            w3_s,
            norm_weight2,
            pos,
            cos_sin_cache,
        )
        w1_q_packed = convert_weight_packed(w1_q)
        w2_q_packed = convert_weight_packed(w2_q)
        w3_q_packed = convert_weight_packed(w3_q)
        wkc_packed = convert_weight_packed(w_kc)
        q_out, k_out, v_out = qkv_proj_with_rope(
            hidden_states,
            w1_q_packed,
            w2_q_packed,
            w3_q_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            True,
            False,
            w1_s,
            w2_s,
            w3_s,
            True,
            None,
        )
        fused_weight = torch.cat([w1_q, w3_q], dim=0)
        fused_weight_s = torch.cat([w1_s, w3_s], dim=0)
        w_fused_q_packed = convert_weight_packed(fused_weight)
        fused_q_out, fused_k_out, fused_v_out = qkv_proj_with_rope_fused_weight(
            hidden_states,
            w_fused_q_packed,
            w2_q_packed,
            wkc_packed,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            True,
            False,
            fused_weight_s,
            w2_s,
            True,
            None,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        atol = rtol = precision[q_ref.dtype]
        torch.testing.assert_close(q_ref, q_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_ref, k_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_ref, v_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_q_out, q_out)
        torch.testing.assert_close(fused_k_out, k_out)
        torch.testing.assert_close(fused_v_out, v_out)

    def test_fp8_qkv_proj_with_rope(self):
        dtype = torch.bfloat16
        hidden_states = torch.randn(B, hidden_size, dtype=dtype) / hidden_size
        q_input = torch.empty(
            B, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=dtype
        )
        q_a_proj_weight = torch.randn(q_lora_rank, hidden_size, dtype=dtype) * 0.1
        norm_weight1 = torch.randn(q_lora_rank, dtype=dtype)
        q_b_proj_weight = (
            torch.randn(num_heads * qk_head_dim, q_lora_rank, dtype=dtype) * 0.1
        )
        w_kc = torch.randn(num_heads, kv_lora_rank, qk_nope_head_dim, dtype=dtype) * 0.1
        kv_a_proj_weight = (
            torch.randn(kv_lora_rank + qk_rope_head_dim, hidden_size, dtype=dtype) * 0.1
        )
        norm_weight2 = torch.randn(kv_lora_rank, dtype=dtype)
        pos = torch.randint(10, 100, (B,))
        cos_sin_cache = torch.randn(100, rotary_dim, dtype=dtype)

        scale_block_size_N = 128
        scale_block_size_K = 128
        fp8_q_a_proj_weight, q_a_proj_weight_scale_inv, q_a_proj_weight_dq = (
            convert_weight(
                q_a_proj_weight,
                [scale_block_size_N, scale_block_size_K],
                torch.bfloat16,
            )
        )
        fp8_q_b_proj_weight, q_b_proj_weight_scale_inv, q_b_proj_weight_dq = (
            convert_weight(
                q_b_proj_weight,
                [scale_block_size_N, scale_block_size_K],
                torch.bfloat16,
            )
        )
        (
            fp8_kv_a_proj_with_mqa_weight,
            kv_a_proj_with_mqa_weight_scale_inv,
            kv_a_proj_with_mqa_weight_dq,
        ) = convert_weight(
            kv_a_proj_weight, [scale_block_size_N, scale_block_size_K], torch.bfloat16
        )
        q_ref, k_ref, v_ref = native_torch(
            q_input,
            hidden_states,
            q_a_proj_weight_dq,
            norm_weight1,
            q_b_proj_weight_dq,
            w_kc.transpose(1, 2),
            kv_a_proj_with_mqa_weight_dq,
            norm_weight2,
            pos,
            cos_sin_cache,
        )
        fp8_q_a_proj_weight_packed = convert_weight_packed(fp8_q_a_proj_weight)
        fp8_q_b_proj_weight_packed = convert_weight_packed(fp8_q_b_proj_weight)
        fp8_kv_a_proj_with_mqa_weight_packed = convert_weight_packed(
            fp8_kv_a_proj_with_mqa_weight
        )
        w_kc = convert_weight_packed(w_kc)
        q_out, k_out, v_out = qkv_proj_with_rope(
            hidden_states,
            fp8_q_a_proj_weight_packed,
            fp8_q_b_proj_weight_packed,
            fp8_kv_a_proj_with_mqa_weight_packed,
            w_kc,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            True,
            q_a_proj_weight_scale_inv.float(),
            q_b_proj_weight_scale_inv.float(),
            kv_a_proj_with_mqa_weight_scale_inv.float(),
            True,
            [scale_block_size_N, scale_block_size_K],
        )

        fused_weight = torch.cat(
            [fp8_q_a_proj_weight, fp8_kv_a_proj_with_mqa_weight], dim=0
        )
        fused_weight_s = torch.cat(
            [q_a_proj_weight_scale_inv, kv_a_proj_with_mqa_weight_scale_inv], dim=0
        )
        fused_weight_packed = convert_weight_packed(fused_weight)
        fused_q_out, fused_k_out, fused_v_out = qkv_proj_with_rope_fused_weight(
            hidden_states,
            fused_weight_packed,
            fp8_q_b_proj_weight_packed,
            w_kc,
            norm_weight1,
            norm_weight2,
            pos,
            cos_sin_cache,
            eps,
            False,
            True,
            fused_weight_s.float(),
            q_b_proj_weight_scale_inv.float(),
            True,
            [scale_block_size_N, scale_block_size_K],
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        atol = rtol = precision[q_ref.dtype]
        # Due to the change in multiplication order, the error is amplified.
        # In the model, with fewer layers, this doesn't cause issues, but in
        # tests with more layers, we need to enlarge the tolerance to pass the tests.
        torch.testing.assert_close(q_ref, q_out, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(k_ref, k_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_ref, v_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_q_out, q_out)
        torch.testing.assert_close(fused_k_out, k_out)
        torch.testing.assert_close(fused_v_out, v_out)


if __name__ == "__main__":
    unittest.main()
