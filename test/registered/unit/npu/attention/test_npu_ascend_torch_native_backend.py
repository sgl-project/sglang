"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend.
"""

import math
import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")


class TestInit(unittest.TestCase):
    def test_construction(self):
        backend = AscendTorchNativeAttnBackend()
        self.assertIsNotNone(backend)


class TestSupportTriton(unittest.TestCase):
    def test_returns_false(self):
        backend = AscendTorchNativeAttnBackend()
        self.assertFalse(backend.support_triton())


class TestScaledDotProductAttentionWithSoftcapping(unittest.TestCase):
    def setUp(self):
        self.backend = AscendTorchNativeAttnBackend()

    def test_basic_attention(self):
        L, S, d = 4, 4, 8
        q = torch.randn(1, 2, L, d)
        k = torch.randn(1, 2, S, d)
        v = torch.randn(1, 2, S, d)
        out = self.backend.scaled_dot_product_attention_with_softcapping(q, k, v)

        scale = 1 / math.sqrt(d)
        ref = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_causal_mask(self):
        L, S, d = 4, 4, 8
        q = torch.randn(1, 2, L, d)
        k = torch.randn(1, 2, S, d)
        v = torch.randn(1, 2, S, d)
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, is_causal=True
        )

        scale = 1 / math.sqrt(d)
        attn = q @ k.transpose(-2, -1) * scale
        mask = torch.ones(L, S, dtype=torch.bool).tril()
        attn = attn.masked_fill(~mask, float("-inf"))
        ref = torch.softmax(attn, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_causal_mask_asserts_when_attn_mask_given(self):
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        v = torch.randn(1, 1, 2, 4)
        with self.assertRaises(AssertionError):
            self.backend.scaled_dot_product_attention_with_softcapping(
                q, k, v, attn_mask=torch.ones(2, 2), is_causal=True
            )

    def test_explicit_scale(self):
        d = 8
        q = torch.randn(1, 1, 2, d)
        k = torch.randn(1, 1, 2, d)
        v = torch.randn(1, 1, 2, d)
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, scale=0.5
        )
        ref = torch.softmax(q @ k.transpose(-2, -1) * 0.5, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_gqa(self):
        H_q, H_kv, L, S, d = 4, 2, 3, 3, 8
        q = torch.randn(1, H_q, L, d)
        k = torch.randn(1, H_kv, S, d)
        v = torch.randn(1, H_kv, S, d)
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, enable_gqa=True
        )
        self.assertEqual(out.shape, (1, H_q, L, d))

        k_exp = k.repeat_interleave(H_q // H_kv, -3)
        v_exp = v.repeat_interleave(H_q // H_kv, -3)
        scale = 1 / math.sqrt(d)
        ref = torch.softmax(q @ k_exp.transpose(-2, -1) * scale, dim=-1) @ v_exp
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_tanh_softcapping(self):
        d = 8
        q = torch.randn(1, 1, 2, d)
        k = torch.randn(1, 1, 2, d)
        v = torch.randn(1, 1, 2, d)
        cap = 10.0
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, logit_cap=cap
        )

        scale = 1 / math.sqrt(d)
        attn = q @ k.transpose(-2, -1) * scale
        attn = cap * torch.tanh(attn / cap)
        ref = torch.softmax(attn, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_no_softcapping_when_cap_zero(self):
        d = 8
        q = torch.randn(1, 1, 2, d)
        k = torch.randn(1, 1, 2, d)
        v = torch.randn(1, 1, 2, d)
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, logit_cap=0.0
        )
        scale = 1 / math.sqrt(d)
        ref = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_boolean_attn_mask(self):
        d = 8
        q = torch.randn(1, 1, 3, d)
        k = torch.randn(1, 1, 3, d)
        v = torch.randn(1, 1, 3, d)
        mask = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        )
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, attn_mask=mask
        )

        scale = 1 / math.sqrt(d)
        attn = q @ k.transpose(-2, -1) * scale
        attn = attn.masked_fill(~mask, float("-inf"))
        ref = torch.softmax(attn, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_additive_attn_mask(self):
        d = 8
        q = torch.randn(1, 1, 2, d)
        k = torch.randn(1, 1, 2, d)
        v = torch.randn(1, 1, 2, d)
        mask = torch.tensor([[0.0, -1e9], [0.0, 0.0]])
        out = self.backend.scaled_dot_product_attention_with_softcapping(
            q, k, v, attn_mask=mask
        )
        scale = 1 / math.sqrt(d)
        attn = q @ k.transpose(-2, -1) * scale + mask
        ref = torch.softmax(attn, dim=-1) @ v
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


class TestRunSdpaForwardExtend(unittest.TestCase):
    def setUp(self):
        self.backend = AscendTorchNativeAttnBackend()

    def _make_caches(self, num_tokens, num_heads, head_size, dtype=torch.float32):
        k_cache = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
        v_cache = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
        return k_cache, v_cache

    def _ref_extend(
        self,
        query,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_prefix_lens,
        extend_seq_lens,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        H, D = query.shape[1], query.shape[2]
        outputs = []
        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            ext_len = int(extend_seq_lens[seq_idx].item())
            pre_len = int(extend_prefix_lens[seq_idx].item())
            seq_len_kv = int(seq_lens[seq_idx].item())
            end_q = start_q + ext_len

            req_pool_idx = req_pool_indices[seq_idx]
            tokens = req_to_token[req_pool_idx, :seq_len_kv]

            # Build padded query [1, H, seq_len_kv, D] with extend tokens at pre_len
            q_pad = torch.zeros(1, H, seq_len_kv, D, dtype=query.dtype)
            q_pad[0, :, pre_len : pre_len + ext_len, :] = query[start_q:end_q].movedim(
                0, 1
            )
            # Key/value [1, H, seq_len_kv, D]
            k_u = k_cache[tokens].movedim(0, 1).unsqueeze(0)
            v_u = v_cache[tokens].movedim(0, 1).unsqueeze(0)

            if enable_gqa and k_u.size(-3) != q_pad.size(-3):
                rep = q_pad.size(-3) // k_u.size(-3)
                k_u = k_u.repeat_interleave(rep, -3)
                v_u = v_u.repeat_interleave(rep, -3)

            out = scaled_dot_product_attention(
                q_pad, k_u, v_u, scale=scaling, is_causal=causal
            )
            # [1, H, seq_len_kv, D] → slice [pre_len:pre_len+ext_len] → [ext, H, D]
            out = out[0, :, pre_len : pre_len + ext_len, :].movedim(1, 0)
            outputs.append(out)
            start_q = end_q
        return torch.cat(outputs, dim=0)

    def test_basic_extend(self):
        H, D = 2, 8
        num_seqs = 2
        extend_prefix_lens = torch.tensor([0, 0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([3, 4], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = int(extend_seq_lens.sum().item())
        max_ctx = int(seq_lens.max().item())

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = (
            torch.arange(max_ctx).unsqueeze(0).expand(num_seqs, -1).contiguous()
        )
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=False,
        )
        ref = self._ref_extend(
            query,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=False,
        )
        self.assertTrue(torch.allclose(result, ref, atol=1e-5))

    def test_causal_extend(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([5], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 5
        max_ctx = 5

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=True,
        )
        ref = self._ref_extend(
            query,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=True,
        )
        self.assertTrue(torch.allclose(result, ref, atol=1e-5))

    def test_extend_with_prefix_lens(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([2], dtype=torch.int32)
        extend_seq_lens = torch.tensor([3], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 3
        max_ctx = 5

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=True,
        )
        ref = self._ref_extend(
            query,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=True,
        )
        self.assertTrue(torch.allclose(result, ref, atol=1e-5))

    def test_extend_with_gqa(self):
        H_q, H_kv, D = 4, 2, 8
        extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([4], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 4
        max_ctx = 4

        query = torch.randn(total_q, H_q, D)
        output = torch.empty(total_q, H_q, D)
        k_cache, v_cache = self._make_caches(max_ctx, H_kv, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=True,
            causal=False,
        )
        self.assertEqual(result.shape, (total_q, H_q, D))

    def test_extend_with_logit_cap(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([4], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 4
        max_ctx = 4
        cap = 10.0

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=True,
            logit_cap=cap,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_extend_with_cross_attention(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([2], dtype=torch.int32)
        extend_seq_lens = torch.tensor([3], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 3
        max_ctx = 5

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        encoder_lens = torch.tensor([4], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            encoder_lens=encoder_lens,
            is_cross_attention=True,
            enable_gqa=False,
            causal=False,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_extend_with_encoder_self_attention(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([3], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 3
        max_ctx = 6

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        encoder_lens = torch.tensor([3], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            encoder_lens=encoder_lens,
            is_cross_attention=False,
            enable_gqa=False,
            causal=False,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_extend_with_sw(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([4], dtype=torch.int32)
        extend_seq_lens = torch.tensor([3], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 3
        max_ctx = 7

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=False,
            sliding_window_size=3,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_extend_with_full_to_swa_mapping(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([4], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 4
        max_ctx = 4

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        # Identity mapping: full index == SWA index
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        full_to_swa = torch.arange(max_ctx)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=False,
            full_to_swa_mapping=full_to_swa,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_extend_dtype_casting(self):
        H, D = 2, 8
        extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
        extend_seq_lens = torch.tensor([3], dtype=torch.int32)
        seq_lens = extend_prefix_lens + extend_seq_lens
        total_q = 3
        max_ctx = 3

        query = torch.randn(total_q, H, D, dtype=torch.float32)
        output = torch.empty_like(query)
        k_cache = torch.randn(max_ctx, H, D, dtype=torch.float64)
        v_cache = torch.randn(max_ctx, H, D, dtype=torch.float64)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_extend(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_prefix_lens,
            extend_seq_lens,
            enable_gqa=False,
            causal=False,
        )
        self.assertEqual(result.dtype, torch.float32)


class TestRunSdpaForwardDecode(unittest.TestCase):
    def setUp(self):
        self.backend = AscendTorchNativeAttnBackend()

    def _make_caches(self, num_tokens, num_heads, head_size, dtype=torch.float32):
        k_cache = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
        v_cache = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
        return k_cache, v_cache

    def test_basic_decode(self):
        H, D = 2, 8
        num_seqs = 2
        seq_lens = torch.tensor([3, 5], dtype=torch.int32)
        total_q = num_seqs
        max_ctx = int(seq_lens.max().item())

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = (
            torch.arange(max_ctx).unsqueeze(0).expand(num_seqs, -1).contiguous()
        )
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=False,
            causal=False,
        )

        # Manual reference
        outputs = []
        for i in range(num_seqs):
            kv_len = int(seq_lens[i].item())
            tokens = req_to_token[i, :kv_len]
            # query[i]: [H, D] → [1, H, 1, D]
            q_req = query[i : i + 1].movedim(0, 1).unsqueeze(0)
            # k/v: [kv, H, D] → [1, H, kv, D]
            k_req = k_cache[tokens].movedim(0, 1).unsqueeze(0)
            v_req = v_cache[tokens].movedim(0, 1).unsqueeze(0)
            out = scaled_dot_product_attention(q_req, k_req, v_req)
            # [1, H, 1, D] → [1, H, D]
            out = out.squeeze(0).movedim(1, 0)
            outputs.append(out)
        ref = torch.cat(outputs, dim=0)
        self.assertTrue(torch.allclose(result, ref, atol=1e-5))

    def test_decode_causal(self):
        H, D = 2, 8
        seq_lens = torch.tensor([5], dtype=torch.int32)
        total_q = 1
        max_ctx = 5

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=False,
            causal=True,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_decode_with_gqa(self):
        H_q, H_kv, D = 4, 2, 8
        seq_lens = torch.tensor([4], dtype=torch.int32)
        total_q = 1
        max_ctx = 4

        query = torch.randn(total_q, H_q, D)
        output = torch.empty(total_q, H_q, D)
        k_cache, v_cache = self._make_caches(max_ctx, H_kv, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=True,
            causal=False,
        )
        self.assertEqual(result.shape, (total_q, H_q, D))

    def test_decode_with_logit_cap(self):
        H, D = 2, 8
        seq_lens = torch.tensor([4], dtype=torch.int32)
        total_q = 1
        max_ctx = 4
        cap = 10.0

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=False,
            causal=False,
            logit_cap=cap,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_decode_with_encoder_lens_cross(self):
        H, D = 2, 8
        seq_lens = torch.tensor([5], dtype=torch.int32)
        total_q = 1
        max_ctx = 5

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        encoder_lens = torch.tensor([3], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            encoder_lens=encoder_lens,
            is_cross_attention=True,
            enable_gqa=False,
            causal=False,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_decode_with_encoder_lens_self(self):
        H, D = 2, 8
        seq_lens = torch.tensor([6], dtype=torch.int32)
        total_q = 1
        max_ctx = 6

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        encoder_lens = torch.tensor([3], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            encoder_lens=encoder_lens,
            is_cross_attention=False,
            enable_gqa=False,
            causal=False,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_decode_with_sw(self):
        H, D = 2, 8
        seq_lens = torch.tensor([10], dtype=torch.int32)
        total_q = 1
        max_ctx = 10

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=False,
            causal=False,
            sliding_window_size=3,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_decode_with_full_to_swa_mapping(self):
        H, D = 2, 8
        seq_lens = torch.tensor([4], dtype=torch.int32)
        total_q = 1
        max_ctx = 4

        query = torch.randn(total_q, H, D)
        output = torch.empty_like(query)
        k_cache, v_cache = self._make_caches(max_ctx, H, D)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        full_to_swa = torch.arange(max_ctx)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=False,
            causal=False,
            full_to_swa_mapping=full_to_swa,
        )
        self.assertEqual(result.shape, (total_q, H, D))

    def test_decode_dtype_casting(self):
        H, D = 2, 8
        seq_lens = torch.tensor([3], dtype=torch.int32)
        total_q = 1
        max_ctx = 3

        query = torch.randn(total_q, H, D, dtype=torch.float32)
        output = torch.empty_like(query)
        k_cache = torch.randn(max_ctx, H, D, dtype=torch.float64)
        v_cache = torch.randn(max_ctx, H, D, dtype=torch.float64)
        req_to_token = torch.arange(max_ctx).unsqueeze(0)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)

        result = self.backend.run_sdpa_forward_decode(
            query,
            output,
            k_cache,
            v_cache,
            req_to_token,
            req_pool_indices,
            seq_lens,
            enable_gqa=False,
            causal=False,
        )
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
