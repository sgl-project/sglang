"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend.
"""

import math
import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
    reference_swa_decode_mask,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")


class TestReferenceSWADecodeMask(CustomTestCase):
    def test_verify_ancestry_uses_logical_positions_and_avoids_evicted_kv(self):
        """A branch's storage index is not its position; freed KV may be NaN."""
        torch.manual_seed(73)
        prefix, reference, window, nodes = 20, 7, 4, 6
        parents = [-1, 0, 1, 2, 0, 4]
        depth, ancestry = [], torch.ones(nodes, prefix + nodes, dtype=torch.bool)
        ancestry[:, prefix:] = False
        for node, parent in enumerate(parents):
            depth.append(0 if parent < 0 else depth[parent] + 1)
            current = node
            while current >= 0:
                ancestry[node, prefix + current] = True
                current = parents[current]
        q = torch.randn(nodes, 2, 4)
        raw_k, raw_v = (
            torch.randn(prefix + nodes, 2, 4),
            torch.randn(prefix + nodes, 2, 4),
        )
        # A permutation makes slot identity independent of the token position.
        slots = torch.randperm(prefix + nodes)
        cache_k, cache_v = torch.empty_like(raw_k), torch.empty_like(raw_v)
        cache_k[slots], cache_v[slots] = raw_k, raw_v
        cache_k[slots[reference : prefix - window + 1]] = float("nan")
        cache_v[slots[reference : prefix - window + 1]] = float("nan")
        positions = torch.tensor([prefix + d for d in depth])
        mask = torch.zeros_like(ancestry)
        for node, query_position in enumerate(positions.tolist()):
            for key in range(prefix + nodes):
                key_position = key if key < prefix else prefix + depth[key - prefix]
                mask[node, key] = bool(ancestry[node, key]) and (
                    key < reference or key_position > query_position - window
                )
        expected = scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            raw_k.transpose(0, 1)[None],
            raw_v.transpose(0, 1)[None],
            attn_mask=mask,
        )[0].transpose(0, 1)
        output = torch.empty_like(q)
        AscendTorchNativeAttnBackend().run_sdpa_forward_reference_verify(
            q,
            output,
            cache_k,
            cache_v,
            slots[None],
            torch.tensor([0]),
            torch.tensor([prefix]),
            torch.tensor([reference]),
            ancestry.flatten(),
            positions,
            nodes,
            window,
        )
        torch.testing.assert_close(output, expected)

    def test_non_aligned_references_suffix_and_padding(self):
        """References survive past W; holes and graph padding remain invisible."""
        lengths = torch.tensor([0, 1, 11, 139, 140, 257, 386, 900])
        references = torch.tensor([0, 1, 11, 11, 11, 257, 257, 257])
        mask = reference_swa_decode_mask(lengths, references, torch.arange(1024), 128)
        for row, (length, reference) in enumerate(
            zip(lengths.tolist(), references.tolist())
        ):
            visible = set(range(min(reference, length)))
            visible.update(range(max(reference, length - 128), length))
            expected = torch.tensor([pos not in visible for pos in range(1024)])
            self.assertTrue(torch.equal(mask[row, 0], expected))


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


class TestReferenceSlidingWindowDecode(CustomTestCase):
    def test_reprefill_generated_tokens_keep_the_original_reference_boundary(self):
        generator = torch.Generator().manual_seed(17)
        backend = AscendTorchNativeAttnBackend()
        prompt_len, total_len = 11, 276
        key = torch.randn(total_len, 2, 8, generator=generator)
        value = torch.randn(total_len, 2, 8, generator=generator)
        for prefix_len in (0, 149):
            with self.subTest(prefix_len=prefix_len):
                query = torch.randn(total_len - prefix_len, 2, 8, generator=generator)
                actual = torch.empty_like(query)
                backend.run_sdpa_forward_extend(
                    query,
                    actual,
                    key,
                    value,
                    torch.arange(total_len).unsqueeze(0),
                    torch.tensor([0]),
                    torch.tensor([total_len]),
                    torch.tensor([prefix_len]),
                    torch.tensor([total_len - prefix_len]),
                    causal=True,
                    sliding_window_size=128,
                    reference_lens=torch.tensor([prompt_len]),
                )
                positions = torch.arange(total_len)
                for offset, q in enumerate(query):
                    position = prefix_len + offset
                    visible = (positions <= position) & (
                        (positions < prompt_len) | (position - positions < 128)
                    )
                    scores = q.unsqueeze(1) @ key.transpose(0, 1).transpose(-2, -1)
                    scores = scores / math.sqrt(8)
                    expected = (
                        scores.masked_fill(~visible, float("-inf")).softmax(-1)
                        @ value.transpose(0, 1)
                    ).squeeze(1)
                    torch.testing.assert_close(actual[offset], expected)

    def test_reference_prefix_and_generated_window_against_full_mask(self):
        generator = torch.Generator().manual_seed(13)
        backend = AscendTorchNativeAttnBackend()
        # Request pool slots deliberately differ from batch rows; scattered
        # cache locations must be gathered in each request's logical order.
        req_indices = torch.tensor([3, 1])
        req_to_token = torch.stack(
            [torch.randperm(1024, generator=generator) for _ in range(4)]
        )
        key = torch.randn(1024, 2, 8, generator=generator)
        value = torch.randn(1024, 2, 8, generator=generator)
        query = torch.randn(2, 2, 8, generator=generator)
        reference_lens = torch.tensor([257, 11])
        for generated in (1, 127, 128, 129, 265):
            with self.subTest(generated=generated):
                seq_lens = reference_lens + torch.tensor([generated, generated + 7])
                actual = torch.empty_like(query)
                backend.run_sdpa_forward_decode(
                    query,
                    actual,
                    key,
                    value,
                    req_to_token,
                    req_indices,
                    seq_lens,
                    sliding_window_size=128,
                    reference_lens=reference_lens,
                )
                for batch_idx, slot in enumerate(req_indices):
                    length = int(seq_lens[batch_idx])
                    positions = torch.arange(length)
                    visible = (positions < reference_lens[batch_idx]) | (
                        positions >= length - 128
                    )
                    cache_locs = req_to_token[slot, :length]
                    # Independent full-cache attention oracle: no compaction.
                    q = query[batch_idx].unsqueeze(1)
                    k = key[cache_locs].transpose(0, 1)
                    v = value[cache_locs].transpose(0, 1)
                    scores = (q @ k.transpose(-2, -1)) / math.sqrt(8)
                    scores = scores.masked_fill(~visible, float("-inf"))
                    expected = (scores.softmax(-1) @ v).squeeze(1)
                    torch.testing.assert_close(actual[batch_idx], expected)


if __name__ == "__main__":
    unittest.main()
