import math
import os
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, suite="nightly-4-gpu-b200", nightly=True)


def reference_attention(q, k, v, rel_bias, scale):
    batch, prediction, heads_q, _ = q.shape
    seqlen = k.shape[1]
    heads_k = k.shape[2]
    grouped_heads = heads_q // heads_k
    k = k.repeat_interleave(grouped_heads, dim=2)
    v = v.repeat_interleave(grouped_heads, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale

    query = torch.arange(prediction, device=q.device)[:, None]
    key = torch.arange(seqlen, device=q.device)[None, :]
    relative = query + seqlen - prediction - key
    causal_mask = relative < 0
    in_bias = (relative >= 0) & (relative < rel_bias.shape[-1])
    bias_by_head = rel_bias.permute(0, 2, 1, 3)
    gathered_bias = torch.stack(
        [
            bias_by_head[
                :,
                :,
                query_idx,
                relative[query_idx].clamp(0, rel_bias.shape[-1] - 1),
            ]
            for query_idx in range(prediction)
        ],
        dim=2,
    )
    scores = scores + gathered_bias.float() * in_bias[None, None]
    scores.masked_fill_(causal_mask[None, None], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities.float(), v.float())


class TestFlashAttentionSm100BiasDecode(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires SM100")

    def test_compact_bias_decode(self):
        torch.manual_seed(0)
        batch, prediction, seqlen = 2, 4, 1024
        heads_q, heads_k, head_dim, rel_extent = 8, 1, 128, 128
        shape_q = (batch, prediction, heads_q, head_dim)
        shape_kv = (batch, seqlen, heads_k, head_dim)
        q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
        rel_bias = torch.randn(
            batch,
            prediction,
            heads_q,
            rel_extent,
            device="cuda",
            dtype=torch.bfloat16,
        )
        scale = 1.0 / math.sqrt(head_dim)

        with mock.patch.dict(os.environ, {"SGLANG_FA4_BIAS_DECODE": "force"}):
            actual, lse = _flash_attn_fwd(
                q,
                k,
                v,
                softmax_scale=scale,
                causal=True,
                num_splits=4,
                rel_bias=rel_bias,
            )
        expected = reference_attention(q, k, v, rel_bias, scale)

        self.assertIsNone(lse)
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

    def test_paged_bias_decode_matches_dense_bitwise(self):
        torch.manual_seed(0)
        batch, prediction, seqlen, page_size = 2, 4, 1024, 128
        heads_q, heads_k, head_dim, rel_extent = 8, 1, 128, 128
        q = torch.randn(
            batch, prediction, heads_q, head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(
            batch, seqlen, heads_k, head_dim, device="cuda", dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        rel_bias = torch.randn(
            batch,
            prediction,
            heads_q,
            rel_extent,
            device="cuda",
            dtype=torch.bfloat16,
        )
        scale = 1.0 / math.sqrt(head_dim)
        common = dict(softmax_scale=scale, causal=True, num_splits=4)

        pages_per_seq = seqlen // page_size
        permutation = torch.randperm(batch * pages_per_seq, device="cuda")
        # Column-sliced view of a wider buffer, like the backend's CUDA-graph
        # page-table metadata (row stride > row width).
        page_table_buf = torch.zeros(
            batch, pages_per_seq + 3, dtype=torch.int32, device="cuda"
        )
        page_table_buf[:, :pages_per_seq] = permutation.to(torch.int32).view(
            batch, pages_per_seq
        )
        page_table = page_table_buf[:, :pages_per_seq]
        k_paged = torch.empty(
            batch * pages_per_seq,
            page_size,
            heads_k,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_paged = torch.empty_like(k_paged)
        k_paged[permutation] = k.view(-1, page_size, heads_k, head_dim)
        v_paged[permutation] = v.view(-1, page_size, heads_k, head_dim)
        seqused_k = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)

        with mock.patch.dict(os.environ, {"SGLANG_FA4_BIAS_DECODE": "force"}):
            dense, _ = _flash_attn_fwd(q, k, v, rel_bias=rel_bias, **common)
            paged, _ = _flash_attn_fwd(
                q,
                k_paged,
                v_paged,
                rel_bias=rel_bias,
                page_table=page_table,
                seqused_k=seqused_k,
                **common,
            )

        self.assertEqual(
            (paged.view(torch.uint16) != dense.view(torch.uint16)).sum().item(), 0
        )

    def test_paged_bias_decode_mxfp8(self):
        torch.manual_seed(0)
        from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_sm100_bias_decode import (
            create_mxfp8_scale_factor_tensor,
        )

        batch, prediction, seqlen, page_size = 2, 1, 1024, 64
        heads_q, heads_k, head_dim, rel_extent = 8, 1, 128, 128
        sf_groups = head_dim // 32
        pages_per_seq = seqlen // page_size
        pages = batch * pages_per_seq
        scale = 1.0 / math.sqrt(head_dim)
        choices = torch.tensor([0.5, 1.0, 2.0], device="cuda")

        q8 = torch.randn(batch, prediction, heads_q, head_dim, device="cuda").to(
            torch.float8_e4m3fn
        )
        # Blocked 128-row-atom SFQ storage; rows are the packed
        # (grouped_head_tile, prediction) plane per (heads_k, batch).
        sfq_elem, _, sfq = create_mxfp8_scale_factor_tensor(
            heads_q * prediction, head_dim, heads_k * batch, pattern="grid"
        )
        sfq = sfq.view(torch.float8_e8m0fnu)
        q_deq = q8.float() * sfq_elem.permute(2, 0, 1).view(
            batch, prediction, heads_q, head_dim
        )

        k8 = torch.randn(batch, seqlen, heads_k, head_dim, device="cuda").to(
            torch.float8_e4m3fn
        )
        v8 = torch.randn(batch, seqlen, heads_k, head_dim, device="cuda").to(
            torch.float8_e4m3fn
        )
        k_sf = choices[
            torch.randint(0, 3, (batch, seqlen, heads_k, sf_groups), device="cuda")
        ]
        v_sf = choices[
            torch.randint(0, 3, (batch, seqlen // 32, heads_k, head_dim), device="cuda")
        ]
        k_deq = k8.float() * k_sf.repeat_interleave(32, dim=-1)
        v_deq = v8.float() * v_sf.repeat_interleave(32, dim=1)

        permutation = torch.randperm(pages, device="cuda")
        page_table = permutation.to(torch.int32).view(batch, pages_per_seq)
        k_paged = torch.empty(
            pages, page_size, heads_k, head_dim, device="cuda", dtype=k8.dtype
        )
        v_paged = torch.empty_like(k_paged)
        k_paged[permutation] = k8.view(-1, page_size, heads_k, head_dim)
        v_paged[permutation] = v8.view(-1, page_size, heads_k, head_dim)
        # Compact page-local SFK: contiguous (page, head) planes viewed as
        # (pages, page_size, heads_k, groups); SFV is plain contiguous.
        sfk = (
            torch.empty(
                pages,
                heads_k,
                page_size,
                sf_groups,
                device="cuda",
                dtype=torch.float8_e8m0fnu,
            )
        ).permute(0, 2, 1, 3)
        sfk[permutation] = k_sf.view(-1, page_size, heads_k, sf_groups).to(
            torch.float8_e8m0fnu
        )
        sfv = torch.empty(
            pages,
            page_size // 32,
            heads_k,
            head_dim,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        sfv[permutation] = v_sf.view(-1, page_size // 32, heads_k, head_dim).to(
            torch.float8_e8m0fnu
        )
        seqused_k = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)
        rel_bias = torch.randn(
            batch,
            prediction,
            heads_q,
            rel_extent,
            device="cuda",
            dtype=torch.bfloat16,
        )

        local_cache: dict = {}
        with mock.patch.dict(
            os.environ, {"SGLANG_FA4_BIAS_DECODE": "force"}
        ), mock.patch.object(_flash_attn_fwd, "compile_cache_bias_decode", local_cache):
            actual, lse = _flash_attn_fwd(
                q8,
                k_paged,
                v_paged,
                softmax_scale=scale,
                causal=True,
                num_splits=4,
                rel_bias=rel_bias,
                page_table=page_table,
                seqused_k=seqused_k,
                sfq=sfq,
                sfk=sfk,
                sfv=sfv,
                qk_sf_vec_size=32,
                v_sf_vec_size=32,
            )

        self.assertIsNone(lse)
        # decode_key[-1] is the bias_decode_mxfp8 flag: proves the decode
        # kernel (not the general FA4 fallback) served this call.
        self.assertTrue(any(key[-1] for key in local_cache))
        expected = reference_attention(
            q_deq.to(torch.bfloat16),
            k_deq.to(torch.bfloat16),
            v_deq.to(torch.bfloat16),
            rel_bias,
            scale,
        )
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
