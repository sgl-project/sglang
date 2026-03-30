"""Standalone validation of FA3 MLA attention with simulated DCP sharding.

Step 1: Run FA3 MLA on a single GPU with full KV (ground truth).
Step 2: Simulate DCP by sharding KV across N virtual ranks, running
        attention per-shard, and combining with LSE-weighted merge.
        Result must match the full-KV ground truth.
"""

import unittest

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-large")


def run_fa3_mla_decode(
    q_rope,
    qv,
    k_rope_cache,
    v_cache,
    page_table,
    cache_seqlens,
    softmax_scale=None,
):
    result = flash_attn_with_kvcache(
        q=q_rope,
        k_cache=k_rope_cache,
        v_cache=v_cache,
        qv=qv,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        causal=True,
        return_softmax_lse=True,
        softmax_scale=softmax_scale,
    )
    out, lse, *_ = result
    return out, lse


def build_kv_cache(total_tokens, num_kv_heads, d_rope, d_nope, page_size, device):
    num_pages = (total_tokens + page_size - 1) // page_size
    k_rope = torch.randn(
        num_pages, page_size, num_kv_heads, d_rope, device=device, dtype=torch.bfloat16
    )
    v_nope = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        d_nope,
        device=device,
        dtype=torch.bfloat16,
    )
    return k_rope, v_nope, num_pages


def build_page_table_identity(B, seq_len, page_size, device):
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    pt = torch.zeros(B, num_pages_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        for p in range(num_pages_per_seq):
            pt[b, p] = b * num_pages_per_seq + p
    return pt


def shard_kv_for_dcp(
    k_rope_full, v_nope_full, seq_len, page_size, dcp_rank, dcp_size, B, device
):
    """Shard a full KV cache for a simulated DCP rank.

    DCP interleaved ownership: token at position `pos` belongs to rank `pos % dcp_size`.
    """
    num_kv_heads = k_rope_full.shape[2]
    d_rope = k_rope_full.shape[3]
    d_nope = v_nope_full.shape[3]
    full_pages_per_seq = (seq_len + page_size - 1) // page_size

    local_tokens_per_seq = []
    for b in range(B):
        count = sum(1 for pos in range(seq_len) if pos % dcp_size == dcp_rank)
        local_tokens_per_seq.append(count)

    max_local_tokens = max(local_tokens_per_seq)
    local_pages_per_seq = (max_local_tokens + page_size - 1) // page_size

    k_local = torch.zeros(
        B * local_pages_per_seq,
        page_size,
        num_kv_heads,
        d_rope,
        device=device,
        dtype=torch.bfloat16,
    )
    v_local = torch.zeros(
        B * local_pages_per_seq,
        page_size,
        num_kv_heads,
        d_nope,
        device=device,
        dtype=torch.bfloat16,
    )
    local_page_table = torch.zeros(
        B, local_pages_per_seq, dtype=torch.int32, device=device
    )

    for b in range(B):
        local_idx = 0
        for pos in range(seq_len):
            if pos % dcp_size == dcp_rank:
                src_page = b * full_pages_per_seq + pos // page_size
                src_offset = pos % page_size
                dst_page_idx = b * local_pages_per_seq + local_idx // page_size
                dst_offset = local_idx % page_size
                k_local[dst_page_idx, dst_offset] = k_rope_full[src_page, src_offset]
                v_local[dst_page_idx, dst_offset] = v_nope_full[src_page, src_offset]
                local_idx += 1
        for p in range(local_pages_per_seq):
            local_page_table[b, p] = b * local_pages_per_seq + p

    local_seqlens = torch.tensor(local_tokens_per_seq, dtype=torch.int32, device=device)
    return k_local, v_local, local_page_table, local_seqlens


class TestFA3MLASingleGPU(CustomTestCase):
    """FA3 MLA on single GPU -- ground truth."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def test_basic_mla_output(self):
        torch.manual_seed(42)
        B, H_q, H_kv, d_rope, d_nope = 2, 8, 1, 64, 512
        seq_len, page_size = 32, 16

        q_rope = torch.randn(
            B, 1, H_q, d_rope, device=self.device, dtype=torch.bfloat16
        )
        qv = torch.randn(B, 1, H_q, d_nope, device=self.device, dtype=torch.bfloat16)

        k_cache, v_cache, _ = build_kv_cache(
            B * seq_len, H_kv, d_rope, d_nope, page_size, self.device
        )
        page_table = build_page_table_identity(B, seq_len, page_size, self.device)
        cache_seqlens = torch.full((B,), seq_len, dtype=torch.int32, device=self.device)

        out, lse = run_fa3_mla_decode(
            q_rope, qv, k_cache, v_cache, page_table, cache_seqlens
        )

        self.assertEqual(out.shape, (B, 1, H_q, d_nope))
        self.assertEqual(lse.shape, (B, H_q, 1))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isnan(lse).any())

    def test_deterministic(self):
        torch.manual_seed(7)
        B, H_q, H_kv, d_rope, d_nope = 1, 4, 1, 64, 512
        seq_len, page_size = 16, 8

        q_rope = torch.randn(
            B, 1, H_q, d_rope, device=self.device, dtype=torch.bfloat16
        )
        qv = torch.randn(B, 1, H_q, d_nope, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, _ = build_kv_cache(
            B * seq_len, H_kv, d_rope, d_nope, page_size, self.device
        )
        page_table = build_page_table_identity(B, seq_len, page_size, self.device)
        cache_seqlens = torch.full((B,), seq_len, dtype=torch.int32, device=self.device)

        out1, _ = run_fa3_mla_decode(
            q_rope, qv, k_cache, v_cache, page_table, cache_seqlens
        )
        out2, _ = run_fa3_mla_decode(
            q_rope, qv, k_cache, v_cache, page_table, cache_seqlens
        )

        torch.testing.assert_close(out1, out2, atol=0, rtol=0)


class TestFA3MLASimulatedDCP(CustomTestCase):
    """Simulate DCP sharding, combine partials, compare to ground truth."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def _run_dcp_simulation(
        self, B, H_q, H_kv, d_rope, d_nope, seq_len, page_size, dcp_size
    ):
        from sglang.srt.layers.attention.dcp_a2a import dcp_lse_combine_triton

        torch.manual_seed(42)

        q_rope = torch.randn(
            B, 1, H_q, d_rope, device=self.device, dtype=torch.bfloat16
        )
        qv = torch.randn(B, 1, H_q, d_nope, device=self.device, dtype=torch.bfloat16)

        total_tokens = B * seq_len
        k_full, v_full, _ = build_kv_cache(
            total_tokens, H_kv, d_rope, d_nope, page_size, self.device
        )
        pt_full = build_page_table_identity(B, seq_len, page_size, self.device)
        seqlens_full = torch.full((B,), seq_len, dtype=torch.int32, device=self.device)

        scale = (d_rope + d_nope) ** (-0.5)

        out_full, _ = run_fa3_mla_decode(
            q_rope, qv, k_full, v_full, pt_full, seqlens_full, softmax_scale=scale
        )

        partial_outputs = []
        partial_lses = []

        for rank in range(dcp_size):
            k_local, v_local, pt_local, seqlens_local = shard_kv_for_dcp(
                k_full, v_full, seq_len, page_size, rank, dcp_size, B, self.device
            )
            out_r, lse_r = run_fa3_mla_decode(
                q_rope,
                qv,
                k_local,
                v_local,
                pt_local,
                seqlens_local,
                softmax_scale=scale,
            )
            partial_outputs.append(out_r.squeeze(1))
            partial_lses.append(lse_r.squeeze(-1))

        stacked_out = torch.stack(partial_outputs, dim=0)
        stacked_lse = torch.stack(partial_lses, dim=0)

        combined, _ = dcp_lse_combine_triton(
            stacked_out, stacked_lse, is_lse_base_on_e=True
        )

        return out_full.squeeze(1), combined

    def test_dcp2_matches_full(self):
        out_full, out_dcp = self._run_dcp_simulation(
            B=2,
            H_q=8,
            H_kv=1,
            d_rope=64,
            d_nope=512,
            seq_len=32,
            page_size=16,
            dcp_size=2,
        )
        torch.testing.assert_close(
            out_full.float(), out_dcp.float(), atol=0.05, rtol=0.02
        )

    def test_dcp4_matches_full(self):
        out_full, out_dcp = self._run_dcp_simulation(
            B=2,
            H_q=16,
            H_kv=1,
            d_rope=64,
            d_nope=512,
            seq_len=64,
            page_size=16,
            dcp_size=4,
        )
        torch.testing.assert_close(
            out_full.float(), out_dcp.float(), atol=0.05, rtol=0.02
        )

    def test_dcp8_matches_full(self):
        out_full, out_dcp = self._run_dcp_simulation(
            B=1,
            H_q=8,
            H_kv=1,
            d_rope=64,
            d_nope=512,
            seq_len=128,
            page_size=16,
            dcp_size=8,
        )
        torch.testing.assert_close(
            out_full.float(), out_dcp.float(), atol=0.05, rtol=0.02
        )

    def test_dcp2_large_seq(self):
        out_full, out_dcp = self._run_dcp_simulation(
            B=4,
            H_q=8,
            H_kv=1,
            d_rope=64,
            d_nope=512,
            seq_len=256,
            page_size=16,
            dcp_size=2,
        )
        torch.testing.assert_close(
            out_full.float(), out_dcp.float(), atol=0.05, rtol=0.02
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
