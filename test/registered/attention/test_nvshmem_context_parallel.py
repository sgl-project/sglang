import unittest

import torch

from sglang.srt.layers.attention.nvshmem_context_parallel import (
    build_sequence_chunks,
    chunks_for_rank,
    context_parallel_attention_reference,
    context_parallel_attention_triton_forward,
    local_kv_chunks,
    query_positions_for_chunks,
    shard_sequence_tensor,
)


def _full_attention(q, k, v, query_positions, *, causal):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    if causal:
        key_positions = torch.arange(k.shape[2], device=q.device)
        mask = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v.float())
    return out.to(q.dtype), torch.logsumexp(scores, dim=-1)


class TestNvshmemContextParallelReference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260506)

    def _make_local_kv(self, k_full, v_full, chunks, world_size):
        local_k = []
        local_v = []
        for rank in range(world_size):
            owned = chunks_for_rank(chunks, rank)
            local_k.append(
                shard_sequence_tensor(k_full, owned, seq_dim=2)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            local_v.append(
                shard_sequence_tensor(v_full, owned, seq_dim=2)
                .detach()
                .clone()
                .requires_grad_(True)
            )
        return local_k, local_v

    def test_multi_head_contiguous_matches_full_attention_and_lse(self):
        bsz, heads, seq_len, dim, world_size = 2, 4, 9, 8, 3
        q_full = torch.randn(bsz, heads, seq_len, dim)
        k_full = torch.randn(bsz, heads, seq_len, dim)
        v_full = torch.randn(bsz, heads, seq_len, dim)

        chunks = build_sequence_chunks(seq_len, world_size, "contiguous")
        local_k, local_v = self._make_local_kv(k_full, v_full, chunks, world_size)
        kv_chunks = local_kv_chunks(local_k, local_v, chunks)

        rank = 1
        q_chunks = chunks_for_rank(chunks, rank)
        q_local = shard_sequence_tensor(q_full, q_chunks, seq_dim=2)
        query_positions = query_positions_for_chunks(q_chunks, device=q_full.device)

        out, lse = context_parallel_attention_reference(
            q_local, kv_chunks, query_positions=query_positions, return_lse=True
        )
        out_ref, lse_ref = _full_attention(
            q_local, k_full, v_full, query_positions, causal=True
        )

        self.assertTrue(torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5))

    def test_headtail_load_balanced_chunks_match_full_attention(self):
        bsz, heads, seq_len, dim, world_size = 1, 3, 13, 8, 2
        q_full = torch.randn(bsz, heads, seq_len, dim)
        k_full = torch.randn(bsz, heads, seq_len, dim)
        v_full = torch.randn(bsz, heads, seq_len, dim)

        chunks = build_sequence_chunks(seq_len, world_size, "headtail")
        local_k, local_v = self._make_local_kv(k_full, v_full, chunks, world_size)
        kv_chunks = local_kv_chunks(local_k, local_v, chunks)

        rank = 0
        q_chunks = chunks_for_rank(chunks, rank)
        self.assertGreater(len(q_chunks), 1)
        q_local = shard_sequence_tensor(q_full, q_chunks, seq_dim=2)
        query_positions = query_positions_for_chunks(q_chunks, device=q_full.device)

        out = context_parallel_attention_reference(
            q_local, kv_chunks, query_positions=query_positions, causal=True
        )
        out_ref, _ = _full_attention(q_local, k_full, v_full, query_positions, causal=True)

        self.assertTrue(torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5))

    def test_backward_grads_stay_on_owned_kv_chunks(self):
        bsz, heads, seq_len, dim, world_size = 1, 2, 7, 4, 2
        q_full = torch.randn(bsz, heads, seq_len, dim)
        k_ref = torch.randn(bsz, heads, seq_len, dim, requires_grad=True)
        v_ref = torch.randn(bsz, heads, seq_len, dim, requires_grad=True)

        chunks = build_sequence_chunks(seq_len, world_size, "headtail")
        local_k, local_v = self._make_local_kv(k_ref.detach(), v_ref.detach(), chunks, world_size)
        kv_chunks = local_kv_chunks(local_k, local_v, chunks)

        rank = 1
        q_chunks = chunks_for_rank(chunks, rank)
        q_local = shard_sequence_tensor(q_full, q_chunks, seq_dim=2).detach().clone()
        q_local.requires_grad_(True)
        q_ref = q_local.detach().clone().requires_grad_(True)
        query_positions = query_positions_for_chunks(q_chunks, device=q_full.device)

        out = context_parallel_attention_reference(
            q_local, kv_chunks, query_positions=query_positions, causal=True
        )
        out.square().sum().backward()

        out_ref, _ = _full_attention(q_ref, k_ref, v_ref, query_positions, causal=True)
        out_ref.square().sum().backward()

        self.assertTrue(torch.allclose(q_local.grad, q_ref.grad, atol=1e-5, rtol=1e-5))
        for rank_id in range(world_size):
            owned = chunks_for_rank(chunks, rank_id)
            for chunk in owned:
                local_slice = slice(chunk.local_start, chunk.local_end)
                global_slice = slice(chunk.global_start, chunk.global_end)
                self.assertTrue(
                    torch.allclose(
                        local_k[rank_id].grad[:, :, local_slice, :],
                        k_ref.grad[:, :, global_slice, :],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        local_v[rank_id].grad[:, :, local_slice, :],
                        v_ref.grad[:, :, global_slice, :],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )

    def test_non_causal_attention_matches_full_attention(self):
        bsz, heads, seq_len, dim, world_size = 2, 2, 8, 8, 4
        q_full = torch.randn(bsz, heads, seq_len, dim)
        k_full = torch.randn(bsz, heads, seq_len, dim)
        v_full = torch.randn(bsz, heads, seq_len, dim)

        chunks = build_sequence_chunks(seq_len, world_size, "headtail")
        local_k, local_v = self._make_local_kv(k_full, v_full, chunks, world_size)
        kv_chunks = local_kv_chunks(local_k, local_v, chunks)
        q_chunks = chunks_for_rank(chunks, 2)
        q_local = shard_sequence_tensor(q_full, q_chunks, seq_dim=2)
        query_positions = query_positions_for_chunks(q_chunks, device=q_full.device)

        out = context_parallel_attention_reference(
            q_local, kv_chunks, query_positions=query_positions, causal=False
        )
        out_ref, _ = _full_attention(q_local, k_full, v_full, query_positions, causal=False)

        self.assertTrue(torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_triton_forward_matches_reference(self):
        device = torch.device("cuda")
        bsz, heads, seq_len, dim, world_size = 2, 3, 17, 32, 2
        q_full = torch.randn(bsz, heads, seq_len, dim, device=device)
        k_full = torch.randn(bsz, heads, seq_len, dim, device=device)
        v_full = torch.randn(bsz, heads, seq_len, dim, device=device)

        chunks = build_sequence_chunks(seq_len, world_size, "headtail")
        local_k, local_v = self._make_local_kv(k_full, v_full, chunks, world_size)
        kv_chunks = local_kv_chunks(local_k, local_v, chunks)
        q_chunks = chunks_for_rank(chunks, 0)
        q_local = shard_sequence_tensor(q_full, q_chunks, seq_dim=2)
        query_positions = query_positions_for_chunks(q_chunks, device=device)

        out, lse = context_parallel_attention_triton_forward(
            q_local,
            kv_chunks,
            query_positions=query_positions,
            causal=True,
            return_lse=True,
            block_q=16,
            block_k=16,
        )
        out_ref, lse_ref = context_parallel_attention_reference(
            q_local,
            kv_chunks,
            query_positions=query_positions,
            causal=True,
            return_lse=True,
        )

        self.assertTrue(torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(lse, lse_ref, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
