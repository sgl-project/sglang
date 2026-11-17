import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import (
    CHUNK_SIZE,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=14, stage="stage-b", runner_config="1-gpu-large-amd")
register_xpu_ci(est_time=930, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
class TestChunkGatedDeltaRule(unittest.TestCase):
    """Test chunk_gated_delta_rule against token-by-token fused_recurrent reference."""

    ATOL = 2e-2
    RTOL = 1e-2

    def _run_reference(self, pool_init, cache_indices, q, k, v, g, beta):
        """Per-batch token-by-token reference using fused_recurrent_gated_delta_rule.

        initial_state shape: [N, H, V, K] (native layout on this branch).
        """
        B = cache_indices.shape[0]
        T_per_seq = q.shape[1] // B
        pool = pool_init.clone()
        h_cur = pool[cache_indices].contiguous().clone()

        o_list = []
        for b in range(B):
            sl = slice(b * T_per_seq, (b + 1) * T_per_seq)
            o_b, h_b = fused_recurrent_gated_delta_rule(
                q=q[0, sl].unsqueeze(0),
                k=k[0, sl].unsqueeze(0),
                v=v[0, sl].unsqueeze(0),
                g=g[0, sl].unsqueeze(0),
                beta=beta[0, sl].unsqueeze(0),
                initial_state=h_cur[b : b + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o_list.append(o_b)
            h_cur[b] = h_b[0]

        pool[cache_indices] = h_cur
        return torch.cat(o_list, dim=1), pool

    def _run_chunk(self, pool_init, cache_indices, q, k, v, g, beta, cu_seqlens):
        """Run chunk_gated_delta_rule with native [V, K] pool."""
        pool = pool_init.clone()
        o, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return o, pool

    def _check_shape(
        self, B, T_per_seq, H, K, V, pool_size, sequential_indices=False, seed=42
    ):
        """Run correctness check for one (B, T_per_seq, H, K, V, pool_size) config."""
        device = get_device()
        dtype = torch.bfloat16
        T = B * T_per_seq

        torch.manual_seed(seed)

        if sequential_indices:
            cache_indices = torch.arange(B, dtype=torch.int32, device=device)
        else:
            perm = torch.randperm(pool_size, device=device)[:B]
            cache_indices = perm.to(torch.int32)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )

        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        o_ref, pool_ref = self._run_reference(
            pool_init, cache_indices, q, k, v, g, beta
        )
        o_new, pool_new = self._run_chunk(
            pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
        )

        self.assertTrue(
            torch.allclose(
                o_ref.float(), o_new.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"Output mismatch: max_diff="
            f"{(o_ref.float() - o_new.float()).abs().max().item():.2e}",
        )

        ref_slots = pool_ref[cache_indices].contiguous()
        new_slots = pool_new[cache_indices].contiguous()
        self.assertTrue(
            torch.allclose(
                ref_slots.float(), new_slots.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"State mismatch: max_diff="
            f"{(ref_slots.float() - new_slots.float()).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Production-style configs (Qwen3-Next)
    # ------------------------------------------------------------------
    def test_production_nt1(self):
        self._check_shape(B=4, T_per_seq=64, H=16, K=128, V=128, pool_size=32)

    def test_production_nt2(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_production_nt4(self):
        self._check_shape(B=4, T_per_seq=256, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Batch size sweep
    # ------------------------------------------------------------------
    def test_batch_1(self):
        self._check_shape(B=1, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_2(self):
        self._check_shape(B=2, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_8(self):
        self._check_shape(B=8, T_per_seq=128, H=16, K=128, V=128, pool_size=64)

    def test_batch_16(self):
        self._check_shape(B=16, T_per_seq=64, H=16, K=128, V=128, pool_size=128)

    def test_batch_32(self):
        self._check_shape(B=32, T_per_seq=32, H=16, K=128, V=128, pool_size=256)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "The read-only initial-state path is not implemented by the XPU chunk kernel",
    )
    def test_read_only_initial_state_supports_duplicate_indices(self):
        """MIS item branches may share one query-end state without updating it."""
        device = get_device()
        dtype = torch.bfloat16
        batch_size, tokens_per_item = 3, 65
        num_heads, key_dim, value_dim = 4, 32, 32
        total_tokens = batch_size * tokens_per_item

        torch.manual_seed(1234)
        pool_init = torch.randn(
            4,
            num_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=device,
        )
        duplicate_indices = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            tokens_per_item,
            dtype=torch.int32,
            device=device,
        )
        q = torch.randn(1, total_tokens, num_heads, key_dim, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn(
            1, total_tokens, num_heads, value_dim, dtype=dtype, device=device
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, total_tokens, num_heads, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(
            torch.randn(1, total_tokens, num_heads, dtype=dtype, device=device)
        )

        expected, _ = self._run_reference(
            pool_init, duplicate_indices, q, k, v, g, beta
        )
        actual_pool = pool_init.clone()
        actual, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=actual_pool,
            initial_state_indices=duplicate_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            inplace_update=False,
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), atol=self.ATOL, rtol=self.RTOL
        )
        self.assertTrue(torch.equal(actual_pool, pool_init))

        updating_pool = pool_init.clone()
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=updating_pool,
            initial_state_indices=torch.arange(
                batch_size, dtype=torch.int32, device=device
            ),
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertFalse(
            torch.equal(updating_pool[:batch_size], pool_init[:batch_size])
        )

    # ------------------------------------------------------------------
    # Head count sweep
    # ------------------------------------------------------------------
    def test_heads_4(self):
        self._check_shape(B=4, T_per_seq=128, H=4, K=128, V=128, pool_size=32)

    def test_heads_8(self):
        self._check_shape(B=4, T_per_seq=128, H=8, K=128, V=128, pool_size=32)

    def test_heads_32(self):
        self._check_shape(B=4, T_per_seq=128, H=32, K=128, V=128, pool_size=32)

    def test_heads_64(self):
        self._check_shape(B=4, T_per_seq=128, H=64, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # K != V  (exercises that [V,K] != [K,V] byte-order matters)
    # ------------------------------------------------------------------
    def test_dim_64x64(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=64, pool_size=32)

    def test_dim_k_lt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=128, pool_size=32)

    def test_dim_k_gt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=64, pool_size=32)

    @unittest.skipIf(
        is_hip(),
        "K=V=256 head dim exceeds the FLA chunk triton kernel's shared-memory "
        "budget on ROCm (out-of-resource at launch); smaller head dims pass.",
    )
    def test_dim_256x256(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=256, V=256, pool_size=32)

    # ------------------------------------------------------------------
    # Short sequences (T < chunk_size=64)
    # ------------------------------------------------------------------
    def test_seqlen_1(self):
        self._check_shape(B=4, T_per_seq=1, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_7(self):
        self._check_shape(B=4, T_per_seq=7, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_16(self):
        self._check_shape(B=4, T_per_seq=16, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_32(self):
        self._check_shape(B=4, T_per_seq=32, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Multi-chunk and large pool
    # ------------------------------------------------------------------
    def test_multi_chunk_nt8(self):
        self._check_shape(B=4, T_per_seq=512, H=16, K=128, V=128, pool_size=32)

    def test_large_pool(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=512)

    # ------------------------------------------------------------------
    # Long prompts (many chunks; regression test for cross-chunk errors)
    # ------------------------------------------------------------------
    def test_long_prompt(self):
        for B, T_per_seq in [(1, 1024), (1, 1536), (1, 2048), (2, 1024)]:
            with self.subTest(T_per_seq=T_per_seq):
                self._check_shape(
                    B=B, T_per_seq=T_per_seq, H=16, K=128, V=128, pool_size=32
                )

    # ------------------------------------------------------------------
    # Combined stress
    # ------------------------------------------------------------------
    def test_stress(self):
        self._check_shape(B=32, T_per_seq=128, H=32, K=128, V=128, pool_size=256)

    # ------------------------------------------------------------------
    # Sequential-index variants (pool_size == B, indices = 0..B-1)
    # ------------------------------------------------------------------
    def test_seq_idx_b4(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_b8(self):
        self._check_shape(
            B=8,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=8,
            sequential_indices=True,
        )

    def test_seq_idx_h32(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_h64(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=64,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_stress(self):
        self._check_shape(
            B=32,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=32,
            sequential_indices=True,
        )

    # ------------------------------------------------------------------
    # Padded rows (idle DP ranks under breakable CUDA graph prefill)
    # ------------------------------------------------------------------

    def test_padded_state_index_is_skipped(self):
        """Rows carrying state index -1 must be skipped, not addressed.

        Without the guard the sentinel reaches pointer arithmetic and
        addresses before the state pool.
        """
        device = get_device()
        dtype = torch.bfloat16
        B, T_per_seq, H, K, V, pool_size = 4, 64, 4, 128, 128, 8
        T = B * T_per_seq
        torch.manual_seed(0)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )
        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        # "all padded" is the idle-DP-rank case that faulted.
        for label, indices in (("mixed", [0, -1, 2, -1]), ("all_padded", [-1] * B)):
            with self.subTest(label):
                cache_indices = torch.tensor(indices, dtype=torch.int32, device=device)
                o, pool = self._run_chunk(
                    pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
                )
                # Forces a sync: an out-of-bounds access surfaces here.
                self.assertTrue(torch.isfinite(o.float()).all())

                untouched = [s for s in range(pool_size) if s not in indices]
                self.assertTrue(
                    torch.equal(pool[untouched], pool_init[untouched]),
                    f"{label}: padded rows wrote into the state pool",
                )

    # ------------------------------------------------------------------
    # Long packed prefills (chunk-state offsets beyond the int32 range)
    # ------------------------------------------------------------------

    def test_chunk_state_slot_beyond_int32_offset(self):
        """A sequence whose chunk-state offset exceeds int32 must fill its own slots.

        Chunk states are addressed by chunk_offsets[i] * H * V * K elements.
        Computed in int32, that product wraps once it passes 2**31: between
        2**31 and 2**32 the sequence writes before the buffer, and at 2**32 it
        lands on slot 0, over the first sequence's chunk states. The test sits
        exactly at 2**32 so the bug shows up as a failed comparison rather
        than an illegal memory access.

        Two sequences with identical inputs and identical initial states must
        produce identical chunk states, whatever offset they sit at.
        """
        device = get_device()
        dtype = torch.bfloat16
        Hg = H = 32
        K = V = 128
        T_per_seq = 2 * CHUNK_SIZE
        nt_seq = T_per_seq // CHUNK_SIZE
        # First chunk slot whose element offset lands on a multiple of 2**32.
        overflow_slot = 2**32 // (H * V * K)
        num_chunks = overflow_slot + nt_seq
        torch.manual_seed(0)

        k_seq = torch.randn(1, T_per_seq, Hg, K, dtype=dtype, device=device)
        w_seq = 0.1 * torch.randn(1, T_per_seq, H, K, dtype=dtype, device=device)
        u_seq = torch.randn(1, T_per_seq, H, V, dtype=dtype, device=device)
        # g enters this kernel already cumulatively summed within each chunk.
        g_seq = (
            torch.nn.functional.logsigmoid(
                torch.randn(1, T_per_seq, H, dtype=torch.float32, device=device) + 4.0
            )
            .view(1, nt_seq, CHUNK_SIZE, H)
            .cumsum(dim=2)
            .view(1, T_per_seq, H)
        )
        k = torch.cat([k_seq, k_seq], dim=1)
        w = torch.cat([w_seq, w_seq], dim=1)
        u = torch.cat([u_seq, u_seq], dim=1)
        g = torch.cat([g_seq, g_seq], dim=1)

        state = torch.randn(1, H, V, K, dtype=torch.float32, device=device) * 0.1
        pool = state.repeat(2, 1, 1, 1)
        state_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor(
            [0, T_per_seq, 2 * T_per_seq], dtype=torch.long, device=device
        )
        # The second sequence sits behind the chunks of earlier, already
        # scheduled sequences; only their slots are reserved here, not their
        # tokens, which keeps the test to one chunk-state buffer.
        chunk_offsets = cu_seqlens.new_tensor([0, overflow_slot])
        # chunk_indices never reaches the kernel; its length sizes that buffer.
        chunk_indices = cu_seqlens.new_zeros(num_chunks, 2)

        buffer_bytes = num_chunks * H * V * K * dtype.itemsize
        free_bytes, _ = getattr(torch, device).mem_get_info()
        if free_bytes < buffer_bytes + 2**30:
            self.skipTest(
                f"needs {buffer_bytes / 2**30 + 1:.1f} GiB of free device memory"
            )

        # Poison the block the buffer is about to be carved out of, so an
        # unwritten slot cannot pass for a correctly written one. This relies
        # on the caching allocator handing the same-sized block back.
        poison = torch.empty(num_chunks * H * V * K, dtype=dtype, device=device)
        poison.fill_(float("nan"))
        del poison
        h, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=pool,
            initial_state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
        same = torch.equal(h[0, overflow_slot : overflow_slot + nt_seq], h[0, :nt_seq])
        del h
        getattr(torch, device).empty_cache()

        self.assertTrue(
            same,
            "chunk states of the sequence past the int32 offset were written "
            "somewhere other than its own slots",
        )


if __name__ == "__main__":
    unittest.main()
