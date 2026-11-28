"""A padded chunk table must not change what the GDN extend kernels write.

A breakable prefill CUDA graph hands the chunk kernels a table padded to a fixed
capacity, and `gdn_prefill_graph.py::_write_pairs` fills the tail with the chunk
one past sequence 0's end. Those rows carry no token, so every kernel masks or
skips them, and the extend must produce exactly what an exactly-sized table
produces: same output, same recurrent state, and no write outside the batch.

Covers `chunk_fwd_o` and `recompute_w_u_fwd` on their own, the whole
`chunk_gated_delta_rule` extend over a state pool, the non-varlen path where the
chunk index comes from the grid instead of the table, and one capture whose
`cu_seqlens` and chunk table are rewritten between replays.
"""

import unittest

import torch
import triton

from sglang.kernels.ops.attention.fla.chunk import (
    CHUNK_SIZE,
    chunk_gated_delta_rule,
)
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

BT = CHUNK_SIZE
HG, H, K, V = 16, 32, 128, 128
POOL = 16
# length 1, exactly one chunk, one token into a second chunk, a multi-chunk
# sequence with an incomplete tail, and mixed-length batches.
LEN_CASES = (
    [1],
    [BT],
    [BT + 1],
    [200],
    [BT, 1, 3 * BT - 7],
    [17, 129, 64, 1, 255],
)


def _cu_seqlens(lens, device):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    out[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
    return out


def _padded_chunk_indices(lens, max_rows, device):
    """What `_write_pairs` builds: the real (seq, chunk) rows, then rows naming
    the chunk one past sequence 0's end."""
    rows = [(s, c) for s, n in enumerate(lens) for c in range(triton.cdiv(n, BT))]
    assert len(rows) <= max_rows
    rows += [(0, triton.cdiv(lens[0], BT))] * (max_rows - len(rows))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _chunk_offsets(lens, device):
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + triton.cdiv(n, BT))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestFlaPaddedChunkTables(unittest.TestCase):
    DEVICE = "cuda"

    def _inputs(self, lens, h_rows):
        """``h`` gets ``h_rows`` rows because chunk_gated_delta_rule_fwd sizes it
        by ``len(chunk_indices)``, and chunk_fwd_kernel_o indexes it by the grid
        row rather than by the token."""
        torch.manual_seed(0)
        total = sum(lens)
        f = dict(device=self.DEVICE, dtype=torch.bfloat16)
        return dict(
            q=torch.randn(1, total, HG, K, **f) * 0.1,
            k=torch.randn(1, total, HG, K, **f) * 0.1,
            v=torch.randn(1, total, H, V, **f) * 0.1,
            h=torch.randn(h_rows, H, V, K, **f) * 0.1,
            g=torch.randn(1, total, H, device=self.DEVICE, dtype=torch.float32) * 0.1,
            beta=torch.randn(1, total, H, **f) * 0.1,
            A=torch.randn(1, total, H, BT, **f) * 0.1,
        )

    def test_chunk_fwd_o_ignores_padded_rows(self):
        for lens in LEN_CASES:
            for extra in (1, 8, 115):
                with self.subTest(lens=lens, extra_rows=extra):
                    cu = _cu_seqlens(lens, self.DEVICE)
                    exact = prepare_chunk_indices(cu, BT).to(torch.int32)
                    padded = _padded_chunk_indices(
                        lens, exact.shape[0] + extra, self.DEVICE
                    )
                    self.assertTrue(torch.equal(padded[: exact.shape[0]], exact))
                    t = self._inputs(lens, padded.shape[0])
                    args = dict(q=t["q"], k=t["k"], v=t["v"], h=t["h"], g=t["g"])
                    ref = chunk_fwd_o(cu_seqlens=cu, chunk_indices=exact, **args)
                    got = chunk_fwd_o(cu_seqlens=cu, chunk_indices=padded, **args)
                    self.assertTrue(torch.equal(ref, got))
                    self.assertFalse(torch.isnan(got).any())
                    self.assertFalse(torch.isinf(got).any())

    def test_recompute_w_u_ignores_padded_rows(self):
        for lens in LEN_CASES:
            for extra in (1, 8, 115):
                with self.subTest(lens=lens, extra_rows=extra):
                    cu = _cu_seqlens(lens, self.DEVICE)
                    exact = prepare_chunk_indices(cu, BT).to(torch.int32)
                    padded = _padded_chunk_indices(
                        lens, exact.shape[0] + extra, self.DEVICE
                    )
                    t = self._inputs(lens, padded.shape[0])
                    a = (t["k"], t["v"], t["beta"], t["g"], t["A"])
                    w0, u0 = recompute_w_u_fwd(*a, cu_seqlens=cu, chunk_indices=exact)
                    w1, u1 = recompute_w_u_fwd(*a, cu_seqlens=cu, chunk_indices=padded)
                    self.assertTrue(torch.equal(w0, w1))
                    self.assertTrue(torch.equal(u0, u1))
                    for x in (w1, u1):
                        self.assertFalse(torch.isnan(x).any())
                        self.assertFalse(torch.isinf(x).any())

    def test_recompute_w_u_padded_rows_with_k_ne_v(self):
        """K != V exercises a different mask path in this kernel; issue #50249
        reports it leaving part of `u` unwritten on XPU. The padded-table
        invariant must hold there too."""
        lens = [17, 129, 64, 1, 255]
        total = sum(lens)
        f = dict(device=self.DEVICE, dtype=torch.bfloat16)
        for k_dim, v_dim in ((128, 64), (64, 128), (128, 192)):
            with self.subTest(K=k_dim, V=v_dim):
                torch.manual_seed(3)
                cu = _cu_seqlens(lens, self.DEVICE)
                exact = prepare_chunk_indices(cu, BT).to(torch.int32)
                padded = _padded_chunk_indices(lens, exact.shape[0] + 64, self.DEVICE)
                kk = torch.randn(1, total, HG, k_dim, **f) * 0.1
                vv = torch.randn(1, total, H, v_dim, **f) * 0.1
                beta = torch.randn(1, total, H, **f) * 0.1
                gg = (
                    torch.randn(1, total, H, device=self.DEVICE, dtype=torch.float32)
                    * 0.1
                )
                A = torch.randn(1, total, H, BT, **f) * 0.1
                w0, u0 = recompute_w_u_fwd(
                    kk, vv, beta, gg, A, cu_seqlens=cu, chunk_indices=exact
                )
                w1, u1 = recompute_w_u_fwd(
                    kk, vv, beta, gg, A, cu_seqlens=cu, chunk_indices=padded
                )
                self.assertTrue(torch.equal(w0, w1))
                self.assertTrue(torch.equal(u0, u1))
                for x in (w1, u1):
                    self.assertFalse(torch.isnan(x).any())
                    self.assertFalse(torch.isinf(x).any())

    def test_non_varlen_matches_varlen_single_sequence(self):
        """IS_VARLEN=False takes the chunk index from the grid, whose extent is
        cdiv(T, BT), so the guard must stay inert there."""
        lens = [200]
        cu = _cu_seqlens(lens, self.DEVICE)
        t = self._inputs(lens, triton.cdiv(lens[0], BT))
        args = dict(q=t["q"], k=t["k"], v=t["v"], h=t["h"], g=t["g"])
        varlen = chunk_fwd_o(
            cu_seqlens=cu,
            chunk_indices=prepare_chunk_indices(cu, BT).to(torch.int32),
            **args,
        )
        self.assertTrue(torch.equal(varlen, chunk_fwd_o(**args)))

    def test_extend_over_state_pool_matches_exact_table(self):
        """The whole extend, including the recurrent state the pool holds."""
        for lens in ([1], [BT + 1], [200], [BT, 1, 3 * BT - 7], [17, 129, 64, 1, 255]):
            for extra in (0, 8, 64):
                with self.subTest(lens=lens, extra_rows=extra):
                    torch.manual_seed(11)
                    total = sum(lens)
                    f = dict(device=self.DEVICE, dtype=torch.bfloat16)
                    q = torch.randn(1, total, HG, K, **f) * 0.1
                    k = torch.randn(1, total, HG, K, **f) * 0.1
                    v = torch.randn(1, total, H, V, **f) * 0.1
                    g = (
                        torch.randn(
                            1, total, H, device=self.DEVICE, dtype=torch.float32
                        )
                        * 0.1
                    )
                    beta = torch.randn(1, total, H, **f).sigmoid()
                    cu = _cu_seqlens(lens, self.DEVICE)
                    # Slot 0 is the padding sink; live requests take later slots.
                    pool0 = (
                        torch.randn(
                            POOL, H, V, K, device=self.DEVICE, dtype=torch.float32
                        )
                        * 0.1
                    )
                    pool0[0].zero_()
                    idx = torch.arange(
                        1, len(lens) + 1, dtype=torch.int32, device=self.DEVICE
                    )
                    exact = prepare_chunk_indices(cu, BT).to(torch.int32)
                    padded = _padded_chunk_indices(
                        lens, exact.shape[0] + extra, self.DEVICE
                    )
                    offsets = _chunk_offsets(lens, self.DEVICE)

                    def run(chunk_indices, chunk_offsets):
                        pool = pool0.clone()
                        # inplace_update writes the final state back into the
                        # pool, so comparing the pool covers the state too.
                        ret = chunk_gated_delta_rule(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            scale=K**-0.5,
                            initial_state=pool,
                            initial_state_indices=idx,
                            cu_seqlens=cu,
                            chunk_indices=chunk_indices,
                            chunk_offsets=chunk_offsets,
                            use_qk_l2norm_in_kernel=True,
                            inplace_update=True,
                        )
                        o = ret[0] if isinstance(ret, tuple) else ret
                        return o, pool

                    o0, p0 = run(exact, None)
                    o1, p1 = run(padded, offsets)
                    self.assertTrue(torch.equal(o0, o1))
                    live = idx.to(torch.int64)
                    self.assertTrue(torch.equal(p0[live], p1[live]))
                    others = [i for i in range(POOL) if i not in live.tolist()]
                    self.assertTrue(torch.equal(p0[others], p1[others]))
                    self.assertFalse(torch.isnan(o1).any())
                    self.assertFalse(torch.isinf(o1).any())

    def test_capture_replay_rewrites_tables(self):
        """One capture at a fixed capacity, then replays for other batches."""
        cap_rows, cap_tokens, cap_seqs = 64, 512, 8
        f = dict(device=self.DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(7)
        q = torch.zeros(1, cap_tokens, HG, K, **f)
        k = torch.zeros(1, cap_tokens, HG, K, **f)
        v = torch.zeros(1, cap_tokens, H, V, **f)
        h = torch.zeros(cap_rows, H, V, K, **f)
        g = torch.zeros(1, cap_tokens, H, device=self.DEVICE, dtype=torch.float32)
        out = torch.zeros(1, cap_tokens, H, V, **f)
        cu = torch.zeros(cap_seqs + 1, dtype=torch.int32, device=self.DEVICE)
        ci = torch.zeros(cap_rows, 2, dtype=torch.int32, device=self.DEVICE)
        dst = torch.arange(cap_tokens, dtype=torch.int32, device=self.DEVICE)

        def fill(lens):
            for t in (q, k, v, h):
                t.normal_(0, 0.1)
            g.normal_(0, 0.1)
            cu.zero_()
            total = sum(lens)
            cu[1 : len(lens) + 1] = torch.tensor(
                lens, dtype=torch.int32, device=self.DEVICE
            ).cumsum(0)
            cu[len(lens) + 1 :] = total
            ci.copy_(_padded_chunk_indices(lens, cap_rows, self.DEVICE))

        def run():
            return chunk_fwd_o(
                q=q,
                k=k,
                v=v,
                h=h,
                g=g,
                cu_seqlens=cu,
                chunk_indices=ci,
                out=out,
                output_indices=dst,
            )

        fill([cap_tokens])
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

        for lens in ([cap_tokens], [100, 23, 1], [BT, BT, BT], [7]):
            with self.subTest(lens=lens):
                fill(lens)
                total = sum(lens)
                graph.replay()
                torch.cuda.synchronize()
                replayed = out[:, :total].clone()
                cu_ref = _cu_seqlens(lens, self.DEVICE)
                ref = chunk_fwd_o(
                    q=q[:, :total],
                    k=k[:, :total],
                    v=v[:, :total],
                    h=h,
                    g=g[:, :total],
                    cu_seqlens=cu_ref,
                    chunk_indices=prepare_chunk_indices(cu_ref, BT).to(torch.int32),
                )
                self.assertTrue(torch.equal(replayed, ref))
                self.assertFalse(torch.isnan(replayed).any())


if __name__ == "__main__":
    unittest.main()
