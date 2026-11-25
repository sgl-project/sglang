"""One pull over the K3 [latent | shared] buffer (a MegaMoE kernel call) against the two
pulls of a stock call, without a GPU: the index arithmetic of both ar_fusion.cuh kernels
is transcribed and run for every rank, and the source lines it rests on are pinned. The
model: a multicast reduce load returns the same value whichever rank issues it."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest

import torch

from sglang.kernels.ops.communication import all_reduce_residual as arr
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.test.test_utils import CustomTestCase

WORLD = 8
NORM_DIM = k3_ar_fusion.NORM_DIM  # 3584
ROW_VECS = NORM_DIM // 8  # 448 16-byte vectors per norm-width row
PULL_BLOCK = 512
WARP = 32
CUH = os.path.join(
    os.path.dirname(k3_ar_fusion.__file__),
    "../../../kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh",
)


def pull_res_vecs(num_vecs, rank, grid, unroll):
    """Vector ids that all_reduce_pull_res_kernel<unroll> loads and stores on `rank`."""
    avg, rem = divmod(num_vecs, WORLD)
    bias = avg * rank + min(rank, rem)
    mine = avg + (1 if rank < rem else 0)
    step = PULL_BLOCK * grid
    out = []
    # start = bx * 512 + tx; starts beyond the shard do nothing
    for start in range(min(step, mine)):
        vid, width = start, unroll
        while width >= 1:  # pull_reduce_pass<kWidth> then <kWidth / 2> ... <1>
            while vid + (width - 1) * step < mine:
                out.extend(bias + vid + u * step for u in range(width))
                vid += width * step
            width //= 2
    return out


def pull_norm_rows(num_vecs, num_norm_rows, rank, grid, unroll):
    """(row, normed) pairs all_reduce_pull_norm_kernel<unroll> handles on `rank`."""
    assert num_vecs % ROW_VECS == 0
    avg, rem = divmod(num_vecs // ROW_VECS, WORLD)
    bias = avg * rank + min(rank, rem)
    mine = avg + (1 if rank < rem else 0)
    out = []
    for bx in range(grid):
        row0 = bx * unroll
        while row0 < mine:
            for u in range(min(unroll, mine - row0)):
                row = bias + row0 + u
                out.append((row, row < num_norm_rows))
            row0 += grid * unroll
    return out


def geometry(table, nbytes, blocks=None, unroll=None):
    tuned = arr._resolve_tuning(table, nbytes=nbytes, num_blocks=blocks, unroll=unroll)
    return tuned.num_blocks, tuned.unroll


def plan_split(n):
    """Two pulls: pull_norm on the n latent rows, pull_res on the shared vectors."""
    latent_vecs, shared_vecs = n * ROW_VECS, 2 * n * ROW_VECS
    g_norm = geometry(arr.NORM_TUNING, latent_vecs * 16)
    g_res = geometry(arr.RES_TUNING, shared_vecs * 16)
    normed, plain = [], []
    for rank in range(WORLD):
        for row, is_norm in pull_norm_rows(latent_vecs, n, rank, *g_norm):
            assert is_norm
            normed.append(row)
        plain.extend(latent_vecs + v for v in pull_res_vecs(shared_vecs, rank, *g_res))
    return normed, plain


def plan_fused(n, blocks=None, unroll=None):
    """One pull_norm over the buffer as 3n norm-width rows, num_norm_rows = n."""
    num_vecs = 3 * n * ROW_VECS
    g = geometry(arr.NORM_TUNING, num_vecs * 16, blocks, unroll)
    normed, plain = [], []
    for rank in range(WORLD):
        for row, is_norm in pull_norm_rows(num_vecs, n, rank, *g):
            if is_norm:
                normed.append(row)
            else:
                plain.extend(range(row * ROW_VECS, (row + 1) * ROW_VECS))
    return normed, plain


def mc_reduce(per_rank):
    """The multicast reduce load: fp32 accumulation over the ranks, bf16 result."""
    return per_rank.float().sum(0).to(torch.bfloat16)


def norm_row(reduced_row, weight, eps):
    """The pull_norm epilogue on one reduced row: a function of the row only."""
    x = reduced_row.float()
    per_thread = (x * x).view(ROW_VECS, 8).sum(1)  # 448 threads, one vector each
    total = per_thread.view(ROW_VECS // WARP, WARP).sum(1).sum()
    factor = torch.rsqrt(total / NORM_DIM + eps)
    return (x * factor * weight.float()).to(torch.bfloat16)


def execute(plan, per_rank, weight, eps):
    normed, plain = plan
    out = torch.full((per_rank.shape[1],), float("nan"), dtype=torch.bfloat16)
    reduced = mc_reduce(per_rank)
    for row in normed:
        out[row * NORM_DIM : (row + 1) * NORM_DIM] = norm_row(
            reduced[row * NORM_DIM : (row + 1) * NORM_DIM], weight, eps
        )
    for vid in plain:
        out[vid * 8 : (vid + 1) * 8] = reduced[vid * 8 : (vid + 1) * 8]
    return out


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


class TestFusedPullEquivalence(CustomTestCase):
    def check(self, n, plan, what):
        normed, plain = plan
        # exactly the latent rows normed and every shared vector reduced, each once
        self.assertEqual(sorted(normed), list(range(n)), (what, n))
        self.assertEqual(
            sorted(plain), list(range(n * ROW_VECS, 3 * n * ROW_VECS)), (what, n)
        )

    def test_kernel_source_is_what_was_transcribed(self):
        with open(CUH) as f:
            src = f.read()
        norm_at = src.index("void all_reduce_pull_norm_kernel(")
        norm = src[norm_at : src.index("inline auto choose_block_size")]
        res = src[src.index("pull_reduce_pass(uint32_t& vid") : norm_at]
        self.assertIn("inline constexpr uint32_t kPullBlockSize = 512;", src)
        guard = "if (row_bias + row0 + u < params.num_norm_rows) {"
        for line in (
            "const auto row_bias = avg_rows * r + min(r, rem_rows);",
            "const auto my_rows = avg_rows + (r < rem_rows ? 1 : 0);",
            "for (auto row0 = bx * kUnroll; row0 < my_rows; row0 += row_step) {",
            "const auto vid0 = int64_t(row_bias + row0) * kNormRowVecs + tx;",
            guard,
        ):
            self.assertIn(line, norm, line)
        # a non-normed row is the load then the store, nothing in between: as pull_res
        plain = norm[norm.index("} else {", norm.index(guard)) :].split("\n")[1]
        store = "ptx::st_multimem_16B(vec[u], params.input_mc, vid0 + u * kNormRowVecs)"
        self.assertTrue(plain.strip().startswith(store))
        for line in (
            "for (; vid + (kWidth - 1) * step < num_vecs; vid += kWidth * step) {",
            "const auto vec_bias = int64_t(avg_vecs) * r + min(r, rem_vecs);",
            "const auto step = kPullBlockSize * gridDim.x;",
            # without a residual nothing touches the loaded value before the store
            "if constexpr (kHasResidual) {",
        ):
            self.assertIn(line, res, line)

    def test_every_vector_reduced_once_and_only_the_latent_rows_normed(self):
        for n in (1, 5, 12, 13, 40, 73, 74, 80, 81, 120, 160, 200, 235, 256):
            self.check(n, plan_split(n), "split")
            self.check(n, plan_fused(n), "fused")
        # the fused pull at other launch geometries of the kernel
        for blocks, unroll in ((1, 2), (12, 4), (16, 8), (64, 16)):
            for n in (80, 235):
                self.check(n, plan_fused(n, blocks, unroll), (blocks, unroll))

    def test_split_and_fused_give_the_same_buffer(self):
        g = torch.Generator().manual_seed(20261124)
        eps = 1e-6
        weight = (torch.rand(NORM_DIM, generator=g) + 0.5).to(torch.bfloat16)
        for n in (3, 8, 13, 21):
            per_rank = (torch.randn(WORLD, 3 * n * NORM_DIM, generator=g) * 0.37).to(
                torch.bfloat16
            )
            split = execute(plan_split(n), per_rank, weight, eps)
            fused = execute(plan_fused(n), per_rank, weight, eps)
            self.assertFalse(torch.isnan(split.float()).any())
            self.assertTrue(bits_equal(split, fused), n)
            # and both are the plain definition: all-reduce, then RMSNorm of the n rows
            reduced = mc_reduce(per_rank)
            self.assertTrue(bits_equal(fused[n * NORM_DIM :], reduced[n * NORM_DIM :]))
            lat = reduced[: n * NORM_DIM].view(n, NORM_DIM).float()
            ref = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + eps) * weight
            got = fused[: n * NORM_DIM].view(n, NORM_DIM).float()
            torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
