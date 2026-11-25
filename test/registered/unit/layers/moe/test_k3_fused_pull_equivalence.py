"""One pull over the K3 [latent | shared] buffer (SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER=fused) against the two pulls it replaces
(all_reduce_pull_res on the shared part, all_reduce_pull_norm on the latent), without a GPU.

The index arithmetic of the two kernels of csrc/kimi_k3/comm/ar_fusion.cuh is transcribed below (rank shards, block / thread
strides, the unroll cascade, the normed-row predicate) and run for every rank. Checked: both ways reduce every 16-byte vector
of the buffer exactly once, norm exactly the latent rows, and give bit-identical buffers under the model that a multicast
reduce load returns the same value whichever rank issues it (the load sums the same eight addresses with fp32 accumulation).
The source lines the transcription rests on are pinned, so a kernel change fails here instead of going unnoticed.
What this cannot show: the hardware reduce itself. That part is the engine's determinism set against the two-pull arm."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest

import torch

from sglang.kernels.ops.communication import all_reduce_residual as arr
from sglang.srt.layers.communication import k3_ar_fusion

WORLD = 8
NORM_DIM = k3_ar_fusion.NORM_DIM  # 3584
ROW_VECS = NORM_DIM // 8  # 448 16-byte vectors per norm-width row
PULL_BLOCK = 512
WARP = 32
CUH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(k3_ar_fusion.__file__)))
        )
    ),
    "kernels",
    "jit",
    "csrc",
    "kimi_k3",
    "comm",
    "ar_fusion.cuh",
)


def pull_res_vecs(num_vecs, rank, grid, unroll, world=WORLD):
    """Vector ids (within the reduced tensor) that all_reduce_pull_res_kernel<unroll> loads and stores on `rank`."""
    avg, rem = divmod(num_vecs, world)
    bias = avg * rank + min(rank, rem)
    mine = avg + (1 if rank < rem else 0)
    step = PULL_BLOCK * grid
    out = []
    for start in range(
        min(step, mine)
    ):  # start = bx * 512 + tx; starts beyond the shard do nothing
        vid, width = start, unroll
        while width >= 1:  # pull_reduce_pass<kWidth> then <kWidth / 2> ... <1>
            while vid + (width - 1) * step < mine:
                out.extend(bias + vid + u * step for u in range(width))
                vid += width * step
            width //= 2
    return out


def pull_norm_rows(num_vecs, num_norm_rows, rank, grid, unroll, world=WORLD):
    """(row, normed) pairs that all_reduce_pull_norm_kernel<unroll> handles on `rank`; every row is 448 threads x one vector."""
    assert num_vecs % ROW_VECS == 0
    avg, rem = divmod(num_vecs // ROW_VECS, world)
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
    """Two pulls: latent rows [0, n) normed by pull_norm on the latent tensor; shared vectors by pull_res on the shared tensor."""
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
    """One pull_norm over the whole buffer viewed as 3n rows of the norm width, num_norm_rows = n."""
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
    """The multicast reduce load: fp32 accumulation over the ranks, bf16 result; the same for every issuing rank."""
    return per_rank.float().sum(0).to(torch.bfloat16)


def norm_row(reduced_row, weight, eps):
    """The epilogue of all_reduce_pull_norm_kernel on one reduced row: per-warp sums of squares, their total over the 14
    warps, rsqrt, scale by the weight, bf16. A function of the row only: no rank, block or unroll index enters."""
    x = reduced_row.float()
    per_thread = (
        (x * x).view(ROW_VECS, 8).sum(1)
    )  # 448 threads, one 16-byte vector each
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


class TestKernelSourceIsWhatWasTranscribed(unittest.TestCase):
    def test_pinned_lines(self):
        with open(CUH) as f:
            src = f.read()
        norm = src[
            src.index("void all_reduce_pull_norm_kernel(") : src.index(
                "inline auto choose_block_size"
            )
        ]
        res = src[
            src.index("pull_reduce_pass(uint32_t& vid") : src.index(
                "void all_reduce_pull_norm_kernel("
            )
        ]
        for line in (
            "constexpr uint32_t kNormDim = 3584;",
            "inline constexpr uint32_t kPullBlockSize = 512;",
        ):
            self.assertIn(line, src)
        for line in (
            "const auto num_rows = params.num_vecs / kNormRowVecs;",
            "const auto row_bias = avg_rows * r + min(r, rem_rows);",
            "const auto my_rows = avg_rows + (r < rem_rows ? 1 : 0);",
            "const auto row_step = gridDim.x * kUnroll;",
            "for (auto row0 = bx * kUnroll; row0 < my_rows; row0 += row_step) {",
            "const auto vid0 = int64_t(row_bias + row0) * kNormRowVecs + tx;",
            "if (u < cnt) ptx::ld_multimem_16B(vec[u], params.input_mc, vid0 + u * kNormRowVecs);",
            "if (row_bias + row0 + u < params.num_norm_rows) {",
            "const auto norm_factor = math::rsqrt(total / kNormDim + params.norm_eps);",
        ):
            self.assertIn(line, norm, line)
        # a non-normed row is the load followed by the store, nothing in between: what pull_res does to the shared output
        plain = norm[
            norm.index(
                "} else {",
                norm.index("if (row_bias + row0 + u < params.num_norm_rows) {"),
            ) :
        ]
        self.assertTrue(
            plain.split("\n")[1]
            .strip()
            .startswith(
                "ptx::st_multimem_16B(vec[u], params.input_mc, vid0 + u * kNormRowVecs);"
            )
        )
        self.assertEqual(
            norm.count("barrier.arrive_relaxed();"), 1
        )  # one entry barrier
        self.assertEqual(norm.count("barrier.arrive_rel_acq();"), 1)  # one exit barrier
        for line in (
            "for (; vid + (kWidth - 1) * step < num_vecs; vid += kWidth * step) {",
            "device::ptx::ld_multimem_16B(vec[u], mc_ptr, vid + u * step);",
            "device::ptx::st_multimem_16B(vec[u], mc_ptr, vid + u * step);",
            "const auto vec_bias = int64_t(avg_vecs) * r + min(r, rem_vecs);",
            "const auto step = kPullBlockSize * gridDim.x;",
        ):
            self.assertIn(line, res, line)
        # without a residual nothing touches the loaded value before the store
        self.assertIn("if constexpr (kHasResidual) {", res)

    def test_python_side(self):
        self.assertEqual(NORM_DIM, 3584)
        # the K3 buffer at the widths the kernel takes, as the fused call sees it
        self.assertEqual(geometry(arr.NORM_TUNING, 80 * 3 * NORM_DIM * 2), (12, 2))
        self.assertEqual(geometry(arr.NORM_TUNING, 160 * 3 * NORM_DIM * 2), (24, 2))
        self.assertEqual(geometry(arr.NORM_TUNING, 80 * NORM_DIM * 2), (12, 2))
        self.assertEqual(geometry(arr.RES_TUNING, 80 * 2 * NORM_DIM * 2), (16, 8))
        self.assertEqual(
            geometry(arr.NORM_TUNING, 80 * 3 * NORM_DIM * 2, 16, 4), (16, 4)
        )


class TestCoverage(unittest.TestCase):
    ROWS = (1, 5, 12, 13, 40, 73, 74, 80, 81, 120, 160, 200, 235, 256)

    def check(self, n, plan, what):
        normed, plain = plan
        self.assertEqual(
            sorted(normed), list(range(n)), (what, n)
        )  # exactly the latent rows, each once
        self.assertEqual(
            sorted(plain), list(range(n * ROW_VECS, 3 * n * ROW_VECS)), (what, n)
        )  # every shared vector once

    def test_both_ways_reduce_every_vector_once_and_norm_the_latent_rows(self):
        for n in self.ROWS:
            self.check(n, plan_split(n), "split")
            self.check(n, plan_fused(n), "fused")

    def test_fused_with_geometry_overrides(self):
        for n in (80, 160, 235):
            for blocks in (1, 8, 12, 16, 24, 32, 64):
                for unroll in (2, 4, 8, 16):
                    self.check(
                        n, plan_fused(n, blocks, unroll), ("fused", blocks, unroll)
                    )

    def test_work_per_rank(self):
        # fused: every rank handles 3n/8 rows; the normed rows sit on the first ranks (rank 0-2 at n = 80)
        n = 80
        num_vecs = 3 * n * ROW_VECS
        g = geometry(arr.NORM_TUNING, num_vecs * 16)
        rows = [pull_norm_rows(num_vecs, n, rank, *g) for rank in range(WORLD)]
        self.assertEqual([len(r) for r in rows], [30] * WORLD)
        self.assertEqual(
            [sum(is_norm for _, is_norm in r) for r in rows],
            [30, 30, 20, 0, 0, 0, 0, 0],
        )


class TestBitIdentical(unittest.TestCase):
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
            tuned = execute(plan_fused(n, 16, 4), per_rank, weight, eps)
            self.assertFalse(torch.isnan(split.float()).any())
            self.assertTrue(
                torch.equal(split.view(torch.int16), fused.view(torch.int16)), n
            )
            self.assertTrue(
                torch.equal(split.view(torch.int16), tuned.view(torch.int16)), n
            )
            # and both are the plain definition: all-reduce, then RMSNorm over the first n rows
            reduced = mc_reduce(per_rank)
            lat = reduced[: n * NORM_DIM].view(n, NORM_DIM).float()
            ref = (
                lat
                * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + eps)
                * weight.float()
            ).to(torch.bfloat16)
            self.assertTrue(
                torch.equal(
                    fused[n * NORM_DIM :].view(torch.int16),
                    reduced[n * NORM_DIM :].view(torch.int16),
                )
            )
            torch.testing.assert_close(
                fused[: n * NORM_DIM].view(n, NORM_DIM).float(),
                ref.float(),
                atol=2e-2,
                rtol=2e-2,
            )


if __name__ == "__main__":
    unittest.main()
