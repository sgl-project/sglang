"""Numerics of the SM90 mixed-input MXFP4 x BF16 grouped GEMM, and its metadata.

The collective reads three independently derived layouts -- packed E2M1 nibbles
straight from the checkpoint, ``Array<uint8_t, 4>`` scales transposed to
``[E, K/128, N*4]`` with ``+126`` folded in, and the transposed A/B strides that
make it compute ``D^T = B^T A^T`` -- plus a per-expert row range taken from
``expert_offsets`` / ``problem_sizes``. Getting any one wrong shifts results by a
power of two, scrambles rows, or bleeds one expert's rows into the next, so this
differential test pins all of them against a dequantized bf16 matmul.

Tolerance is set by bf16 rounding of ``q * scale`` on the reference side plus
fp32-accumulator order, not by any quantization error: both sides see exactly
the same numbers.

``mxfp4_moe_grouped_metadata`` produces those row ranges, and is covered here
because it is the same subsystem and the same CUDA fixture.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

NUM_EXPERTS = 4
# Both dims must be multiples of the 128-wide K tile / 128-wide output tile.
N = 256
K = 256
ROWS_PER_EXPERT = 37

# E2M1 code -> value, in code order (sign bit is the high bit of each nibble).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e2m1_lut(device: str) -> torch.Tensor:
    return torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float32, device=device)


def _dequant_weight(
    b_q: torch.Tensor, b_scales: torch.Tensor, device: str
) -> torch.Tensor:
    """``[E, N, K/2]`` nibbles + ``[E, N, K/32]`` E8M0 bytes -> ``[E, N, K]`` fp32."""
    lut = _e2m1_lut(device)
    low = lut[(b_q & 0xF).long()]
    high = lut[(b_q >> 4).long()]
    values = torch.stack((low, high), dim=-1).reshape(*b_q.shape[:-1], -1)
    scale = torch.exp2(b_scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return values * scale


class TestMxfp4A16MoeMm(CustomTestCase):
    def _run(self, *, rows: list[int]):
        from sglang.kernels.ops.moe.mxfp4_a16_moe_mm import mxfp4_a16_moe_mm
        from sglang.srt.layers.quantization.mxfp4_cutlass_moe import (
            _pack_mxfp4_scales_for_cutlass,
        )

        device = "cuda"
        g = torch.Generator(device=device).manual_seed(0)
        total_rows = sum(rows)

        b_q = torch.randint(
            0,
            256,
            (NUM_EXPERTS, N, K // 2),
            dtype=torch.uint8,
            device=device,
            generator=g,
        )
        # Narrow E8M0 band (2**-2 .. 2**1) keeps the bf16 products well inside range;
        # 128 is also the largest byte the collective's scale decode can carry.
        b_scales = torch.randint(
            125,
            129,
            (NUM_EXPERTS, N, K // 32),
            dtype=torch.uint8,
            device=device,
            generator=g,
        )
        a = torch.randn(
            (total_rows, K), dtype=torch.float32, device=device, generator=g
        ).to(torch.bfloat16)

        offsets = torch.tensor(
            [0] + torch.tensor(rows).cumsum(0).tolist(),
            dtype=torch.int32,
            device=device,
        )
        problem_sizes = torch.tensor(
            [[N, r, K] for r in rows], dtype=torch.int32, device=device
        )

        out = torch.zeros((total_rows, N), dtype=torch.bfloat16, device=device)
        mxfp4_a16_moe_mm(
            out,
            a,
            b_q,
            _pack_mxfp4_scales_for_cutlass(b_scales),
            offsets[:-1],
            problem_sizes,
        )

        weight = _dequant_weight(b_q, b_scales, device).to(torch.bfloat16)
        for e, num_rows in enumerate(rows):
            if num_rows == 0:
                continue
            lo, hi = int(offsets[e]), int(offsets[e + 1])
            want = (a[lo:hi].float() @ weight[e].float().t()).to(torch.bfloat16)
            torch.testing.assert_close(
                out[lo:hi], want, rtol=2e-2, atol=2e-2, msg=f"expert {e}"
            )

    def test_uniform_rows(self):
        self._run(rows=[ROWS_PER_EXPERT] * NUM_EXPERTS)

    def test_ragged_and_empty_experts(self):
        # An empty expert and a ragged tail are the two cases where a wrong
        # offset or problem size silently reads a neighbour's rows.
        self._run(rows=[0, 1, 128 + 5, ROWS_PER_EXPERT])

    def test_token_tile_dispatch(self):
        """Every token-tile width the entry point selects must give the same result.

        The entry dispatches on the average rows per expert to one of four separate
        collective instantiations (16/32/64/128 tokens), and prefill only reaches
        the 128-wide one. The other cases here all land in the 64-wide tile, so a
        tile whose epilogue or predication is wrong outside that width fails
        nothing else.
        """
        for rows_per_expert in (8, 24, 200):
            with self.subTest(rows_per_expert=rows_per_expert):
                self._run(rows=[rows_per_expert] * NUM_EXPERTS)


METADATA_NUM_EXPERTS = 8
# Only carried through to problem_sizes, so any positive pair exercises the packing.
METADATA_N = 384
METADATA_K = 2048


def _run_metadata(topk_ids: torch.Tensor, num_experts: int, n: int, k: int):
    from sglang.kernels.ops.moe.mxfp4_moe_grouped_metadata import (
        mxfp4_moe_grouped_metadata,
    )

    device = topk_ids.device
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    src2dst = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    mxfp4_moe_grouped_metadata(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        src2dst,
        n,
        k,
    )
    return expert_offsets, problem_sizes1, problem_sizes2, src2dst


class TestMxfp4MoeGroupedMetadata(CustomTestCase):
    """The fused replacement for torch.sort + three metadata kernels.

    Nothing downstream can detect a wrong permutation on its own: the GEMM
    consumes whatever ranges it is handed and the post-reorder kernel gathers
    back through the same map, so a scan or a cursor that groups the rows wrong
    shows up only as degraded accuracy.
    """

    def _check(self, topk_ids: torch.Tensor, *, num_experts: int):
        n, k = METADATA_N, METADATA_K
        offsets, sizes1, sizes2, src2dst = _run_metadata(topk_ids, num_experts, n, k)

        flat = topk_ids.flatten().cpu()
        valid = (flat >= 0) & (flat < num_experts)
        counts = torch.bincount(flat[valid].to(torch.int64), minlength=num_experts)
        want_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])

        self.assertEqual(offsets.cpu().tolist(), want_offsets.tolist())
        self.assertEqual(sizes1.cpu().tolist(), [[2 * n, int(c), k] for c in counts])
        self.assertEqual(sizes2.cpu().tolist(), [[k, int(c), n] for c in counts])

        dst = src2dst.cpu()
        self.assertEqual(dst[~valid].tolist(), [-1] * int((~valid).sum()))

        # A bijection onto the rows the GEMM will visit, grouped by expert: every
        # row is written exactly once, and inside its own expert's range.
        assigned = dst[valid].to(torch.int64)
        self.assertEqual(sorted(assigned.tolist()), list(range(int(counts.sum()))))
        owner = flat[valid].to(torch.int64)
        in_range = (assigned >= want_offsets[owner]) & (
            assigned < want_offsets[owner + 1]
        )
        self.assertTrue(bool(in_range.all()))

    def test_grouping_and_problem_sizes(self):
        g = torch.Generator().manual_seed(0)
        topk_ids = torch.randint(
            0, METADATA_NUM_EXPERTS, (17, 3), dtype=torch.int32, generator=g
        )
        # An expert with no rows must still get an offset and a zero-row problem size.
        topk_ids[topk_ids == 5] = 2
        self.assertEqual(int((topk_ids == 5).sum()), 0)
        self._check(topk_ids.cuda(), num_experts=METADATA_NUM_EXPERTS)

    def test_out_of_range_ids_are_dropped(self):
        """Ids outside [0, num_experts) must not consume a row.

        ``num_experts`` is the sentinel the pre/post reorder kernels skip on, so
        an id the histogram counted anyway would hand the GEMM a row that nobody
        gathers back.
        """
        topk_ids = torch.tensor(
            [
                [0, METADATA_NUM_EXPERTS, 3],
                [7, -1, 0],
                [METADATA_NUM_EXPERTS + 100, 3, 3],
            ],
            dtype=torch.int32,
        )
        self._check(topk_ids.cuda(), num_experts=METADATA_NUM_EXPERTS)

    def test_max_expert_width(self):
        """At the documented cap the scan spans the whole CTA.

        One expert per thread means ``num_experts == kMetadataBlock`` is the only
        shape where the block scan carries a count in its last lane.
        """
        from sglang.kernels.ops.moe.mxfp4_moe_grouped_metadata import (
            MXFP4_MOE_FUSED_METADATA_MAX_EXPERTS as max_experts,
        )

        g = torch.Generator().manual_seed(0)
        topk_ids = torch.randint(
            0, max_experts, (64, 8), dtype=torch.int32, generator=g
        )
        # Pin the bottom and top lanes of the scan.
        topk_ids[0, 0] = 0
        topk_ids[0, 1] = max_experts - 1
        self._check(topk_ids.cuda(), num_experts=max_experts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
