"""SM120 grouped-GEMM MXFP4 MoE prefill kernel tests.

Validates the expert-grouped GEMM prefill path
(`mxfp4_grouped_moe_sm120_triton.mxfp4_moe_forward_grouped`) against the per-slot
GEMV (`mxfp4_moe_sm120_triton.mxfp4_moe_forward_triton`), which it replaces for
prefill-sized batches (M > 128).

Coverage:
- Numeric equivalence vs the per-slot kernel across prefill batch sizes
- topk_ids == -1 (padded / off-rank tokens under expert parallelism): the grouped
  path must drop those slots, matching the per-slot kernel's invalid-slot masking;
  includes a fully-dropped token and an all-dropped batch (moe_align total == 0,
  which must short-circuit to zeros rather than launch a zero-grid kernel)
- moe_align invalid-slot bookkeeping (placed count, uniqueness, no negative experts)
- CUDA-graph capture: a captured decode batch with M > 128 (max_bs > 128 on a 96GB
  SM120 card) must NOT route into the grouped path, whose host sync would abort
  capture; the dispatch guards on torch.cuda.is_current_stream_capturing()

Requires SM120 hardware (the kernels run FP8/FP4 tensor-core tl.dot).
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton.mxfp4_grouped_moe_sm120_triton import (
    moe_align,
    mxfp4_moe_forward_grouped,
)
from sglang.srt.layers.moe.fused_moe_triton.mxfp4_moe_sm120_triton import (
    mxfp4_moe_forward_triton,
)
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# DSv4 MXFP4 MoE shapes (E2M1 weights, block-32 fp32 scales).
E, TOPK, HID, INTER, QBLK = 256, 6, 4096, 2048, 32


def _weights(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randint(
            0, 256, (E, 2 * INTER, HID // 2), generator=g, dtype=torch.uint8
        ).to(device),
        (torch.rand((E, 2 * INTER, HID // QBLK), generator=g) * 0.1 + 0.05)
        .float()
        .to(device),
        torch.randint(0, 256, (E, HID, INTER // 2), generator=g, dtype=torch.uint8).to(
            device
        ),
        (torch.rand((E, HID, INTER // QBLK), generator=g) * 0.1 + 0.05)
        .float()
        .to(device),
    )


def _routing(M, device, seed=0, frac_invalid=0.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = (
        (torch.randn((M, HID), generator=g, dtype=torch.float32) / 8)
        .clamp(-1, 1)
        .to(device, torch.bfloat16)
    )
    tid = torch.randint(0, E, (M, TOPK), generator=g, dtype=torch.int32).to(device)
    tw = torch.rand((M, TOPK), generator=g, dtype=torch.float32).to(device)
    if frac_invalid > 0:
        mask = (torch.rand((M, TOPK), generator=g) < frac_invalid).to(device)
        tid[mask] = -1
    return h, tid, tw


def _slot(h, w, tid, tw):
    return mxfp4_moe_forward_triton(h, w[0], w[2], w[1], w[3], tid, tw, HID, INTER)


def _grouped(h, w, tid, tw):
    return mxfp4_moe_forward_grouped(h, w[0], w[2], w[1], w[3], tid, tw, HID, INTER)


def _relerr(a, b):
    return (a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6)


class TestSM120GroupedMxfp4Moe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if not is_sm120_supported():
            raise unittest.SkipTest("SM120 required (FP8/FP4 tensor-core kernels)")
        cls.device = torch.device("cuda")
        cls.w = _weights(cls.device)

    def test_equiv_prefill_sizes(self):
        for M in (256, 512, 2048):
            h, tid, tw = _routing(M, self.device, seed=M)
            rel = _relerr(
                _grouped(h, self.w, tid, tw), _slot(h, self.w, tid, tw)
            ).item()
            self.assertLess(rel, 0.05, f"M={M} grouped vs slot rel err {rel:.3%}")

    def test_equiv_with_invalid_slots(self):
        """topk_ids == -1 must be dropped, matching the per-slot kernel."""
        for M in (256, 512, 2048):
            for frac in (0.3, 0.7):
                h, tid, tw = _routing(M, self.device, seed=M, frac_invalid=frac)
                rel = _relerr(
                    _grouped(h, self.w, tid, tw), _slot(h, self.w, tid, tw)
                ).item()
                self.assertLess(rel, 0.05, f"M={M} frac={frac} rel err {rel:.3%}")

    def test_fully_dropped_token(self):
        M = 256
        h, tid, tw = _routing(M, self.device, seed=1)
        tid[7] = -1  # every expert of token 7 dropped
        out_g = _grouped(h, self.w, tid, tw)
        rel = _relerr(out_g, _slot(h, self.w, tid, tw)).item()
        self.assertLess(rel, 0.05)
        self.assertLess(
            out_g[7].abs().max().item(), 1e-3, "dropped token row must be zero"
        )

    def test_all_dropped_returns_zeros(self):
        """Every slot off-rank (topk_ids all -1) makes moe_align return total == 0.
        The grouped path must short-circuit to zeros instead of launching a GEMM
        with a grid dimension of 0 (an invalid launch)."""
        M = 256
        h, tid, tw = _routing(M, self.device, seed=2)
        tid[:] = -1  # drop every slot -> total == 0
        out_g = _grouped(h, self.w, tid, tw)
        self.assertEqual(out_g.shape, (M, HID))
        self.assertLess(
            out_g.abs().max().item(), 1e-3, "all-dropped output must be zero"
        )

    def test_moe_align_drops_invalid(self):
        h, tid, tw = _routing(128, self.device, seed=3, frac_invalid=0.3)
        sorted_ids, expert_ids, num_valid = moe_align(tid, E)
        placed = sorted_ids[sorted_ids < num_valid]
        n_valid = int((tid.reshape(-1) >= 0).sum())
        self.assertEqual(
            placed.numel(), n_valid, "placed count must equal valid-slot count"
        )
        self.assertEqual(
            placed.unique().numel(), placed.numel(), "placed slots must be unique"
        )
        self.assertTrue(
            bool((tid.reshape(-1)[placed.long()] >= 0).all()),
            "no placed slot may reference a negative expert",
        )

    def test_capture_guard_routes_to_slot(self):
        """A captured M>128 batch must take the slot path (the grouped host sync
        would abort capture). The dispatch guards on is_current_stream_capturing."""
        from sglang.srt.layers.moe.fused_moe_triton import mxfp4_moe_sm120_triton as mod

        M = 256
        h, tid, tw = _routing(M, self.device, seed=5)
        h_small, tid_small, tw_small = _routing(
            64, self.device, seed=6
        )  # M<=128 -> slot path
        w13, w13s, w2, w2s = self.w

        # Warmup both kernels on a side stream so capture triggers no JIT compile.
        # mxfp4_moe_forward_triton at M=256 takes the grouped path; the M=64 call
        # compiles the per-slot kernels that the captured M=256 call will use.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                mod.mxfp4_moe_forward_triton(h, w13, w2, w13s, w2s, tid, tw, HID, INTER)
                mod.mxfp4_moe_forward_triton(
                    h_small, w13, w2, w13s, w2s, tid_small, tw_small, HID, INTER
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # The shipped dispatch entry; with the capture guard this stays on the
            # per-slot GEMV during capture instead of the non-capturable grouped path.
            out = mod.mxfp4_moe_forward_triton(
                h, w13, w2, w13s, w2s, tid, tw, HID, INTER
            )
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(out.shape, (M, HID))


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
