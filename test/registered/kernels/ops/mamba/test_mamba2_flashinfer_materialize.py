"""Validate eager materialization from pinned FlashInfer compact records."""

import argparse
import unittest

import test_mamba2_flashinfer_replay as replay_tests
import torch
from sglang.kernels.ops.mamba.flashinfer_replay_materialize import (
    make_replay_pointer_table,
    materialize_flashinfer_mamba2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
from test_mamba2_flashinfer_replay import FlashInferCase, check_numerics


class TestFlashInferMaterialization(CustomTestCase):
    @torch.inference_mode()
    def test_invalid_endpoints_do_not_write(self):
        # Mask invalid lengths before the native kernel (its GPU metadata is
        # unchecked), without letting a padding row hide later valid rows.
        c = FlashInferCase()
        c.verify(True)
        last = torch.tensor([-1, 4, -7, 99, -1], device="cuda", dtype=torch.int32)
        c.slots[-1] = c.capacity
        self.commit(c, last, torch.zeros_like(last), torch.zeros_like(last))
        torch.testing.assert_close(c.state, c.initial, rtol=0, atol=0)

    def commit(self, c, last, tracks=None, steps=None, seeds=None):
        if not hasattr(c, "pointer_table"):
            c.pointer_table = make_replay_pointer_table(
                c.state.unsqueeze(0),
                c.old_x.unsqueeze(0),
                c.old_B.unsqueeze(0),
                c.old_dt.unsqueeze(0),
                c.A_base.unsqueeze(0),
            )
        materialize_flashinfer_mamba2(
            c.state.unsqueeze(0),
            c.old_x.unsqueeze(0),
            c.old_B.unsqueeze(0),
            c.old_dt.unsqueeze(0),
            c.A_base.unsqueeze(0),
            c.pointer_table,
            c.slots,
            last,
            tracks,
            steps,
            seed=seeds,
            philox_rounds=5 if seeds is not None else 0,
        )

    @torch.inference_mode()
    def test_every_endpoint_and_tracking(self):
        for rounding in (False, True):
            torch.manual_seed(137)
            c = FlashInferCase(rounding=rounding)
            c.verify(False)
            c.verify(True)
            last = torch.arange(c.batch, device="cuda", dtype=torch.int32) - 1
            tracks = c.slots + 1
            for position in ("first", "middle", "last", "none"):
                c.state.copy_(c.initial)
                steps = {
                    "first": torch.where(last >= 0, 0, -1),
                    "middle": last // 2,
                    "last": last.clone(),
                    "none": torch.full_like(last, -1),
                }[position]
                self.commit(c, last, tracks, steps, c.seed)
                expected = c.initial.clone()
                for row in range(c.batch):
                    if last[row] >= 0:
                        expected[c.slots[row]] = c.snapshots[row, last[row]]
                    if steps[row] >= 0:
                        expected[tracks[row]] = c.snapshots[row, steps[row]]
                touched = torch.cat((c.slots[1:], tracks[1:])).long()
                check_numerics(
                    "materialize_endpoints",
                    c.state[touched],
                    expected[touched],
                    rounding=rounding,
                    position=position,
                )
                untouched = torch.ones(c.capacity, device="cuda", dtype=torch.bool)
                untouched[c.slots[1:].long()] = False
                if position != "none":
                    untouched[tracks[1:].long()] = False
                torch.testing.assert_close(
                    c.state[untouched], c.initial[untouched], rtol=0, atol=0
                )
                if position == "last":
                    torch.testing.assert_close(
                        c.state[c.slots[1:].long()],
                        c.state[tracks[1:].long()],
                        rtol=0,
                        atol=0,
                    )

    @torch.inference_mode()
    def test_materialize_matches_native_flush(self):
        torch.manual_seed(149)
        c = FlashInferCase(rounding=False)
        c.verify(True)
        last = torch.arange(c.batch, device="cuda", dtype=torch.int32) - 1
        self.commit(c, last)
        materialized = c.state.clone()
        c.state.copy_(c.initial)
        c.pending[c.slots.long()] = last + 1
        c.randomize()
        c.verify(True)
        # Both use BF16 scaled-B operands; require much tighter parity here
        # than against scalar verification. Different exp/MMA reduction orders
        # permit low FP32 error and FP16 rounding ties, not BF16-level drift.
        torch.testing.assert_close(c.state, materialized, rtol=1e-3, atol=5e-4)
        check_numerics("materialize_vs_native_flush", materialized, c.state)

    @torch.inference_mode()
    def test_repeated_windows_and_graph(self):
        for rounding in (False, True):
            torch.manual_seed(151)
            c = FlashInferCase(rounding=rounding)
            c.slots[-1] = -1
            last = torch.tensor([0, 1, 2, 3, -1], device="cuda", dtype=torch.int32)
            tracks = torch.tensor([2, 4, 6, 8, -1], device="cuda", dtype=torch.int32)
            steps = torch.minimum(last, torch.ones_like(last))
            c.verify(True)
            self.commit(c, last, tracks, steps, c.seed)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                c.verify(True)
                self.commit(c, last, tracks, steps, c.seed)
            c.state.copy_(c.initial)
            for iteration in range(128):
                c.randomize()
                if c.seed is not None:
                    c.seed.add_(1)
                last[:4].copy_(torch.roll(last[:4], 1))
                steps.copy_(torch.minimum(last, torch.ones_like(last)))
                c.verify(False)
                graph.replay()
                for row in range(4):
                    c.reference[c.slots[row]] = c.snapshots[row, last[row]]
                    c.reference[tracks[row]] = c.snapshots[row, steps[row]]
                if iteration in (0, 15, 31, 47, 63, 64, 95, 127):
                    check_numerics(
                        "repeated_outputs",
                        c.out[:-1],
                        c.ref_out[:-1],
                        iteration=iteration,
                        rounding=rounding,
                    )
                    touched = torch.cat((c.slots[:-1], tracks[:-1])).long()
                    check_numerics(
                        "repeated_states",
                        c.state[touched],
                        c.reference[touched],
                        iteration=iteration,
                        rounding=rounding,
                    )
                if iteration == 63:
                    # Simulate a new request reusing old physical state/record
                    # slots; overwritten record positions must be sufficient.
                    c.state.copy_(c.initial)
                    c.reference.copy_(c.initial)
            torch.testing.assert_close(
                c.old_x[-1], torch.zeros_like(c.old_x[-1]), rtol=0, atol=0
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.diagnostic:
        replay_tests.DIAGNOSTIC = True
        print(
            "DIAGNOSTIC ONLY: numerical gates reported, not enforced; not a validation pass.",
            flush=True,
        )
    unittest.main(argv=[__file__, *remaining])
