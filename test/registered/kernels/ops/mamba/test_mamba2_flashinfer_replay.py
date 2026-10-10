"""Isolated validation of the supported FlashInfer replay contract.

No model or server is loaded. JSON measurements distinguish numerical error
from exact state/buffer lifetime invariants. Run directly with Python on CUDA.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import unittest
from pathlib import Path

import torch
import triton
from flashinfer.mamba import checkpointing_ssu, selective_state_update
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DIAGNOSTIC = False


def error_summary(actual, expected):
    a, b = actual.float(), expected.float()
    delta = a - b
    scale = b.square().mean().sqrt().clamp_min(1e-20)
    worst = delta.abs().reshape(-1).argmax()
    eps = torch.finfo(torch.bfloat16).eps
    return {
        "rms_relative": (delta.square().mean().sqrt() / scale).item(),
        "max_relative_to_rms": (delta.abs().max() / scale).item(),
        "max_absolute": delta.abs().max().item(),
        "finite": bool(torch.isfinite(a).all()),
        "worst_reference": b.reshape(-1)[worst].item(),
        "worst_actual": a.reshape(-1)[worst].item(),
        "max_bf16_scaled_error": (delta.abs() / (eps * (b.abs() + scale))).max().item(),
    }


def check_numerics(label, actual, expected, **metadata):
    stats = error_summary(actual, expected)
    print(json.dumps({"comparison": label, **metadata, **stats}), flush=True)
    # Tensor-core BF16 arithmetic is not bitwise equal to the scalar recurrence.
    # RMS <= one BF16 epsilon. Peak error uses four BF16 eps times
    # (abs(reference) + RMS), accounting for scaled-B/output rounding and
    # cancellation. The original magnitude-independent peak screen rejected
    # a 0.125 difference on a -44.4375 value; preserve it in the JSON report.
    # This is a kernel screening ceiling, not a model-accuracy qualification.
    eps = torch.finfo(torch.bfloat16).eps
    assert stats["finite"], label
    if DIAGNOSTIC:
        return
    assert stats["rms_relative"] <= eps, (label, stats)
    assert stats["max_bf16_scaled_error"] <= 4, (label, stats)


class FlashInferCase:
    def __init__(self, batch=5, width=4, window=None, rounding=False):
        self.batch, self.width = batch, width
        self.window = window or width
        self.h, self.p, self.g, self.n = 128, 64, 8, 128
        h, p, g, n = self.h, self.p, self.g, self.n
        self.capacity = 2 * batch + 3
        k, w = self.capacity, self.window
        dtype, device = torch.bfloat16, "cuda"
        self.initial = torch.randn(k, h, p, n, device=device, dtype=torch.float16)
        self.state = self.initial.clone()
        self.reference = self.initial.clone()
        self.slots = torch.arange(batch, device=device, dtype=torch.int32) * 2 + 1
        self.rows = torch.arange(batch, device=device, dtype=torch.int32)
        self.ring_start = torch.zeros(k, device=device, dtype=torch.int32)
        self.pending = torch.zeros_like(self.ring_start)
        self.old_x = torch.zeros(k, h, w + width, p, device=device, dtype=dtype)
        self.old_B = torch.zeros(k, g, w + width, n, device=device, dtype=dtype)
        self.old_dt = torch.zeros(k, h, w + width, device=device)
        self.A_base = -torch.rand(h, device=device) - 0.5
        self.A = self.A_base[:, None, None].expand(h, p, n)
        self.bias_base = torch.randn(h, device=device, dtype=dtype) - 2
        self.bias = self.bias_base[:, None].expand(h, p)
        self.D = torch.randn(h, device=device, dtype=dtype)[:, None].expand(h, p)
        self.seed = (
            torch.tensor([81723], device=device, dtype=torch.int64)
            if rounding
            else None
        )
        self.snapshots = torch.empty(
            batch, width, h, p, n, device=device, dtype=torch.float16
        )
        self.out = torch.empty(batch, width, h, p, device=device, dtype=dtype)
        self.ref_out = torch.empty_like(self.out)
        # Noncontiguous inputs like the mixer projection/split views.
        self.packed = torch.empty(
            batch, width, h * p + 2 * g * n + h, device=device, dtype=dtype
        )
        self.x = self.packed[..., : h * p].view(batch, width, h, p)
        self.B = self.packed[..., h * p : h * p + g * n].view(batch, width, g, n)
        self.C = self.packed[..., h * p + g * n : h * p + 2 * g * n].view(
            batch, width, g, n
        )
        self.raw_dt = self.packed[..., -h:]
        self.dt = self.raw_dt[..., None].expand(batch, width, h, p)
        self.randomize()

    def randomize(self):
        self.packed.normal_()

    def verify(self, replay):
        shared = dict(
            x=self.x,
            dt=self.dt,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            dt_bias=self.bias,
            dt_softplus=True,
            state_batch_indices=self.slots,
            rand_seed=self.seed,
            philox_rounds=5,
        )
        if replay:
            checkpointing_ssu(
                self.state,
                self.old_x,
                self.old_B,
                self.old_dt,
                self.ring_start,
                self.pending,
                out=self.out,
                algorithm="monolith",
                **shared,
            )
        else:
            selective_state_update(
                self.reference,
                out=self.ref_out,
                disable_state_update=True,
                intermediate_states_buffer=self.snapshots,
                intermediate_state_indices=self.rows,
                cache_steps=self.width,
                **shared,
            )

    def check_records(self):
        valid = self.slots >= 0
        slots = self.slots[valid].long()
        torch.testing.assert_close(
            self.old_x[slots, :, : self.width],
            self.x[valid].transpose(1, 2),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            self.old_B[slots, :, : self.width],
            self.B[valid].transpose(1, 2),
            rtol=0,
            atol=0,
        )
        dt = torch.nn.functional.softplus(
            self.raw_dt[valid].float() + self.bias_base.float()
        )
        for name, actual, expected in (
            (
                "processed_dt",
                self.old_dt[slots, :, : self.width],
                dt.transpose(1, 2),
            ),
        ):
            if DIAGNOSTIC:
                print(
                    json.dumps({"comparison": name, **error_summary(actual, expected)}),
                    flush=True,
                )
            else:
                # CUDA's fast log/exp softplus is not PyTorch's log1p path.
                # Charge four FP32 eps per term and accumulation length;
                # keep copied x/B byte-exact and report numerical errors.
                terms = 1 if name == "processed_dt" else self.width
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-6,
                    atol=4 * torch.finfo(torch.float32).eps * terms,
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlashInferReplayContract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.mamba.mamba2_spec_replay import checkpointing_kernel

        version = importlib.metadata.version("flashinfer-python")
        # Exercise the same compatibility gate as memory-pool construction.
        checkpointing_kernel()
        print(
            json.dumps({"flashinfer": version, "gpu": torch.cuda.get_device_name()}),
            flush=True,
        )

    @torch.inference_mode()
    def test_verify_without_state_writes(self):
        for rounding in (False, True):
            torch.manual_seed(71)
            c = FlashInferCase(rounding=rounding)
            c.verify(False)
            c.verify(True)
            torch.testing.assert_close(c.state, c.initial, rtol=0, atol=0)
            torch.testing.assert_close(c.reference, c.initial, rtol=0, atol=0)
            c.check_records()
            check_numerics("no_history_outputs", c.out, c.ref_out, rounding=rounding)

    @torch.inference_mode()
    def test_native_replay_each_accepted_length(self):
        for rounding in (False, True):
            torch.manual_seed(73)
            c = FlashInferCase(rounding=rounding)
            c.verify(False)
            c.verify(True)
            counts = torch.arange(c.batch, device="cuda", dtype=torch.int32)
            # K=0 leaves the initial state; K=1..4 chooses the matching snapshot.
            expected = c.initial.clone()
            for row in range(c.batch):
                if row:
                    expected[c.slots[row]] = c.snapshots[row, row - 1]
            c.reference.copy_(expected)
            c.pending[c.slots.long()] = counts
            c.randomize()
            c.verify(False)
            c.verify(True)
            check_numerics(
                "native_accepted_states",
                c.state[c.slots.long()],
                expected[c.slots.long()],
                rounding=rounding,
            )
            check_numerics("native_next_outputs", c.out, c.ref_out, rounding=rounding)
            # No history means no flush, so row zero must remain exactly intact.
            torch.testing.assert_close(
                c.state[c.slots[0]], c.initial[c.slots[0]], rtol=0, atol=0
            )

    @torch.inference_mode()
    def test_graph_padding_and_reuse(self):
        torch.manual_seed(79)
        c = FlashInferCase(rounding=True)
        c.slots[-1] = -1
        c.out.fill_(41)
        c.verify(True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            c.verify(True)
        for step in range(32):
            c.randomize()
            c.slots[:-1].copy_(torch.roll(c.slots[:-1], 1))
            c.verify(False)
            graph.replay()
            torch.testing.assert_close(c.state, c.initial, rtol=0, atol=0)
            torch.testing.assert_close(
                c.out[-1], torch.full_like(c.out[-1], 41), rtol=0, atol=0
            )
            c.check_records()
            if step in (0, 31):
                check_numerics(
                    "graph_reordered_outputs", c.out[:-1], c.ref_out[:-1], step=step
                )


@torch.inference_mode()
def benchmark():
    for batch in (16, 32, 64):
        c = FlashInferCase(batch=batch, rounding=True)
        c.verify(False)
        c.verify(True)
        baseline = triton.testing.do_bench_cudagraph(lambda: c.verify(False), rep=200)
        replay = triton.testing.do_bench_cudagraph(lambda: c.verify(True), rep=200)
        print(
            json.dumps(
                {
                    "benchmark": "verify_only_per_layer",
                    "batch": batch,
                    "baseline_ms": baseline,
                    "checkpointing_ms": replay,
                    "note": "excludes materialization; not end-to-end speedup",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--all-stages", action="store_true")
    options, remaining = parser.parse_known_args()
    DIAGNOSTIC = options.diagnostic
    if DIAGNOSTIC:
        print(
            "DIAGNOSTIC ONLY: numerical gates reported but not enforced; not a validation pass.",
            flush=True,
        )
    if options.all_stages:
        # Sequential gates in one container invocation: do not pay Pyxis startup
        # again for each stage, and never run a later stage after a failed gate.
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            TestFlashInferReplayContract
        )
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        from test_mamba2_flashinfer_materialize import TestFlashInferMaterialization

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            TestFlashInferMaterialization
        )
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        from test_mamba2_spec_replay import TestMamba2SpecReplay

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMamba2SpecReplay)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        sizing_path = (
            Path(__file__).resolve().parents[3]
            / "unit/mem_cache/test_mamba2_replay_sizing.py"
        )
        spec = importlib.util.spec_from_file_location(
            "replay_sizing_tests", sizing_path
        )
        sizing_tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sizing_tests)
        suite = unittest.defaultTestLoader.loadTestsFromModule(sizing_tests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        benchmark()
        from bench_mamba2_flashinfer_pipeline import benchmark as pipeline_benchmark

        pipeline_benchmark()
    elif options.benchmark:
        benchmark()
    else:
        unittest.main(argv=[__file__, *remaining])
