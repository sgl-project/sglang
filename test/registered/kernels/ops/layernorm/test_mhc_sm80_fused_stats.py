"""Diagnostic qualification only: existing SM80 mHC stats vs fused Sinkhorn.

No production selector is changed. Run on an otherwise idle SM80 GPU:
  PYTHONPATH=python python test/registered/kernels/ops/layernorm/test_mhc_sm80_fused_stats.py
  ...test_mhc_sm80_fused_stats.py --benchmark
  ...test_mhc_sm80_fused_stats.py --benchmark --rows 1 4 32

The deployed V4.1 geometry is HC4, K20480, MIX24, 20 Sinkhorn iterations.
Both configured epsilons default to 1e-6 (deepseek_v4.py config defaults).
This path uses the existing batch-invariant tf32x3 partial projection, NOT
mhc_pre's unrelated T2049 TileLang/TF32 path. Numerical tolerance is fixed
before execution: atol=rtol=1e-5. Bitwise parity is reported first and can be
required with --require-bitwise; passing numerical tests does not imply it.
"""

from __future__ import annotations

import argparse
import json
import sys
import unittest

import torch

HC, K, MIX, ITERS = 4, 20480, 24, 20
RMS_EPS, HC_EPS = 1e-6, 1e-6
ATOL, RTOL = 1e-5, 1e-5
REQUIRE_BITWISE = False


def supported():
    return (
        torch.version.cuda is not None
        and torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
    )


def functions():
    from sglang.kernels.ops.layernorm.mhc import (
        hc_mix_stats,
        hc_mix_stats_sinkhorn,
        hc_split_sinkhorn,
    )

    def unfused(x, fn, scale, base):
        mixes = hc_mix_stats(x, fn, RMS_EPS).unsqueeze(1)
        return tuple(
            y.squeeze(1)
            for y in hc_split_sinkhorn(mixes, scale, base, HC, ITERS, HC_EPS)
        )

    def fused(x, fn, scale, base):
        return hc_mix_stats_sinkhorn(
            x, fn, scale, base, HC, ITERS, RMS_EPS, HC_EPS
        )

    return unfused, fused


def inputs(rows, case="signed", seed=41):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, generator=generator, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(MIX, K, generator=generator, device="cuda", dtype=torch.float32) * 0.01
    scale = torch.tensor([0.5, 0.25, 0.25], device="cuda", dtype=torch.float32)
    base = torch.randn(MIX, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    if case == "large":
        x.mul_(1024)
    elif case == "tiny":
        x.mul_(2.0**-20)
    elif case == "zeros":
        x.zero_()
    elif case == "wide_coefficients":
        scale.copy_(torch.tensor([8.0, -4.0, 12.0], device="cuda"))
        base.mul_(10)
    return x, fn, scale, base


def oracle(args):
    """Independent CPU FP64 projection, RMS, sigmoid and exact iteration count."""
    x, fn, scale, base = [a.detach().cpu().double() for a in args]
    mix = (x @ fn.T) / torch.sqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)
    pre = torch.sigmoid(mix[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(mix[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    comb = (mix[:, 2 * HC :] * scale[2] + base[2 * HC :]).reshape(-1, HC, HC)
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    return pre, post, comb


def ordered_fp32(value):
    bits = value.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(bits < 0, -(2**31) - bits, bits)


def compare(actual, expected, label, *, bitwise_report=False):
    for name, a, b in zip(("pre", "post", "comb"), actual, expected):
        if a.shape != b.shape:
            raise AssertionError(f"{label}/{name}: {a.shape} != {b.shape}")
        if a.dtype != torch.float32:
            raise AssertionError(f"{label}/{name}: expected FP32 output, got {a.dtype}")
        ac, bc = a.detach().cpu(), b.detach().cpu()
        if not torch.isfinite(ac).all() or not torch.isfinite(bc).all():
            raise AssertionError(f"{label}/{name}: nonfinite output/reference")
        exact = False
        ulp = None
        if bc.dtype == torch.float32:
            exact = torch.equal(ac.view(torch.int32), bc.view(torch.int32))
            ulp = int((ordered_fp32(ac) - ordered_fp32(bc)).abs().max())
        error = (ac.double() - bc.double()).abs()
        print(json.dumps({
            "comparison": label, "output": name,
            "bitwise_equal": exact if bitwise_report else None,
            "max_abs": error.max().item(), "max_fp32_ulp": ulp,
            "atol": ATOL, "rtol": RTOL,
        }), flush=True)
        if bitwise_report and REQUIRE_BITWISE and not exact:
            raise AssertionError(f"{label}/{name}: strict bitwise qualification failed")
        torch.testing.assert_close(ac.double(), bc.double(), atol=ATOL, rtol=RTOL)


@unittest.skipUnless(supported(), "requires actual CUDA SM80 (not HIP/SM86)")
class TestMHCSM80FusedStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unfused, cls.fused = map(staticmethod, functions())

    def qualify(self, args, label):
        baseline, fused = self.unfused(*args), self.fused(*args)
        # Bitwise comparison precedes the CPU oracle and numerical qualification.
        compare(fused, baseline, label + "/fused_vs_unfused", bitwise_report=True)
        reference = oracle(args)
        compare(baseline, reference, label + "/unfused_vs_fp64")
        compare(fused, reference, label + "/fused_vs_fp64")
        self.assertEqual(tuple(y.shape for y in fused), ((args[0].shape[0], HC),
                         (args[0].shape[0], HC), (args[0].shape[0], HC, HC)))

    def test_single_row_real_geometry_synthetic_ranges(self):
        for case in ("signed", "large", "tiny", "zeros", "wide_coefficients"):
            with self.subTest(case=case):
                self.qualify(inputs(1, case), "M1/" + case)

    def test_batch_geometry_is_diagnostic_not_production_gate(self):
        for rows in (4, 32):
            with self.subTest(rows=rows):
                self.qualify(inputs(rows), f"M{rows}/signed")

    def test_single_row_graph_replay_changed_all_inputs(self):
        args = inputs(1)
        # Compile both complete chains before entering either capture.
        for _ in range(3):
            self.unfused(*args)
            self.fused(*args)
        torch.cuda.synchronize()
        graphs, outputs = [], []
        for func in (self.unfused, self.fused):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = func(*args)
            graphs.append(graph)
            outputs.append(result)
        pointers = tuple(a.data_ptr() for a in args)
        for step, case in enumerate(("zeros", "large", "tiny", "signed")):
            fresh = inputs(1, case, seed=410 + step)
            fresh[2].add_(0.125 * (step + 1))
            fresh[3].add_(0.05 * (step + 1))
            # All input mutation is outside capture; storage addresses stay fixed.
            for dst, src in zip(args, fresh):
                dst.copy_(src)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(tuple(a.data_ptr() for a in args), pointers)
            compare(outputs[1], outputs[0], f"graph{step}/fused_vs_unfused", bitwise_report=True)
            reference = oracle(args)
            compare(outputs[0], reference, f"graph{step}/unfused_vs_fp64")
            compare(outputs[1], reference, f"graph{step}/fused_vs_fp64")


def benchmark(rows):
    """Full chains, no CPU/oracle in captures; fixed-input graph throughput."""
    unfused, fused = functions()
    for m in rows:
        args = inputs(m)
        # Validate each benchmarked shape before compiling/capturing timed graphs.
        a, b = unfused(*args), fused(*args)
        compare(b, a, f"benchmarkM{m}/parity", bitwise_report=True)
        reference = oracle(args)
        compare(a, reference, f"benchmarkM{m}/unfused_fp64")
        compare(b, reference, f"benchmarkM{m}/fused_fp64")
        times = {}
        inner = 100
        for name, func in (("unfused", unfused), ("fused", fused)):
            for _ in range(10):
                func(*args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(inner):
                    output = func(*args)
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / (inner * 20))
            times[name] = sorted(samples)[len(samples) // 2]
            # Keep output live through timing; don't include a host read in capture.
            assert all(torch.isfinite(y).all().item() for y in output)
        print(json.dumps({"benchmark": "graph_full_chain", "rows": m,
                          "warm_graph_replays": 100, "calls_per_graph": inner,
                          "unfused_us": times["unfused"], "fused_us": times["fused"],
                          "speedup": times["unfused"] / times["fused"],
                          "saved_us_per_call": times["unfused"] - times["fused"],
                          "projected_80_calls_saved_ms":
                          (times["unfused"] - times["fused"]) * 80 / 1000}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--rows", nargs="+", type=int, default=[1])
    parser.add_argument("--require-bitwise", action="store_true")
    options, rest = parser.parse_known_args()
    REQUIRE_BITWISE = options.require_bitwise
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMHCSM80FusedStats)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    if options.benchmark:
        if not supported():
            print("Benchmark skipped: requires actual CUDA SM80", flush=True)
        else:
            benchmark(options.rows)
