"""M1 HC4/H5120 diagnostic: existing combine+norm fusion versus production kernels.

No runtime selection change. Run on otherwise idle SM80:
 PYTHONPATH=python python test/registered/kernels/ops/layernorm/test_mhc_sm80_combine_norm.py --benchmark
Baseline calls hc_combine + sgl_kernel.rmsnorm (the actual forward_cuda API,
whose implementation appears as FlashInfer/CUTLASS RMSNorm in deployed traces).
Fixed predeclared tolerance rtol=.015/atol=.004 on BF16 outputs; report bitwise,
ULP, and relative RMS as well. Intermediate BF16 combine rounding is retained.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import unittest

import torch

HC, H, K = 4, 5120, 20480
EPS = 1e-6
ATOL, RTOL = 0.004, 0.015


def supported():
    return (torch.version.cuda is not None and torch.version.hip is None
            and torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0))


def functions():
    from sgl_kernel import rmsnorm
    from sglang.kernels.ops.layernorm.mhc import hc_combine
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm, _hc_combine_norm

    def baseline(x, pre, weight):
        return rmsnorm(hc_combine(x, pre, HC, torch.bfloat16), weight, EPS)

    def fused(x, pre, weight):
        return hc_combine_norm(x, pre, weight, EPS)

    def variant(x, pre, weight, out, parts, warps):
        _hc_combine_norm[(1, parts)](x, pre, weight, out, x.stride(0), pre.stride(0),
                                     EPS, parts, num_warps=warps)
        return out

    return baseline, fused, variant


def fixture(case):
    g = torch.Generator(device='cuda').manual_seed(20200 + case)
    x = torch.randn((1, K), generator=g, device='cuda').bfloat16()
    pre = torch.randn((1, HC), generator=g, device='cuda') * 0.5
    weight = (torch.randn((H,), generator=g, device='cuda') * 0.25 + 1).bfloat16()
    mode = case % 10
    if mode == 1:
        x.mul_(2**-20)
    elif mode == 2:
        x.mul_(2**12)
    elif mode == 3:
        x.zero_()
    elif mode == 4:
        pre.copy_(torch.tensor([[8., -4., 2., -1.]], device='cuda'))
    elif mode == 5:
        weight.mul_(16)
    elif mode == 6:
        x.mul_(2**-60)  # finite/subnormal-sensitive RMS, never square overflow
    elif mode == 7:
        x.mul_(2**40)  # FP32 square still finite after combine
        pre.mul_(2**-8)
    elif mode == 8:
        x[:, H:2*H].copy_(x[:, :H]);pre.copy_(torch.tensor([[1., -1., 0., 0.]], device='cuda'))
    elif mode == 9:
        weight.copy_(torch.linspace(-16, 16, H, device='cuda').bfloat16())
        x[:, ::2].neg_()
    return x, pre, weight


def oracle(args):
    x, pre, weight = [v.detach().cpu().double() for v in args]
    x = x.reshape(1, HC, H)
    value = torch.zeros((1, H), dtype=torch.float64)
    for c in range(HC):
        value = value + pre[:, c:c+1] * x[:, c]
    # Precisely the architecture's materialized intermediate storage dtype.
    value = value.bfloat16().double()
    return (value / torch.sqrt(value.square().mean(-1, keepdim=True) + EPS) * weight).bfloat16()


def ordered_bf16(x):
    bits = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


def compare(label, got, ref):
    got, ref = got.detach().cpu(), ref.detach().cpu()
    assert bool(torch.isfinite(got).all()) and bool(torch.isfinite(ref).all()), label
    diff = got.float() - ref.float()
    rms = diff.double().square().mean().sqrt()
    ref_rms = ref.double().square().mean().sqrt().clamp_min(1e-30)
    result = dict(comparison=label, bitwise_equal=torch.equal(got.view(torch.int16), ref.view(torch.int16)),
                  max_abs=float(diff.abs().max()), relative_rms=float(rms/ref_rms),
                  max_bf16_ulp=int((ordered_bf16(got)-ordered_bf16(ref)).abs().max()),
                  atol=ATOL, rtol=RTOL)
    print(json.dumps(result), flush=True)
    torch.testing.assert_close(got, ref, atol=ATOL, rtol=RTOL)


@unittest.skipUnless(supported(), 'actual CUDA SM80 required')
class TestMHCCombineNormSM80(unittest.TestCase):
    def test_twenty_finite_cases_production_and_fp64(self):
        baseline, fused, _ = functions()
        for case in range(20):
            with self.subTest(case=case):
                args = fixture(case)
                ref = oracle(args)
                a, b = baseline(*args), fused(*args)
                compare(f'case{case}/fused_vs_production', b, a)
                compare(f'case{case}/production_vs_fp64', a, ref)
                compare(f'case{case}/fused_vs_fp64', b, ref)

    def test_graph_replay_all_inputs_changed(self):
        baseline, fused, _ = functions()
        args = fixture(0)
        for _ in range(3):
            fused(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused(*args)
        addresses = tuple(a.data_ptr() for a in args)
        for case in [1, 4, 7, 8, 9, 13, 19, 0]:
            new = fixture(case)
            for dst, src in zip(args, new):
                dst.copy_(src)
            self.assertEqual(addresses, tuple(a.data_ptr() for a in args))
            graph.replay()
            compare(f'graph{case}/fused_vs_production', out, baseline(*args))
            compare(f'graph{case}/fused_vs_fp64', out, oracle(args))

    def test_parts_warps_diagnostic_only(self):
        baseline, _, variant = functions()
        for case in [0, 1, 4, 7, 8, 9]:
            args = fixture(case)
            out = torch.empty((1, H), device='cuda', dtype=torch.bfloat16)
            ref = baseline(*args)
            for parts in [1, 2, 4]:
                for warps in [4, 8]:
                    variant(*args, out, parts, warps)
                    compare(f'case{case}/parts{parts}_warps{warps}', out, ref)


def benchmark(routes=1536, calls=80, replays=4, repeats=5):
    baseline, fused, variant = functions()
    assert routes * K * 2 > 40 * 1024**2, 'residual workset must exceed A100 L2'
    g = torch.Generator(device='cuda').manual_seed(20201)
    residuals = torch.randn((routes, 1, K), generator=g, device='cuda').bfloat16()
    pres = torch.randn((routes, 1, HC), generator=g, device='cuda') * .5
    weights = (torch.randn((routes, H), generator=g, device='cuda') * .25 + 1).bfloat16()
    out = torch.empty((1, H), device='cuda', dtype=torch.bfloat16)
    # Independent graph per rotating route block; combined sweeps exceed L2.
    configs = [('production', None, None), ('existing_fused', None, None)]
    configs += [(f'parts{p}_warps{w}', p, w) for p in [1, 2, 4] for w in [4, 8]]
    results = []
    for name, parts, warps in configs:
        def fn(i):
            args = residuals[i], pres[i], weights[i]
            if name == 'production':
                return baseline(*args)
            if name == 'existing_fused':
                return fused(*args)
            return variant(*args, out, parts, warps)
        for i in range(3):
            fn(i)
        graphs = []
        stride = calls
        for start in range(0, routes, stride):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(start, min(start + calls, routes)):
                    fn(i)
            graphs.append(graph)
        for _ in range(2):
            for graph in graphs:
                graph.replay()
        samples = []
        for _ in range(repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(replays):
                for graph in graphs:
                    graph.replay()
            b.record();b.synchronize()
            samples.append(a.elapsed_time(b) * 1000 / (routes * replays))
        med = statistics.median(samples)
        row = dict(variant=name, median_us_per_call=med, samples_us=samples,
                   residual_workset_MiB=residuals.numel()*2/1024**2,
                   total_workset_MiB=(residuals.numel()*2+pres.numel()*4+weights.numel()*2)/1024**2,
                   routes=routes, calls_per_graph=calls, graph_count=len(graphs),
                   replays=replays, repeats=repeats)
        results.append(row);print(json.dumps(row), flush=True)
    baseline_us = results[0]['median_us_per_call']
    for r in results[1:]:
        print(json.dumps(dict(variant=r['variant'], speedup=baseline_us/r['median_us_per_call'],
                              projected_saved_ms_per_80_calls=(baseline_us-r['median_us_per_call'])*.08)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--routes', type=int, default=1536)
    parser.add_argument('--replays', type=int, default=4)
    options, remaining = parser.parse_known_args()
    if not supported():
        raise SystemExit('actual CUDA SM80 required; no CPU surrogate qualification')
    if not options.benchmark_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMHCCombineNormSM80)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    if options.benchmark or options.benchmark_only:
        benchmark(routes=options.routes, replays=options.replays)
