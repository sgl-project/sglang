"""ONLY diagnostic M1 FP32-SIMT mHC partial; unchanged fused reduce/Sinkhorn.

 PYTHONPATH=python python test/registered/kernels/ops/layernorm/test_mhc_sm80_simt_stats.py --benchmark
M1/K20480/MIX24/HC4; 80 slices of K256 retain current partial interface.
No weight downcast, no plain TF32, no runtime routing edit. Fixed prior
FP32 pre/post/comb thresholds atol=rtol=1e-5; no tolerance auto-adjustment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from unittest.mock import patch
from pathlib import Path
import statistics
import unittest

import torch
import triton
import triton.language as tl

spec = importlib.util.spec_from_file_location('_mhc_existing_diagnostic',
    Path(__file__).with_name('test_mhc_sm80_fused_stats.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
K, MIX, HC, SLICES = 20480, 24, 4, 80
EPS, ATOL, RTOL = 1e-6, 1e-5, 1e-5


def candidate(args, warps=4):
    from sglang.kernels.ops.layernorm.mhc import (
        _hc_mix_reduce_sinkhorn_kernel, _hc_mix_stats_partial_sm80_simt,
    )
    x, fn, scale, bias = args
    assert x.shape == (1, K) and fn.shape == (MIX, K)
    assert x.dtype == torch.bfloat16 and fn.dtype == torch.float32
    assert x.stride(1) == fn.stride(1) == 1
    pm = torch.empty((SLICES, 1, MIX), device=x.device, dtype=torch.float32)
    ps = torch.empty((SLICES, 1), device=x.device, dtype=torch.float32)
    pre = torch.empty((1, HC), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((1, HC, HC), device=x.device, dtype=torch.float32)
    _hc_mix_stats_partial_sm80_simt[(SLICES,)](x, fn, pm, ps, num_warps=warps,
                             enable_fp_fusion=False)
    _hc_mix_reduce_sinkhorn_kernel[(1,)](pm, ps, scale, bias, pre, post, comb,
        1, 1.0/K, EPS, MIX=MIX, HC=HC, NUM_SLICES=SLICES,
        ITERS=20, EPS=EPS, num_warps=1)
    return pre, post, comb


def fixtures(case, seed=2100):
    mode = ['signed', 'large', 'tiny', 'zeros', 'wide_coefficients'][case % 5]
    args = base.inputs(1, mode, seed + case)
    if case >= 5:
        # Larger finite dynamic range/cancellation without square overflow.
        x, fn, scale, bias = args
        if case == 5:
            x.mul_(2**20);fn.mul_(2**-20)
        elif case == 6:
            x.mul_(2**-40)
        elif case == 7:
            fn[:, 1::2].neg_();fn[:, ::2].mul_(16)
        elif case == 8:
            scale.copy_(torch.tensor([4., -2., 8.], device='cuda'))
            bias.mul_(4)
        elif case == 9:
            fn.zero_();bias.zero_()
    return args


@unittest.skipUnless(base.supported(), 'actual CUDA SM80 required')
class TestMHCStatsSIMT(unittest.TestCase):
    def test_production_literal_gate_matches_qualified_kernel(self):
        _, fused = base.functions()
        args = fixtures(0)
        with patch.dict(os.environ, {
            'SGLANG_DSV41_SM80_EXPERIMENTAL': '1',
            'SGLANG_DSV41_SM80_MHC_SIMT_STATS': '1',
        }):
            out = fused(*args)
        base.compare(out, candidate(args, 8), 'production_gate_vs_kernel', bitwise_report=True)
        base.compare(out, base.oracle(args), 'production_gate_vs_fp64')

    def test_ten_real_geometry_cases_fixed_tolerance(self):
        _, fused = base.functions()
        for case in range(10):
            args = fixtures(case)
            ref, previous = base.oracle(args), fused(*args)
            for warps in [4, 8]:
                with self.subTest(case=case, warps=warps):
                    out = candidate(args, warps)
                    base.compare(out, previous, f'case{case}/warps{warps}/vs_existing', bitwise_report=True)
                    base.compare(out, ref, f'case{case}/warps{warps}/vs_fp64')
            base.compare(previous, ref, f'case{case}/existing_vs_fp64')

    def test_dynamic_graph_replay_changes_all_inputs(self):
        _, fused = base.functions()
        args = fixtures(0)
        for warps in [4, 8]:
            for _ in range(3):candidate(args, warps)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out = candidate(args, warps)
            pointers = tuple(v.data_ptr() for v in args)
            for case in [3, 1, 6, 7, 8, 9, 0]:
                for dst, src in zip(args, fixtures(case, seed=2200)):
                    dst.copy_(src)
                self.assertEqual(pointers, tuple(v.data_ptr() for v in args))
                graph.replay()
                base.compare(out, fused(*args), f'graph{case}/warps{warps}/vs_existing', bitwise_report=True)
                base.compare(out, base.oracle(args), f'graph{case}/warps{warps}/vs_fp64')


def benchmark(routes=32, calls=8, replays=10, repeats=5):
    _, fused = base.functions()
    assert routes * MIX * K * 4 > 40 * 1024**2, 'FN working set must exceed A100 L2'
    args = [fixtures(0, seed=2300+i) for i in range(routes)]
    configs = [('tf32x3_existing', lambda a:fused(*a)),
               ('simt_warps4', lambda a:candidate(a, 4)),
               ('simt_warps8', lambda a:candidate(a, 8))]
    results = []
    for name, fn in configs:
        for a in args[:3]:fn(a)
        graphs = []
        for start in range(0, routes, calls):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for a in args[start:start+calls]:fn(a)
            graphs.append(graph)
        for _ in range(5):
            for graph in graphs:graph.replay()
        samples = []
        for _ in range(repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(replays):
                for graph in graphs:graph.replay()
            b.record();b.synchronize()
            samples.append(a.elapsed_time(b)*1000/(routes*replays))
        med = statistics.median(samples)
        row = dict(variant=name, median_us_per_full_chain=med, samples_us=samples,
                   fn_working_set_MiB=routes*MIX*K*4/1024**2, routes=routes,
                   graph_count=len(graphs), replays=replays, repeats=repeats)
        print(json.dumps(row), flush=True);results.append(row)
    baseline = results[0]['median_us_per_full_chain']
    for r in results[1:]:
        print(json.dumps(dict(variant=r['variant'], speedup=baseline/r['median_us_per_full_chain'],
            projected_saved_ms_per_80_calls=(baseline-r['median_us_per_full_chain'])*.08)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--benchmark-only', action='store_true')
    options = p.parse_args()
    if not base.supported():raise SystemExit('actual CUDA SM80 required')
    if not options.benchmark_only:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestMHCStatsSIMT))
        if not result.wasSuccessful():raise SystemExit(1)
    if options.benchmark or options.benchmark_only:benchmark()
