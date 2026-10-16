"""Offline SM80 dense FP8 group32 tile test; never launches or edits a model.

PYTHONPATH=python MAX_JOBS=1 python test/manual/layers/quantization/bench_fp8_sm80_dense_tiles.py
Requires patched gptq_marlin.cuh, actual A100, ~1GiB spare. Rotates 32 distinct
weights (>40MiB L2) and keeps BF16 output, FP32 reduction, atomic=False.
CPU FP64 oracle uses original FP8 values and BF16 block scales, not Marlin layout.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace, marlin_permute_scales
from sglang.srt.layers.quantization.marlin_utils_fp8 import pack_fp8_to_int32, fp8_fused_exponent_bias_into_scales

FLAG = 'SGLANG_DSV41_SM80_DENSE_TILE'
SHAPES = ((1792, 5120), (4096, 1280), (5120, 1024))
# The shared-expert pair reads through a separate opt-in, so they are measured
# with their own list and flag.
SMALL_SHAPES = ((576, 5120), (5120, 320))
VARIANTS = ('default', 'n64', 'n128k64', 'n64x2')
# Variants the shared-expert pair understands through SGLANG_DSV41_SM80_DENSE_TILE_SMALL.
SMALL_VARIANTS = ('default', 'n64', 'n128k64', 'n32', 'n64k64', 'n32k64')


def metrics(got, ref):
    diff = got.float() - ref.float()
    return {'max_abs': diff.abs().max().item(), 'relative_rms':
            (diff.square().mean().sqrt() / ref.float().square().mean().sqrt().clamp_min(1e-30)).item(),
            'bitwise_equal': torch.equal(got.view(torch.int16), ref.view(torch.int16))}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--routes', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--variants', default=None)
    parser.add_argument('--shapes', default=None,
                        help="comma separated NxK list; defaults to the three dense shapes")
    parser.add_argument('--flag', default=FLAG)
    args = parser.parse_args()
    shapes = SHAPES if args.shapes is None else tuple(
        tuple(int(v) for v in spec.split('x')) for spec in args.shapes.split(','))
    variants = tuple((args.variants or ','.join(VARIANTS)).split(','))
    assert 'default' in variants and set(variants) <= set(VARIANTS) | set(SMALL_VARIANTS)
    assert args.routes >= 16 and args.rounds >= 2
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
    assert torch.version.hip is None
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['SGLANG_DSV41_SM80_EXPERIMENTAL'] = '1'
    torch.manual_seed(41200)
    for n, k in shapes:
        assert args.routes * n * k > 40 * 2**20
        perm = torch.empty(0, dtype=torch.int32, device='cuda')
        workspace = marlin_make_workspace(torch.device('cuda'))
        packed, scales = [], []
        # Three fixtures include mixed signed values, alternating cancellation,
        # and a wider finite exponent range. All values preserve FP8 encoding.
        fixtures, xs = [], []
        for route in range(args.routes):
            w = (torch.randn((n, k), device='cuda') * .25).to(torch.float8_e4m3fn)
            exponent = torch.randint(-4, 2, (n // 32, k // 32), device='cuda')
            sf = torch.exp2(exponent.float()).to(torch.bfloat16)
            x = (torch.randn((1, k), device='cuda') * .1).to(torch.bfloat16)
            if route == 1:
                signs = torch.where(torch.arange(k, device='cuda') % 2 == 0, 1., -1.)
                w = (w.float().abs() * signs).to(torch.float8_e4m3fn)
                x.fill_(.125)
            elif route == 2:
                w = (torch.randn((n, k), device='cuda') * 2).to(torch.float8_e4m3fn)
                sf = torch.exp2(torch.randint(-8, 4, sf.shape, device='cuda').float()).to(torch.bfloat16)
                x.mul_(.01)
            packed.append(gptq_marlin_repack(b_q_weight=pack_fp8_to_int32(w, False).T.contiguous(),
                          perm=perm, size_k=k, size_n=n, num_bits=8))
            groups = sf.T.contiguous().repeat_interleave(32, 1)
            scales.append(fp8_fused_exponent_bias_into_scales(marlin_permute_scales(
                          s=groups, size_k=k, size_n=n, group_size=32)))
            xs.append(x)
            if route < 3:
                # Group scales applied in BF16 before Marlin BF16 MMA. Power-of-two
                # scales make FP8*scale exactly BF16 representable in this fixture.
                wc = w.cpu().float().double()
                sc = sf.cpu().float().double().repeat_interleave(32, 0).repeat_interleave(32, 1)
                fixtures.append((wc * sc).matmul(x.cpu().double().squeeze(0)).to(torch.bfloat16).unsqueeze(0))
        outputs = [torch.empty((1, n), device='cuda', dtype=torch.bfloat16) for _ in xs]

        def call(i):
            return gptq_marlin_gemm(a=xs[i], c=outputs[i], b_q_weight=packed[i],
                b_scales=scales[i], global_scale=None, b_zeros=None, g_idx=None, perm=None,
                workspace=workspace, b_q_type=scalar_types.float8_e4m3fn,
                size_m=1, size_n=n, size_k=k, is_k_full=True,
                use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

        graphs, snapshots = {}, {}
        for variant in variants:
            if variant == 'default': os.environ.pop(args.flag, None)
            else: os.environ[args.flag] = variant
            for i in range(args.routes): call(i)
            torch.cuda.synchronize()
            snapshots[variant] = [out.cpu().clone() for out in outputs]
            for i, ref in enumerate(fixtures):
                got = snapshots[variant][i]
                torch.testing.assert_close(got, ref, rtol=.015, atol=.004)
                print(json.dumps({'kind': 'oracle', 'MNK': [1,n,k], 'variant': variant,
                      'fixture': i, **metrics(got, ref)}), flush=True)
            graphs[variant] = []
            for i in range(args.routes):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): call(i)
                graphs[variant].append(graph)
            for i in range(args.routes):
                torch.testing.assert_close(snapshots[variant][i], snapshots['default'][i], rtol=.015, atol=.004)
            print(json.dumps({'kind':'cross_tile', 'MNK':[1,n,k], 'variant':variant,
                  'routes_checked':args.routes, 'max_abs':max(metrics(a,b)['max_abs'] for a,b in
                  zip(snapshots[variant], snapshots['default']))}), flush=True)
        # Mutation checks prove graph re-reads activations and scales rather than
        # replaying a constant output. Original inputs restored before timing.
        for variant in variants:
            x_old = xs[0].clone(); xs[0].neg_(); graphs[variant][0].replay()
            torch.testing.assert_close(outputs[0].cpu(), -snapshots[variant][0], rtol=.015, atol=.004)
            xs[0].copy_(x_old)
            s_old = scales[0].clone(); scales[0].mul_(.5); graphs[variant][0].replay()
            torch.testing.assert_close(outputs[0].cpu(), snapshots[variant][0] * .5, rtol=.015, atol=.004)
            scales[0].copy_(s_old)
        for variant in variants:
            for _ in range(3):
                for g in graphs[variant]: g.replay()
        torch.cuda.synchronize()
        timings = {v:[] for v in variants}
        for round_id in range(args.rounds):
            order = variants if round_id % 2 == 0 else tuple(reversed(variants))
            for variant in order:
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for g in graphs[variant]:g.replay()
                end.record(); end.synchronize()
                timings[variant].append(start.elapsed_time(end) * 1000 / args.routes)
        for variant in variants:
            median = statistics.median(timings[variant]); baseline = statistics.median(timings['default'])
            print(json.dumps({'kind':'benchmark', 'MNK':[1,n,k], 'variant':variant,
                'weight_MiB_per_rotation':args.routes*n*k/2**20, 'median_us':median,
                'speedup':baseline/median, 'projected_40_calls_saved_ms':(baseline-median)*40/1000,
                'samples_us':timings[variant]}),flush=True)
        del graphs, packed, scales, xs, outputs, snapshots, fixtures
    os.environ.pop(FLAG,None)


if __name__ == '__main__':
    main()
