"""Offline FP8/group32 Marlin vs resident BF16 GEMV. No runtime model changes.

PYTHONPATH=python MAX_JOBS=1 python test/manual/layers/quantization/bench_fp8_sm80_predecode_gemv.py
Original raw weights retained. Predecode adds BF16 storage but no extra precision
reduction beyond existing BF16 scaled Marlin fragments. Qualification is finite
normal-range only; no claim about FTZ/extreme overflow or whole-model bit parity.
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

SHAPES = ((1792,5120),(4096,1280),(5120,1024),(576,5120),(5120,320))
VARIANTS = ('marlin_default','marlin_n64','bf16_mm','bf16_mv')
FLAG = 'SGLANG_DSV41_SM80_DENSE_TILE'


def errors(got, ref):
    a,b = got.float(),ref.float(); diff=a-b
    return {'max_abs':diff.abs().max().item(), 'relative_rms':
        (diff.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30)).item(),
        'bitwise_equal':torch.equal(got.view(torch.int16),ref.view(torch.int16))}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--routes',type=int,default=32)
    parser.add_argument('--rounds',type=int,default=20)
    args=parser.parse_args()
    assert args.routes>=32 and args.rounds>=2
    assert torch.cuda.get_device_capability()==(8,0) and torch.version.hip is None
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    os.environ['SGLANG_DSV41_SM80_EXPERIMENTAL']='1'
    torch.manual_seed(41201)
    for n,k in SHAPES:
        assert args.routes*n*k>40*2**20
        pack=[];scale=[];dense=[];xs=[];refs=[]
        workspace=marlin_make_workspace(torch.device('cuda'))
        perm=torch.empty(0,device='cuda',dtype=torch.int32)
        for route in range(args.routes):
            # Three raw-byte fixture families: normal signed, full finite FP8
            # codebook, cancellation. Arbitrary BF16 scales exercise multiply
            # rounding, in addition to exact power-of-two scale routes.
            if route==1:
                codes=torch.randint(0,256,(n,k),device='cuda',dtype=torch.uint8)
                codes=torch.where((codes&127)==127,torch.zeros_like(codes),codes)
                w=codes.view(torch.float8_e4m3fn)
            else:
                w=(torch.randn(n,k,device='cuda')*.25).to(torch.float8_e4m3fn)
                if route==2:
                    sign=torch.where(torch.arange(k,device='cuda')%2==0,1.,-1.)
                    w=(w.float().abs()*sign).to(torch.float8_e4m3fn)
            if route%2==0:
                sf=torch.exp2(torch.randint(-6,1,(n//32,k//32),device='cuda').float()).to(torch.bfloat16)
            else:
                sf=(.01+torch.rand(n//32,k//32,device='cuda')*1.99).to(torch.bfloat16)
            x=(torch.randn(1,k,device='cuda')*.05).to(torch.bfloat16)
            if route==2:x.fill_(.125)
            if route==1:x.mul_(.01)
            # Reproduce exponent-bias folding: Marlin decode shifts FP8 BF16
            # exponent by -120 and scales are multiplied by +120 before BF16
            # hmul2. Both intermediate values are normal in these fixtures.
            raw_bits=w.view(torch.uint8).to(torch.int32)
            shifted_bits=((raw_bits & 128) << 8) | ((raw_bits & 127) << 4)
            shifted=shifted_bits.to(torch.int16).view(torch.bfloat16)
            biased=(sf.float()*2.**120).to(torch.bfloat16)
            scaled=(shifted.float()*biased.float().repeat_interleave(32,0).repeat_interleave(32,1)).to(torch.bfloat16)
            direct=(w.float()*sf.float().repeat_interleave(32,0).repeat_interleave(32,1)).to(torch.bfloat16)
            assert torch.equal(scaled.view(torch.int16),direct.view(torch.int16)), 'biased decode weight mismatch'
            dense.append(scaled.contiguous())
            xs.append(x)
            pack.append(gptq_marlin_repack(b_q_weight=pack_fp8_to_int32(w,False).T.contiguous(),
                perm=perm,size_k=k,size_n=n,num_bits=8))
            expanded=sf.T.contiguous().repeat_interleave(32,1)
            scale.append(fp8_fused_exponent_bias_into_scales(marlin_permute_scales(
                s=expanded,size_k=k,size_n=n,group_size=32)))
            if route<3:
                refs.append(direct.cpu().double().matmul(x.cpu().double().squeeze()).to(torch.bfloat16).unsqueeze(0))
        out=[torch.empty(1,n,device='cuda',dtype=torch.bfloat16) for _ in xs]
        def run(v,i):
            if v.startswith('marlin'):
                return gptq_marlin_gemm(a=xs[i],c=out[i],b_q_weight=pack[i],b_scales=scale[i],
                    global_scale=None,b_zeros=None,g_idx=None,perm=None,workspace=workspace,
                    b_q_type=scalar_types.float8_e4m3fn,size_m=1,size_n=n,size_k=k,
                    use_fp32_reduce=True,use_atomic_add=False,is_k_full=True,is_zp_float=False)
            if v=='bf16_mm':return torch.mm(xs[i],dense[i].T,out=out[i])
            return torch.mv(dense[i],xs[i].squeeze(0),out=out[i].squeeze(0))
        graphs={};snap={}
        for v in VARIANTS:
            if v=='marlin_n64':os.environ[FLAG]='n64'
            else:os.environ.pop(FLAG,None)
            for _ in range(2):
                for i in range(args.routes):run(v,i)
            torch.cuda.synchronize()
            snap[v]=[o.cpu().clone() for o in out]
            for i,r in enumerate(refs):
                torch.testing.assert_close(snap[v][i],r,rtol=.015,atol=.004)
                print(json.dumps({'kind':'oracle','MNK':[1,n,k],'variant':v,'fixture':i,
                    **errors(snap[v][i],r)}),flush=True)
            graphs[v]=[]
            # Capture requires a non-default stream; initialize cuBLAS handles
            # on that exact stream, not merely on the default eager stream.
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    for i in range(args.routes):run(v,i)
            stream.synchronize()
            for i in range(args.routes):
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=stream):run(v,i)
                graphs[v].append(g)
            torch.cuda.current_stream().wait_stream(stream)
            cross=[]
            for i in range(args.routes):
                torch.testing.assert_close(snap[v][i],snap['marlin_default'][i],rtol=.015,atol=.004)
                cross.append(errors(snap[v][i],snap['marlin_default'][i]))
            print(json.dumps({'kind':'cross','MNK':[1,n,k],'variant':v,'routes':args.routes,
                'max_abs':max(t['max_abs'] for t in cross),
                'max_relative_rms':max(t['relative_rms'] for t in cross),
                'all_bitwise':all(t['bitwise_equal'] for t in cross)}),flush=True)
            # Dynamic activation qualification under replay, same retained weight.
            old=xs[0].clone();xs[0].neg_();graphs[v][0].replay()
            torch.testing.assert_close(out[0].cpu(),-snap[v][0],rtol=.015,atol=.004)
            xs[0].copy_(old)
        for v in VARIANTS:
            for _ in range(3):
                for g in graphs[v]:g.replay()
        torch.cuda.synchronize()
        times={v:[] for v in VARIANTS}
        for r in range(args.rounds):
            for v in VARIANTS if r%2==0 else tuple(reversed(VARIANTS)):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for g in graphs[v]:g.replay()
                b.record();b.synchronize();times[v].append(a.elapsed_time(b)*1000/args.routes)
        base=statistics.median(times['marlin_default'])
        for v in VARIANTS:
            t=statistics.median(times[v])
            print(json.dumps({'kind':'benchmark','MNK':[1,n,k],'variant':v,'median_us':t,
                'speedup':base/t,'projected_40_calls_saved_ms':(base-t)*40/1000,
                'raw_weight_MiB_rotation':args.routes*n*k/2**20,
                'BF16_weight_MiB_rotation':args.routes*n*k*2/2**20,'samples_us':times[v]}),flush=True)
        del graphs,snap,pack,scale,dense,xs,refs,out
    os.environ.pop(FLAG,None)
    print('PREDECODE_ALL_SHAPES_ORACLE_CROSS_REPLAY_PASS',flush=True)


if __name__=='__main__':main()
