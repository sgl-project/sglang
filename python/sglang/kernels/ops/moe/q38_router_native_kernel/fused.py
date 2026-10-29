"""Experimental native split-K GEMM with NVFP4 quantization in its idle warp.

Original MMA, DSMEM reduction, BF16 rounding, and epilogue are unchanged.
Only M4/M8, N512,K2560, contiguous inputs are supported by this prototype.
"""
from functools import cache
import hashlib
from pathlib import Path
import sys
import types

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op


def quant_asm():
    lines = ['{', '.reg .b32 a<8>, m, t, s, lo, hi;',
             '.reg .f32 v<16>, mx, gs, r6, scale, sf, inv, z;',
             '.reg .b16 pair, sl, sh;', '.reg .b8 p<8>;', '.reg .pred zero;',
             'ld.global.v4.b32 {a0,a1,a2,a3}, [$0];',
             'ld.global.v4.b32 {a4,a5,a6,a7}, [$0+16];', 'mov.b32 m, 0;']
    for i in range(8):
        lines += [f'and.b32 t, a{i}, 0x7fff;', 'max.u32 m, m, t;',
                  f'shr.u32 t, a{i}, 16;', 'and.b32 t, t, 0x7fff;', 'max.u32 m,m,t;',
                  f'shl.b32 t,a{i},16;', f'mov.b32 v{i*2},t;',
                  f'and.b32 t,a{i},0xffff0000;', f'mov.b32 v{i*2+1},t;']
    lines += ['shl.b32 t,m,16;', 'mov.b32 mx,t;', 'mov.f32 gs,$3;',
              'rcp.approx.ftz.f32 r6, 0f40c00000;', 'mul.rn.f32 scale,mx,r6;',
              'mul.rn.f32 scale,gs,scale;', 'mov.f32 z,0f00000000;',
              'cvt.rn.satfinite.e4m3x2.f32 pair,z,scale;', 'cvt.u32.u16 s,pair;',
              'and.b32 s,s,255;', 'cvt.u16.u32 pair,s;',
              'cvt.rn.f16x2.e4m3x2 t,pair;', 'mov.b32 {sl,sh},t;',
              'cvt.f32.f16 sf,sl;', 'rcp.approx.ftz.f32 inv,gs;',
              'mul.rn.f32 inv,sf,inv;', 'rcp.approx.ftz.f32 inv,inv;',
              'setp.eq.f32 zero,sf,0f00000000;', '@zero mov.f32 inv,0f00000000;']
    for i in range(16):
        lines.append(f'mul.rn.f32 v{i},v{i},inv;')
    for i in range(8):
        lines.append(f'cvt.rn.satfinite.e2m1x2.f32 p{i},v{i*2+1},v{i*2};')
    lines += ['mov.b32 lo,{p0,p1,p2,p3};', 'mov.b32 hi,{p4,p5,p6,p7};',
              'st.global.v2.b32 [$1],{lo,hi};', 'st.global.u8 [$2],s;', '}']
    return '\n'.join(lines)


_ASM = quant_asm()


@dsl_user_op
def quant16(xptr, qptr, sptr, gs, *, loc=None, ip=None):
    llvm.inline_asm(None, [xptr.toint(loc=loc, ip=ip).ir_value(),
                          qptr.toint(loc=loc, ip=ip).ir_value(),
                          sptr.toint(loc=loc, ip=ip).ir_value(), gs.ir_value(loc=loc, ip=ip)],
                    _ASM, 'l,l,l,f', has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@cache
def core():
    import importlib.util
    path=Path(__file__).parent/'source/native_splitk_quant_generated.py'
    native=path.with_name('dense_bf16_gemm_sm100_splitk.py')
    spec=importlib.util.spec_from_file_location('_q38_native_splitk_quant_v2',path)
    module=importlib.util.module_from_spec(spec)
    module._q38_quant16=quant16
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.q38_native_source_sha256=hashlib.sha256(native.read_bytes()).hexdigest()
    module.q38_patched_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    return module

@ext.jit
def invoke(op: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
           gs: cute.Tensor, q: cute.Tensor, sf: cute.Tensor, stream: cuda.CUstream):
    c=cute.make_tensor(c.iterator,cute.select(c.layout,mode=[1,2,0]))
    op(cute.make_tensor(a.iterator,cute.select(a.layout,mode=[1,2,0])),
       cute.make_tensor(b.iterator,cute.select(b.layout,mode=[2,1,0])),
       c, c, gs, q, sf, stream)


_COMPILED={}


def router_gemm_quant(x, weight, gs):
    m=x.shape[0]
    assert m in (4,8) and x.shape==(m,2560) and weight.shape==(512,2560)
    assert x.dtype==weight.dtype==torch.bfloat16 and gs.dtype==torch.float32 and gs.numel()==1
    assert x.is_contiguous() and weight.is_contiguous() and gs.is_contiguous()
    assert x.device==weight.device==gs.device
    mod=core()
    out=torch.empty((m,512),dtype=x.dtype,device=x.device)
    q=torch.empty((m,1280),dtype=torch.uint8,device=x.device)
    sf=torch.empty((m,160),dtype=torch.uint8,device=x.device)
    tensors=mod._to_cute_swap(x,weight.T,out,None)
    args=(*tensors[:3],from_dlpack(gs),from_dlpack(q.flatten()),from_dlpack(sf.flatten()))
    stream=cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key=(x.get_device(),m)
    if key not in _COMPILED:
        tactic=mod.SplitKTactic(64,8,4,10)
        mod.validate_tactic(tactic,m,512,2560)
        op=mod.SplitKDenseGemmKernel(tactic=tactic,use_pdl=True,has_bias=False)
        _COMPILED[key]=ext.compile(invoke,op,*args,stream)
    _COMPILED[key](*args,stream)
    return out,q,sf
