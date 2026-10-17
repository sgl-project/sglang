"""Read 4 rows per original expert matrix; verify byte decoding on SM80.

Standalone diagnostic, not CI/model execution. Does not load full tensors or
write model data. Run with CUDA torch/Triton and --model-dir CHECKPOINT.
"""
import argparse
import importlib.util
import json
import pathlib
import struct
import sys

import torch
import triton
import triton.language as tl

ROOT=pathlib.Path(__file__).resolve().parents[5]
SOURCE=ROOT/'python/sglang/kernels/ops/attention/dsv4'
PACKAGE='sglang.kernels.ops.attention.dsv4'
for name in ('kv_layout','dequant_k_cache_sm80'):
    spec=importlib.util.spec_from_file_location(PACKAGE+'.'+name,SOURCE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
codec=module


@triton.jit
def decode_rows(P,S,O,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    packed=tl.load(P+i//2,mask=i<N,other=0).to(tl.int32)
    code=(packed>>((i&1)*4))&15
    scale=tl.load(S+i//32,mask=i<N,other=127)
    value=codec._scaled_bf16(codec._e2m1_to_fp32(code),codec._ue8m0_to_fp32(scale))
    tl.store(O+i,value,mask=i<N)


def read_rows(root,index,key,rows=4):
    with (root/index[key]).open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        meta=json.loads(f.read(length))[key]
        if len(meta['shape'])!=2:raise ValueError('expected matrix')
        rows=min(rows,meta['shape'][0]);width=meta['shape'][1]
        f.seek(8+length+meta['data_offsets'][0])
        data=f.read(rows*width)
    if len(data)!=rows*width:raise ValueError('incomplete sample')
    return torch.frombuffer(bytearray(data),dtype=torch.uint8).reshape(rows,width),meta


def main():
    args=argparse.ArgumentParser();args.add_argument('--model-dir',required=True)
    root=pathlib.Path(args.parse_args().model_dir)
    if torch.cuda.get_device_capability()!=(8,0):raise RuntimeError('requires SM80')
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    lookup=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
    results=[]
    for matrix in ('w1','w3','w2'):
        prefix='layers.0.ffn.experts.0.'+matrix
        packed,wm=read_rows(root,index,prefix+'.weight')
        scales,sm=read_rows(root,index,prefix+'.scale')
        if wm['dtype']!='I8' or sm['dtype']!='F8_E8M0':raise ValueError('unexpected format')
        width=packed.shape[1]*2
        if width%32 or scales.shape!=(packed.shape[0],width//32):raise ValueError('unexpected scale shape')
        codes=torch.stack((packed&15,packed>>4),-1).flatten(1).long()
        ref=(lookup[codes]*scales.contiguous().view(torch.float8_e8m0fnu).float().double().repeat_interleave(32,1)).bfloat16()
        p=packed.cuda();s=scales.cuda();out=torch.empty_like(ref,device='cuda')
        decode_rows[(triton.cdiv(out.numel(),256),)](p,s,out,out.numel(),256)
        cpu=out.cpu();finite=~ref.isnan()
        assert torch.equal(cpu.isnan(),ref.isnan())
        assert torch.equal(cpu.view(torch.int16)[finite],ref.view(torch.int16)[finite])
        # A tiny projection using the sampled actual rows, not a full expert.
        torch.manual_seed(4111)
        x=(torch.randn(2,width)*.02).bfloat16()
        got=x.cuda().float()@out.float().T
        expected=x.float()@ref.float().T
        torch.testing.assert_close(got.cpu(),expected,rtol=1e-4,atol=2e-4)
        results.append({'matrix':matrix,'checkpoint_weight_shape':wm['shape'],'rows_sampled':packed.shape[0],
                        'payload_bytes':packed.numel()+scales.numel(),'decode_bit_exact':True,
                        'projection_max_abs':float((got.cpu()-expected).abs().max())})
    print(json.dumps({'scope':'4 real rows per matrix; software decode and tiny FP32 projection, not full MoE',
                      'gpu':torch.cuda.get_device_name(),'results':results},indent=2))

if __name__=='__main__':main()
