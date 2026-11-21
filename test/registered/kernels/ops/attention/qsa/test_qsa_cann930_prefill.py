"""Opt-in numerical checks for the GBSA implementation bundled with sgl-kernel-npu."""
import os
import pytest
if os.getenv("SGLANG_TEST_CANN930_QSA_NATIVE") != "1":
    pytest.skip("Requires an A3 and an sgl-kernel-npu wheel with GBSA", allow_module_level=True)

import sgl_kernel_npu
import torch
from sglang.kernels.ops.attention.qsa_gbsa_prefill import qsa_gbsa_prefill
from sgl_kernel_npu.operators.attention import qsa_sparse_attention


def case(rows, prefix):
    torch.manual_seed(52)
    length = rows + prefix
    pool = ((length + 127)//128 + 2) * 128
    q = torch.randn(rows,3,256,device='npu',dtype=torch.bfloat16)*.2
    k = torch.randn(pool,1,256,device='npu',dtype=torch.bfloat16)*.2
    v = torch.randn_like(k)
    page = torch.randperm(pool//128)
    table = (page[:,None]*128 + torch.arange(128)).flatten()[None,:length].int()
    logical = torch.full((rows,2051),-1,dtype=torch.int32)
    for row in range(rows):
        visible = prefix + row + 1
        selected = torch.randperm(visible//4)[:512]
        full = (selected[:,None]*4 + torch.arange(4)).flatten()
        tail = torch.arange(visible//4*4,visible)
        ids = torch.cat([full,tail])
        logical[row,:ids.numel()] = ids
    physical = table[0,logical.clamp_min(0).long()]
    physical[logical<0] = -1
    return q,k,v,logical.npu(),table.npu(),torch.tensor([length],device='npu',dtype=torch.int32),torch.tensor([rows],device='npu',dtype=torch.int32),physical.npu()


@pytest.mark.parametrize('rows,prefix',[(128,0),(512,1777),(128,11000)])
def test_native_prefill(rows,prefix):
    q,k,v,indices,table,lens,extend,physical = case(rows,prefix)
    ref = qsa_sparse_attention(q,k,v,physical,impl='torch_native')
    out = qsa_gbsa_prefill(q,k,v,indices,table,lens,extend,rows)
    torch.npu.synchronize()
    torch.testing.assert_close(out,ref,atol=.004,rtol=.02)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph,auto_dispatch_capture=False):
        captured = qsa_gbsa_prefill(q,k,v,indices,table,lens,extend,rows)
    q.mul_(.7)
    v.mul_(.6)
    graph.replay()
    torch.npu.synchronize()
    ref = qsa_sparse_attention(q,k,v,physical,impl='torch_native')
    torch.testing.assert_close(captured,ref,atol=.004,rtol=.02)


def test_multiple_sequences():
    first = case(128,0)
    second = case(384,7001)
    q = torch.cat([first[0],second[0]])
    k = torch.cat([first[1],second[1]])
    v = torch.cat([first[2],second[2]])
    indices = torch.cat([first[3],second[3]])
    table = torch.zeros((2,second[4].shape[1]),device='npu',dtype=torch.int32)
    table[0,:first[4].shape[1]] = first[4][0]
    table[1] = second[4][0] + first[1].shape[0]
    lens = torch.cat([first[5],second[5]])
    extend = torch.cat([first[6],second[6]])
    physical = torch.cat([first[7],torch.where(second[7]>=0,second[7]+first[1].shape[0],-1)])
    out = qsa_gbsa_prefill(q,k,v,indices,table,lens,extend,512)
    ref = qsa_sparse_attention(q,k,v,physical,impl='torch_native')
    torch.testing.assert_close(out,ref,atol=.004,rtol=.02)
