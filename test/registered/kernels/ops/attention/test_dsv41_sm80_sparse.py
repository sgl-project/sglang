"""SM80 paged attention component tests, not end-to-end model enablement.

Run directly with CUDA torch/Triton; leaf imports avoid unrelated SGLang deps.
The reference decodes formats on CPU and computes attention in float64.
"""
import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT=pathlib.Path(__file__).resolve().parents[5]
SOURCE=ROOT/'python/sglang/kernels/ops/attention/dsv4'
PACKAGE='sglang.kernels.ops.attention.dsv4'

def leaf(name):
    spec=importlib.util.spec_from_file_location(PACKAGE+'.'+name,SOURCE/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

KVLayout=leaf('kv_layout').KVLayout
leaf('dequant_k_cache_sm80')
attention=leaf('sparse_attention_sm80').sparse_attention_sm80


def cache_fixture(layout,page,n=3):
    raw=torch.randint(0,256,(n,layout.page_bytes(page)+128),dtype=torch.uint8)
    if layout==KVLayout.V41:
        data=raw[:,:page*512]
        data[(data&127)==127]=0
        # Moderate values make comparisons interpretable rather than overflow.
        raw[:,page*512:page*528]=torch.randint(119,123,(n,page*16),dtype=torch.uint8)
    else:
        raw[:,page*256:page*288]=torch.randint(24,56,(n,page*32),dtype=torch.uint8)
    return raw.cuda()


def decode_cpu(raw,ids,page,layout):
    c=raw.cpu();ids=ids.cpu().long()
    pages=c[ids//page];slots=ids%page
    data=pages.gather(1,slots[:,None]*layout.data_bytes+torch.arange(layout.data_bytes))
    sb=pages.gather(1,layout.scale_offset(page)+slots[:,None]*layout.scale_bytes+torch.arange(layout.scale_bytes))
    if layout==KVLayout.V41:
        values=data.contiguous().view(torch.float8_e4m3fn).float().double()
        scales=sb.contiguous().view(torch.float8_e8m0fnu).float().double()
    else:
        code=torch.stack((data&15,data>>4),-1).flatten(1).long()
        table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
        values=table[code]
        scales=sb.contiguous().view(torch.float8_e4m3fn).float().double()
    return (values*scales.repeat_interleave(layout.tile_size,1)).bfloat16().double()


def reference(q,raw,indices,page,layout,scale,topk=None,sink=None,extra=None):
    q=q.cpu().double();indices=indices.cpu()
    out=torch.zeros_like(q);lse=torch.full(q.shape[:2],-torch.inf,dtype=torch.float64)
    for b in range(q.shape[0]):
        pieces=[]
        for c,idx,p,l,lengths in [(raw,indices,page,layout,topk)]+([] if extra is None else [extra]):
            ids=idx[b].cpu()
            if lengths is not None:ids=ids[:max(0,int(lengths[b]))]
            ids=ids[ids>=0]
            if ids.numel():pieces.append(decode_cpu(c,ids,p,l))
        if not pieces:continue
        kv=torch.cat(pieces)
        score=q[b]@kv.T*scale
        lse[b]=torch.logsumexp(score,-1)
        den=lse[b] if sink is None else torch.logaddexp(lse[b],sink.cpu().double())
        out[b]=torch.exp(score-den[:,None])@kv
    return out.bfloat16(),lse.float()


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0),'SM80 required')
class TestSparseSM80(unittest.TestCase):
    def assert_attention(self,a,b):
        torch.testing.assert_close(a[0].cpu(),b[0],rtol=.015,atol=.004)
        torch.testing.assert_close(a[1].cpu(),b[1],rtol=2e-5,atol=2e-5)

    def test_mixed_pages_formats_lengths(self):
        torch.manual_seed(4101)
        for main in (KVLayout.V41,KVLayout.V41_FP4):
          for other in (KVLayout.V41,KVLayout.V41_FP4):
            for dtype in (torch.int32,torch.int64):
                q=(torch.randn(3,8,512,device='cuda')*.15).bfloat16()
                raw=cache_fixture(main,64);ex=cache_fixture(other,128)
                ids=torch.tensor([[0,63,64,-1,191,0],[-1,-1,-1,-1,-1,-1],[190,2,5,8,190,60]],device='cuda',dtype=dtype)
                ei=torch.tensor([[383,0,128,-1],[-1,-1,-1,-1],[4,4,127,128]],device='cuda',dtype=dtype)
                lengths=torch.tensor([4,6,5],device='cuda',dtype=torch.int32)
                el=torch.tensor([3,0,4],device='cuda',dtype=torch.int32)
                sink=torch.linspace(-2,2,8,device='cuda')
                kw=dict(topk_length=lengths,attn_sink=sink,extra_cache=ex,extra_indices=ei,extra_page_size=128,extra_layout=other,extra_topk_length=el)
                got=attention(q,raw,ids,64,main,.04419417382415922,**kw)
                expected=reference(q,raw,ids,64,main,.04419417382415922,lengths,sink,(ex,ei,128,other,el))
                self.assert_attention(got,expected)

    def test_empty_and_no_sink(self):
        q=torch.zeros(2,4,512,device='cuda',dtype=torch.bfloat16)
        cache=cache_fixture(KVLayout.V41,64)
        ids=torch.empty((2,0),device='cuda',dtype=torch.int64)
        for sink in (None,torch.full((4,),-torch.inf,device='cuda'),torch.zeros(4,device='cuda')):
            got=attention(q,cache,ids,64,KVLayout.V41,1.,attn_sink=sink)
            self.assertTrue(bool(torch.all(got[0]==0)))
            self.assertTrue(bool(torch.all(got[1]==-torch.inf)))
        ids=torch.tensor([[0,0],[1,2]],device='cuda')
        self.assert_attention(attention(q,cache,ids,64,KVLayout.V41,.1),reference(q,cache,ids,64,KVLayout.V41,.1))
        empty=attention(q[:0],cache,ids[:0],64,KVLayout.V41,.1)
        self.assertEqual(empty[0].numel(),0)

    def test_large_heads_candidates_and_stability(self):
        torch.manual_seed(4102)
        q=(torch.randn(2,64,512,device='cuda')*.2).bfloat16()
        raw=cache_fixture(KVLayout.V41,64)
        ids=torch.randint(0,192,(2,128),device='cuda',dtype=torch.int32)
        ids[0,::7]=-1
        # Very positive/negative sinks exercise stable denominator rescaling.
        sink=torch.linspace(-1000,1000,64,device='cuda')
        got=attention(q,raw,ids,64,KVLayout.V41,.04419417382415922,attn_sink=sink)
        self.assert_attention(got,reference(q,raw,ids,64,KVLayout.V41,.04419417382415922,sink=sink))

    def test_extra_only_and_duplicate_sink_math(self):
        # Q=0 => all scores=0. One identical key twice plus sink log(2)
        # must give exactly half the decoded value. Duplicates MUST count twice.
        q=torch.zeros(1,4,512,device='cuda',dtype=torch.bfloat16)
        raw=torch.zeros((1,KVLayout.V41.page_bytes(1)),device='cuda',dtype=torch.uint8)
        raw[:,:512]=56 # E4M3 1.0
        raw[:,512:528]=127 # UE8M0 1.0
        ids=torch.empty((1,0),device='cuda',dtype=torch.int64)
        extra_ids=torch.tensor([[0,0]],device='cuda')
        sink=torch.full((4,),__import__('math').log(2),device='cuda')
        got=attention(q,raw,ids,1,KVLayout.V41,1.,attn_sink=sink,
                      extra_cache=raw,extra_indices=extra_ids,extra_page_size=1,extra_layout=KVLayout.V41)
        torch.testing.assert_close(got[0],torch.full_like(q,.5),rtol=0,atol=0)
        torch.testing.assert_close(got[1],torch.full_like(got[1],__import__('math').log(2)),rtol=1e-6,atol=1e-6)
        lengths=torch.tensor([-1],device='cuda',dtype=torch.int32)
        masked=attention(q,raw,extra_ids,1,KVLayout.V41,1.,topk_length=lengths)
        self.assertTrue(bool(torch.all(masked[0]==0)))
        self.assertTrue(bool(torch.all(masked[1]==-torch.inf)))

    def test_graph_replay(self):
        q=torch.randn(1,8,512,device='cuda',dtype=torch.bfloat16)*.1
        raw=cache_fixture(KVLayout.V41_FP4,64)
        ids=torch.tensor([[0,1,2,-1]],device='cuda',dtype=torch.int32)
        sink=torch.zeros(8,device='cuda')
        out=torch.empty_like(q);lse=torch.empty(1,8,device='cuda')
        kw=dict(attn_sink=sink,out=out,lse=lse)
        attention(q,raw,ids,64,KVLayout.V41_FP4,.05,**kw)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):attention(q,raw,ids,64,KVLayout.V41_FP4,.05,**kw)
        graph.replay()
        self.assert_attention((out,lse),reference(q,raw,ids,64,KVLayout.V41_FP4,.05,sink=sink))
        ids.copy_(torch.tensor([[127,64,0,0]],device='cuda',dtype=torch.int32))
        q.mul_(.5)
        graph.replay()
        self.assert_attention((out,lse),reference(q,raw,ids,64,KVLayout.V41_FP4,.05,sink=sink))

if __name__=='__main__':unittest.main(verbosity=2)
