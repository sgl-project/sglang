"""Tests for runtime-shaped SM80 cache views; no complete runtime import."""
import importlib.util
import pathlib
import sys
import unittest

import torch

HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sm80_sparse_test_helpers',HERE/'test_dsv41_sm80_sparse.py')
helpers=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=helpers
spec.loader.exec_module(helpers)
KVLayout=helpers.KVLayout
adapter=helpers.leaf('flash_mla_sm80').flash_mla_with_kvcache_sm80


def cache_view(raw,page,layout,carrier):
    view=raw[:,:page*layout.bytes_per_token].view(raw.shape[0],page,1,layout.bytes_per_token)
    return view if carrier==torch.uint8 else view.view(carrier)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0),'SM80 required')
class TestAdapter(unittest.TestCase):
    def assert_result(self,actual,expected):
        torch.testing.assert_close(actual[0].squeeze(1).cpu(),expected[0],rtol=.015,atol=.004)
        torch.testing.assert_close(actual[1].squeeze(-1).cpu(),expected[1],rtol=2e-5,atol=2e-5)

    def test_runtime_mixed_views(self):
        torch.manual_seed(4110)
        for carrier in (torch.uint8,torch.float8_e4m3fn):
            for ml,el in ((KVLayout.V41,KVLayout.V41_FP4),(KVLayout.V41_FP4,KVLayout.V41)):
                raw=helpers.cache_fixture(ml,64)
                extra=helpers.cache_fixture(el,128)
                q=(torch.randn(2,1,16,512,device='cuda')*.15).bfloat16()
                ids=torch.tensor([[[0,63,64,-1]],[[191,1,2,1]]],device='cuda',dtype=torch.int32)
                ei=torch.tensor([[[383,0]],[[128,-1]]],device='cuda',dtype=torch.int32)
                length=torch.tensor([[3],[4]],device='cuda',dtype=torch.int32)
                sink=torch.linspace(-2,2,16,device='cuda')
                out=torch.empty_like(q);lse=torch.empty(2,16,1,device='cuda')
                result=adapter(q,cache_view(raw,64,ml,carrier),512,.04419417382415922,ids,
                               topk_length=length,attn_sink=sink,
                               extra_k_cache=cache_view(extra,128,el,carrier),extra_indices_in_kvcache=ei,out=out,lse=lse)
                self.assertIs(result[0],out);self.assertIs(result[1],lse)
                expected=helpers.reference(q.squeeze(1),raw,ids.squeeze(1),64,ml,.04419417382415922,length.flatten(),sink,(extra,ei.squeeze(1),128,el,None))
                self.assert_result(result,expected)

    def test_nonzero_storage_offset_and_graph(self):
        layout=KVLayout.V41;page=64
        raw=helpers.cache_fixture(layout,page)
        # Reserve a full prefix row; views begin at a nonzero storage offset.
        parent=torch.zeros((raw.shape[0]+1,raw.shape[1]),device='cuda',dtype=torch.uint8)
        parent[1:].copy_(raw)
        shifted=parent[1:]
        cv=cache_view(shifted,page,layout,torch.float8_e4m3fn)
        self.assertGreater(cv.storage_offset(),0)
        q=torch.zeros(1,1,8,512,device='cuda',dtype=torch.bfloat16)
        ids=torch.tensor([[[0,127,-1]]],device='cuda',dtype=torch.int64)
        out=torch.empty_like(q);lse=torch.empty(1,8,1,device='cuda')
        adapter(q,cv,512,.05,ids,out=out,lse=lse)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):adapter(q,cv,512,.05,ids,out=out,lse=lse)
        graph.replay()
        self.assert_result((out,lse),helpers.reference(q.squeeze(1),shifted,ids.squeeze(1),page,layout,.05))
        ids.copy_(torch.tensor([[[191,0,0]]],device='cuda'))
        graph.replay()
        self.assert_result((out,lse),helpers.reference(q.squeeze(1),shifted,ids.squeeze(1),page,layout,.05))

    def test_empty_and_rejections(self):
        layout=KVLayout.V41;page=64
        raw=helpers.cache_fixture(layout,page)
        cv=cache_view(raw,page,layout,torch.uint8)
        q=torch.zeros(1,1,4,512,device='cuda',dtype=torch.bfloat16)
        ids=torch.zeros(1,1,2,device='cuda',dtype=torch.int32)
        with self.assertRaises((ValueError,TypeError)):adapter(q,cv,448,.05,ids)
        with self.assertRaises((ValueError,TypeError)):adapter(q.expand(1,2,4,512),cv,512,.05,ids)
        with self.assertRaises((ValueError,TypeError)):adapter(q,cv,512,.05,ids.squeeze(1))
        with self.assertRaises((ValueError,TypeError)):adapter(q,cv[...,::2],512,.05,ids)
        pitch=raw.shape[1]
        visible=(raw.shape[0]-1)*pitch+page*layout.bytes_per_token
        short_storage=torch.zeros(visible,device='cuda',dtype=torch.uint8)
        short_view=short_storage.as_strided((raw.shape[0],page,1,layout.bytes_per_token),(pitch,layout.bytes_per_token,layout.bytes_per_token,1))
        with self.assertRaises((ValueError,TypeError)):adapter(q,short_view,512,.05,ids)
        result=adapter(q[:0],cv,512,.05,ids[:0])
        self.assertEqual(result[0].shape,(0,1,4,512));self.assertEqual(result[1].shape,(0,4,1))

if __name__=='__main__':unittest.main(verbosity=2)
