"""Correctness tests for the isolated SM80 V4.1 cache fallback.

Can run without installing SGLang by loading the two leaf modules directly.
This does not test complete DeepSeek-V4.1 model support.
"""
import ast
import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/kernels/ops/attention/dsv4"


def load_leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layout_module = load_leaf(
    "sglang.kernels.ops.attention.dsv4.kv_layout", SOURCE / "kv_layout.py"
)
KVLayout = layout_module.KVLayout
fallback = load_leaf("sglang.kernels.ops.attention.dsv4.dequant_k_cache_sm80", SOURCE / "dequant_k_cache_sm80.py")


def decode(cache, ids, page, layout, out=None):
    return fallback.dequantize_k_cache_paged_sm80(cache, ids, page, out, layout)


def reference(cache, ids, page, layout):
    c = cache.cpu()
    ids = ids.cpu().long()
    selected = c[ids // page]
    slots = ids % page
    data = selected.gather(
        1, slots[:, None] * layout.data_bytes + torch.arange(layout.data_bytes)
    )
    sb = selected.gather(
        1,
        layout.scale_offset(page)
        + slots[:, None] * layout.scale_bytes
        + torch.arange(layout.scale_bytes),
    )
    if layout == KVLayout.V41:
        values = data.contiguous().view(torch.float8_e4m3fn).float()
        scales = sb.contiguous().view(torch.float8_e8m0fnu).float()
    else:
        codes = torch.stack((data & 15, data >> 4), -1).flatten(1).long()
        table = torch.tensor(
            [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]
        )
        values = table[codes]
        scales = sb.contiguous().view(torch.float8_e4m3fn).float()
    # Float64 avoids FTZ in the reference and represents every operand exactly.
    result = (values.double() * scales.double().repeat_interleave(layout.tile_size, 1)).bfloat16()
    return result[:, None, :]


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0),
    "requires an SM80 CUDA GPU",
)
class TestSM80Dequant(unittest.TestCase):
    def assert_bits(self, actual, expected):
        actual = actual.cpu()
        self.assertTrue(torch.equal(actual.isnan(), expected.isnan()))
        finite = ~expected.isnan()
        self.assertTrue(torch.equal(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]))

    def test_exhaustive_formats(self):
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            n = 256
            cache = torch.zeros(n, layout.page_bytes(1), dtype=torch.uint8)
            if layout == KVLayout.V41:
                cache[:, :512] = torch.arange(512).remainder(256).to(torch.uint8)
            else:
                codes = torch.arange(512).remainder(16).to(torch.uint8)
                cache[:, :256] = codes[::2] | (codes[1::2] << 4)
            cache[:, layout.data_bytes:layout.bytes_per_token] = torch.arange(256,dtype=torch.uint8)[:,None]
            ids = torch.arange(n,device="cuda")
            result = decode(cache.cuda(), ids, 1, layout)
            self.assert_bits(result, reference(cache,ids,1,layout))

    def test_pages_strides_and_empty(self):
        torch.manual_seed(732)
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for page in (1,64,128):
                for dtype in (torch.int32,torch.int64):
                    cache = torch.randint(0,256,(3,layout.page_bytes(page)+128),dtype=torch.uint8,device="cuda")
                    ids = torch.tensor([3*page-1,0,page,0,page-1],dtype=dtype,device="cuda")
                    storage = torch.full((5,1,640),123.,dtype=torch.bfloat16,device="cuda")
                    out = storage[:,:,:512]
                    returned = decode(cache,ids,page,layout,out)
                    self.assertIs(returned,out)
                    self.assert_bits(out,reference(cache,ids,page,layout))
                    self.assertTrue(bool(torch.all(storage[:,:,512:]==123)))
                    empty=out[:0]
                    self.assertIs(decode(cache,ids[:0],page,layout,empty),empty)

    def test_public_wrapper_dispatch(self):
        # Execute the actual public wrapper in isolation. Full SGLang imports
        # require optional runtime dependencies unavailable in this leaf test.
        tree=ast.parse((SOURCE / "dequant_k_cache.py").read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="dequantize_k_cache_paged_v41")
        namespace={"torch":torch,"Optional":__import__('typing').Optional,"KVLayout":KVLayout,
                   "is_hip_runtime":lambda:False,"__package__":"sglang.kernels.ops.attention.dsv4"}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"public-wrapper","exec"),namespace)
        wrapper=namespace[fn.name]
        cache=torch.zeros((1,KVLayout.V41.page_bytes(1)),device="cuda",dtype=torch.uint8)
        cache[:,:512]=56
        cache[:,512:528]=127
        ids=torch.tensor([0],device="cuda")
        self.assert_bits(wrapper(cache,ids,1),reference(cache,ids,1,KVLayout.V41))

    def test_graph_replay(self):
        layout=KVLayout.V41
        page=64
        cache=torch.zeros(2,layout.page_bytes(page),device="cuda",dtype=torch.uint8)
        cache[:,:page*512]=56
        cache[:,page*512:page*528]=127
        ids=torch.tensor([0,127],device="cuda")
        out=torch.empty(2,1,512,device="cuda",dtype=torch.bfloat16)
        decode(cache,ids,page,layout,out)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode(cache,ids,page,layout,out)
        graph.replay()
        self.assert_bits(out,reference(cache,ids,page,layout))
        cache[1,:page*512]=64
        ids.copy_(torch.tensor([127,0],device="cuda"))
        graph.replay()
        self.assert_bits(out,reference(cache,ids,page,layout))


if __name__ == "__main__":
    unittest.main(verbosity=2)
