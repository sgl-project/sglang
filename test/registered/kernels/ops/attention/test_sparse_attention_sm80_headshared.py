"""Candidate regression + graph geometry autotuning; no model route changes."""
import functools
import importlib.util
import pathlib
import sys
import unittest
import torch

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_headshared_tiled_tests', HERE / 'test_dsv41_sm80_sparse_tiled.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
base = original.base
candidate = base.leaf('sparse_attention_sm80_headshared').sparse_attention_sm80_headshared


@unittest.skipUnless(original.SM80, 'real SM80 required')
class TestGeometry(original.TestSparseTiledSM80):
    def setUp(self):
        self.saved_tiled = original.tiled
        original.tiled = candidate
        super().setUp()

    def tearDown(self):
        super().tearDown()
        original.tiled = self.saved_tiled

    def test_every_geometry_large_finite_cardinality(self):
        for block in (8, 16, 32, 64):
            for warps in (4, 8):
                with self.subTest(block=block, warps=warps):
                    original.tiled = functools.partial(candidate, block=block, warps=warps)
                    original.TestSparseTiledSM80.test_large_finite_scores_preserve_chunk_cardinality(self)
        original.tiled = candidate


def benchmark():
    if not original.SM80:
        raise RuntimeError('real SM80 required')
    torch.manual_seed(41061)
    check = original.TestSparseTiledSM80()
    print('mainK extraK page BLOCK warps graph_ms baseline_graph_ms speedup', flush=True)
    for k in (128, 512):
        for page in ((None,) if '--main-only' in sys.argv else (None, 128, 256)):
            q = (torch.randn(1, 8, 512, device='cuda') * .15).bfloat16()
            raw = base.cache_fixture(base.KVLayout.V41, 256)
            ids = torch.randint(0, 768, (1, k), device='cuda', dtype=torch.int32)
            ids[:, ::11] = -1
            ids[:, 30:33] = ids[:, 1]  # duplicates cross chunks
            sink = torch.linspace(-2, 2, 8, device='cuda')
            kw = dict(attn_sink=sink)
            extra_oracle = None
            if page is not None:
                extra = base.cache_fixture(base.KVLayout.V41_FP4, page)
                ei = torch.randint(0, page * 3, (1, 512), device='cuda', dtype=torch.int32)
                ei[:, ::17] = -3
                kw.update(extra_cache=extra, extra_indices=ei,
                          extra_page_size=page, extra_layout=base.KVLayout.V41_FP4)
                extra_oracle = (extra, ei, page, base.KVLayout.V41_FP4, None)
            expected = base.reference(q, raw, ids, 256, base.KVLayout.V41, 512 ** -.5,
                                      sink=sink, extra=extra_oracle)
            baseline_ms = None
            configurations = [(32, 4)] + [(b, w) for b in (8, 16, 32, 64) for w in (4, 8) if (b, w) != (32, 4)]
            for block, warps in configurations:
                out = torch.empty_like(q)
                lse = torch.empty((1, 8), device='cuda')
                def call():
                    return candidate(q, raw, ids, 256, base.KVLayout.V41, 512 ** -.5,
                                     out=out, lse=lse, block=block, warps=warps, **kw)
                try:
                    check.assert_attention(call(), expected)
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        call()
                    graph.replay()
                    check.assert_attention((out, lse), expected)
                    ms = original._milliseconds(graph.replay, repeats=100)
                    if baseline_ms is None:
                        baseline_ms = ms
                    print(f'{k} {512 if page is not None else 0} {page} {block} {warps} {ms:.6f} {baseline_ms:.6f} {baseline_ms/ms:.3f}', flush=True)
                    del graph
                except Exception as error:
                    print(f'FAILED k={k} page={page} block={block} warps={warps}: {type(error).__name__}: {error}', flush=True)
                    raise


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main(verbosity=2)
