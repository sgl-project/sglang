"""Standalone tiled SM80 correctness tests and serial comparison benchmark.

No heavy SRT imports; reuses the existing independent CPU float64 format/math
oracle and tests via a leaf import. Run on the real SM80, not an architecture
spoof. Tests: python test/registered/kernels/ops/attention/test_dsv41_sm80_sparse_tiled.py
Benchmark (checks oracle before timing): same command with --benchmark.
"""

import importlib.util
import math
import pathlib
import sys
import unittest
from unittest import mock

import torch

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_dsv41_sm80_serial_tests", HERE / "test_dsv41_sm80_sparse.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
tiled_module = base.leaf("sparse_attention_sm80_tiled")
tiled = tiled_module.sparse_attention_sm80_tiled
serial = base.attention
KVLayout = base.KVLayout
SM80 = (torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0))


@unittest.skipUnless(SM80, "real NVIDIA SM80 required")
class TestSparseTiledSM80(base.TestSparseSM80):
    """Inherited tests execute tiled, retaining the serial CPU oracle tolerance."""

    def setUp(self):
        self.previous = base.attention
        base.attention = tiled

    def tearDown(self):
        base.attention = self.previous

    def test_model_pages_chunks_and_strided_cache(self):
        torch.manual_seed(4103)
        for main in (KVLayout.V41, KVLayout.V41_FP4):
            for extra_page in (128, 256):
                q = (torch.randn(3, 8, 512, device="cuda") * .15).bfloat16()
                raw = base.cache_fixture(main, 256)
                ex = base.cache_fixture(KVLayout.V41_FP4, extra_page)
                # Actual stride != visible pitch, nonzero storage offset too.
                backing = torch.zeros((3, raw.shape[1] + 256),
                                      device="cuda", dtype=torch.uint8)
                strided = backing[:, 64:64 + raw.shape[1]]
                strided.copy_(raw)
                self.assertFalse(strided.is_contiguous())
                ids = torch.randint(0, 768, (3, 65), device="cuda", dtype=torch.int64)
                ei = torch.randint(0, extra_page * 3, (3, 33),
                                   device="cuda", dtype=torch.int32)
                ids[:, ::9] = -7
                ids[0, 31:34] = ids[0, 1]  # duplicate across chunk boundary
                ei[:, ::8] = -1
                lengths = torch.tensor([1000, -3, 32], device="cuda")
                el = torch.tensor([33, 33, -1], device="cuda", dtype=torch.int32)
                sink = torch.linspace(-1000, 1000, 8, device="cuda")
                kw = dict(topk_length=lengths, attn_sink=sink,
                          extra_cache=ex, extra_indices=ei,
                          extra_page_size=extra_page, extra_layout=KVLayout.V41_FP4,
                          extra_topk_length=el)
                got = tiled(q, strided, ids, 256, main, 512 ** -.5, **kw)
                expected = base.reference(
                    q, strided, ids, 256, main, 512 ** -.5, lengths, sink,
                    (ex, ei, extra_page, KVLayout.V41_FP4, el),
                )
                self.assert_attention(got, expected)
                self.assert_attention(got, tuple(x.cpu() for x in
                                      serial(q, raw, ids, 256, main, 512 ** -.5, **kw)))

    def test_prefill_decode_512_candidates(self):
        torch.manual_seed(4104)
        for batch, k in ((1, 128), (1, 512), (64, 512)):
            q = (torch.randn(batch, 8, 512, device="cuda") * .15).bfloat16()
            raw = base.cache_fixture(KVLayout.V41, 256)
            ids = torch.randint(0, 768, (batch, k), device="cuda", dtype=torch.int32)
            ids[:, ::11] = -1
            sink = torch.linspace(-2, 2, 8, device="cuda")
            got = tiled(q, raw, ids, 256, KVLayout.V41, 512 ** -.5, attn_sink=sink)
            self.assert_attention(got, base.reference(
                q, raw, ids, 256, KVLayout.V41, 512 ** -.5, sink=sink,
            ))

    def test_large_finite_scores_preserve_chunk_cardinality(self):
        # FP32 max+log(32) rounds back to max=2**26. Merging partial LSEs
        # would erroneously weight the 32-key and 1-key chunks equally.
        page = 64
        q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
        q[:, :, 0] = 2 ** 26
        raw = torch.zeros((1, KVLayout.V41.page_bytes(page)),
                          device="cuda", dtype=torch.uint8)
        raw[:, :page * 512] = 56  # E4M3 1.0
        raw[:, page * 512:] = 127  # UE8M0 1.0
        raw[:, 32 * 512 + 1:33 * 512] = 68  # E4M3 3.0, same score
        ids = torch.arange(33, device="cuda", dtype=torch.int32)[None, :]
        for separate_pool in (False, True):
            for sink in (None, torch.full((8,), float(2 ** 26), device="cuda")):
                main_ids = ids[:, :32].contiguous() if separate_pool else ids
                kw = dict(attn_sink=sink)
                if separate_pool:
                    kw.update(extra_cache=raw, extra_indices=ids[:, 32:].contiguous(),
                              extra_page_size=page, extra_layout=KVLayout.V41)
                got = tiled(q, raw, main_ids, page, KVLayout.V41, 1., **kw)
                reference = serial(q, raw, main_ids, page, KVLayout.V41, 1., **kw)
                self.assert_attention(got, tuple(x.cpu() for x in reference))
                denominator = 33 + (sink is not None)
                expected = torch.full_like(q, 35 / denominator)
                expected[:, :, 0] = 33 / denominator
                torch.testing.assert_close(got[0], expected, rtol=0, atol=0)
                torch.testing.assert_close(got[1],
                                           torch.full_like(got[1], float(2 ** 26)),
                                           rtol=0, atol=0)

    def test_empty_pool_and_no_launch_t0(self):
        q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
        raw = torch.empty((0, KVLayout.V41.page_bytes(256)),
                          device="cuda", dtype=torch.uint8)
        ids = torch.full((1, 33), -1, device="cuda", dtype=torch.int64)
        self.assert_attention(tiled(q, raw, ids, 256, KVLayout.V41, .1),
                              (torch.zeros_like(q).cpu(),
                               torch.full((1, 8), -torch.inf)))
        for empty_q, empty_ids in ((q[:0], ids[:0]), (q[:, :0], ids)):
            with mock.patch.object(tiled_module, "_partial_kernel") as first:
                with mock.patch.object(tiled_module, "_merge_kernel") as second:
                    out, lse = tiled(empty_q, raw, empty_ids, 256, KVLayout.V41, .1)
                    self.assertEqual(out.numel(), 0)
                    self.assertEqual(lse.numel(), 0)
                    first.__getitem__.assert_not_called()
                    second.__getitem__.assert_not_called()

    def test_graph_prefill_mixed_replay_buffers(self):
        torch.manual_seed(4105)
        q = (torch.randn(64, 8, 512, device="cuda") * .15).bfloat16()
        raw = base.cache_fixture(KVLayout.V41, 256)
        extra = base.cache_fixture(KVLayout.V41_FP4, 128)
        ids = torch.randint(0, 768, (64, 65), device="cuda", dtype=torch.int32)
        ei = torch.randint(0, 384, (64, 33), device="cuda", dtype=torch.int64)
        lengths = torch.full((64,), 65, device="cuda", dtype=torch.int32)
        sink = torch.zeros(8, device="cuda")
        out = torch.empty_like(q)
        lse = torch.empty((64, 8), device="cuda", dtype=torch.float32)
        kw = dict(topk_length=lengths, attn_sink=sink, extra_cache=extra,
                  extra_indices=ei, extra_page_size=128,
                  extra_layout=KVLayout.V41_FP4, out=out, lse=lse)
        returned = tiled(q, raw, ids, 256, KVLayout.V41, .05, **kw)
        self.assertIs(returned[0], out)
        self.assertIs(returned[1], lse)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tiled(q, raw, ids, 256, KVLayout.V41, .05, **kw)
        for replay in range(2):
            if replay:
                q.mul_(.5)
                ids[:, ::4] = -1
                ei.copy_(torch.randint(0, 384, ei.shape, device="cuda"))
                lengths[::2] = -2
                sink.copy_(torch.linspace(-10, 10, 8, device="cuda"))
            graph.replay()
            expected = base.reference(q, raw, ids, 256, KVLayout.V41, .05,
                                      lengths, sink,
                                      (extra, ei, 128, KVLayout.V41_FP4, None))
            self.assert_attention((out, lse), expected)
            before = out.clone(), lse.clone()
            graph.replay()
            torch.testing.assert_close(out, before[0], rtol=0, atol=0)
            torch.testing.assert_close(lse, before[1], rtol=0, atol=0)


def _milliseconds(fn, repeats=10):
    for _ in range(2):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def benchmark():
    if not SM80:
        raise RuntimeError("Benchmark requires real NVIDIA SM80 (8,0)")
    torch.manual_seed(4106)
    print(f"device={torch.cuda.get_device_name()} capability=(8,0) "
          f"torch={torch.__version__}")
    print("B H main_K extra_K extra_page mode serial_ms tiled_ms speedup workspace_MiB")
    for batch, k in ((1, 128), (1, 512), (64, 512)):
        for extra_page in (None, 128, 256):
            q = (torch.randn(batch, 8, 512, device="cuda") * .15).bfloat16()
            raw = base.cache_fixture(KVLayout.V41, 256)
            ids = torch.randint(0, 768, (batch, k), device="cuda", dtype=torch.int32)
            sink = torch.linspace(-2, 2, 8, device="cuda")
            kw = dict(attn_sink=sink)
            extra_oracle = None
            if extra_page is not None:
                extra = base.cache_fixture(KVLayout.V41_FP4, extra_page)
                ei = torch.randint(0, extra_page * 3, (batch, 512),
                                   device="cuda", dtype=torch.int64)
                kw.update(extra_cache=extra, extra_indices=ei,
                          extra_page_size=extra_page, extra_layout=KVLayout.V41_FP4)
                extra_oracle = (extra, ei, extra_page, KVLayout.V41_FP4, None)
            expected = base.reference(q, raw, ids, 256, KVLayout.V41,
                                      512 ** -.5, sink=sink, extra=extra_oracle)
            checker = TestSparseTiledSM80()
            calls = []
            graphs = []
            for function in (serial, tiled):
                out = torch.empty_like(q)
                lse = torch.empty(q.shape[:2], device="cuda")
                def call(function=function, out=out, lse=lse):
                    return function(q, raw, ids, 256, KVLayout.V41, 512 ** -.5,
                                    out=out, lse=lse, **kw)
                checker.assert_attention(call(), expected)
                calls.append(call)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                graph.replay()
                checker.assert_attention((out, lse), expected)
                graphs.append(graph)
            chunks = math.ceil(k / 32) + (16 if extra_page is not None else 0)
            workspace = batch * 8 * chunks * 514 * 4 / (1024 ** 2)
            for mode, functions in (("eager", calls),
                                    ("graph", [g.replay for g in graphs])):
                serial_ms, tiled_ms = [_milliseconds(fn) for fn in functions]
                print(f"{batch} 8 {k} {512 if extra_page else 0} "
                      f"{extra_page or '-'} {mode} {serial_ms:.4f} {tiled_ms:.4f} "
                      f"{serial_ms / tiled_ms:.2f} {workspace:.3f}", flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main(verbosity=2)
