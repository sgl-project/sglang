#!/usr/bin/env python3
"""Qualify the production SM80 selected-index sort/gather/mask/store kernel.

Runtime remains opt-in. Input IDs follow torch.topk's contract: nonnegative,
representable int32 indices; lmax <= INT_MAX and 0 <= lens <= lmax. This does
not select top-k or change scores/ties: it only sorts already selected IDs.
Duplicate selected IDs are deliberately retained. As in runtime reset_outputs,
the output tail outside k is -1. No live GPU values are read by the wrapper.
"""
import argparse
import json
import statistics
import unittest

import torch


from sglang.kernels.ops.attention.dsv4.indexer_finish_sm80 import (
    finish_indexer_selected_sm80 as fused_finish,
)


def torch_finish(idx, slots, lens, out):
    k = idx.shape[1]
    ids = idx.sort(dim=-1).values
    reach = ids < lens[:, None]
    out[:, :k] = torch.where(reach, slots.gather(1, ids.clamp_max(slots.shape[1] - 1)), -1).to(torch.int32)
    # Runtime resets the output before write_decode; benchmark defaults k=width.
    if k < out.shape[1]:
        out[:, k:] = -1
    return out


def cpu_oracle(idx, slots, lens, width):
    ids = idx.cpu().to(torch.int64).sort(dim=-1).values
    slot = slots.cpu().to(torch.int64)
    lengths = lens.cpu().to(torch.int64)
    out = torch.full((idx.shape[0], width), -1, dtype=torch.int32)
    out[:, :idx.shape[1]] = torch.where(ids < lengths[:, None],
        slot.gather(1, ids.clamp_max(slot.shape[1] - 1)), -1).to(torch.int32)
    return out


class TestFinish(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(200)
        assert torch.cuda.get_device_capability() == (8, 0), 'qualification limited to actual SM80'

    def test_geometry_dtypes_duplicates_and_strides(self):
        cases = 0
        for bs, k, lmax in [(1, 1, 1), (1, 64, 128), (1, 127, 512),
                             (1, 256, 32000), (1, 511, 65536),
                             (1, 512, 65536), (3, 512, 66048)]:
            for dtype in (torch.int32, torch.int64):
                for strided in (False, True):
                    factor = 2 if strided else 1
                    idx = torch.randint(lmax, (bs, k * factor), device='cuda', dtype=dtype)[:, ::factor]
                    if k > 3:
                        idx[:, 1:4] = idx[:, :1]  # retain duplicate contributions
                    slots = torch.randint(0, 2147483647, (bs, lmax * factor), device='cuda', dtype=dtype)[:, ::factor]
                    out = torch.empty((bs, 512 * factor), device='cuda', dtype=torch.int32)[:, ::factor]
                    lens = torch.empty(bs * factor, device='cuda', dtype=dtype)[::factor]
                    for length in sorted(set([0, 1, min(511, lmax), min(512, lmax), lmax])):
                        lens.fill_(length)
                        reference = cpu_oracle(idx, slots, lens, 512)
                        baseline = torch.empty_like(out)
                        torch_finish(idx.to(torch.int64), slots, lens, baseline)
                        self.assertTrue(torch.equal(baseline.cpu(), reference))
                        for warps in (4, 8):
                            out.fill_(123)
                            fused_finish(idx, slots, lens, out, warps=warps)
                            self.assertTrue(torch.equal(out.cpu(), reference), (bs, k, lmax, dtype, strided, length, warps))
                            cases += 1
        print(json.dumps({'numeric': 'exact PASS', 'geometry_cases': cases}), flush=True)

    def test_graph_replay_changes_indices_slots_and_lengths(self):
        idx = torch.randint(65536, (1, 512), device='cuda', dtype=torch.int64)
        slots = torch.arange(65536, device='cuda', dtype=torch.int64)[None, :] * 3
        lens = torch.tensor([65536], device='cuda', dtype=torch.int32)
        out = torch.empty((1, 512), device='cuda', dtype=torch.int32)
        for warps in (4, 8):
            for _ in range(3):
                fused_finish(idx, slots, lens, out, warps=warps)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_finish(idx, slots, lens, out, warps=warps)
            for length in (0, 1, 511, 512, 32000, 65536):
                idx.copy_(torch.randint(65536, idx.shape, device='cuda', dtype=idx.dtype))
                idx[:, :4] = 0
                slots.add_(7)
                lens.fill_(length)
                out.fill_(123)
                graph.replay()
                self.assertTrue(torch.equal(out.cpu(), cpu_oracle(idx, slots, lens, 512)))
        print(json.dumps({'graph_replay': 'exact PASS', 'changed_inputs': ['idx', 'slots', 'lens'], 'replays': 12}), flush=True)


def graph_time(fn, calls, repeats):
    # CUDA capture requires a nondefault stream. Warm and capture on exactly
    # that stream, so Torch sort/gather temporary state is initialized there.
    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(caller)
    graph = torch.cuda.CUDAGraph()
    timings = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                fn()
        for _ in range(10):
            graph.replay()
        stream.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            stop.record(stream)
            stop.synchronize()
            timings.append(start.elapsed_time(stop) * 1000 / calls)
    caller.wait_stream(stream)
    return {'median_us': statistics.median(timings), 'samples_us': timings}


def benchmark(calls, repeats):
    torch.manual_seed(201)
    for lmax in (4096, 65536):
        idx = torch.randint(lmax, (1, 512), device='cuda', dtype=torch.int64)
        slots = torch.randint(0, 2000000, (1, lmax), device='cuda', dtype=torch.int64)
        lens = torch.tensor([lmax], device='cuda', dtype=torch.int32)
        out = torch.empty((1, 512), device='cuda', dtype=torch.int32)
        reference = cpu_oracle(idx, slots, lens, 512)
        baseline = graph_time(lambda: torch_finish(idx, slots, lens, out), calls, repeats)
        assert torch.equal(out.cpu(), reference)
        for warps in (4, 8):
            fused = graph_time(lambda: fused_finish(idx, slots, lens, out, warps=warps), calls, repeats)
            assert torch.equal(out.cpu(), reference)
            print(json.dumps({'benchmark': 'selected_topk_finish_only', 'device': torch.cuda.get_device_name(),
                'shape': [1, 512], 'lmax': lmax, 'warps': warps, 'calls_per_graph': calls,
                'baseline': baseline, 'fused': fused,
                'speedup': baseline['median_us'] / fused['median_us'],
                'projected_saved_ms_per_8_owners': (baseline['median_us'] - fused['median_us']) * 8 / 1000,
                'scope': 'excludes scoring/topk/reset; local projection, not whole-model speedup'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--calls', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestFinish))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.benchmark:
        benchmark(args.calls, args.repeats)
