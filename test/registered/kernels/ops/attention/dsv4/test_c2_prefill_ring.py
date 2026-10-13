"""The ratio-2 compressor's eager-extend path against its decode path.

``c2_prefill_norm_rope_store`` pairs rows inside a chunk and parks each request's
last ``ring_size`` rows with a separate write-back kernel, while
``c2_decode_norm_rope_store`` takes every partner from the ring and parks every
pending row there. Driving one token stream both ways must leave the same main-KV
bytes and the same latents, so a chunk boundary that parks the wrong rows shows up
in the next chunk's output instead of staying invisible in the first chunk's.
"""

import unittest
from typing import List, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c2_decode_norm_rope_store,
    c2_prefill_norm_rope_store,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The c2 JIT kernels are the SM100 build; the only Blackwell runner config is the
# four-GPU one.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
NUM_PAGES = 16
EPS = 1e-6
# Request r's full slots start here, even so that a (2k, 2k+1) pair shares a
# compressed slot; never 0, which marks a padded row.
SLOT_BASE = 64


def _sm100() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


class Stream:
    """One token stream: per-request prefix and chunk sizes, laid out as the rows
    the backend would hand each call."""

    def __init__(self, prefixes: List[int], chunks: List[List[int]]):
        self.prefixes = prefixes
        self.chunks = chunks  # chunks[c][r] = rows of request r in chunk c

    def chunk_rows(self, c: int) -> List[Tuple[int, int]]:
        """(request, position) of every row of chunk ``c``, requests in order."""
        rows = []
        for r, n in enumerate(self.chunks[c]):
            start = self.prefixes[r] + sum(self.chunks[j][r] for j in range(c))
            rows.extend((r, start + i) for i in range(n))
        return rows


def _make_inputs(gen, rows: int) -> torch.Tensor:
    """``[rows, 2 * HEAD_DIM]`` fp32: the kv half then the score half."""
    return torch.randn(
        rows, 2 * HEAD_DIM, generator=gen, device="cuda", dtype=torch.float32
    )


def _row_tensors(rows: List[Tuple[int, int]], pad: int):
    """``req``, ``positions`` and ``raw_out_loc`` for ``rows``, plus ``pad`` padded
    rows. A padded row still has to address a live ring slot, so it takes request 0
    and position 0; only ``raw_out_loc == 0`` marks it."""
    req = [r for r, _ in rows] + [0] * pad
    pos = [p for _, p in rows] + [0] * pad
    loc = [SLOT_BASE * (1 + r) + p for r, p in rows] + [0] * pad
    to = lambda v, d: torch.tensor(v, device="cuda", dtype=d)
    return to(req, torch.int64), to(pos, torch.int32), to(loc, torch.int32)


class TestC2PrefillRing(CustomTestCase):
    @unittest.skipUnless(_sm100(), "the c2 JIT kernels are the SM100 build")
    def test_graph_replay_global_offsets_padding_and_ring_carry(self):
        # CP materializes request-major global rows before compression. The
        # request axis must stay fixed even when a token bucket replays B=1/5/1.
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            with self.subTest(layout=layout.name):
                self._run_graph_replay(layout)

    def _run_graph_replay(self, layout):
        gen = torch.Generator(device="cuda").manual_seed(19)
        bucket, capacity, ring_size = 32, 8, 8
        inputs = _make_inputs(gen, bucket)
        weight = torch.randn(
            HEAD_DIM, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        angles = torch.randn(128, ROPE_DIM // 2, generator=gen, device="cuda")
        freqs = torch.view_as_real(
            torch.polar(torch.ones_like(angles), angles)
        ).flatten(-2)
        state = _make_inputs(gen, capacity * ring_size + 1)
        eager_state = state.clone()
        cache = torch.zeros(
            NUM_PAGES, layout.page_bytes(PAGE_SIZE), device="cuda", dtype=torch.uint8
        )
        eager_cache = cache.clone()
        req = torch.zeros(bucket, device="cuda", dtype=torch.int64)
        pos = torch.zeros(bucket, device="cuda", dtype=torch.int64)
        loc = torch.zeros(bucket, device="cuda", dtype=torch.int64)
        starts = torch.zeros(capacity, device="cuda", dtype=torch.int32)
        lengths = torch.zeros_like(starts)
        latent = torch.full(
            (bucket, HEAD_DIM), 123, device="cuda", dtype=torch.bfloat16
        )

        def run(ring, dst_cache, offsets, out=None):
            return c2_prefill_norm_rope_store(
                inputs,
                ring,
                weight,
                pos,
                req,
                loc,
                EPS,
                freqs,
                dst_cache,
                *offsets,
                page_size=PAGE_SIZE,
                ring_size=ring_size,
                layout=layout,
                out=out,
            )

        # Warm on a side stream before capturing; all rows initially pad.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(state, cache, (starts, lengths), latent)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(state, cache, (starts, lengths), latent)

        request_ids = [4, 1, 6, 2, 0]
        next_pos = {4: 1, 1: 8, 6: 7, 2: 0, 0: 3}
        for sizes in ([4], [9, 0, 3, 4, 2], [5], []):
            rows, begin = [], []
            for rid, n in zip(request_ids, sizes):
                begin.append(len(rows))
                rows.extend((rid, next_pos[rid] + i) for i in range(n))
                next_pos[rid] += n
            n = len(rows)
            req.zero_()
            pos.zero_()
            loc.zero_()
            starts.zero_()
            lengths.zero_()
            if n:
                req[:n].copy_(torch.tensor([r for r, _ in rows], device="cuda"))
                pos[:n].copy_(torch.tensor([p for _, p in rows], device="cuda"))
                loc[:n].copy_(
                    torch.tensor(
                        [SLOT_BASE * (1 + r) + p for r, p in rows], device="cuda"
                    )
                )
            m = len(sizes)
            starts[:m].copy_(torch.tensor(begin, device="cuda", dtype=torch.int32))
            lengths[:m].copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
            inputs.copy_(_make_inputs(gen, bucket))
            before = eager_state.clone()
            latent.fill_(123)
            graph.replay()
            eager = run(eager_state, eager_cache, (starts, lengths))
            self.assertTrue(torch.equal(cache, eager_cache))
            self.assertTrue(torch.equal(state, eager_state))
            odd = [i for i, (_, p) in enumerate(rows) if p % 2]
            self.assertTrue(torch.equal(latent[odd], eager[odd]))
            untouched = [i for i in range(bucket) if i not in odd]
            self.assertTrue(torch.all(latent[untouched] == 123).item())

            # Independently check Torch's pair selection/softmax/finish, not
            # only another invocation of the fused kernel. FP32 reduction order
            # can cross a bf16 rounding boundary, so do not require byte equality.
            for i in odd:
                rid, p = rows[i]
                previous = (
                    inputs[i - 1]
                    if i and rows[i - 1] == (rid, p - 1)
                    else before[rid * ring_size + (p - 1) % ring_size]
                )
                pair = torch.stack((previous, inputs[i]))
                kv, score = pair[:, :HEAD_DIM], pair[:, HEAD_DIM:]
                pooled = (kv * score.softmax(0)).sum(0).to(torch.bfloat16).float()
                expected = (
                    pooled * torch.rsqrt(pooled.square().mean() + EPS) * weight.float()
                ).to(torch.bfloat16)
                torch.testing.assert_close(latent[i], expected, rtol=0.016, atol=0.016)
            # The ring contains exactly each request's most recent rows, even
            # when a request has more new rows than its ring capacity.
            for start, size in zip(begin, sizes):
                for i in range(start + max(0, size - ring_size), start + size):
                    rid, p = rows[i]
                    torch.testing.assert_close(
                        state[rid * ring_size + p % ring_size],
                        inputs[i],
                        rtol=0,
                        atol=0,
                    )

    def _run(self, stream: Stream, layout: KVLayout, ring_size: int, seed: int):
        """The fused chunked path and the row-at-a-time decode path over the same
        stream; returns (cache, latents) for each."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        num_reqs = len(stream.prefixes)
        weight = torch.randn(
            HEAD_DIM, generator=gen, device="cuda", dtype=torch.float32
        ).to(torch.bfloat16)
        angles = torch.randn(4096, ROPE_DIM // 2, generator=gen, device="cuda")
        freqs = torch.view_as_real(
            torch.polar(torch.ones_like(angles), angles)
        ).flatten(-2)

        # One fp32 input row per (request, position) in the stream, shared by both
        # paths so any difference is the pairing, not the data.
        all_rows = [
            row for c in range(len(stream.chunks)) for row in stream.chunk_rows(c)
        ]
        inputs = {row: _make_inputs(gen, 1)[0] for row in all_rows}

        state_rows = num_reqs * ring_size + ring_size
        caches, latents = [], []
        for fused in (True, False):
            cache = torch.zeros(
                NUM_PAGES,
                layout.page_bytes(PAGE_SIZE),
                dtype=torch.uint8,
                device="cuda",
            )
            state = torch.zeros(
                state_rows, 2 * HEAD_DIM, dtype=torch.float32, device="cuda"
            )
            out = {}
            if fused:
                for c in range(len(stream.chunks)):
                    rows = stream.chunk_rows(c)
                    pad = 2
                    req, pos, loc = _row_tensors(rows, pad)
                    kv = torch.stack(
                        [inputs[row] for row in rows]
                        + [torch.zeros(2 * HEAD_DIM, device="cuda")] * pad
                    )
                    starts, lens = [], []
                    at = 0
                    for r in range(num_reqs):
                        starts.append(at)
                        lens.append(stream.chunks[c][r])
                        at += stream.chunks[c][r]
                    latent = c2_prefill_norm_rope_store(
                        kv,
                        state,
                        weight,
                        pos,
                        req,
                        loc,
                        EPS,
                        freqs,
                        cache,
                        torch.tensor(starts, device="cuda", dtype=torch.int32),
                        torch.tensor(lens, device="cuda", dtype=torch.int32),
                        page_size=PAGE_SIZE,
                        ring_size=ring_size,
                        layout=layout,
                    )
                    for i, row in enumerate(rows):
                        out[row] = latent[i].clone()
            else:
                for row in all_rows:
                    req, pos, loc = _row_tensors([row], 0)
                    latent = c2_decode_norm_rope_store(
                        inputs[row][None, :],
                        state,
                        weight,
                        pos,
                        req,
                        loc,
                        EPS,
                        freqs,
                        cache,
                        page_size=PAGE_SIZE,
                        ring_size=ring_size,
                        layout=layout,
                    )
                    out[row] = latent[0].clone()
            caches.append(cache)
            # Only a completing (odd position) row writes a latent.
            latents.append(
                torch.stack([out[row] for row in all_rows if row[1] % 2 == 1])
            )
        return caches, latents

    @unittest.skipUnless(_sm100(), "the c2 JIT kernels are the SM100 build")
    def test_chunked_extend_matches_decode(self):
        # Request 0 starts even, request 1 odd and spans more rows than the ring,
        # request 2 odd with a single row in its first chunk.
        stream = Stream(
            prefixes=[0, 7, 1],
            chunks=[[5, 4, 1], [3, 6, 2], [1, 1, 1]],
        )
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            for ring_size in (2, 8):
                with self.subTest(layout=layout.name, ring_size=ring_size):
                    caches, latents = self._run(stream, layout, ring_size, seed=7)
                    self.assertTrue(
                        torch.equal(caches[0], caches[1]),
                        f"main-KV bytes differ: {(caches[0] != caches[1]).sum()} of "
                        f"{caches[0].numel()}",
                    )
                    self.assertTrue(torch.equal(latents[0], latents[1]))


if __name__ == "__main__":
    unittest.main()
