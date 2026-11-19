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

# The c2 JIT kernels serve SM90 and SM100; the only Blackwell runner config is the
# four-GPU one.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
NUM_PAGES = 16
EPS = 1e-6
# Request r's full slots start here, even so that a (2k, 2k+1) pair shares a
# compressed slot; never 0, which marks a padded row.
SLOT_BASE = 64


def _fused_c2_device() -> bool:
    return is_cuda() and torch.cuda.get_device_capability()[0] in (9, 10)


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

    @unittest.skipUnless(_fused_c2_device(), "the c2 JIT kernels need SM90 or SM100")
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
