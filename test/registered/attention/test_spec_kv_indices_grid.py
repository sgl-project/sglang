"""2D (token-block) launches of the KV index-copy kernels must be
bit-identical to the historical 1D launches, with no unwritten or
overwritten bytes (sentinel-checked over the full buffers), and the draft
kernel's output must match a Python reference."""

import unittest

import torch
import triton

from sglang.kernels.ops.attention.utils import (
    create_flashinfer_kv_indices_triton,
    kv_indices_num_token_blocks,
)
from sglang.kernels.ops.speculative.cache_locs import (
    generate_draft_decode_kv_indices,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-small-amd")

SENTINEL = 0x7EADBEEF
POOL_LEN = 262_144
LENSETS = [
    [0, 1, 511, 512, 513, 8191, 8192, 8193],
    [100_000, 33, 4096],
    [1000] * 7 + [100_000],
]


def _npo2(x: int) -> int:
    return max(1, 1 << (max(1, x) - 1).bit_length())


def _draft_inputs(seqs, topk, steps, device, idx_dtype=torch.int64):
    bs = len(seqs)
    req_pool = torch.arange(bs, dtype=idx_dtype, device=device)
    r2t = torch.randint(
        0, 6_000_000, (bs + 1, POOL_LEN), dtype=torch.int32, device=device
    )
    lens = torch.tensor(seqs, dtype=idx_dtype, device=device)
    width = topk * (max(seqs) + steps) + 64
    return bs, req_pool, r2t, lens, width, bs * topk


def _run_draft(kern_inputs, topk, steps, page_size, nb, kw):
    bs, req_pool, r2t, lens, width, tot = kern_inputs
    dev = lens.device
    kv_i = torch.full((steps, bs * width), SENTINEL, dtype=torch.int32, device=dev)
    kv_p = torch.full((steps, tot + 1), SENTINEL, dtype=torch.int32, device=dev)
    # positions is per draft token in production (bs * topk entries); the
    # kernel reads positions[:bs * topk] for the kv_indptr prefix sums.
    positions = torch.repeat_interleave(lens, topk)
    generate_draft_decode_kv_indices[(steps * nb, bs, topk)](
        req_pool,
        r2t,
        lens,
        kv_i,
        kv_p,
        positions,
        POOL_LEN,
        kv_i.shape[1],
        kv_p.shape[1],
        _npo2(bs),
        _npo2(steps),
        _npo2(tot),
        page_size,
        **kw,
    )
    torch.cuda.synchronize()
    return kv_i, kv_p


class TestSpecKvIndicesGrid(CustomTestCase):
    def test_draft_grid_equivalence(self):
        torch.manual_seed(0)
        for seqs in LENSETS:
            for topk, page_size in [(1, 1), (4, 1), (4, 16)]:
                for steps, idx_dtype in [
                    (2, torch.int64),
                    (3, torch.int32),
                    (4, torch.int64),
                ]:
                    inputs = _draft_inputs(seqs, topk, steps, "cuda", idx_dtype)
                    ref = _run_draft(inputs, topk, steps, page_size, 1, {})
                    for nb in [
                        kv_indices_num_token_blocks(POOL_LEN, steps * len(seqs) * topk),
                        triton.cdiv(POOL_LEN, 8192),
                    ]:
                        out = _run_draft(
                            inputs, topk, steps, page_size, nb, {"NUM_STEPS": steps}
                        )
                        self.assertTrue(torch.equal(ref[0], out[0]), (seqs, topk, nb))
                        self.assertTrue(torch.equal(ref[1], out[1]), (seqs, topk, nb))

    def test_draft_reference(self):
        torch.manual_seed(1)
        seqs, steps = [100_000, 33, 4096, 16], 3
        for topk, page_size in [(1, 1), (4, 1), (4, 16)]:
            inputs = _draft_inputs(seqs, topk, steps, "cuda")
            bs, _, r2t, _, _, tot = inputs
            nb = kv_indices_num_token_blocks(POOL_LEN, steps * bs * topk)
            kv_i, kv_p = _run_draft(
                inputs, topk, steps, page_size, nb, {"NUM_STEPS": steps}
            )
            for it in range(steps):
                iters = it + 1
                for s in range(bs):
                    ln = seqs[s]
                    for k in range(topk):
                        off = sum(seqs[:s]) * topk + s * iters * topk + k * (ln + iters)
                        self.assertTrue(
                            torch.equal(r2t[s, :ln], kv_i[it, off : off + ln]),
                            (it, s, k, topk, page_size),
                        )
                        if page_size == 1 or topk == 1:
                            src = ln + k * steps
                        else:
                            last = ln % page_size
                            pages = -(-(last + steps) // page_size)
                            src = (
                                ln // page_size * page_size
                                + k * pages * page_size
                                + last
                            )
                        self.assertTrue(
                            torch.equal(
                                r2t[s, src : src + iters],
                                kv_i[it, off + ln : off + ln + iters],
                            ),
                            (it, s, k, topk, page_size),
                        )
                positions = [ln for ln in seqs for _ in range(topk)]
                for z in range(1, tot + 1):
                    self.assertEqual(
                        int(kv_p[it, z]),
                        sum(positions[:z]) + z * iters,
                        (it, z, topk, page_size),
                    )

    def test_flat_grid_equivalence(self):
        torch.manual_seed(0)
        dev = "cuda"
        for seqs in LENSETS:
            for use_start, entry_page_size in [(False, 1), (True, 1), (False, 16)]:
                bs = len(seqs)
                req_pool = torch.arange(bs, dtype=torch.int64, device=dev)
                r2t = torch.randint(
                    0, 6_000_000, (bs + 1, POOL_LEN), dtype=torch.int32, device=dev
                )
                lens = torch.tensor(
                    seqs,
                    dtype=torch.int32 if use_start else torch.int64,
                    device=dev,
                )
                indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
                indptr[1:] = torch.cumsum(lens, 0).to(torch.int32)
                start = (
                    torch.full((bs,), 7, dtype=torch.int32, device=dev)
                    if use_start
                    else None
                )
                n = int(indptr[-1]) + (7 * bs if use_start else 0) + 8
                outs = []
                for grid in [
                    (bs,),
                    (bs, kv_indices_num_token_blocks(POOL_LEN, bs)),
                    (bs, triton.cdiv(POOL_LEN, 8192)),
                ]:
                    kv_i = torch.full((n,), SENTINEL, dtype=torch.int32, device=dev)
                    create_flashinfer_kv_indices_triton[grid](
                        r2t,
                        req_pool,
                        lens,
                        indptr,
                        start,
                        kv_i,
                        r2t.shape[1],
                        ENTRY_PAGE_SIZE=entry_page_size,
                    )
                    torch.cuda.synchronize()
                    outs.append(kv_i)
                for kv_i in outs[1:]:
                    self.assertTrue(
                        torch.equal(outs[0], kv_i), (seqs, use_start, entry_page_size)
                    )


if __name__ == "__main__":
    unittest.main()
