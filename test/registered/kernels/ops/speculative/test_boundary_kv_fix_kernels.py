from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

"""Boundary-KV fix kernels (SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX) vs a pure-torch reference.

Covers the three pieces behind the pool-free chain-MTP boundary-KV exactness
fix (widened draft-extend windows that rewrite draft KV rows keyed on rejected
chain proposals with their committed keys):
  - compute_widened_draft_extend_locs_positions: batched out_cache_loc /
    positions for the widened window (runs BEFORE the forward batch is built;
    data-invalid rows zeroed, conv warm-up rows routed to the sacrificial
    cache slot 0, sacrificial zone capped at the front);
  - fill_widened_draft_extend_inputs: widened depth-0 window token/hidden
    materialization (stash fronts | predict rows);
  - stash_append_boundary_state: rolling per-request (token, base-hidden)
    stash append, in both decode (accumulate valid_len) and prefill-seed
    (SET valid_len, varlen sources) modes.
"""

import unittest

import torch

from sglang.srt.speculative.multi_layer_eagle_utils import (
    compute_widened_draft_extend_locs_positions,
    fill_widened_draft_extend_inputs_triton,
    stash_append_boundary_state_triton,
)
from sglang.test.test_utils import CustomTestCase

DEV = "cuda"

# (bs, W, front, warmup, hidden, seq_lens, valid, accept)
CASES = [
    # steady state: fully seeded stash, long sequences
    (4, 4, 5, 3, 64, [100, 33, 200, 17], [5, 5, 5, 5], [1, 4, 2, 3]),
    # fresh/short: partially seeded stash, seq_lens < front (degenerate rows)
    (4, 4, 5, 3, 64, [3, 1, 6, 2], [2, 0, 5, 1], [1, 1, 4, 2]),
    # d66-like 8-step chain shape without conv warm-up
    (2, 9, 7, 0, 128, [50, 9], [7, 3], [9, 1]),
    # minimal chain: 2 steps, W=3, front=1
    (3, 3, 1, 0, 32, [10, 11, 12], [1, 1, 0], [1, 2, 3]),
]


def _ref_stash_append(
    stash_t, stash_h, valid, src_t, src_h, ends, avail, rpis, set_valid
):
    front = stash_t.shape[1]
    for i, rpi in enumerate(rpis.tolist()):
        m = min(int(avail[i]), front)
        end = int(ends[i])
        keep = front - m
        stash_t[rpi, :keep] = stash_t[rpi, m:].clone()
        stash_h[rpi, :keep] = stash_h[rpi, m:].clone()
        stash_t[rpi, keep:] = src_t[end - m : end].to(stash_t.dtype)
        stash_h[rpi, keep:] = src_h[end - m : end].to(stash_h.dtype)
        if set_valid:
            valid[rpi] = m
        else:
            valid[rpi] = min(int(valid[rpi]) + m, front)


def _ref_fill_and_locs(
    predict, vhid, stash_t, stash_h, valid, seq_lens, rpis, req_to_token, W, warmup
):
    bs = rpis.shape[0]
    front = stash_t.shape[1]
    width = W + front
    h = stash_h.shape[2]
    ids = torch.zeros(bs * width, dtype=torch.int64, device=DEV)
    hid = torch.zeros(bs * width, h, dtype=stash_h.dtype, device=DEV)
    pos = torch.zeros(bs * width, dtype=torch.int64, device=DEV)
    loc = torch.zeros(bs * width, dtype=torch.int64, device=DEV)
    for i, rpi in enumerate(rpis.tolist()):
        seq_len = int(seq_lens[i])
        first_valid = max(front - int(valid[rpi]), front - seq_len, 0)
        # Sacrificial zone is capped at the front: original rows always write.
        first_real = min(first_valid + warmup, front)
        for j in range(width):
            row = i * width + j
            p = seq_len - front + j
            if j >= front:
                src = i * W + j - front
                ids[row] = predict[src]
                hid[row] = vhid[src]
                pos[row] = p
                loc[row] = req_to_token[rpi, p]
            else:
                if j >= first_valid:
                    ids[row] = stash_t[rpi, j]
                    hid[row] = stash_h[rpi, j]
                    pos[row] = p
                if j >= first_real:
                    loc[row] = req_to_token[rpi, p]
    return ids, hid, pos, loc


class TestBoundaryKvFixKernels(CustomTestCase):
    def test_kernels_vs_reference(self):
        for case_i, (
            bs,
            W,
            front,
            warmup,
            hidden,
            seq_lens,
            valid,
            accept,
        ) in enumerate(CASES):
            with self.subTest(case=case_i):
                self._run_case(bs, W, front, warmup, hidden, seq_lens, valid, accept)

    def _run_case(self, bs, W, front, warmup, hidden, seq_lens, valid, accept):
        torch.manual_seed(0)
        pool, max_ctx = 32, 512
        rpis = torch.randperm(pool, device=DEV)[:bs].to(torch.int64)
        req_to_token = (
            torch.arange(pool * max_ctx, device=DEV, dtype=torch.int32).reshape(
                pool, max_ctx
            )
            + 1000
        )
        seq_lens_t = torch.tensor(seq_lens, device=DEV, dtype=torch.int64)

        stash_t = torch.randint(5, 900, (pool, front), device=DEV, dtype=torch.int64)
        stash_h = torch.randn(pool, front, hidden, device=DEV, dtype=torch.bfloat16)
        valid_t = torch.zeros(pool, dtype=torch.int32, device=DEV)
        for i, rpi in enumerate(rpis.tolist()):
            valid_t[rpi] = valid[i]

        predict = torch.randint(5, 900, (bs * W,), device=DEV, dtype=torch.int64)
        vhid = torch.randn(bs * W, hidden, device=DEV, dtype=torch.bfloat16)

        r_ids, r_hid, r_pos, r_loc = _ref_fill_and_locs(
            predict,
            vhid,
            stash_t,
            stash_h,
            valid_t,
            seq_lens_t,
            rpis,
            req_to_token,
            W,
            warmup,
        )

        # --- locs / positions (pre-forward-batch torch path) ---
        loc, pos = compute_widened_draft_extend_locs_positions(
            seq_lens_t,
            rpis,
            req_to_token,
            valid_t,
            draft_token_num=W,
            num_front_tokens=front,
            num_warmup_tokens=warmup,
        )
        self.assertTrue(torch.equal(loc, r_loc), "locs mismatch")
        self.assertTrue(torch.equal(pos, r_pos), "positions mismatch")

        # --- widened window token/hidden fill ---
        width = W + front
        ids = torch.zeros(bs * width, dtype=torch.int64, device=DEV)
        hid = torch.zeros(bs * width, hidden, dtype=torch.bfloat16, device=DEV)
        fill_widened_draft_extend_inputs_triton(
            ids,
            hid,
            predict,
            vhid,
            stash_t,
            stash_h,
            valid_t,
            seq_lens_t,
            rpis,
            draft_token_num=W,
        )
        self.assertTrue(torch.equal(ids, r_ids), "fill mismatch on ids")
        self.assertTrue(torch.equal(hid, r_hid), "fill mismatch on hid")

        # --- decode-style stash roll-forward (accumulating valid_len) ---
        accept_t = torch.tensor(accept, device=DEV, dtype=torch.int32)
        ends = torch.arange(bs, device=DEV, dtype=torch.int64) * W + accept_t.to(
            torch.int64
        )
        ref_t, ref_h, ref_v = stash_t.clone(), stash_h.clone(), valid_t.clone()
        _ref_stash_append(
            ref_t, ref_h, ref_v, predict, vhid, ends, accept_t, rpis, False
        )
        stash_append_boundary_state_triton(
            predict,
            vhid,
            ends,
            accept_t,
            rpis,
            stash_t,
            stash_h,
            valid_t,
            set_valid=False,
        )
        self.assertTrue(torch.equal(stash_t, ref_t), "decode stash tokens mismatch")
        self.assertTrue(torch.equal(stash_h, ref_h), "decode stash hiddens mismatch")
        self.assertTrue(torch.equal(valid_t, ref_v), "decode stash valid_len mismatch")

        # --- prefill-style seed (varlen segments, SET valid_len) ---
        lens = torch.tensor(
            [max(1, (i * 7) % (W + front)) for i in range(bs)],
            device=DEV,
            dtype=torch.int32,
        )
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, device=DEV, dtype=torch.int32), lens[:-1]]), 0
        )
        total = int(lens.sum())
        src_t = torch.randint(5, 900, (total,), device=DEV, dtype=torch.int64)
        src_h = torch.randn(total, hidden, device=DEV, dtype=torch.bfloat16)
        ends2 = (starts + lens).to(torch.int64)
        ref_t, ref_h, ref_v = stash_t.clone(), stash_h.clone(), valid_t.clone()
        _ref_stash_append(ref_t, ref_h, ref_v, src_t, src_h, ends2, lens, rpis, True)
        stash_append_boundary_state_triton(
            src_t, src_h, ends2, lens, rpis, stash_t, stash_h, valid_t, set_valid=True
        )
        self.assertTrue(torch.equal(stash_t, ref_t), "seed stash tokens mismatch")
        self.assertTrue(torch.equal(stash_h, ref_h), "seed stash hiddens mismatch")
        self.assertTrue(torch.equal(valid_t, ref_v), "seed stash valid_len mismatch")


if __name__ == "__main__":
    unittest.main()
