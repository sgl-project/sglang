# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""GPU tests for SGLANG_OPT_K31_SEAM_SMALL_COPY: the DFlash round's small seam copies done as one batched SM copy kernel.

Each case builds the exact views a served site copies (shapes, strides, dtypes), poisons every destination, runs the batched
form, and compares against values constructed independently of the code under test (index arithmetic on the sources), including
the rows and columns a copy must not touch.
"""

import unittest

import torch

from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.srt.environ import envs
from sglang.srt.speculative import dflash_utils

POISON = -7


def _ids(n, base, device):
    return torch.arange(base, base + n, dtype=torch.int64, device=device)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestSeamSmallCopy(unittest.TestCase):
    def setUp(self):
        self.dev = torch.device("cuda")

    def test_forced_acceptance_block_matches_hand_built_rows(self):
        # Bug caught: the batched forced-AL build writes the wrong bonus, misses the accepted drafts, or leaves stale tokens.
        block = 8
        for bs in (1, 2):
            self._forced(bs, block)

    def _forced(self, bs, block):
        candidates = (_ids(bs * block, 1000, self.dev)).view(bs, block)
        target = (_ids(bs * block, 5000, self.dev)).view(bs, block)
        for commit in (1, 4, block):
            k = commit - 1
            want = torch.zeros(bs, block, dtype=torch.int64)
            for r in range(bs):
                for j in range(k):
                    want[r, j] = 1000 + r * block + 1 + j
                want[r, k] = 5000 + r * block + k
            for knob in (False, True):
                self._forced_one(bs, block, commit, k, knob, candidates, target, want)

    def _forced_one(self, bs, block, commit, k, knob, candidates, target, want):
        with envs.SGLANG_OPT_K31_SEAM_SMALL_COPY.override(knob):
            out = torch.full((bs, block), POISON, dtype=torch.int64, device=self.dev)
            bonus = torch.full((bs,), POISON, dtype=torch.int64, device=self.dev)
            acc = torch.empty(bs, dtype=torch.int32, device=self.dev)
            com = torch.empty(bs, dtype=torch.int32, device=self.dev)
            orig = dflash_utils.sample_simulated_acc_len
            dflash_utils.sample_simulated_acc_len = lambda *a, **kw: commit
            try:
                dflash_utils.apply_dflash_simulated_acceptance(
                    candidates=candidates, target_predict=target, accept_len=acc, commit_lens=com,
                    bonus=bonus, out_tokens=out, simulate_acc_len=float(commit),
                    simulate_acc_method="match-expected", simulate_acc_token_mode="real-draft-token")
            finally:
                dflash_utils.sample_simulated_acc_len = orig
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out.cpu(), want), f"out_tokens knob={knob} commit={commit}")
            self.assertTrue(torch.equal(bonus.cpu(), want[:, k]), f"bonus knob={knob} commit={commit}")
            self.assertTrue(bool((acc == k).all()) and bool((com == commit).all()))

    def test_draft_token_block_views(self):
        # Bug caught: a strided column view (draft_tokens[:, 0]) or the [:, 1:] block copied with the wrong stride, or rows >= bs written.
        cap, block = 16, 8
        for bs in (1, 2, 16):
            buf = torch.full((cap, block), POISON, dtype=torch.int64, device=self.dev)
            block_ids = _ids(bs * block, 300, self.dev).view(bs, block)
            draft_next = _ids(bs * (block - 1), 9000, self.dev).view(bs, block - 1)
            d = buf[:bs]
            ok = try_small_copy([d[:, 0], d[:, 1:]], [block_ids[:, 0], draft_next])
            if bs == 1:
                self.assertTrue(ok, "bs 1 must take the batched copy")
            if not ok:  # the site's eager fallback
                d[:, 0].copy_(block_ids[:, 0])
                d[:, 1:].copy_(draft_next)
            torch.cuda.synchronize()
            want = torch.full((cap, block), POISON, dtype=torch.int64)
            for r in range(bs):
                want[r, 0] = 300 + r * block
                for j in range(block - 1):
                    want[r, 1 + j] = 9000 + r * (block - 1) + j
            self.assertTrue(torch.equal(buf.cpu(), want), f"bs={bs}")

    def test_aux_staging_and_replay_inputs(self):
        # Bug caught: mixed int64 / int32 fields batched with the wrong width, or a slice prefix overrun into the static buffers' tails.
        for n in (8, 16, 128):
            loc, pos = _ids(n, 77, self.dev), _ids(n, 4000, self.dev)
            commit = torch.arange(3, 3 + n // 8, dtype=torch.int32, device=self.dev)
            s_loc = torch.full((n + 4,), POISON, dtype=torch.int64, device=self.dev)
            s_pos = torch.full((n + 4,), POISON, dtype=torch.int64, device=self.dev)
            s_com = torch.full((n // 8,), POISON, dtype=torch.int32, device=self.dev)
            self.assertTrue(try_small_copy([s_loc[:n], s_pos[:n], s_com], [loc, pos, commit]))
            torch.cuda.synchronize()
            for got, src in ((s_loc, loc), (s_pos, pos)):
                self.assertTrue(torch.equal(got[:n].cpu(), src.cpu()))
                self.assertTrue(bool((got[n:] == POISON).all()), "tail written")
            self.assertTrue(torch.equal(s_com.cpu(), commit.cpu()))


if __name__ == "__main__":
    unittest.main()
