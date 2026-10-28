"""FlashInfer's kernel over a sliding-window layer's window read, for a target
verify and a draft extend, against attention computed from positions alone.

A verify tree's rows sit at their tree depth, so the window is folded into the
tree's mask and the kernel is planned without one; a draft extend's queries
are consecutive positions, so the kernel is planned with the layer's window.
Each reads only the window, from its first token on.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import window_read
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

HEADS, DIM, WINDOW = 4, 64, 32


def _branched():
    parents = [-1, 0, 0, 1, 2, 4]
    tree = torch.zeros(len(parents), len(parents), dtype=torch.bool)
    for row, parent in enumerate(parents):
        tree[row, row] = True
        while parent >= 0:
            tree[row, parent] = True
            parent = parents[parent]
    return tree


def _reference(q, k, v, visible):
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / DIM**0.5
    scores = scores.masked_fill(~visible[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFlashInferWindowRead(CustomTestCase):
    def setUp(self):
        import flashinfer

        torch.manual_seed(0)
        self.flashinfer = flashinfer
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")

    def _run(self, seq_lens, width, read, mask, window_left, total_lens):
        """Plan and run over the window read of each request's KV, whose
        tokens sit at slots ``[base_i, base_i + total_lens_i)``."""
        bs = len(seq_lens)
        n = sum(total_lens)
        k = torch.randn(n, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        q = torch.randn(bs * width, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        bases = torch.tensor([0] + total_lens[:-1]).cumsum(0).tolist()
        lens = (read.lens + width).tolist()
        indices = torch.cat(
            [
                torch.arange(b + s, b + s + l)
                for b, s, l in zip(bases, read.start.tolist(), lens)
            ]
        ).to(torch.int32)
        kv_indptr = torch.tensor([0] + lens).cumsum(0).to(torch.int32)
        qo_indptr = torch.arange(0, (bs + 1) * width, width, dtype=torch.int32)
        wrapper = self.flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD"
        )
        wrapper.plan(
            qo_indptr.cuda(),
            kv_indptr.cuda(),
            indices.cuda(),
            torch.ones(bs, dtype=torch.int32, device="cuda"),
            HEADS,
            HEADS,
            DIM,
            1,
            custom_mask=mask,
            causal=mask is None,
            window_left=window_left,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        out = wrapper.run(q, (k, v))
        return q, k[:, 0], v[:, 0], bases, out

    def test_verify_tree_reads_its_window(self):
        """BUG REGRESSION. The verify read its window wrapper from column 0;
        and a row-indexed kernel window misplaces a branched tree's rows."""
        tree = _branched().cuda()
        width = tree.shape[0]
        seq_lens = [10, 100, 33]
        seq = torch.tensor(seq_lens, device="cuda")
        read = window_read(
            seq_lens=seq, prefix_lens=seq, window=WINDOW, prefix_only=False
        )
        full = torch.cat(
            [
                torch.cat(
                    [torch.ones(width, s, dtype=torch.bool, device="cuda"), tree], 1
                ).flatten()
                for s in seq_lens
            ]
        )
        mask = read.tree_mask(full, width=width, read_lens_sum=int(read.lens.sum()))
        total = [s + width for s in seq_lens]
        q, k, v, bases, out = self._run(seq_lens, width, read, mask, -1, total)
        depth = tree.sum(-1) - 1
        for i, s in enumerate(seq_lens):
            keys = torch.arange(s + width, device="cuda")
            pos = s + depth
            prefix = (keys[None, :] < s) & (keys[None, :] >= pos[:, None] - WINDOW)
            visible = prefix.clone()
            visible[:, s:] = tree
            sl = slice(bases[i], bases[i] + s + width)
            expected = _reference(q[i * width : (i + 1) * width], k[sl], v[sl], visible)
            torch.testing.assert_close(
                out[i * width : (i + 1) * width].float(),
                expected,
                atol=2e-2,
                rtol=2e-2,
            )

    def test_draft_extend_reads_its_window(self):
        """A draft extend's queries are its sequence's last tokens; planned
        with the window, each sees the keys from its position minus the
        window on."""
        width = 3
        seq_lens = [5, 90, 40]
        seq = torch.tensor(seq_lens, device="cuda")
        read = window_read(
            seq_lens=seq, prefix_lens=seq - width, window=WINDOW, prefix_only=False
        )
        # The read already counts the queries' own keys.
        read_prefix = type(read)(
            start=read.start,
            lens=read.lens - width,
            window=read.window,
            window_left=read.window_left,
        )
        q, k, v, bases, out = self._run(
            seq_lens, width, read_prefix, None, read.window_left, seq_lens
        )
        for i, s in enumerate(seq_lens):
            keys = torch.arange(s, device="cuda")
            pos = torch.arange(s - width, s, device="cuda")
            visible = (keys[None, :] <= pos[:, None]) & (
                keys[None, :] >= pos[:, None] - WINDOW
            )
            sl = slice(bases[i], bases[i] + s)
            expected = _reference(q[i * width : (i + 1) * width], k[sl], v[sl], visible)
            torch.testing.assert_close(
                out[i * width : (i + 1) * width].float(),
                expected,
                atol=2e-2,
                rtol=2e-2,
            )


if __name__ == "__main__":
    unittest.main()
